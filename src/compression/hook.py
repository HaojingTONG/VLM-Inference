"""Hook the visual token compressor into the Qwen2.5-VL forward pass.

Strategy: pre-compute compressed visual embeddings, then call
``model.generate`` with the original input_ids/pixel_values flow, but
monkey-patch ``get_image_features`` so the inner model receives our
already-compressed embeddings. This keeps M-RoPE / get_rope_index in
the trained code path (it operates on input_ids + image_grid_thw), so
we get the same per-token cost reduction as if the image had truly
fewer visual tokens.

To make the math match we also:
  * Rewrite input_ids / attention_mask so each image_pad run shrinks
    from the original count to the compressed count.
  * Synthesize a new image_grid_thw whose ``prod / merge**2`` equals
    the new per-image token count, so the model's count check and
    M-RoPE position assignment line up.

Falls back to the older inputs_embeds path is *not* implemented — the
new path is strictly better (preserves M-RoPE, reuses the trained
codepath).
"""

import math
import torch


class CompressedVLM:
    """Wrap a Qwen2.5-VL model + processor to apply visual token compression."""

    def __init__(self, model, processor, compressor):
        self.model = model
        self.processor = processor
        self.compressor = compressor
        self.image_token_id = self._resolve_image_token_id()
        self.spatial_merge_size = self._resolve_spatial_merge_size()

    @torch.no_grad()
    def generate(self, inputs, **gen_kwargs):
        """Run generate with optional visual token compression."""
        if self.compressor is None or inputs.get("pixel_values") is None:
            return self.model.generate(**inputs, **gen_kwargs)
        return self._generate_with_compression(inputs, **gen_kwargs)

    @torch.no_grad()
    def _generate_with_compression(self, inputs, **gen_kwargs):
        prepared, precomputed_embeds = self._prepare_compressed_inputs(inputs)
        target, restore_state = self._patch_get_image_features(precomputed_embeds)
        try:
            return self.model.generate(**prepared, **gen_kwargs)
        finally:
            self._unpatch_get_image_features(target, restore_state)

    # --- core: precompute compressed embeds + reshape input_ids/grid_thw ---
    @torch.no_grad()
    def _prepare_compressed_inputs(self, inputs):
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        pixel_values = inputs["pixel_values"]
        image_grid_thw = inputs["image_grid_thw"]

        # Run vision tower once and split per-image
        per_image = self._compute_image_embeds(pixel_values, image_grid_thw)

        merge = self.spatial_merge_size
        old_lens = (image_grid_thw.prod(dim=-1) // (merge * merge)).tolist()

        # Compress each image's visual tokens
        compressed_per_image = [
            self.compressor.compress(emb.unsqueeze(0)).squeeze(0) for emb in per_image
        ]
        new_lens = [emb.shape[0] for emb in compressed_per_image]
        new_image_embeds = torch.cat(compressed_per_image, dim=0)

        # Cast to model dtype/device so masked_scatter inside forward succeeds
        target_dtype = self.model.get_input_embeddings().weight.dtype
        new_image_embeds = new_image_embeds.to(self.model.device, dtype=target_dtype)

        # Shrink each image_pad run in input_ids/attention_mask to match
        new_input_ids, new_attention_mask = self._rewrite_image_spans(
            input_ids, attention_mask, old_lens, new_lens
        )

        # Synthesize a grid_thw that satisfies prod/merge**2 == new_lens[i]
        new_image_grid_thw = self._synthesize_grid_thw(
            new_lens, old_lens, image_grid_thw
        )

        prepared = {
            "input_ids": new_input_ids,
            "attention_mask": new_attention_mask,
            # pixel_values must stay non-None so the model's forward calls
            # get_image_features (which we have patched). Its actual contents
            # are ignored by the patched function.
            "pixel_values": pixel_values,
            "image_grid_thw": new_image_grid_thw,
        }
        return prepared, new_image_embeds

    # --- monkey-patch helpers ---
    def _patch_get_image_features(self, precomputed_embeds):
        """Patch the inner Qwen2.5-VL model so get_image_features returns precomputed embeds.

        We patch on the instance level so we don't disturb the class.
        """
        target = self._find_get_image_features_owner()

        # Closure-captured precomputed embeds; ignore positional args from caller
        def patched(*args, **kwargs):
            return precomputed_embeds

        had_instance_attr = "get_image_features" in target.__dict__
        original_attr = target.__dict__.get("get_image_features")
        target.get_image_features = patched
        return target, (had_instance_attr, original_attr)

    def _unpatch_get_image_features(self, target, state):
        had_instance_attr, original_attr = state
        if had_instance_attr:
            target.get_image_features = original_attr
        else:
            try:
                delattr(target, "get_image_features")
            except AttributeError:
                pass

    def _find_get_image_features_owner(self):
        # Forward goes through the inner model (Qwen2_5_VLModel), so patch
        # the deepest module that exposes get_image_features.
        candidates = [getattr(self.model, "model", None), self.model]
        for c in candidates:
            if c is not None and callable(getattr(c, "get_image_features", None)):
                return c
        raise AttributeError("Could not locate get_image_features on model.")

    # --- input_ids/attention_mask rewriting ---
    def _rewrite_image_spans(self, input_ids, attention_mask, old_lens, new_lens):
        """Replace each run of image_token_id (length old_lens[i]) with new_lens[i] tokens."""
        batch, _ = input_ids.shape
        rebuilt_ids, rebuilt_masks = [], []

        cursor_per_batch = 0
        for b in range(batch):
            ids = input_ids[b]
            mask = attention_mask[b]
            spans = self._find_image_spans(ids)

            pieces_ids, pieces_mask = [], []
            cursor = 0
            for span_i, (start, end) in enumerate(spans):
                pieces_ids.append(ids[cursor:start])
                pieces_mask.append(mask[cursor:start])

                idx = span_i if batch == 1 else cursor_per_batch + span_i
                n_new = new_lens[idx]
                pieces_ids.append(
                    torch.full((n_new,), self.image_token_id, dtype=ids.dtype, device=ids.device)
                )
                pieces_mask.append(torch.ones(n_new, dtype=mask.dtype, device=mask.device))
                cursor = end
            pieces_ids.append(ids[cursor:])
            pieces_mask.append(mask[cursor:])

            rebuilt_ids.append(torch.cat(pieces_ids))
            rebuilt_masks.append(torch.cat(pieces_mask))
            cursor_per_batch += len(spans)

        max_len = max(x.shape[0] for x in rebuilt_ids)
        padded_ids = torch.zeros((batch, max_len), dtype=input_ids.dtype, device=input_ids.device)
        padded_mask = torch.zeros((batch, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        for b, (ids_b, mask_b) in enumerate(zip(rebuilt_ids, rebuilt_masks)):
            n = ids_b.shape[0]
            padded_ids[b, :n] = ids_b
            padded_mask[b, :n] = mask_b
        return padded_ids, padded_mask

    def _find_image_spans(self, ids):
        is_img = (ids == self.image_token_id)
        spans = []
        seq_len = ids.shape[0]
        i = 0
        while i < seq_len:
            if is_img[i]:
                start = i
                while i < seq_len and is_img[i]:
                    i += 1
                spans.append((start, i))
            else:
                i += 1
        return spans

    # --- grid_thw synthesis ---
    @staticmethod
    def _factor_pair(k):
        """Return (a, b) with a*b == k, a <= b, a as close to sqrt(k) as possible."""
        if k <= 0:
            return 1, 1
        a = int(math.isqrt(k))
        while a > 0 and k % a != 0:
            a -= 1
        if a == 0:
            a = 1
        return a, k // a

    def _synthesize_grid_thw(self, new_lens, old_lens, original_grid_thw):
        """Build (B, 3) grid_thw so that prod/merge**2 == new_lens[i].

        If new_lens[i] == old_lens[i] (no compression on that image), keep the
        original grid_thw[i] to preserve exact M-RoPE positions.
        """
        merge = self.spatial_merge_size
        rows = []
        for i, k in enumerate(new_lens):
            if k == old_lens[i]:
                rows.append(original_grid_thw[i].tolist())
                continue
            T = int(original_grid_thw[i, 0].item())
            if T == 1 or (k % T) != 0:
                a, b = self._factor_pair(k)
                rows.append([1, a * merge, b * merge])
            else:
                per_frame = k // T
                a, b = self._factor_pair(per_frame)
                rows.append([T, a * merge, b * merge])
        return torch.tensor(
            rows, dtype=original_grid_thw.dtype, device=original_grid_thw.device
        )

    # --- vision tower path used during _prepare_compressed_inputs ---
    @torch.no_grad()
    def _compute_image_embeds(self, pixel_values, image_grid_thw):
        """Return per-image merged visual embeddings as a list of (N_i, hidden) tensors."""
        merge = self.spatial_merge_size
        post_merge_sizes = (image_grid_thw.prod(dim=-1) // (merge * merge)).tolist()

        # Path 1: model's own get_image_features (handles PatchMerger correctly)
        for obj in (self.model, getattr(self.model, "model", None)):
            if obj is None:
                continue
            fn = getattr(obj, "get_image_features", None)
            if not callable(fn):
                continue
            try:
                out = fn(pixel_values, image_grid_thw)
            except TypeError:
                continue
            if isinstance(out, (list, tuple)):
                return [t for t in out]
            if isinstance(out, torch.Tensor):
                if out.shape[0] == sum(post_merge_sizes):
                    return list(torch.split(out, post_merge_sizes))

        # Path 2: visual tower directly + merger fallback
        visual = self._get_visual_module()
        vision_dtype = next(visual.parameters()).dtype
        vis_out = visual(pixel_values.to(vision_dtype), grid_thw=image_grid_thw)
        embeds = self._unwrap_vision_output(vis_out)

        total = embeds.shape[0]
        if total == sum(post_merge_sizes):
            return list(torch.split(embeds, post_merge_sizes))

        pre_merge_sizes = image_grid_thw.prod(dim=-1).tolist()
        if total == sum(pre_merge_sizes):
            merger = getattr(visual, "merger", None)
            if merger is None:
                raise RuntimeError(
                    "Vision tower returned pre-merge tokens but no .merger submodule found."
                )
            merged = merger(embeds)
            if merged.shape[0] != sum(post_merge_sizes):
                raise RuntimeError(
                    f"After merger: got {merged.shape[0]} tokens, "
                    f"expected {sum(post_merge_sizes)}."
                )
            return list(torch.split(merged, post_merge_sizes))

        raise RuntimeError(
            f"Cannot reconcile vision output size {total} with grid_thw "
            f"(pre={sum(pre_merge_sizes)}, post={sum(post_merge_sizes)})."
        )

    @staticmethod
    def _unwrap_vision_output(out):
        if isinstance(out, torch.Tensor):
            return out
        for attr in ("last_hidden_state", "image_embeds", "hidden_states"):
            val = getattr(out, attr, None)
            if isinstance(val, torch.Tensor):
                return val
        if isinstance(out, (tuple, list)) and len(out) > 0 and isinstance(out[0], torch.Tensor):
            return out[0]
        raise TypeError(f"Unexpected vision tower output type: {type(out)}")

    def _get_visual_module(self):
        if hasattr(self.model, "visual"):
            return self.model.visual
        if hasattr(self.model, "model") and hasattr(self.model.model, "visual"):
            return self.model.model.visual
        raise AttributeError("Could not locate the vision tower on the model (expected .visual).")

    # --- config resolution helpers ---
    def _resolve_image_token_id(self):
        cfg = self.model.config
        for attr in ("image_token_id", "image_token_index"):
            if hasattr(cfg, attr) and getattr(cfg, attr) is not None:
                return getattr(cfg, attr)
        tok = self.processor.tokenizer
        tid = tok.convert_tokens_to_ids("<|image_pad|>")
        if tid is None or tid == tok.unk_token_id:
            raise ValueError("Could not resolve image token id from config or tokenizer.")
        return tid

    def _resolve_spatial_merge_size(self):
        vcfg = getattr(self.model.config, "vision_config", None)
        if vcfg is not None and hasattr(vcfg, "spatial_merge_size"):
            return vcfg.spatial_merge_size
        return 2  # Qwen2.5-VL default
