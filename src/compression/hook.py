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
from types import SimpleNamespace

import torch


class CompressedVLM:
    """Wrap a Qwen2.5-VL model + processor to apply visual token compression."""

    def __init__(self, model, processor, compressor, enable_debug=False):
        self.model = model
        self.processor = processor
        self.compressor = compressor
        self.image_token_id = self._resolve_image_token_id()
        self.spatial_merge_size = self._resolve_spatial_merge_size()
        self.enable_debug = enable_debug
        self.last_debug = {}

    @torch.no_grad()
    def generate(self, inputs, **gen_kwargs):
        """Run generate with optional visual token compression."""
        self.last_debug = {}
        if self.compressor is None or inputs.get("pixel_values") is None:
            if self.enable_debug:
                self.last_debug = self._baseline_debug(inputs)
            return self.model.generate(**inputs, **gen_kwargs)
        return self._generate_with_compression(inputs, **gen_kwargs)

    @torch.no_grad()
    def _generate_with_compression(self, inputs, **gen_kwargs):
        prepared, compressed_per_image = self._prepare_compressed_inputs(inputs)
        target, restore_state = self._patch_get_image_features(compressed_per_image)
        try:
            out = self.model.generate(**prepared, **gen_kwargs)
        finally:
            self._unpatch_get_image_features(target, restore_state)
        # Slice off the prompt portion using the PREPARED input length.
        # We rewrite input_ids so each image_pad run shrinks from old_lens[i]
        # to new_lens[i], so prepared.shape[1] != original.shape[1] whenever
        # any compression actually happens. Returning only the new tokens
        # gives callers a stable contract regardless of compression ratio:
        # they should NOT slice with the original input_ids length.
        prepared_len = prepared["input_ids"].shape[1]
        return out[:, prepared_len:]

    # --- core: precompute compressed embeds + reshape input_ids/grid_thw ---
    @torch.no_grad()
    def _prepare_compressed_inputs(self, inputs):
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        pixel_values = inputs["pixel_values"]
        image_grid_thw = inputs["image_grid_thw"]
        debug = self._input_debug(inputs) if self.enable_debug else {}

        # Run vision tower once and split per-image
        per_image = self._compute_image_embeds(pixel_values, image_grid_thw)
        if self.enable_debug:
            debug["visual_embedding_shapes_before_compression"] = [
                list(emb.shape) for emb in per_image
            ]

        merge = self.spatial_merge_size
        old_lens = (image_grid_thw.prod(dim=-1) // (merge * merge)).tolist()
        if self.enable_debug:
            debug["visual_tokens_before_per_image"] = [int(x) for x in old_lens]
            debug["visual_tokens_before_total"] = int(sum(old_lens))

        # Compress each image's visual tokens
        compressed_per_image = [
            self.compressor.compress(emb.unsqueeze(0)).squeeze(0) for emb in per_image
        ]
        if self.enable_debug:
            debug["visual_embedding_shapes_after_compression_raw"] = [
                list(emb.shape) for emb in compressed_per_image
            ]

        # Cast each per-image tensor to model dtype/device so the model's
        # downstream masked_scatter / cat / .to() in forward all succeed.
        target_dtype = self.model.get_input_embeddings().weight.dtype
        compressed_per_image = [
            emb.to(self.model.device, dtype=target_dtype)
            for emb in compressed_per_image
        ]
        new_lens = [emb.shape[0] for emb in compressed_per_image]
        if self.enable_debug:
            debug["visual_tokens_after_per_image"] = [int(x) for x in new_lens]
            debug["visual_tokens_after_total"] = int(sum(new_lens))
            debug["visual_embedding_shapes_after_compression"] = [
                list(emb.shape) for emb in compressed_per_image
            ]

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
        if self.enable_debug:
            debug.update(self._prepared_debug(prepared))
            debug["effective_retention_ratio"] = (
                debug["visual_tokens_after_total"] / debug["visual_tokens_before_total"]
                if debug["visual_tokens_before_total"]
                else None
            )
            debug["nominal_retention_ratio"] = float(
                getattr(self.compressor, "retention_ratio", 1.0)
            )
            debug["compression_method"] = type(self.compressor).__name__
            debug["input_sequence_length_delta"] = (
                debug["prepared_input_ids_shape"][1] - debug["input_ids_shape"][1]
            )
            self.last_debug = debug
        return prepared, compressed_per_image

    def inspect_prepared_inputs(self, inputs):
        """Return the actual inputs that would enter generate plus debug metadata.

        This is intended for diagnostics. For compressed runs it executes the
        same preparation path as ``generate`` up to, but not including, the
        final LLM ``generate`` call. For baseline it returns the original
        inputs and a comparable debug dictionary.
        """
        old_debug = self.enable_debug
        self.enable_debug = True
        try:
            if self.compressor is None or inputs.get("pixel_values") is None:
                debug = self._baseline_debug(inputs)
                self.last_debug = debug
                return inputs, [], debug
            prepared, compressed_per_image = self._prepare_compressed_inputs(inputs)
            return prepared, compressed_per_image, dict(self.last_debug)
        finally:
            self.enable_debug = old_debug

    # --- monkey-patch helpers ---
    def _patch_get_image_features(self, compressed_per_image):
        """Patch the inner Qwen2.5-VL model so get_image_features returns
        our precomputed compressed embeddings.

        Different transformers versions call this differently:
          * old: ``image_embeds = self.get_image_features(pv, grid)``  -> tensor
          * new: ``image_embeds = self.get_image_features(pv, grid, return_dict=True).pooler_output``
                 followed by ``torch.cat(image_embeds, dim=0)``

        We support both: when return_dict=True we return an object with a
        ``pooler_output`` attribute set to the per-image tuple; otherwise
        we return the concatenated tensor.
        """
        target = self._find_get_image_features_owner()
        per_image_tuple = tuple(compressed_per_image)
        cat_tensor = torch.cat(compressed_per_image, dim=0) if compressed_per_image \
            else torch.empty(0)

        def patched(*args, **kwargs):
            if kwargs.get("return_dict", False):
                # New transformers path: callsite does
                #   .pooler_output -> torch.cat(..., dim=0)
                # so pooler_output must be an iterable of per-image tensors.
                return SimpleNamespace(pooler_output=per_image_tuple)
            # Legacy path: callsite expects a single concatenated tensor.
            return cat_tensor

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
        """Return per-image merged visual embeddings as a list of (N_i, hidden) tensors.

        This calls the vision tower (and merger if needed) directly. We
        deliberately bypass ``model.get_image_features`` because:
          * On new transformers it returns a wrapper object whose internal
            shape varies across versions, and earlier defensive isinstance
            checks caused us to retry the call up to 3 times -- each retry
            re-runs the vision tower and accounted for the entire 325ms
            wrapper overhead seen in Section 8b.
          * Calling visual + merger directly is exactly what
            get_image_features does internally, so we don't lose
            functionality.
        Result: vision tower runs exactly ONCE per call.
        """
        merge = self.spatial_merge_size
        post_merge_sizes = (image_grid_thw.prod(dim=-1) // (merge * merge)).tolist()

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

    # --- diagnostics helpers ---
    @staticmethod
    def _shape(value):
        return list(value.shape) if isinstance(value, torch.Tensor) else None

    def _input_debug(self, inputs):
        input_ids = inputs.get("input_ids")
        attention_mask = inputs.get("attention_mask")
        image_grid_thw = inputs.get("image_grid_thw")
        pixel_values = inputs.get("pixel_values")
        debug = {
            "input_ids_shape": self._shape(input_ids),
            "attention_mask_shape": self._shape(attention_mask),
            "image_grid_thw_shape": self._shape(image_grid_thw),
            "pixel_values_shape": self._shape(pixel_values),
            "spatial_merge_size": int(self.spatial_merge_size),
        }
        if input_ids is not None:
            debug["input_sequence_length"] = int(input_ids.shape[1])
            debug["image_placeholder_count_total"] = int(
                (input_ids == self.image_token_id).sum().item()
            )
        if attention_mask is not None:
            debug["attention_mask_active_tokens"] = int(attention_mask.sum().item())
        if image_grid_thw is not None:
            debug["image_grid_thw"] = image_grid_thw.detach().cpu().tolist()
        return debug

    def _prepared_debug(self, prepared):
        input_ids = prepared.get("input_ids")
        attention_mask = prepared.get("attention_mask")
        image_grid_thw = prepared.get("image_grid_thw")
        debug = {
            "prepared_input_ids_shape": self._shape(input_ids),
            "prepared_attention_mask_shape": self._shape(attention_mask),
            "prepared_image_grid_thw_shape": self._shape(image_grid_thw),
        }
        if input_ids is not None:
            debug["prepared_sequence_length"] = int(input_ids.shape[1])
            debug["prepared_image_placeholder_count_total"] = int(
                (input_ids == self.image_token_id).sum().item()
            )
        if attention_mask is not None:
            debug["prepared_attention_mask_active_tokens"] = int(attention_mask.sum().item())
        if image_grid_thw is not None:
            debug["prepared_image_grid_thw"] = image_grid_thw.detach().cpu().tolist()
        return debug

    def _baseline_debug(self, inputs):
        debug = self._input_debug(inputs)
        debug.update(
            {
                "compression_method": "none",
                "nominal_retention_ratio": 1.0,
                "effective_retention_ratio": 1.0,
                "visual_tokens_before_per_image": [],
                "visual_tokens_after_per_image": [],
            }
        )
        image_grid_thw = inputs.get("image_grid_thw")
        if image_grid_thw is not None:
            merge = self.spatial_merge_size
            lens = (image_grid_thw.prod(dim=-1) // (merge * merge)).tolist()
            debug["visual_tokens_before_per_image"] = [int(x) for x in lens]
            debug["visual_tokens_after_per_image"] = [int(x) for x in lens]
            debug["visual_tokens_before_total"] = int(sum(lens))
            debug["visual_tokens_after_total"] = int(sum(lens))
        else:
            count = debug.get("image_placeholder_count_total")
            debug["visual_tokens_before_total"] = count
            debug["visual_tokens_after_total"] = count
        debug.update(
            {
                "prepared_input_ids_shape": debug.get("input_ids_shape"),
                "prepared_attention_mask_shape": debug.get("attention_mask_shape"),
                "prepared_image_grid_thw_shape": debug.get("image_grid_thw_shape"),
                "prepared_sequence_length": debug.get("input_sequence_length"),
                "prepared_attention_mask_active_tokens": debug.get(
                    "attention_mask_active_tokens"
                ),
                "prepared_image_placeholder_count_total": debug.get(
                    "image_placeholder_count_total"
                ),
                "input_sequence_length_delta": 0,
            }
        )
        return debug
