"""Hook the visual token compressor into the Qwen2.5-VL forward pass.

Qwen2.5-VL flow:
    pixel_values -> model.visual(..) -> image_embeds  (shape: [sum_tokens, hidden])
    input_ids (with <|image_pad|> runs) -> inputs_embeds
    inputs_embeds.masked_scatter_(image_mask, image_embeds)
    -> language model

The scatter requires sum(image_mask) == image_embeds.shape[0]. So to compress
visual tokens we must also shrink the number of <|image_pad|> slots in the
input_ids. This wrapper does that end-to-end and feeds `inputs_embeds` directly
to generate(), bypassing the model's own vision forward.
"""

import torch


class CompressedVLM:
    """Wraps a Qwen2.5-VL model + processor to apply visual token compression."""

    def __init__(self, model, processor, compressor):
        self.model = model
        self.processor = processor
        self.compressor = compressor
        self.image_token_id = self._resolve_image_token_id()
        self.spatial_merge_size = self._resolve_spatial_merge_size()

    @torch.no_grad()
    def generate(self, inputs, **gen_kwargs):
        """Run generate with optional visual token compression.

        Args:
            inputs: dict from the processor (input_ids, attention_mask,
                pixel_values, image_grid_thw, ...).
            **gen_kwargs: passed through to model.generate.
        """
        if self.compressor is None or inputs.get("pixel_values") is None:
            return self.model.generate(**inputs, **gen_kwargs)

        prepared = self._prepare_compressed_inputs(inputs)
        return self.model.generate(**prepared, **gen_kwargs)

    @torch.no_grad()
    def _prepare_compressed_inputs(self, inputs):
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        pixel_values = inputs["pixel_values"]
        image_grid_thw = inputs["image_grid_thw"]

        visual = self._get_visual_module()
        vision_dtype = next(visual.parameters()).dtype
        image_embeds = visual(pixel_values.to(vision_dtype), grid_thw=image_grid_thw)
        # image_embeds: (sum_visual_tokens, hidden_dim) after spatial merge

        merge = self.spatial_merge_size
        split_sizes = (image_grid_thw.prod(dim=-1) // (merge * merge)).tolist()
        per_image = list(torch.split(image_embeds, split_sizes))

        compressed_per_image = [
            self.compressor.compress(emb.unsqueeze(0)).squeeze(0) for emb in per_image
        ]
        new_lens = [emb.shape[0] for emb in compressed_per_image]
        new_image_embeds = torch.cat(compressed_per_image, dim=0)

        new_input_ids, new_attention_mask = self._rewrite_image_spans(
            input_ids, attention_mask, split_sizes, new_lens
        )

        inputs_embeds = self.model.get_input_embeddings()(new_input_ids)
        image_mask = (new_input_ids == self.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
        new_image_embeds = new_image_embeds.to(inputs_embeds.device, dtype=inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, new_image_embeds)

        return {
            "inputs_embeds": inputs_embeds,
            "attention_mask": new_attention_mask,
        }

    def _rewrite_image_spans(self, input_ids, attention_mask, old_lens, new_lens):
        """Replace each run of image_token_id (length old_lens[i]) with new_lens[i] tokens."""
        batch, _ = input_ids.shape
        rebuilt_ids, rebuilt_masks = [], []

        cursor_per_batch = 0
        for b in range(batch):
            ids = input_ids[b]
            mask = attention_mask[b]
            spans = self._find_image_spans(ids)
            assert len(spans) == len(old_lens) - cursor_per_batch or batch == 1, (
                f"batch>1 with multiple images per sample is not supported yet; "
                f"found {len(spans)} spans in batch {b}"
            )

            pieces_ids, pieces_mask = [], []
            cursor = 0
            for span_i, (start, end) in enumerate(spans):
                pieces_ids.append(ids[cursor:start])
                pieces_mask.append(mask[cursor:start])

                n_new = new_lens[span_i] if batch == 1 else new_lens[cursor_per_batch + span_i]
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

    def _get_visual_module(self):
        if hasattr(self.model, "visual"):
            return self.model.visual
        if hasattr(self.model, "model") and hasattr(self.model.model, "visual"):
            return self.model.model.visual
        raise AttributeError("Could not locate the vision tower on the model (expected .visual).")

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
