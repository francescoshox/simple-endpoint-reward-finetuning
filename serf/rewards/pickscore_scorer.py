# Differentiable counterpart to the ``PickScore`` reward in ``__init__.py``, which
# round-trips images through PIL (breaking gradients). Preprocessing here is
# tensor-native, so with ``no_grad=False`` the returned score keeps a gradient back to
# ``images``.

from contextlib import nullcontext

import torch
import torch.nn as nn

from .clip_scorer import get_image_transform
from .model_ids import (
    LAION_CLIP_H14_ID,
    LAION_CLIP_H14_REVISION,
    PICKSCORE_ID,
    PICKSCORE_REVISION,
)


class PickScoreScorer(nn.Module):
    def __init__(self, device="cpu", no_grad=True):
        super().__init__()
        from transformers import AutoProcessor, CLIPModel

        self.processor = AutoProcessor.from_pretrained(
            LAION_CLIP_H14_ID, revision=LAION_CLIP_H14_REVISION
        )
        self.model = CLIPModel.from_pretrained(
            PICKSCORE_ID, revision=PICKSCORE_REVISION
        ).to(device)
        self.model.requires_grad_(False)
        self.tform = get_image_transform(self.processor.image_processor)
        self.no_grad = no_grad
        self.eval()

    def forward(self, images, prompts):
        device = next(self.model.parameters()).device
        with torch.no_grad() if self.no_grad else nullcontext():
            dtype = images.dtype
            pixels = self.tform(images.to(device)).to(dtype=dtype)

            text_inputs = self.processor(
                text=prompts,
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt",
            ).to(device)

            image_embs = self.model.get_image_features(pixel_values=pixels)
            image_embs = image_embs / image_embs.norm(p=2, dim=-1, keepdim=True)

            text_embs = self.model.get_text_features(**text_inputs)
            text_embs = text_embs / text_embs.norm(p=2, dim=-1, keepdim=True)

            logit_scale = self.model.logit_scale.exp()
            scores = logit_scale * (text_embs @ image_embs.T)
            scores = scores.diag()

            # Keep the native PickScore logit scale (typically around 20--25).
            # In particular, do not divide by 26: first-order reward coefficients
            # are defined against the raw PickScore gradient.
            return scores
