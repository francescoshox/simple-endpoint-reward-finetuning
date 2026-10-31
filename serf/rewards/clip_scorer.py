# Based on https://github.com/RE-N-Y/imscore/blob/main/src/imscore/preference/model.py

from contextlib import nullcontext

import torch
import torch.nn as nn
import torchvision.transforms as T
from transformers import AutoImageProcessor, CLIPModel, CLIPProcessor

from .model_ids import CLIP_L14_ID, CLIP_L14_REVISION


def get_size(size):
    if isinstance(size, int):
        return (size, size)
    elif "height" in size and "width" in size:
        return (size["height"], size["width"])
    elif "shortest_edge" in size:
        return size["shortest_edge"]
    else:
        raise ValueError(f"Invalid size: {size}")


def get_image_transform(processor: AutoImageProcessor):
    config = processor.to_dict()
    resize = (
        T.Resize(get_size(config.get("size")))
        if config.get("do_resize")
        else nn.Identity()
    )
    crop = (
        T.CenterCrop(get_size(config.get("crop_size")))
        if config.get("do_center_crop")
        else nn.Identity()
    )
    normalise = (
        T.Normalize(mean=processor.image_mean, std=processor.image_std)
        if config.get("do_normalize")
        else nn.Identity()
    )

    return T.Compose([resize, crop, normalise])


class ClipScorer(torch.nn.Module):
    """Preprocessing is tensor-native (no PIL/uint8 round-trip), so with ``no_grad=False``
    the returned score keeps a gradient back to ``pixels``."""

    def __init__(self, device="cpu", no_grad=True):
        super().__init__()
        self.model = CLIPModel.from_pretrained(
            CLIP_L14_ID, revision=CLIP_L14_REVISION
        ).to(device)
        self.model.requires_grad_(False)
        self.processor = CLIPProcessor.from_pretrained(
            CLIP_L14_ID, revision=CLIP_L14_REVISION
        )
        self.tform = get_image_transform(self.processor.image_processor)
        self.no_grad = no_grad
        self.eval()

    def _process(self, pixels):
        dtype = pixels.dtype
        pixels = self.tform(pixels)
        pixels = pixels.to(dtype=dtype)

        return pixels

    def forward(self, pixels, prompts, return_img_embedding=False):
        device = next(self.model.parameters()).device
        with torch.no_grad() if self.no_grad else nullcontext():
            texts = self.processor(
                text=prompts, padding="max_length", truncation=True, return_tensors="pt"
            ).to(device)
            pixels = self._process(pixels).to(device)
            outputs = self.model(pixel_values=pixels, **texts)
            if return_img_embedding:
                return outputs.logits_per_image.diagonal() / 100, outputs.image_embeds
            return outputs.logits_per_image.diagonal() / 100
