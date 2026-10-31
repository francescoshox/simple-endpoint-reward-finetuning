"""FP32 decoding, DreamSim, and canonical FID scorer implementations."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import torch
from torch import nn

from serf.rewards.paths import MODELS_DIR


DREAMSIM_IMAGE_SIZE = 224
DREAMSIM_TYPES = {
    "ensemble": ("ensemble", False),
    "dino_vitb16": ("dino_vitb16", False),
    "dino_vitb16_patch": ("dino_vitb16", True),
    "dinov2_vitb14": ("dinov2_vitb14", False),
    "dinov2_vitb14_patch": ("dinov2_vitb14", True),
    "clip_vitb32": ("clip_vitb32", False),
    "open_clip_vitb32": ("open_clip_vitb32", False),
    "synclr_vitb16": ("synclr_vitb16", False),
}
FID_WEIGHTS_FILENAME = "weights-inception-2015-12-05-6726825d.pth"


def decode_latents_fp32(pipe, latents, batch_size=None):
    """Decode SD3 terminal latents with FP32 inputs, weights, and autocast off."""
    floating_state = [
        tensor
        for tensor in (*pipe.vae.parameters(), *pipe.vae.buffers())
        if tensor.is_floating_point()
    ]
    if floating_state and any(
        tensor.dtype != torch.float32 for tensor in floating_state
    ):
        raise RuntimeError(
            "FP32 decoding requires all floating VAE state to be float32"
        )
    batch_size = len(latents) if batch_size is None else int(batch_size)
    if batch_size <= 0:
        raise ValueError("decode batch_size must be positive")
    outputs = []
    device_type = latents.device.type
    with torch.autocast(device_type=device_type, enabled=False):
        for start in range(0, len(latents), batch_size):
            batch = latents[start : start + batch_size].float()
            scaled = batch / pipe.vae.config.scaling_factor
            scaled = scaled + getattr(pipe.vae.config, "shift_factor", 0.0)
            decoded = pipe.vae.decode(scaled, return_dict=False)[0].float()
            outputs.append(((decoded + 1.0) / 2.0).clamp(0.0, 1.0))
    return (
        torch.cat(outputs)
        if outputs
        else latents.new_empty((0, 3, 0, 0), dtype=torch.float32)
    )


def dreamsim_diversity(embeddings):
    """Normalized mean pairwise cosine distance for one prompt group."""
    if embeddings.ndim != 2 or embeddings.shape[0] < 2:
        raise ValueError("DreamSim diversity needs at least two embedding vectors")
    vectors = nn.functional.normalize(embeddings.float(), dim=1, eps=1e-12)
    similarities = vectors @ vectors.T
    indices = torch.triu_indices(
        len(vectors), len(vectors), offset=1, device=vectors.device
    )
    return ((1.0 - similarities[indices[0], indices[1]]) / 2.0).mean()


def per_prompt_fid(features, reference_features):
    """Exact empirical Gaussian FID using an equivalent low-rank SVD."""
    if features.ndim != 2 or reference_features.ndim != 2:
        raise ValueError("FID features must be two-dimensional")
    if features.shape != reference_features.shape or features.shape[0] < 2:
        raise ValueError(
            "FID feature sets must have the same shape and at least two samples"
        )
    generated = features.double()
    reference = reference_features.double()
    generated_mean = generated.mean(dim=0)
    reference_mean = reference.mean(dim=0)
    generated_centered = generated - generated_mean
    reference_centered = reference - reference_mean
    denominator = generated.shape[0] - 1
    mean_term = (generated_mean - reference_mean).square().sum()
    covariance_trace = (
        generated_centered.square().sum() + reference_centered.square().sum()
    ) / denominator
    cross_gram = generated_centered @ reference_centered.T / denominator
    covariance_overlap = torch.linalg.svdvals(cross_gram).sum()
    return (mean_term + covariance_trace - 2.0 * covariance_overlap).clamp_min(0.0)


@contextmanager
def _load_replicated_peft_adapter():
    """Avoid PEFT's incompatible TP hook when loading a replicated adapter.

    PEFT 0.19.1 invokes its tensor-parallel state-dict hook whenever a torch
    process group exists, even for ordinary replicated models. That hook
    imports ``EmbeddingParallel``, which is absent from transformers 4.57.6.
    DreamSim is independently replicated on every validation worker, so TP
    sharding is neither needed nor correct here.
    """
    distributed = (
        torch.distributed.is_available() and torch.distributed.is_initialized()
    )
    if not distributed:
        yield
        return

    from transformers.integrations import tensor_parallel

    if hasattr(tensor_parallel, "EmbeddingParallel"):
        yield
        return

    import peft.utils.save_and_load as peft_save_and_load

    original_hook = peft_save_and_load._maybe_shard_state_dict_for_tp

    def skip_replicated_tp_sharding(model, state_dict, adapter_name):
        del state_dict, adapter_name
        for module in model.modules():
            get_base_layer = getattr(module, "get_base_layer", None)
            base_layer = get_base_layer() if callable(get_base_layer) else module
            if getattr(base_layer, "_hf_tp_plan", None) is not None:
                raise RuntimeError(
                    "DreamSim's validation encoder must not use a tensor-parallel plan"
                )

    peft_save_and_load._maybe_shard_state_dict_for_tp = skip_replicated_tp_sharding
    try:
        yield
    finally:
        peft_save_and_load._maybe_shard_state_dict_for_tp = original_hook


class DreamSimScorer:
    def __init__(self, type_name, cache_dir, batch_size, device):
        from dreamsim import dreamsim

        dreamsim_type, use_patch_model = DREAMSIM_TYPES[type_name]
        with _load_replicated_peft_adapter():
            self.model, _ = dreamsim(
                dreamsim_type=dreamsim_type,
                use_patch_model=use_patch_model,
                device=str(device),
                cache_dir=str(cache_dir),
            )
        self.model.eval().requires_grad_(False)
        self.batch_size = batch_size
        self.device = device

    @staticmethod
    def _reduce(features):
        if features.ndim == 3 and features.shape[1] > 1:
            cls = features[:, 0]
            patches = features[:, 1:]
            patch_count = patches.shape[1]
            side = int(round(patch_count**0.5))
            if side * side == patch_count:
                patches = patches.reshape(
                    len(features), side, side, features.shape[-1]
                ).mean(dim=(1, 2))
                features = torch.cat((cls, patches), dim=-1)
            else:
                features = features.flatten(1)
        elif features.ndim != 2:
            features = features.flatten(1)
        return nn.functional.normalize(features.float(), dim=1, eps=1e-12)

    @torch.no_grad()
    def embed(self, images):
        chunks = []
        for start in range(0, len(images), self.batch_size):
            batch = images[start : start + self.batch_size].to(
                self.device, dtype=torch.float32
            )
            if batch.shape[-2:] != (DREAMSIM_IMAGE_SIZE, DREAMSIM_IMAGE_SIZE):
                batch = nn.functional.interpolate(
                    batch,
                    size=(DREAMSIM_IMAGE_SIZE, DREAMSIM_IMAGE_SIZE),
                    mode="bicubic",
                    align_corners=False,
                    antialias=True,
                ).clamp(0.0, 1.0)
            with torch.autocast(device_type=self.device.type, enabled=False):
                features = self.model.embed(batch)
            if features.ndim == 2 and len(batch) == 1:
                features = features.unsqueeze(0)
            chunks.append(self._reduce(features))
        return torch.cat(chunks)


def canonical_fid_weights_path():
    candidates = (
        Path(torch.hub.get_dir()) / "checkpoints" / FID_WEIGHTS_FILENAME,
        MODELS_DIR / "torch" / "hub" / "checkpoints" / FID_WEIGHTS_FILENAME,
    )
    for path in candidates:
        if path.is_file() and path.stat().st_size:
            return path
    raise FileNotFoundError(
        f"Missing canonical FID weights {FID_WEIGHTS_FILENAME}; run "
        "python -m serf.prepare before training"
    )


class CanonicalFidScorer:
    def __init__(self, batch_size, device):
        from torch_fidelity.feature_extractor_inceptionv3 import (
            FeatureExtractorInceptionV3,
        )

        self.model = (
            FeatureExtractorInceptionV3(
                "inception-v3-compat",
                ["2048"],
                feature_extractor_weights_path=str(canonical_fid_weights_path()),
            )
            .to(device)
            .eval()
        )
        self.batch_size = batch_size
        self.device = device

    @torch.no_grad()
    def embed(self, images):
        chunks = []
        for start in range(0, len(images), self.batch_size):
            batch = images[start : start + self.batch_size]
            batch = (
                (batch * 255.0)
                .round()
                .clamp(0, 255)
                .to(device=self.device, dtype=torch.uint8)
            )
            with torch.autocast(device_type=self.device.type, enabled=False):
                features = self.model(batch)[0]
            chunks.append(features.flatten(1).float())
        return torch.cat(chunks)
