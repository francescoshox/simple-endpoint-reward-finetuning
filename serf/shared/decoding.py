"""Frozen SD3 decoder in full precision."""

import torch


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
