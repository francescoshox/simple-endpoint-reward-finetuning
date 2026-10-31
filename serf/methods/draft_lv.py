"""K=1 DRaFT-LV updates adapted to SD3.5 flow matching."""

from __future__ import annotations

import torch
from serf.shared.sd3 import (
    build_pipeline as build_pipeline,
    latent_to_image,
    _cfg_velocity,
)

DRAFT_LV_IMPLEMENTATION_REVISION = "sd35-k1-lv3-paper"


def _repeat_conditioning(value, copies):
    """Repeat complete prompt batches in the same order as Tensor.repeat."""
    if value is None:
        return None
    return value.repeat((copies,) + (1,) * (value.ndim - 1))


def _draw_lv_noise(clean, generators, copies):
    """Draw deterministic per-prompt LV noise without coupling rank RNG streams."""
    chunks = []
    for _ in range(copies):
        chunks.append(
            torch.stack(
                [
                    torch.randn(
                        clean.shape[1:],
                        generator=generator,
                        device=clean.device,
                        dtype=clean.dtype,
                    )
                    for generator in generators
                ]
            )
        )
    return torch.cat(chunks)


def _make_lv_noised_latents(endpoint, noise, sigma):
    """Forward-noise a detached endpoint along the linear flow path."""
    return (1.0 - sigma) * endpoint.detach() + sigma * noise


def _draft_lv_objective(scores, scale):
    """Average the original and LV reward paths without changing reward scale."""
    return -scale * torch.relu(scores).mean()


def _score_latents(pipe, reward, latents, prompts, batch_size):
    scores = []
    for start in range(0, len(prompts), batch_size):
        stop = start + batch_size
        images = latent_to_image(pipe, latents[start:stop])
        chunk_prompts = prompts[start:stop]
        result = reward(images, chunk_prompts)
        chunk_scores = result[0] if isinstance(result, tuple) else result
        if chunk_scores.ndim != 1 or len(chunk_scores) != len(chunk_prompts):
            raise ValueError("Training reward must return one score per image")
        scores.append(chunk_scores)
    return torch.cat(scores)


def draft_lv_forward(pipe, model, reward, prompts, generators, config, accelerator):
    """Differentiate the last Euler step plus n forward-noised replicas.

    The first ``num_train_inference_steps - 1`` steps form a no-grad rollout. The original last-step endpoint
    supplies the DRaFT-1 term. Its detached value is then forward-noised at the
    last scheduler sigma and reconstructed by n independent trainable last-step
    calls, as in DRaFT-LV. Averaging all n+1 rewards keeps reward scale invariant.
    """
    with torch.no_grad(), accelerator.autocast():
        use_cfg = float(config.train_guidance_scale) > 1.0
        prompt_embeds, negative_embeds, pooled, negative_pooled = pipe.encode_prompt(
            prompt=prompts,
            prompt_2=None,
            prompt_3=None,
            do_classifier_free_guidance=use_cfg,
            device=accelerator.device,
            num_images_per_prompt=1,
        )
        latents = pipe.prepare_latents(
            len(prompts),
            pipe.transformer.config.in_channels,
            config.resolution,
            config.resolution,
            prompt_embeds.dtype,
            accelerator.device,
            generators,
            None,
        )
        pipe.scheduler.set_timesteps(
            config.num_train_inference_steps, device=accelerator.device
        )
        timesteps = pipe.scheduler.timesteps
        prefix_model = pipe.transformer
        for timestep in timesteps[:-1]:
            model_input = latents
            velocity = _cfg_velocity(
                prefix_model,
                model_input,
                timestep,
                prompt_embeds,
                pooled,
                negative_embeds,
                negative_pooled,
                config.train_guidance_scale,
            )
            latents = pipe.scheduler.step(
                velocity, timestep, latents, return_dict=False
            )[0]

    last_index = len(timesteps) - 1
    timestep = timesteps[last_index]
    sigma = pipe.scheduler.sigmas[last_index].to(
        device=latents.device, dtype=torch.float32
    )
    sigma_terminal = pipe.scheduler.sigmas[last_index + 1].to(
        device=latents.device, dtype=torch.float32
    )
    if not torch.equal(sigma_terminal, torch.zeros_like(sigma_terminal)):
        raise ValueError(
            "DRaFT-LV flow noising requires a zero terminal scheduler sigma"
        )

    # K=1: the rollout state is detached immediately before the final Euler step.
    latents = latents.detach()
    with accelerator.autocast():
        model_input = latents
        velocity = _cfg_velocity(
            model,
            model_input,
            timestep,
            prompt_embeds,
            pooled,
            negative_embeds,
            negative_pooled,
            config.train_guidance_scale,
        )
    endpoint = latents.float() + (sigma_terminal - sigma) * velocity

    # LV: stop gradients through the completed sample before forward noising.
    copies = config.draft_lv_num_noised_samples
    repeated_endpoint = endpoint.repeat((copies, 1, 1, 1))
    noise = _draw_lv_noise(endpoint.detach(), generators, copies)
    noised = _make_lv_noised_latents(repeated_endpoint, noise, sigma)
    with accelerator.autocast():
        aux_velocity = _cfg_velocity(
            model,
            noised,
            timestep,
            _repeat_conditioning(prompt_embeds, copies),
            _repeat_conditioning(pooled, copies),
            _repeat_conditioning(negative_embeds, copies),
            _repeat_conditioning(negative_pooled, copies),
            config.train_guidance_scale,
        )
    aux_endpoints = noised.float() + (sigma_terminal - sigma) * aux_velocity

    all_endpoints = torch.cat([endpoint, aux_endpoints])
    all_prompts = list(prompts) * (copies + 1)
    with torch.autocast("cuda", enabled=False):
        scores = _score_latents(
            pipe, reward, all_endpoints, all_prompts, config.reward_batch_size
        )
        loss = _draft_lv_objective(scores, config.draft_reward_loss_scale)
    score_matrix = scores.reshape(copies + 1, len(prompts))
    return loss, score_matrix.detach(), timestep.detach(), sigma.detach()
