"""SD3.5 endpoint rollout and frozen reward differentiation shared by SERF/RSM."""

import os
import torch
from peft import LoraConfig, get_peft_model
from diffusers import StableDiffusion3Pipeline
from . import models as shared
from .distributed_metrics import reduce_tensor_means
from . import decoding as distribution_validation


def build_pipeline(config, accelerator):
    """Load SD3.5 and attach trainable and evaluation-EMA LoRA adapters."""
    # Use the standard Hugging Face cache, separate from reward assets.
    diffusion_cache = os.environ.get("HF_HUB_CACHE")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        config.model_name,
        revision=config.model_revision,
        cache_dir=diffusion_cache,
        local_files_only=os.environ.get("HF_HUB_OFFLINE") == "1",
    )
    pipe.set_progress_bar_config(disable=True)

    modules = [
        pipe.vae,
        pipe.text_encoder,
        pipe.text_encoder_2,
        pipe.text_encoder_3,
        pipe.transformer,
    ]
    for module in modules:
        module.requires_grad_(False)

    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        init_lora_weights="gaussian",
        target_modules=shared.LORA_TARGET_MODULES,
    )
    pipe.transformer = get_peft_model(pipe.transformer, lora_config)
    pipe.transformer.add_adapter("evaluation", lora_config)

    dtype = shared.DTYPE_MAP[str(config.mixed_precision)]
    # Reward gradients traverse the decoder, so the VAE remains float32.
    pipe.vae.to(accelerator.device, dtype=torch.float32)
    for encoder in (pipe.text_encoder, pipe.text_encoder_2, pipe.text_encoder_3):
        encoder.to(accelerator.device, dtype=dtype)
    pipe.transformer.to(accelerator.device)
    for name, parameter in pipe.transformer.named_parameters():
        if "lora_" in name:
            parameter.data = parameter.data.to(torch.float32)
        elif not parameter.requires_grad:
            parameter.data = parameter.data.to(dtype)
    return pipe


def decode_terminal_latents(pipe, terminal_latents):
    """Differentiably decode SD3.5 terminal latents to images in [0, 1]."""
    return distribution_validation.decode_latents_fp32(pipe, terminal_latents)


def compute_endpoint_reward_gradients(
    pipe, terminal_latents, prompts, reward_fn, reward_batch_size, accelerator
):
    """Return detached rewards and gradients with respect to terminal latents."""
    all_rewards = []
    all_gradients = []
    for start in range(0, len(terminal_latents), reward_batch_size):
        stop = start + reward_batch_size
        latent_leaf = terminal_latents[start:stop].detach().float().requires_grad_(True)
        batch_prompts = prompts[start:stop]
        with (
            torch.enable_grad(),
            torch.autocast(device_type=latent_leaf.device.type, enabled=False),
        ):
            images = decode_terminal_latents(pipe, latent_leaf).float()
            rewards = reward_fn(images, batch_prompts).float()
            if rewards.shape != (len(latent_leaf),):
                raise RuntimeError(
                    f"Reward returned shape {tuple(rewards.shape)}, expected {(len(latent_leaf),)}"
                )
            gradients = torch.autograd.grad(rewards.sum(), latent_leaf)[0]
        if not torch.isfinite(rewards).all():
            raise RuntimeError("Reward produced non-finite values")
        if not torch.isfinite(gradients).all():
            raise RuntimeError("Reward produced non-finite endpoint gradients")
        all_rewards.append(rewards.detach())
        all_gradients.append(gradients.detach().float())
        del latent_leaf, images, rewards, gradients

    rewards = torch.cat(all_rewards)
    gradients = torch.cat(all_gradients)
    gradient_norms = torch.linalg.vector_norm(gradients.flatten(1), dim=1)
    metrics = reduce_tensor_means(
        accelerator,
        {
            "rewards": rewards,
            "endpoint_reward_gradient_norm": gradient_norms,
        },
    )
    return gradients, metrics


def sample_epoch(
    pipe,
    prompts,
    image_seeds,
    negative_prompt_embeds,
    negative_pooled_embeds,
    reward_fn,
    config,
    accelerator,
    return_trajectory=False,
):
    """Generate endpoints and optionally retain actual Euler trajectory states."""
    batch_size = (
        config.sampling_batch_size // 2
        if config.train_guidance_scale > 1.0
        else config.sampling_batch_size
    )
    if batch_size <= 0:
        raise ValueError(
            "sampling_batch_size is too small for classifier-free guidance"
        )
    if len(image_seeds) != len(prompts) or any(
        len(seeds) != 1 for seeds in image_seeds
    ):
        raise ValueError("image_seeds must contain exactly one seed per prompt")

    with torch.no_grad(), accelerator.autocast():
        prompt_embeds, _, pooled_embeds, _ = pipe.encode_prompt(
            prompts, prompt_2=None, prompt_3=None, do_classifier_free_guidance=False
        )

    retain_trajectory = config.timestep_sampling == "trajectory_low_variance"
    terminal_batches = []
    trajectory_batches = []
    trajectory_timesteps = None
    for start in range(0, len(prompts), batch_size):
        stop = start + batch_size
        batch_embeds = prompt_embeds[start:stop]
        batch_pooled = pooled_embeds[start:stop]
        if config.train_guidance_scale > 1.0:
            batch_negative_embeds = negative_prompt_embeds.expand(
                batch_embeds.shape[0], -1, -1
            )
            batch_negative_pooled = negative_pooled_embeds.expand(
                batch_pooled.shape[0], -1
            )
        else:
            batch_negative_embeds = batch_negative_pooled = None
        generators = [
            torch.Generator(device=pipe.device).manual_seed(seeds[0])
            for seeds in image_seeds[start:stop]
        ]
        sampling_kwargs = {}
        captured_states = []
        if retain_trajectory:

            def capture_state(_pipe, step_index, _timestep, callback_kwargs):
                # VGGFlow clamps index zero to one and never trains on the
                # initial pure-noise state. State i is available immediately
                # after Euler step i-1, so retain post-step states 1..T-1.
                if step_index < config.num_train_inference_steps - 1:
                    captured_states.append(
                        callback_kwargs["latents"].detach().cpu().clone()
                    )
                return callback_kwargs

            sampling_kwargs.update(
                callback_on_step_end=capture_state,
                callback_on_step_end_tensor_inputs=["latents"],
            )
        with torch.no_grad(), accelerator.autocast():
            terminal = pipe(
                prompt_embeds=batch_embeds,
                negative_prompt_embeds=batch_negative_embeds,
                pooled_prompt_embeds=batch_pooled,
                negative_pooled_prompt_embeds=batch_negative_pooled,
                height=config.resolution,
                width=config.resolution,
                num_inference_steps=config.num_train_inference_steps,
                guidance_scale=config.train_guidance_scale,
                generator=generators,
                output_type="latent",
                return_dict=False,
                **sampling_kwargs,
            )[0]
        terminal_batches.append(terminal.detach())
        if retain_trajectory:
            expected_states = config.num_train_inference_steps - 1
            if len(captured_states) != expected_states:
                raise RuntimeError(
                    f"Euler callback retained {len(captured_states)} states, "
                    f"expected {expected_states}"
                )
            trajectory_batches.append(torch.stack(captured_states, dim=1))
            batch_timesteps = pipe.scheduler.timesteps.detach().cpu().clone()
            if trajectory_timesteps is None:
                trajectory_timesteps = batch_timesteps
            elif not torch.equal(trajectory_timesteps, batch_timesteps):
                raise RuntimeError(
                    "Euler timestep schedule changed between sampling batches"
                )

    terminal_latents = torch.cat(terminal_batches)
    reward_gradients, metrics = compute_endpoint_reward_gradients(
        pipe,
        terminal_latents,
        prompts,
        reward_fn,
        config.reward_batch_size,
        accelerator,
    )
    trajectory_latents = torch.cat(trajectory_batches) if retain_trajectory else None
    result = terminal_latents, prompt_embeds, pooled_embeds, reward_gradients, metrics
    if return_trajectory:
        return result + (trajectory_latents, trajectory_timesteps)
    return result
