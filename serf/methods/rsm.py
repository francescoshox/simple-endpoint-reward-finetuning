"""RSM: endpoint reward gradients matched at retained on-policy trajectory states.

Owns trajectory state selection, CFG loss and clean-time weighting. It never
calls SERF's loss or creates independently noised endpoint training states.
"""

import torch
from serf.shared.loss_utils import adapter_host, model_forward


def select_low_variance_trajectory_states(trajectory_latents, timesteps, config):
    """Select the forced-final state and one state from each VGGFlow partition."""
    num_samples = trajectory_latents.shape[0]
    num_steps = len(timesteps)
    if trajectory_latents.shape[1] != num_steps - 1:
        raise ValueError(
            "Trajectory latents must contain the states for denoising indices 1 through T-1"
        )
    partitions = config.trajectory_num_partitions
    trunk_size, remainder = divmod(num_steps, partitions)
    trunk_sizes = [trunk_size + (index < remainder) for index in range(partitions)]
    step_indices = torch.arange(num_steps, device=trajectory_latents.device)
    trunks = list(torch.split(step_indices, trunk_sizes))
    # VGG constructs a complete interleave even though training consumes only
    # its first pass through the reversed trunks. Retain that RNG behavior for
    # equal trunks and extend it naturally to near-equal partitions.
    trunk_order = list(reversed(range(partitions))) * max(trunk_sizes)
    selected_per_sample = []
    for _ in range(num_samples):
        interleaved = []
        for partition_index in trunk_order:
            offset = torch.randint(
                0,
                len(trunks[partition_index]),
                (1,),
                device=trajectory_latents.device,
            )
            interleaved.append(trunks[partition_index][offset])
        permutation = torch.cat(
            [
                torch.tensor([num_steps - 1], device=trajectory_latents.device),
                *interleaved,
            ]
        ).clamp_min(1)
        selected_per_sample.append(permutation[: partitions + 1])
    selected_indices = torch.stack(selected_per_sample)
    batch_indices = torch.arange(num_samples, device=trajectory_latents.device)[:, None]
    selected_latents = trajectory_latents[batch_indices, selected_indices - 1]
    selected_timesteps = timesteps.to(trajectory_latents.device)[selected_indices]
    return (
        selected_latents.flatten(0, 1),
        selected_timesteps.flatten(),
        selected_indices,
    )


def first_order_model_velocity(
    model,
    sample,
    timestep,
    prompt_embeds,
    pooled_embeds,
    config,
    accelerator,
    negative_prompt_embeds=None,
    negative_pooled_embeds=None,
    detach_uncond=True,
):
    """Evaluate a conditional or VGGFlow-style CFG velocity.

    When ``first_order_loss_cfg`` is enabled, both policy and reference use the
    rollout CFG scale. The policy call detaches the unconditional output so its
    gradient is ``scale * grad(v_cond)``, matching VGGFlow. Reference calls run
    under ``no_grad`` but accept ``detach_uncond=False`` for explicit symmetry.
    """
    if not getattr(config, "first_order_loss_cfg", False):
        return model_forward(
            model, sample, timestep, prompt_embeds, pooled_embeds, accelerator
        )
    if negative_prompt_embeds is None or negative_pooled_embeds is None:
        raise ValueError(
            "first_order_loss_cfg requires negative prompt and pooled embeddings"
        )

    batch_size = sample.shape[0]

    def expand_negative(value, positive, name):
        if value.shape[0] == 1:
            value = value.expand(batch_size, *value.shape[1:])
        if value.shape[0] != batch_size or value.shape[1:] != positive.shape[1:]:
            raise ValueError(
                f"{name} must have one row or match the conditional embedding shape"
            )
        return value

    negative_prompt_embeds = expand_negative(
        negative_prompt_embeds, prompt_embeds, "negative_prompt_embeds"
    )
    negative_pooled_embeds = expand_negative(
        negative_pooled_embeds, pooled_embeds, "negative_pooled_embeds"
    )
    if detach_uncond:
        with torch.no_grad():
            unconditional = model_forward(
                model,
                sample,
                timestep,
                negative_prompt_embeds,
                negative_pooled_embeds,
                accelerator,
            )
    else:
        unconditional = model_forward(
            model,
            sample,
            timestep,
            negative_prompt_embeds,
            negative_pooled_embeds,
            accelerator,
        )
    conditional = model_forward(
        model, sample, timestep, prompt_embeds, pooled_embeds, accelerator
    )
    return unconditional + config.train_guidance_scale * (conditional - unconditional)


def target_velocity(reference, reward_gradient, sigma, beta):
    return reference - beta * (1 - sigma) * reward_gradient.float()


def loss(
    model,
    endpoint,
    embeds,
    pooled,
    gradient,
    timestep,
    config,
    accelerator,
    training_latent=None,
    negative_prompt_embeds=None,
    negative_pooled_embeds=None,
):
    if training_latent is None:
        raise ValueError("RSM requires retained trajectory states")
    host = adapter_host(model)
    host.set_adapter("default")
    sigma = (timestep / 1000).view(-1, 1, 1, 1)
    current = first_order_model_velocity(
        model,
        training_latent,
        timestep,
        embeds,
        pooled,
        config,
        accelerator,
        negative_prompt_embeds,
        negative_pooled_embeds,
        detach_uncond=True,
    )
    with torch.no_grad(), host.disable_adapter():
        reference = first_order_model_velocity(
            model,
            training_latent,
            timestep,
            embeds,
            pooled,
            config,
            accelerator,
            negative_prompt_embeds,
            negative_pooled_embeds,
            detach_uncond=False,
        )
    target = target_velocity(reference, gradient, sigma, config.first_order_coeff)
    host.set_adapter("default")
    per_target = (current - target).square().mean((1, 2, 3))
    value = per_target.view(-1, config.num_loss_targets_per_sample).mean(1).mean()
    return value, {"loss": value.detach().item()}


def prepare(pipe, endpoints, trajectories, times, config, accelerator):
    states, timesteps, indices = select_low_variance_trajectory_states(
        trajectories, times, config
    )
    return states.to(accelerator.device), timesteps.to(accelerator.device)
