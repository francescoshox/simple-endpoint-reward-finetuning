"""SERF: independently re-noised endpoints and unweighted reward-gradient targets.

This module owns SERF's target, loss and timestep sampling. RSM is implemented
independently in rsm.py; no mode flag switches this loss into RSM.
"""

import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from serf.shared.loss_utils import adapter_host, model_forward


def sample_timesteps(scheduler, config, num_samples, num_targets_per_sample, device):
    """Sample SD3.5 timesteps, including clean-end triangular weighting.

    ``weighted_low`` uses probabilities proportional to ``1 - t`` on the
    normalized SD3.5 timestep grid. In the continuous limit this is the
    triangular density ``p(t) = 2(1 - t)``.
    """
    if config.timestep_sampling == "train":
        scheduler.set_timesteps(config.num_train_inference_steps, device=device)
    elif config.timestep_sampling == "eval":
        scheduler.set_timesteps(config.num_eval_inference_steps, device=device)
    else:
        scheduler.set_timesteps(1000, device=device)
    timesteps = scheduler.timesteps

    if config.timestep_sampling == "uniform":
        candidates = timesteps[: len(timesteps) - config.min_timesteps]
    elif config.timestep_sampling == "uniform_open":
        normalized_t = timesteps.float() / 1000.0
        candidates = timesteps[(normalized_t > 0.0) & (normalized_t < 1.0)]
    elif config.timestep_sampling in ("weighted", "weighted_low", "train", "eval"):
        candidates = timesteps
    else:
        raise ValueError(f"Invalid timestep sampling mode: {config.timestep_sampling}")

    candidate_count = len(candidates)
    if candidate_count < num_targets_per_sample:
        raise ValueError(
            f"Cannot draw {num_targets_per_sample} distinct targets from "
            f"{candidate_count} candidate timesteps"
        )
    total = num_samples * num_targets_per_sample
    if config.timestep_sampling == "weighted":
        weights = (
            1.0
            - torch.arange(candidate_count, device=device, dtype=torch.float32)
            / candidate_count
        )
        indices = torch.multinomial(weights, total, replacement=True)
    elif config.timestep_sampling == "weighted_low":
        normalized_t = candidates.float() / 1000.0
        weights = (1.0 - normalized_t).clamp_min(torch.finfo(torch.float32).eps)
        indices = torch.multinomial(weights, total, replacement=True)
    else:
        indices = (
            torch.rand(num_samples, candidate_count, device=device)
            .argsort(dim=1)[:, :num_targets_per_sample]
            .reshape(-1)
        )
    return candidates[indices]


def target_velocity(reference, reward_gradient, beta):
    return reference - beta * reward_gradient.float()


def loss(
    model, endpoint, embeds, pooled, gradient, timestep, config, accelerator, **unused
):
    host = adapter_host(model)
    host.set_adapter("default")
    sigma = (timestep / 1000).view(-1, 1, 1, 1)
    # Independent forward noising is the defining SERF state construction.
    noise = torch.randn_like(endpoint)
    state = (1 - sigma) * endpoint + sigma * noise
    current = model_forward(model, state, timestep, embeds, pooled, accelerator)
    with torch.no_grad(), host.disable_adapter():
        reference = model_forward(model, state, timestep, embeds, pooled, accelerator)
    target = target_velocity(reference, gradient, config.first_order_coeff)
    host.set_adapter("default")
    per_target = (current - target).square().mean((1, 2, 3))
    value = per_target.view(-1, config.num_loss_targets_per_sample).mean(1).mean()
    return value, {"loss": value.detach().item()}


def prepare(pipe, endpoints, trajectories, times, config, accelerator):
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)
    timesteps = sample_timesteps(
        scheduler,
        config,
        len(endpoints),
        config.num_loss_targets_per_sample,
        accelerator.device,
    )
    return None, timesteps
