"""SERF: independently re-noised endpoints and unweighted reward-gradient targets.

This module owns SERF's target, loss and timestep sampling. RSM is implemented
independently in rsm.py; no mode flag switches this loss into RSM.
"""

import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from serf.shared.loss_utils import adapter_host, model_forward


def sample_timesteps(scheduler, config, num_samples, num_targets_per_sample, device):
    """Draw with replacement on the SD3.5 grid with probability proportional to 1-t."""
    if config.timestep_sampling != "weighted_low":
        raise ValueError("The SERF paper recipe uses weighted_low timestep sampling")
    scheduler.set_timesteps(1000, device=device)
    candidates = scheduler.timesteps
    normalized_t = candidates.float() / 1000.0
    weights = (1.0 - normalized_t).clamp_min(torch.finfo(torch.float32).eps)
    indices = torch.multinomial(
        weights, num_samples * num_targets_per_sample, replacement=True
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
