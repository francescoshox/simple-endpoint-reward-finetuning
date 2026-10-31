"""Lean Adjoint Matching for flow models (AM Algorithm 1; FOCUS SD3 adaptation)."""

import torch


def grid(scheduler, steps, device):
    scheduler.set_timesteps(steps, device=device)
    noise = scheduler.sigmas.to(device=device, dtype=torch.float32).clone()
    times = scheduler.timesteps.to(device=device, dtype=torch.float32).clone()
    if len(noise) != steps + 1 or len(times) != steps or noise[-1] != 0:
        raise ValueError("AM requires the standard SD3 Euler grid ending at zero noise")
    if not torch.all(noise[:-1] > noise[1:]):
        raise ValueError("Noise grid must decrease strictly")
    if not torch.allclose(times / scheduler.config.num_train_timesteps, noise[:-1]):
        raise ValueError("Model timesteps and sigma grid disagree")
    return times, noise, 1 - noise, noise[:-1] - noise[1:]


def coefficients(t, epsilon):
    # Paper G.1 uses h regularization; here h_nominal=1/K remains explicit
    # when porting the equations to the nonuniform SD3 grid.
    sigma = torch.sqrt(2 * (1 - t + epsilon) / (t + epsilon))
    kappa = (1 + epsilon) / (t + epsilon)
    return sigma, kappa


def memoryless_step(x, velocity, t, h, noise, epsilon):
    sigma, kappa = coefficients(t, epsilon)
    return x + h * (2 * velocity - kappa * x) + h.sqrt() * sigma * noise


def select_indices(steps, generator=None, device="cpu"):
    if steps != 10:
        raise ValueError("This baseline fixes 10 steps and the approved x8,x9 selector")
    early = torch.randperm(8, generator=generator, device=device)[:2].sort().values
    return torch.cat((early, torch.tensor([8, 9], device=device)))


def lean_adjoint(
    states, progress, intervals, terminal_gradient, reference_velocity, epsilon
):
    """Explicit Euler in reverse time, evaluated at right endpoints (paper/FOCUS).

    reference_velocity(x, s) returns the noise-to-data velocity. Gradients
    are only with respect to x; the detached target has no parameter graph.
    """
    a = terminal_gradient.detach().float()
    result = [None] * len(states)
    result[-1] = a
    for i in reversed(range(len(states) - 1)):
        with torch.enable_grad():
            x = states[i + 1].detach().float().requires_grad_(True)
            _, kappa = coefficients(progress[i + 1], epsilon)
            drift = 2 * reference_velocity(x, progress[i + 1]) - kappa * x
            vjp = torch.autograd.grad(drift, x, grad_outputs=a, create_graph=False)[0]
        a = (a + intervals[i] * vjp).detach()
        if not torch.isfinite(a).all():
            raise FloatingPointError(f"Non-finite lean adjoint at state {i}")
        result[i] = a
    return result


def matching_loss(velocity, reference, adjoint, t, epsilon, reward_scale, clip_factor):
    sigma, _ = coefficients(t, epsilon)
    residual = (
        2 / sigma * (velocity.float() - reference.detach().float())
        + sigma * adjoint.detach().float()
    )
    # Paper uses squared L2 norm, not pixel MSE. Clip before batch averaging.
    per_sample = residual.flatten(1).square().sum(1)
    cap = clip_factor * reward_scale**2
    return (
        per_sample.clamp(max=cap).mean(),
        per_sample.detach(),
        (per_sample >= cap).float().mean(),
    )
