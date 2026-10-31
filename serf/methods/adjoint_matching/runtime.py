"""Memory-bounded AM rollout, terminal reward gradient and matching updates."""

from contextlib import contextmanager
import time
import torch
from .core import grid, memoryless_step, lean_adjoint, select_indices, matching_loss
from serf.shared.sd3 import _cfg_velocity, latent_to_image


@contextmanager
def base_policy(model):
    model.disable_adapters()
    try:
        yield
    finally:
        model.enable_adapters()
        model.set_adapter("default")


def clock():
    torch.cuda.synchronize()
    return time.perf_counter()


def velocity(pipe, x, progress, conditioning, config):
    prompt, negative, pooled, negative_pooled = conditioning
    timestep = (1 - progress) * pipe.scheduler.config.num_train_timesteps
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return -_cfg_velocity(
            pipe.transformer,
            x,
            timestep,
            prompt,
            pooled,
            negative,
            negative_pooled,
            config.train_guidance_scale,
        )


def rollout(pipe, prompts, seeds, config, device):
    generators = [torch.Generator(device=device).manual_seed(seed) for seed in seeds]
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        conditioning = pipe.encode_prompt(
            prompts,
            prompt_2=None,
            prompt_3=None,
            do_classifier_free_guidance=config.train_guidance_scale > 1,
            device=device,
            num_images_per_prompt=1,
        )
        times, noise, progress, intervals = grid(
            pipe.scheduler, config.num_train_inference_steps, device
        )
        x = pipe.prepare_latents(
            len(prompts),
            pipe.transformer.config.in_channels,
            config.resolution,
            config.resolution,
            conditioning[0].dtype,
            device,
            generators,
            None,
        )
        states = [x.detach().cpu()]
        for i in range(config.num_train_inference_steps):
            v = velocity(pipe, x, progress[i], conditioning, config)
            z = torch.stack(
                [
                    torch.randn(
                        x.shape[1:], device=device, generator=g, dtype=torch.float32
                    )
                    for g in generators
                ]
            )
            x = memoryless_step(
                x.float(), v, progress[i], intervals[i], z, config.am_schedule_epsilon
            ).to(x.dtype)
            if not torch.isfinite(x).all():
                raise FloatingPointError(f"Non-finite rollout at step {i}")
            states.append(x.detach().cpu())
    return (
        states,
        tuple(c.detach() if c is not None else None for c in conditioning),
        progress,
        intervals,
    )


def backward_trajectory(
    pipe,
    reward,
    prompt,
    states,
    conditioning,
    progress,
    intervals,
    config,
    denominator,
    accelerator,
):
    """One trajectory; retain one input-VJP graph at a time, then four loss graphs."""
    metrics = {}
    start = clock()

    def reference(x, s):
        with base_policy(pipe.transformer):
            return velocity(pipe, x, s, conditioning, config)

    # Paper G.1: reward gradient at the noiseless reference projection from x[K-1].
    with torch.no_grad():
        endpoint = states[-2].float() + intervals[-1] * reference(
            states[-2].float(), progress[-2]
        )
    endpoint = endpoint.detach().requires_grad_(True)
    with torch.autocast("cuda", enabled=False):
        scores = reward(latent_to_image(pipe, endpoint), [prompt])
        a_terminal = torch.autograd.grad(-config.reward_scale * scores.sum(), endpoint)[
            0
        ].detach()
    if not torch.isfinite(a_terminal).all() or not torch.isfinite(scores).all():
        raise FloatingPointError("Non-finite terminal reward gradient")
    metrics["terminal_seconds"] = clock() - start
    metrics["reward"] = scores.detach().mean().item()
    metrics["terminal_adjoint_norm"] = a_terminal.norm().item()
    start = clock()

    # Keep adapters disabled during each reference VJP; base parameters are frozen.
    def reference_for_vjp(x, s):
        return velocity(pipe, x, s, conditioning, config)

    with base_policy(pipe.transformer):
        adjoints = lean_adjoint(
            states,
            progress,
            intervals,
            a_terminal,
            reference_for_vjp,
            config.am_schedule_epsilon,
        )
    metrics["adjoint_seconds"] = clock() - start
    indices = select_indices(10, device=accelerator.device)
    start = clock()
    loss_total, unclipped, clipped = 0.0, 0.0, 0.0
    for idx in indices.tolist():
        x = states[idx].detach()
        with torch.no_grad():
            ref = reference(x, progress[idx])
        current = velocity(pipe, x, progress[idx], conditioning, config)
        loss, raw, fraction = matching_loss(
            current,
            ref,
            adjoints[idx],
            progress[idx],
            config.am_schedule_epsilon,
            config.reward_scale,
            config.am_loss_clip_factor,
        )
        accelerator.backward(loss / denominator)
        loss_total += loss.detach().item() / 4
        unclipped += raw.mean().item() / 4
        clipped += fraction.item() / 4
    metrics.update(
        matching_seconds=clock() - start,
        loss=loss_total,
        raw_loss=unclipped,
        clipped_fraction=clipped,
        selected_indices=indices.tolist(),
    )
    return metrics
