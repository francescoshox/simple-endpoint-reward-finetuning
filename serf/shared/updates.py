"""Shared orchestration only; method targets and losses live in separate modules."""

import importlib
import torch
from .endpoints import sample_epoch
from .evaluation_ema import collect_policy_params, initialize_evaluation_ema


def build(config, accelerator):
    value = None
    if config.method in ("serf", "rsm"):
        from .endpoints import build_pipeline

        pipe = build_pipeline(config, accelerator)
    elif config.method == "draft_lv":
        from serf.methods.draft_lv import build_pipeline

        pipe = build_pipeline(config, accelerator.device)
    elif config.method == "adjoint_matching":
        from .sd3 import build_pipeline

        pipe = build_pipeline(config, accelerator.device)
    else:
        from serf.methods.vgg_flow import import_upstream, build_models

        import_upstream()
        pipe, policy, value = build_models(config, accelerator.device)
    policy = pipe.transformer
    live, ema = collect_policy_params(policy)
    initialize_evaluation_ema(live, ema)
    parameters = live + (
        [p for p in value.parameters() if p.requires_grad] if value is not None else []
    )
    return pipe, policy, value, live, ema, parameters


def first_order_update(config, pipe, policy, reward, prompts, seeds, accelerator):
    # Same endpoint generation is shared. SERF/RSM own independent losses and
    # target-state construction; no method flag is interpreted by either loss.
    method = importlib.import_module("serf.methods." + config.method)
    policy.eval()
    with torch.no_grad(), accelerator.autocast():
        negative, _, negative_pooled, _ = pipe.encode_prompt(
            [""], prompt_2=None, prompt_3=None, do_classifier_free_guidance=False
        )
    endpoints, embeds, pooled, gradients, metrics, trajectories, times = sample_epoch(
        pipe,
        prompts,
        [[s] for s in seeds],
        negative,
        negative_pooled,
        reward,
        config,
        accelerator,
        return_trajectory=True,
    )
    states, timesteps = method.prepare(
        pipe, endpoints, trajectories, times, config, accelerator
    )
    values = [
        v.repeat_interleave(4, dim=0) for v in (endpoints, embeds, pooled, gradients)
    ]
    endpoints, embeds, pooled, gradients = values
    policy.train()
    n = len(endpoints)
    total = 0.0
    for start in range(0, n, config.loss_batch_size):
        stop = start + config.loss_batch_size
        loss, _ = method.loss(
            policy,
            endpoints[start:stop],
            embeds[start:stop],
            pooled[start:stop],
            gradients[start:stop],
            timesteps[start:stop],
            config,
            accelerator,
            training_latent=states[start:stop] if states is not None else None,
            negative_prompt_embeds=negative,
            negative_pooled_embeds=negative_pooled,
        )
        scaled = loss * (stop - start) / n
        accelerator.backward(scaled)
        total += scaled.detach().item()
    return dict(
        loss=total, **{f"rewards/{config.training_reward_name}": metrics["rewards"]}
    )


def draft_update(config, pipe, policy, reward, prompts, seeds, accelerator):
    from serf.methods.draft_lv import draft_lv_forward

    scores = []
    total = 0.0
    policy.train()
    for i in range(0, len(prompts), config.loss_batch_size):
        batch = prompts[i : i + config.loss_batch_size]
        generators = [
            torch.Generator(device=accelerator.device).manual_seed(s)
            for s in seeds[i : i + len(batch)]
        ]
        loss, score, _, _ = draft_lv_forward(
            pipe, policy, reward, batch, generators, config, accelerator
        )
        scaled = loss * len(batch) / len(prompts)
        accelerator.backward(scaled)
        total += scaled.detach().item()
        scores.append(score.detach().flatten())
    scores = torch.cat(scores)
    return {
        "loss": total,
        f"rewards/{config.training_reward_name}": scores.mean().item(),
    }


def am_update(config, pipe, policy, reward, prompts, seeds, accelerator):
    from serf.methods.adjoint_matching.runtime import rollout, backward_trajectory

    results = []
    policy.eval()
    for start in range(0, len(prompts), config.sampling_batch_size):
        pp = prompts[start : start + config.sampling_batch_size]
        ss = seeds[start : start + len(pp)]
        states, conditioning, progress, intervals = rollout(
            pipe, pp, ss, config, accelerator.device
        )
        for i, prompt in enumerate(pp):
            cs = tuple(v[i : i + 1] if v is not None else None for v in conditioning)
            xs = [v[i : i + 1].to(accelerator.device) for v in states]
            results.append(
                backward_trajectory(
                    pipe,
                    reward,
                    prompt,
                    xs,
                    cs,
                    progress,
                    intervals,
                    config,
                    len(prompts) * 4,
                    accelerator,
                )
            )
    return {
        "loss": sum(r["loss"] for r in results) / len(results),
        "clipped_fraction": sum(r["clipped_fraction"] for r in results) / len(results),
        f"rewards/{config.training_reward_name}": sum(r["reward"] for r in results)
        / len(results),
    }


def vgg_update(
    config, pipe, policy, value, reward, algorithm, prompts, seeds, accelerator
):
    from serf.methods.vgg_flow import sample_epoch, select_states, loss_terms

    policy.eval()
    samples, scores = sample_epoch(
        config, pipe, policy, reward, prompts, seeds, accelerator.device
    )
    samples, selected = select_states(samples, config)
    policy.train()
    value.train()
    total = 0.0
    pieces = {}
    denominator = (len(prompts) // config.loss_batch_size) * selected
    for start in range(0, len(prompts), config.loss_batch_size):
        stop = start + config.loss_batch_size
        batch = {
            k: (
                v[start:stop].to(accelerator.device)
                if torch.is_tensor(v)
                else v[start:stop]
            )
            for k, v in samples.items()
        }
        for step in range(selected):
            terms, _ = loss_terms(algorithm, batch, step, pipe, config, 1.0)
            loss = sum(terms.values()) / denominator
            accelerator.backward(loss)
            total += loss.detach().item()
            for k, v in terms.items():
                pieces[f"loss/{k}"] = (
                    pieces.get(f"loss/{k}", 0) + v.detach().item() / denominator
                )
    return {
        "loss": total,
        **pieces,
        f"rewards/{config.training_reward_name}": scores.mean().item(),
    }
