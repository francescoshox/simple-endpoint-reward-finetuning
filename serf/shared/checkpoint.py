"""Complete checkpoints: mutable weights, optimizer, EMA, counters and every rank's RNG."""

import json, random
from uuid import uuid4
from pathlib import Path
import numpy as np
import torch
from .evaluation_ema import save_policy_exports


def mutable_state(model):
    return {
        n: p.detach().cpu()
        for n, p in model.named_parameters()
        if p.requires_grad or "lora_" in n
    }


def load_mutable(model, state):
    params = dict(model.named_parameters())
    expected = set(mutable_state(model))
    if set(state) != expected:
        raise ValueError("Checkpoint parameter schema mismatch")
    if any(params[n].shape != v.shape for n, v in state.items()):
        raise ValueError("Checkpoint parameter shape mismatch")
    with torch.no_grad():
        for n, v in state.items():
            params[n].copy_(v)


def rng_state():
    return dict(
        python=random.getstate(),
        numpy=np.random.get_state(),
        torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all(),
    )


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])


def save(
    output, step, config, pipe, policy, value, optimizer, accelerator, train_seconds
):
    output = Path(output)
    final = output / f"step_{step}"
    temporary = output / f".step_{step}.partial"
    if final.exists():
        raise FileExistsError(final)
    if accelerator.is_main_process:
        # Retain interrupted writes for inspection without blocking a retry.
        if temporary.exists():
            temporary.rename(output / f".step_{step}.{uuid4().hex}.partial")
        temporary.mkdir(parents=True, exist_ok=False)
    accelerator.wait_for_everyone()
    torch.save(rng_state(), temporary / f"rng_rank_{accelerator.process_index}.pt")
    if accelerator.is_main_process:
        torch.save(
            dict(
                policy=mutable_state(policy),
                value=mutable_state(value) if value is not None else None,
                optimizer=optimizer.state_dict(),
                step=step,
                train_seconds=train_seconds,
                world_size=accelerator.num_processes,
                config=config.to_dict(),
            ),
            temporary / "training.pt",
        )
        from serf.validation.scorers import _load_replicated_peft_adapter

        with _load_replicated_peft_adapter():
            save_policy_exports(pipe, policy, temporary)
        (temporary / "config.json").write_text(
            json.dumps(config.to_dict(), indent=2) + "\n"
        )
        (temporary / "COMPLETE").write_text(f"{config.method}\n")
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        temporary.rename(final)
        latest = output / "latest"
        if latest.exists() and not latest.is_symlink():
            raise FileExistsError(latest)
        link = output / f".latest-{uuid4().hex}"
        link.symlink_to(final.name)
        link.replace(latest)
    accelerator.wait_for_everyone()
    return final


def load(path, config, policy, value, optimizer, accelerator):
    path = Path(path).resolve()
    if path.name.endswith(".partial") or not (path / "COMPLETE").exists():
        raise ValueError("Resume requires a complete checkpoint")
    state = torch.load(path / "training.pt", map_location="cpu", weights_only=False)
    if state["world_size"] != accelerator.num_processes:
        raise ValueError("Exact resume requires the same GPU count")
    mutable = {
        "max_train_steps",
        "num_epochs",
        "wandb_project",
        "wandb_entity",
        "wandb_run_name",
        "checkpoint_every",
    }
    mismatches = [
        k
        for k in state["config"]
        if k not in mutable and state["config"][k] != config.to_dict().get(k)
    ]
    if mismatches:
        raise ValueError(f"Resume configuration changed: {mismatches}")
    load_mutable(policy, state["policy"])
    if value is not None:
        load_mutable(value, state["value"])
    optimizer.load_state_dict(state["optimizer"])
    # Caller restores this after all model, sampler and validation initialization.
    rng = torch.load(
        path / f"rng_rank_{accelerator.process_index}.pt",
        map_location="cpu",
        weights_only=False,
    )
    return state["step"], state["train_seconds"], rng
