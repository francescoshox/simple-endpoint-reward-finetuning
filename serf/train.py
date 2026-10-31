"""Train one explicit paper recipe; validation is enabled in every public recipe."""

import argparse, json, os, subprocess, sys, time
from pathlib import Path
from .config import Config, METHODS, REWARDS, ROOT


def train(config, output, resume=None):
    if config.method == "adjoint_matching" and config.am_deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from accelerate import Accelerator
    from accelerate.utils import set_seed, InitProcessGroupKwargs
    from datetime import timedelta
    from .shared import updates, checkpoint
    from .shared.batch_sampler import PromptSeedBatchSampler
    from .shared.models import load_prompts
    from .shared.distributed_metrics import reduce_max, reduce_scalar_dict
    from .shared.evaluation_ema import update_evaluation_ema
    from .shared.gradients import synchronize_gradients
    from .validation import ValidationRunner, apply_validation_defaults
    from . import rewards

    if not torch.cuda.is_available():
        raise RuntimeError("Training requires CUDA; --check-config needs no GPU")
    if config.method == "adjoint_matching":
        torch.use_deterministic_algorithms(config.am_deterministic)
    ac = Accelerator(
        mixed_precision="bf16",
        kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(hours=2))],
    )
    config.validate(ac.num_processes)
    apply_validation_defaults(config)
    output = Path(output).resolve()
    if not resume and (output / "config.json").exists():
        raise FileExistsError("Existing run requires --resume")
    if ac.is_main_process:
        output.mkdir(parents=True, exist_ok=True)
    ac.wait_for_everyone()
    set_seed(config.seed)
    pipe, policy, value, live, ema, parameters = updates.build(config, ac)
    optimizer = torch.optim.AdamW(
        parameters,
        lr=config.lr,
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay,
        eps=config.adam_epsilon,
    )
    scorer = (
        getattr(rewards, config.training_reward_name)()
        .to(ac.device, dtype=torch.float32)
        .eval()
    )
    if any(p.requires_grad for p in scorer.parameters()):
        raise RuntimeError("Reward weights must be frozen")
    if config.method == "vgg_flow":
        from .methods.vgg_flow import SharedTrainingReward, make_algorithm

        training_reward = SharedTrainingReward(scorer)
        algorithm = make_algorithm(config, pipe, policy, value, training_reward)
    else:
        training_reward = scorer
        algorithm = None
    train_prompts = load_prompts(ROOT / "prompts" / config.prompts / "train.txt")
    sampler = PromptSeedBatchSampler(
        len(train_prompts), 64, 1, ac.num_processes, ac.process_index, config.seed
    )
    # Persistent local JSONL is authoritative; W&B is optional and one row/update.
    tracker = None
    if config.wandb_project and ac.is_main_process:
        import wandb

        id_path = output / "wandb_run_id.txt"
        previous_id = (
            id_path.read_text().strip() if resume and id_path.exists() else None
        )
        tracker = wandb.init(
            id=previous_id,
            resume="must" if previous_id else None,
            project=config.wandb_project,
            entity=config.wandb_entity,
            name=config.wandb_run_name or output.name,
            config=config.to_dict(),
            dir=str(output),
        )
        id_path.write_text(tracker.id + "\n")
    pending = {}

    def log(values, step=None, **kwargs):
        if ac.is_main_process:
            with (output / "metrics.jsonl").open("a") as f:
                f.write(json.dumps(dict(step=step, **values)) + "\n")
            pending.update(values)

    ac.log = log
    ac.log_artifacts = pending.update
    validator = ValidationRunner(
        config,
        pipe,
        policy,
        ac,
        output,
        current_adapter="evaluation",
        task_prompt_path=ROOT / "prompts" / config.prompts / "test.txt",
        reusable_rewards={
            config.training_reward_name: scorer,
            "HPSv2" if config.reward == "hpsv21" else "PickScore": scorer,
        },
    )
    completed, seconds, rng = 0, 0.0, None
    if resume:
        completed, seconds, rng = checkpoint.load(
            resume, config, policy, value, optimizer, ac
        )
    if ac.is_main_process:
        (output / "config.json").write_text(
            json.dumps(config.to_dict(), indent=2) + "\n"
        )
        import yaml

        (output / "config.yaml").write_text(yaml.safe_dump(config.to_dict()))
    set_seed(config.seed + ac.process_index)
    if rng is not None:
        checkpoint.restore_rng(rng)

    def validate_step(step, checkpoint_dir):
        torch.cuda.empty_cache()
        validator.run(
            step,
            training_rewards={config.training_reward_name: scorer},
            train_seconds=seconds,
        )
        if ac.is_main_process:
            (Path(checkpoint_dir) / "VALIDATED").write_text(str(step) + "\n")
        ac.wait_for_everyone()
        torch.cuda.empty_cache()

    # A complete training checkpoint can survive an interruption during eval.
    # Finish that evaluation before taking another optimizer update.
    if (
        resume
        and completed
        and (
            completed % config.eval_every_n_epochs == 0
            or completed == config.max_train_steps
        )
        and not (Path(resume) / "VALIDATED").exists()
    ):
        validate_step(completed, resume)
        if tracker:
            tracker.log(dict(pending), step=completed)
            pending.clear()
    for epoch in range(completed, config.max_train_steps):
        torch.cuda.synchronize()
        started = time.perf_counter()
        sampler.set_epoch(epoch)
        entries = next(iter(sampler))
        prompts = [train_prompts[i] for i, _ in entries]
        seeds = [s[0] for _, s in entries]
        optimizer.zero_grad(set_to_none=True)
        policy.set_adapter("default")
        if config.method in ("serf", "rsm"):
            metrics = updates.first_order_update(
                config, pipe, policy, training_reward, prompts, seeds, ac
            )
        elif config.method == "draft_lv":
            metrics = updates.draft_update(
                config, pipe, policy, training_reward, prompts, seeds, ac
            )
        elif config.method == "adjoint_matching":
            metrics = updates.am_update(
                config, pipe, policy, training_reward, prompts, seeds, ac
            )
        else:
            metrics = updates.vgg_update(
                config,
                pipe,
                policy,
                value,
                training_reward,
                algorithm,
                prompts,
                seeds,
                ac,
            )
        synchronize_gradients(parameters, ac)
        norm = torch.nn.utils.clip_grad_norm_(
            parameters, config.max_grad_norm or float("inf"), error_if_nonfinite=True
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        update_evaluation_ema(live, ema)
        torch.cuda.synchronize()
        elapsed = reduce_max(ac, time.perf_counter() - started)
        seconds += elapsed
        step = epoch + 1
        metrics = reduce_scalar_dict(ac, metrics)
        values = {f"train/{k}": v for k, v in metrics.items()}
        values.update(
            {
                "train/grad_norm": float(norm),
                "train/optimizer_updates": step,
                "train/update_seconds": elapsed,
                "train/gpu_hours": seconds * ac.num_processes / 3600,
                "train/trajectories": step * 64,
                "train/targets": step * 256,
            }
        )
        ac.log(values, step=step)
        ac.print(
            f"{config.method}: update {step}/{config.max_train_steps}, {elapsed:.1f}s, loss={metrics['loss']:.6g}"
        )
        due = step % config.eval_every_n_epochs == 0 or step == config.max_train_steps
        if step % config.checkpoint_every == 0 or due:
            checkpoint.save(
                output, step, config, pipe, policy, value, optimizer, ac, seconds
            )
        if due:
            validate_step(step, output / f"step_{step}")
        if tracker:
            tracker.log(dict(pending), step=step)
            pending.clear()
    if tracker:
        tracker.finish()
    ac.end_training()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", choices=METHODS, required=True)
    p.add_argument("--reward", choices=REWARDS, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=Path)
    p.add_argument("--resume", type=Path)
    p.add_argument("--gpus", type=int, default=4)
    p.add_argument("--wandb-project")
    p.add_argument("--wandb-entity")
    p.add_argument("--check-config", action="store_true")
    args = p.parse_args()
    config = Config(args.method, args.reward, seed=args.seed)
    config.wandb_project = args.wandb_project
    config.wandb_entity = args.wandb_entity
    if args.check_config:
        print(
            json.dumps(
                dict(config=config.to_dict(), accounting=config.validate(args.gpus)),
                indent=2,
            )
        )
        return
    if "LOCAL_RANK" not in os.environ and args.gpus > 1:
        import socket

        # An explicit IPv4 rendezvous also works on clusters without IPv6.
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--master_addr=127.0.0.1",
            f"--master_port={port}",
            f"--nproc_per_node={args.gpus}",
            "-m",
            "serf.train",
            *sys.argv[1:],
        ]
        subprocess.run(command, check=True)
        return
    output = args.output or Path("runs") / f"{args.reward}-{args.method}-s{args.seed}"
    train(config, output, args.resume)


if __name__ == "__main__":
    main()
