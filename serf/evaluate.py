"""Evaluate a saved policy EMA using the complete paper protocol."""

import argparse, json
from pathlib import Path
from .config import Config, ROOT


def evaluate(config, checkpoint, output):
    from accelerate import Accelerator
    from .shared.evaluation_ema import load_policy_export, require_evaluation_artifact
    from .validation import ValidationRunner, apply_validation_defaults
    from .validation.scorers import _load_replicated_peft_adapter

    ac = Accelerator(mixed_precision="bf16")
    apply_validation_defaults(config)
    # Evaluation needs only the policy, never the VGG value network.
    from .methods.am_model import build_pipeline

    pipe = build_pipeline(config, ac.device)
    with _load_replicated_peft_adapter():
        load_policy_export(
            pipe.transformer, require_evaluation_artifact(checkpoint), "evaluation"
        )
    step = int(Path(checkpoint).resolve().name.split("_")[-1])
    validator = ValidationRunner(
        config,
        pipe,
        pipe.transformer,
        ac,
        Path(output),
        current_adapter="evaluation",
        task_prompt_path=ROOT / "prompts" / config.prompts / "test.txt",
    )
    validator.run(step)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    raw = json.loads((a.checkpoint / "config.json").read_text())
    c = Config(raw["method"], raw["reward"], seed=raw["seed"])
    c.__dict__.update(raw)
    evaluate(c, a.checkpoint, a.output)


if __name__ == "__main__":
    main()
