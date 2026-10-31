"""Typed evaluation settings for the explicit paper recipes."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

from serf.config import ROOT

REWARDS = {"PickScore", "HPSv2", "ImageReward", "CLIPScore"}


def _positive_int(name, value, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return value


def _scales(values):
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError("Evaluation guidance scales must be a non-empty list")
    result = tuple(float(value) for value in values)
    if any(not math.isfinite(value) or value < 0 for value in result):
        raise ValueError("Evaluation guidance scales must be finite and non-negative")
    if len(result) != len(set(result)):
        raise ValueError("Evaluation guidance scales must be unique")
    return result


def _cache_root(config):
    value = config.validation_reference_cache_dir or os.environ.get(
        "SERF_VALIDATION_CACHE"
    )
    return Path(value).expanduser() if value else Path("models/validation-cache")


@dataclass(frozen=True)
class DistributionSuite:
    namespace: str
    source: str
    prompt_path: Path | None
    num_prompts: int
    prompt_seed: int
    deduplicate: bool


def _distribution_suites(config):
    raw_suites = config.distribution_eval_suites
    if not isinstance(raw_suites, (list, tuple)) or not raw_suites:
        raise ValueError("The paper recipes require explicit evaluation suites")
    result = []
    for raw in raw_suites:
        namespace = str(raw.get("namespace", "")).strip().strip("/")
        source = raw.get("source", "file")
        path = raw.get("prompt_path")
        if not namespace or source not in {"file", "task_test"}:
            raise ValueError("Each suite needs a namespace and file/task_test source")
        if (source == "file" and not path) or (
            source == "task_test" and path is not None
        ):
            raise ValueError("Only file-based evaluation suites specify prompt_path")
        result.append(
            DistributionSuite(
                namespace=namespace,
                source=source,
                prompt_path=(ROOT / path).resolve() if path else None,
                num_prompts=_positive_int(
                    "num_prompts",
                    raw.get("num_prompts", config.distribution_eval_num_prompts),
                ),
                prompt_seed=_positive_int(
                    "prompt_seed",
                    raw.get("prompt_seed", config.distribution_eval_prompt_seed),
                    minimum=0,
                ),
                deduplicate=bool(raw.get("deduplicate", True)),
            )
        )
    if len({suite.namespace for suite in result}) != len(result):
        raise ValueError("Evaluation suite namespaces must be unique")
    return tuple(result)


def apply_validation_defaults(config):
    """Resolve the optional execution batch size without changing the paper protocol."""
    if config.distribution_eval_batch_size is None:
        config.distribution_eval_batch_size = config.reward_batch_size
    return config


@dataclass(frozen=True)
class ValidationConfig:
    model_name: str
    mixed_precision: str
    resolution: int
    num_steps: int
    sampling_scheduler: str
    wandb_project: str | None
    distribution_samples_per_prompt: int
    distribution_seed_base: int
    distribution_dreamsim_types: tuple[str, ...]
    distribution_batch_size: int
    distribution_rewards: tuple[str, ...]
    distribution_compute_fid: bool
    distribution_log_images: bool
    distribution_suites: tuple[DistributionSuite, ...]
    distribution_guidance_scales: tuple[float, ...]
    distribution_num_steps: int
    distribution_sampling_scheduler: str
    reference_cache_dir: Path

    @classmethod
    def from_namespace(cls, config):
        apply_validation_defaults(config)
        if (
            config.eval_sampling_scheduler != "euler"
            or config.distribution_eval_sampling_scheduler != "euler"
        ):
            raise ValueError("Paper evaluation uses the Euler scheduler")
        if (
            config.distribution_eval_images_per_prompt
            != config.distribution_eval_samples_per_prompt
        ):
            raise ValueError("Evaluation images/samples-per-prompt settings disagree")
        rewards = config.distribution_eval_rewards
        if (
            not isinstance(rewards, (list, tuple))
            or not rewards
            or set(rewards) - REWARDS
        ):
            raise ValueError(
                f"Evaluation rewards must be selected from {sorted(REWARDS)}"
            )
        if config.distribution_eval_dreamsim_types != ["ensemble"]:
            raise ValueError("The paper recipes report DreamSim ensemble diversity")
        return cls(
            model_name=str(config.model_name),
            mixed_precision=str(config.mixed_precision),
            resolution=_positive_int("resolution", config.resolution),
            num_steps=_positive_int(
                "num_eval_inference_steps", config.num_eval_inference_steps
            ),
            sampling_scheduler="euler",
            wandb_project=config.wandb_project,
            distribution_samples_per_prompt=_positive_int(
                "samples_per_prompt",
                config.distribution_eval_samples_per_prompt,
                minimum=2,
            ),
            distribution_seed_base=_positive_int(
                "seed_base", config.distribution_eval_seed_base, minimum=0
            ),
            distribution_dreamsim_types=("ensemble",),
            distribution_batch_size=_positive_int(
                "distribution_batch_size", config.distribution_eval_batch_size
            ),
            distribution_rewards=tuple(dict.fromkeys(rewards)),
            distribution_compute_fid=bool(config.distribution_eval_compute_fid),
            distribution_log_images=bool(config.distribution_eval_log_images),
            distribution_suites=_distribution_suites(config),
            distribution_guidance_scales=_scales(config.distribution_eval_cfgs),
            distribution_num_steps=_positive_int(
                "distribution_num_inference_steps",
                config.distribution_eval_num_inference_steps,
            ),
            distribution_sampling_scheduler="euler",
            reference_cache_dir=_cache_root(config),
        )
