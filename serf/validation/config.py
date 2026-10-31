"""Typed view over the repository's backward-compatible flat validation config."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

from serf.config import ROOT


DEFAULT_DRAWBENCH_REWARDS = ("PickScore", "HPSv2", "ImageReward", "CLIPScore")
VALID_DRAWBENCH_REWARDS = set(DEFAULT_DRAWBENCH_REWARDS)
VALID_SCHEDULERS = {"inherit", "euler", "dpmpp_2m"}
VALID_DREAMSIM_TYPES = {
    "ensemble",
    "dino_vitb16",
    "dino_vitb16_patch",
    "dinov2_vitb14",
    "dinov2_vitb14_patch",
    "clip_vitb32",
    "open_clip_vitb32",
    "synclr_vitb16",
}


def _positive_int(name, value, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return value


def _scales(config):
    values = getattr(config, "eval_guidance_scales", None)
    if values is None:
        values = [getattr(config, "eval_guidance_scale", 1.0)]
    if (
        isinstance(values, (str, bytes))
        or not isinstance(values, (list, tuple))
        or not values
    ):
        raise ValueError("eval_guidance_scales must be a non-empty list")
    result = [float(value) for value in values]
    if any(not math.isfinite(value) or value < 0 for value in result):
        raise ValueError("eval_guidance_scales must contain finite non-negative values")
    if len(set(result)) != len(result):
        raise ValueError("eval_guidance_scales must contain unique values")
    return tuple(result)


def _cache_root(config):
    explicit = getattr(config, "validation_reference_cache_dir", None)
    legacy = getattr(config, "distribution_eval_reference_cache_dir", None)
    if explicit and legacy and Path(explicit) != Path(legacy):
        raise ValueError(
            "validation_reference_cache_dir and distribution_eval_reference_cache_dir disagree"
        )
    if explicit or legacy:
        return Path(explicit or legacy).expanduser()
    environment = os.environ.get("SERF_VALIDATION_CACHE")
    if environment:
        return Path(environment).expanduser()
    weights = os.environ.get("SERF_CACHE_DIR")
    if weights:
        return Path(weights).expanduser() / "validation-cache"
    return Path("models") / "validation-cache"


@dataclass(frozen=True)
class DistributionSuite:
    namespace: str
    source: str
    prompt_path: Path | None
    num_prompts: int
    prompt_seed: int
    deduplicate: bool


def _distribution_suites(config, *, explicit):
    if not explicit:
        if config.distribution_eval_num_prompts == 0:
            return ()
        return (
            DistributionSuite(
                namespace=str(config.distribution_eval_namespace).strip().strip("/"),
                source="file",
                prompt_path=Path(config.distribution_eval_prompt_path),
                num_prompts=_positive_int(
                    "distribution_eval_num_prompts",
                    config.distribution_eval_num_prompts,
                    minimum=0,
                ),
                prompt_seed=_positive_int(
                    "distribution_eval_prompt_seed",
                    config.distribution_eval_prompt_seed,
                    minimum=0,
                ),
                deduplicate=False,
            ),
        )

    raw_suites = config.distribution_eval_suites
    if not isinstance(raw_suites, (list, tuple)) or not raw_suites:
        raise ValueError("distribution_eval_suites must be a non-empty list")
    suites = []
    for index, raw in enumerate(raw_suites):
        if not isinstance(raw, dict):
            raise ValueError(f"distribution_eval_suites[{index}] must be a mapping")
        source = str(raw.get("source", "file"))
        if source not in {"file", "task_test"}:
            raise ValueError(
                f"distribution_eval_suites[{index}].source must be 'file' or 'task_test'"
            )
        namespace = str(raw.get("namespace", "")).strip().strip("/")
        if not namespace:
            raise ValueError(
                f"distribution_eval_suites[{index}].namespace must be non-empty"
            )
        prompt_path = raw.get("prompt_path")
        if source == "file" and not prompt_path:
            raise ValueError(
                f"distribution_eval_suites[{index}].prompt_path is required for file sources"
            )
        if source == "task_test" and prompt_path is not None:
            raise ValueError(
                f"distribution_eval_suites[{index}] cannot set prompt_path for task_test"
            )
        suites.append(
            DistributionSuite(
                namespace=namespace,
                source=source,
                prompt_path=(ROOT / prompt_path).resolve() if prompt_path else None,
                num_prompts=_positive_int(
                    f"distribution_eval_suites[{index}].num_prompts",
                    raw.get("num_prompts", config.distribution_eval_num_prompts),
                ),
                prompt_seed=_positive_int(
                    f"distribution_eval_suites[{index}].prompt_seed",
                    raw.get("prompt_seed", config.distribution_eval_prompt_seed),
                    minimum=0,
                ),
                deduplicate=bool(raw.get("deduplicate", True)),
            )
        )
    namespaces = [suite.namespace for suite in suites]
    if len(namespaces) != len(set(namespaces)):
        raise ValueError("distribution_eval_suites namespaces must be unique")
    return tuple(suites)


def apply_validation_defaults(config):
    """Populate new fields without breaking old YAML files."""
    had_samples_per_prompt = hasattr(config, "distribution_eval_samples_per_prompt")
    defaults = {
        "eval_sampling_scheduler": getattr(config, "sampling_scheduler", "inherit"),
        "eval_sampling_batch_size": None,
        "distribution_eval_num_prompts": 0,
        "distribution_eval_prompt_path": "prompts/drawbench/test.txt",
        "distribution_eval_namespace": "drawbench",
        "distribution_eval_samples_per_prompt": 16,
        "distribution_eval_images_per_prompt": None,
        "distribution_eval_prompt_seed": 20260529,
        "distribution_eval_seed_base": 0,
        "distribution_eval_dreamsim_types": ["dino_vitb16_patch"],
        "distribution_eval_batch_size": None,
        "distribution_eval_rewards": list(DEFAULT_DRAWBENCH_REWARDS),
        "distribution_eval_compute_fid": True,
        "distribution_eval_log_images": True,
        "distribution_eval_suites": None,
        "distribution_eval_cfgs": None,
        "distribution_eval_num_inference_steps": None,
        "distribution_eval_sampling_scheduler": None,
        "distribution_eval_reference_cache_dir": None,
        "validation_reference_cache_dir": None,
    }
    for name, value in defaults.items():
        if not hasattr(config, name):
            setattr(config, name, value)
    if config.distribution_eval_images_per_prompt is not None:
        images_per_prompt = _positive_int(
            "distribution_eval_images_per_prompt",
            config.distribution_eval_images_per_prompt,
            minimum=2,
        )
        if (
            had_samples_per_prompt
            and config.distribution_eval_samples_per_prompt != images_per_prompt
        ):
            raise ValueError(
                "distribution_eval_images_per_prompt and the legacy "
                "distribution_eval_samples_per_prompt disagree"
            )
        config.distribution_eval_samples_per_prompt = images_per_prompt
    if config.distribution_eval_batch_size is None:
        config.distribution_eval_batch_size = config.reward_batch_size
    return config


@dataclass(frozen=True)
class ValidationConfig:
    model_name: str
    mixed_precision: str
    resolution: int
    num_steps: int
    guidance_scales: tuple[float, ...]
    seed: int
    prompt_fraction: float
    sampling_scheduler: str
    sampling_batch_size: int | None
    legacy_sampling_batch_size: int
    reward_batch_size: int
    task_rewards: tuple[str, ...]
    wandb_project: str | None
    distribution_num_prompts: int
    distribution_prompt_path: Path
    distribution_namespace: str
    distribution_samples_per_prompt: int
    distribution_prompt_seed: int
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
    run_task_validation: bool
    reference_cache_dir: Path

    @classmethod
    def from_namespace(cls, config):
        explicit_suites = getattr(config, "distribution_eval_suites", None) is not None
        apply_validation_defaults(config)
        scheduler = str(config.eval_sampling_scheduler)
        if scheduler not in VALID_SCHEDULERS:
            raise ValueError(
                f"eval_sampling_scheduler must be one of {sorted(VALID_SCHEDULERS)}, got {scheduler!r}"
            )
        fraction = float(getattr(config, "eval_prompt_fraction", 1.0))
        if not math.isfinite(fraction) or not 0 < fraction <= 1:
            raise ValueError("eval_prompt_fraction must be finite and in (0, 1]")
        explicit_batch = config.eval_sampling_batch_size
        if explicit_batch is not None:
            explicit_batch = _positive_int("eval_sampling_batch_size", explicit_batch)
        dreamsim = config.distribution_eval_dreamsim_types
        if isinstance(dreamsim, str) or not isinstance(dreamsim, (list, tuple)):
            raise ValueError("distribution_eval_dreamsim_types must be a list")
        unknown_dreamsim = sorted(set(dreamsim) - VALID_DREAMSIM_TYPES)
        if unknown_dreamsim:
            raise ValueError(
                f"Unknown DreamSim types {unknown_dreamsim}; choose from {sorted(VALID_DREAMSIM_TYPES)}"
            )
        rewards = config.distribution_eval_rewards
        if isinstance(rewards, str) or not isinstance(rewards, (list, tuple)):
            raise ValueError("distribution_eval_rewards must be a list")
        unknown_rewards = sorted(set(rewards) - VALID_DRAWBENCH_REWARDS)
        if unknown_rewards:
            raise ValueError(
                f"Unknown DrawBench rewards {unknown_rewards}; choose from {sorted(VALID_DRAWBENCH_REWARDS)}"
            )
        prompt_path = Path(config.distribution_eval_prompt_path)
        namespace = str(config.distribution_eval_namespace).strip().strip("/")
        if not namespace:
            raise ValueError("distribution_eval_namespace must be non-empty")
        distribution_cfgs = config.distribution_eval_cfgs
        if distribution_cfgs is None:
            distribution_cfgs = list(_scales(config))
        elif (
            isinstance(distribution_cfgs, (str, bytes))
            or not isinstance(distribution_cfgs, (list, tuple))
            or not distribution_cfgs
        ):
            raise ValueError("distribution_eval_cfgs must be a non-empty list")
        distribution_cfgs = tuple(float(value) for value in distribution_cfgs)
        if any(not math.isfinite(value) or value < 0 for value in distribution_cfgs):
            raise ValueError(
                "distribution_eval_cfgs must contain finite non-negative values"
            )
        if len(set(distribution_cfgs)) != len(distribution_cfgs):
            raise ValueError("distribution_eval_cfgs must contain unique values")
        distribution_steps = config.distribution_eval_num_inference_steps
        if distribution_steps is None:
            distribution_steps = config.num_eval_inference_steps
        distribution_scheduler = config.distribution_eval_sampling_scheduler
        if distribution_scheduler is None:
            distribution_scheduler = scheduler
        distribution_scheduler = str(distribution_scheduler)
        if distribution_scheduler not in VALID_SCHEDULERS:
            raise ValueError(
                "distribution_eval_sampling_scheduler must be one of "
                f"{sorted(VALID_SCHEDULERS)}, got {distribution_scheduler!r}"
            )
        suites = _distribution_suites(config, explicit=explicit_suites)
        return cls(
            model_name=str(config.model_name),
            mixed_precision=str(config.mixed_precision),
            resolution=_positive_int("resolution", config.resolution),
            num_steps=_positive_int(
                "num_eval_inference_steps", config.num_eval_inference_steps
            ),
            guidance_scales=_scales(config),
            seed=_positive_int("seed", getattr(config, "seed", 0), minimum=0),
            prompt_fraction=fraction,
            sampling_scheduler=scheduler,
            sampling_batch_size=explicit_batch,
            legacy_sampling_batch_size=_positive_int(
                "sampling_batch_size", config.sampling_batch_size
            ),
            reward_batch_size=_positive_int(
                "reward_batch_size", config.reward_batch_size
            ),
            task_rewards=tuple(getattr(config, "validation_rewards", ())),
            wandb_project=getattr(config, "wandb_project", None),
            distribution_num_prompts=_positive_int(
                "distribution_eval_num_prompts",
                config.distribution_eval_num_prompts,
                minimum=0,
            ),
            distribution_prompt_path=prompt_path,
            distribution_namespace=namespace,
            distribution_samples_per_prompt=_positive_int(
                "distribution_eval_samples_per_prompt",
                config.distribution_eval_samples_per_prompt,
                minimum=2,
            ),
            distribution_prompt_seed=_positive_int(
                "distribution_eval_prompt_seed",
                config.distribution_eval_prompt_seed,
                minimum=0,
            ),
            distribution_seed_base=_positive_int(
                "distribution_eval_seed_base",
                config.distribution_eval_seed_base,
                minimum=0,
            ),
            distribution_dreamsim_types=tuple(dict.fromkeys(dreamsim)),
            distribution_batch_size=_positive_int(
                "distribution_eval_batch_size", config.distribution_eval_batch_size
            ),
            distribution_rewards=tuple(dict.fromkeys(rewards)),
            distribution_compute_fid=bool(config.distribution_eval_compute_fid),
            distribution_log_images=bool(config.distribution_eval_log_images),
            distribution_suites=suites,
            distribution_guidance_scales=distribution_cfgs,
            distribution_num_steps=_positive_int(
                "distribution_eval_num_inference_steps", distribution_steps
            ),
            distribution_sampling_scheduler=distribution_scheduler,
            run_task_validation=not explicit_suites,
            reference_cache_dir=_cache_root(config),
        )

    def actual_sampling_batch_size(self, guidance_scale):
        if self.sampling_batch_size is not None:
            return self.sampling_batch_size
        # Preserve the historical memory convention for untouched configs.
        divisor = 2 if guidance_scale > 1 else 1
        value = self.legacy_sampling_batch_size // divisor
        if value <= 0:
            raise ValueError(
                "sampling_batch_size is too small for classifier-free guidance"
            )
        return value
