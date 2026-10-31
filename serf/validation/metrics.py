"""Metric plugins used by task and distribution validation suites."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from importlib.metadata import PackageNotFoundError, version

import torch

import serf.rewards as reward_models
from .scorers import (
    CanonicalFidScorer,
    DreamSimScorer,
    dreamsim_diversity,
    per_prompt_fid,
)
from serf.rewards.paths import MODELS_DIR


REWARD_CLASS_ALIASES = {"PickScore": "PickScoreDiff", "CLIPScore": "ClipScore"}
MAIN_PROCESS_ONLY = set()


def package_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return "missing"


class MetricPlugin(ABC):
    name: str
    main_process_only = False

    @property
    @abstractmethod
    def identity(self):
        """Stable identity used for derived-reference caches."""

    @abstractmethod
    def score(self, images, prompts):
        """Return tensors whose first dimension corresponds to images."""

    def close(self):
        pass


class RewardMetric(MetricPlugin):
    def __init__(self, name, device, batch_size, module=None, aliases=False):
        self.name = name
        self.device = device
        self.batch_size = batch_size
        class_name = REWARD_CLASS_ALIASES.get(name, name) if aliases else name
        if not hasattr(reward_models, class_name):
            raise ValueError(f"Unknown validation reward {name!r}")
        self.class_name = class_name
        self.module = module or getattr(reward_models, class_name)()
        self.module = self.module.to(device, dtype=torch.float32).eval()
        self.main_process_only = name in MAIN_PROCESS_ONLY

    @property
    def identity(self):
        from serf.rewards import model_ids

        revisions = {
            name: value
            for name, value in vars(model_ids).items()
            if name.endswith(("_ID", "_REVISION")) and isinstance(value, str)
        }
        return {
            "kind": "reward",
            "name": self.name,
            "class": self.class_name,
            "implementation": type(self.module).__module__,
            "model_revisions": revisions,
            "packages": {
                name: package_version(name)
                for name in ("transformers", "hpsv2", "image-reward")
            },
        }

    @torch.no_grad()
    def score(self, images, prompts):
        chunks = defaultdict(list)
        for start in range(0, len(images), self.batch_size):
            batch = images[start : start + self.batch_size].to(
                self.device, dtype=torch.float32
            )
            batch_prompts = prompts[start : start + self.batch_size]
            with torch.autocast(device_type=self.device.type, enabled=False):
                result = self.module(batch, batch_prompts)
            if torch.is_tensor(result):
                chunks[self.name].append(result.detach().float().cpu())
            else:
                for key, value in result.items():
                    chunks[f"{self.name}/{key}"].append(value.detach().float().cpu())
        return {name: torch.cat(values) for name, values in chunks.items()}

    def close(self):
        self.module = None


class DreamSimMetric(MetricPlugin):
    def __init__(self, type_name, cache_dir, batch_size, device):
        self.name = f"dreamsim:{type_name}"
        self.type_name = type_name
        self.scorer = DreamSimScorer(type_name, cache_dir, batch_size, device)

    @property
    def identity(self):
        return {
            "kind": "dreamsim",
            "type": self.type_name,
            "version": package_version("dreamsim"),
        }

    def score(self, images, prompts):
        del prompts
        return {self.name: self.scorer.embed(images).detach().float().cpu()}

    def reduce_group(self, features):
        return dreamsim_diversity(features).item()

    def close(self):
        self.scorer = None


class FidMetric(MetricPlugin):
    name = "fid"

    def __init__(self, batch_size, device):
        self.scorer = CanonicalFidScorer(batch_size, device)

    @property
    def identity(self):
        return {
            "kind": "fid",
            "version": package_version("torch-fidelity"),
            "features": "inception-v3-compat-2048",
        }

    def score(self, images, prompts):
        del prompts
        return {self.name: self.scorer.embed(images).detach().float().cpu()}

    @staticmethod
    def compare(features, reference):
        return per_prompt_fid(features, reference).item()

    def close(self):
        self.scorer = None


class MetricRegistry:
    def __init__(self, device, batch_size, reusable=None):
        self.device = device
        self.batch_size = batch_size
        self.reusable = dict(reusable or {})

    def reward(self, name, *, aliases=False):
        reusable = self.reusable.get(name)
        return RewardMetric(
            name, self.device, self.batch_size, module=reusable, aliases=aliases
        )

    def standard_distribution_plugins(self, config):
        plugins = [
            self.reward(name, aliases=True)
            for name in config.distribution_rewards
            if name not in MAIN_PROCESS_ONLY
        ]
        if config.distribution_compute_fid:
            plugins.append(FidMetric(self.batch_size, self.device))
        return plugins

    def dreamsim(self, type_name):
        return DreamSimMetric(type_name, MODELS_DIR, self.batch_size, self.device)


def merge_score_chunks(chunks):
    merged = defaultdict(list)
    for chunk in chunks:
        for name, values in chunk.items():
            merged[name].append(values)
    return {name: torch.cat(values) for name, values in merged.items()}
