"""Shared task and DrawBench validation runner."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from accelerate.utils import gather_object
from PIL import Image

from serf.shared.distributed_metrics import reduce_max

from .backend import SD3GenerationBackend
from .cache import CACHE_SCHEMA, AtomicTensorCache, canonical_metadata, content_key
from .config import ValidationConfig
from .metrics import DreamSimMetric, FidMetric, MAIN_PROCESS_ONLY, MetricRegistry


DRAWBENCH_SEED_OFFSET = 202
PREVIEW_PROMPTS = 4
PREVIEW_SAMPLES = 4


def metadata_differences(local, reference, path="metadata", limit=8):
    """Describe the first cache-metadata disagreements without dumping configs."""

    differences = []

    def visit(left, right, current):
        if len(differences) >= limit:
            return
        if type(left) is not type(right):
            differences.append(
                f"{current}: type {type(left).__name__} != {type(right).__name__}"
            )
        elif isinstance(left, dict):
            for key in sorted(set(left) | set(right)):
                child = f"{current}.{key}"
                if key not in left:
                    differences.append(f"{child}: missing locally")
                elif key not in right:
                    differences.append(f"{child}: missing on rank 0")
                else:
                    visit(left[key], right[key], child)
                if len(differences) >= limit:
                    break
        elif isinstance(left, list):
            if len(left) != len(right):
                differences.append(f"{current}: length {len(left)} != {len(right)}")
            else:
                for index, (left_item, right_item) in enumerate(
                    zip(left, right, strict=True)
                ):
                    visit(left_item, right_item, f"{current}[{index}]")
                    if len(differences) >= limit:
                        break
        elif left != right:
            differences.append(f"{current}: {left!r} != {right!r}")

    visit(local, reference, path)
    return differences


def cfg_label(value):
    return f"{float(value):g}"


def load_prompts(path):
    path = Path(path)
    if (
        not path.exists()
        and path.suffix == ".txt"
        and path.with_suffix(".json").exists()
    ):
        path = path.with_suffix(".json")
    if path.suffix == ".json":
        import json

        records = json.loads(path.read_text())
        if not isinstance(records, list):
            raise ValueError(f"Expected a JSON prompt list in {path}")
        prompts = [
            record["text"].strip()
            for record in records
            if record.get("text", "").strip()
        ]
    else:
        prompts = [
            line.strip() for line in path.read_text().splitlines() if line.strip()
        ]
    return prompts


def select_distribution_prompts(path, count, seed, *, deduplicate=False):
    source = [
        {"source_index": index, "prompt": line.strip()}
        for index, line in enumerate(Path(path).read_text().splitlines())
        if line.strip()
    ]
    if deduplicate:
        unique = {}
        for record in source:
            unique.setdefault(record["prompt"], record)
        source = list(unique.values())
    if count > len(source):
        qualifier = " unique" if deduplicate else ""
        raise ValueError(
            f"Requested {count} prompts from {path}, but found {len(source)}{qualifier} prompts"
        )
    selected = random.Random(seed + DRAWBENCH_SEED_OFFSET).sample(source, count)
    return [dict(record, prompt_index=index) for index, record in enumerate(selected)]


def select_drawbench_prompts(path, count, seed):
    """Backward-compatible selector for the legacy single DrawBench suite."""
    return select_distribution_prompts(path, count, seed)


@dataclass
class ValidationResult:
    scalars: dict[str, float] = field(default_factory=dict)
    tables: dict[str, object] = field(default_factory=dict)
    images: dict[str, list] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)

    def merge(self, other):
        self.scalars.update(other.scalars)
        self.tables.update(other.tables)
        self.images.update(other.images)
        self.metadata.update(other.metadata)
        return self


class AcceleratorSink:
    """The only W&B-aware part of validation."""

    def __init__(self, accelerator, enabled=False):
        self.accelerator = accelerator
        self.enabled = bool(enabled)

    def write(self, result, step):
        if not self.accelerator.is_main_process:
            return
        if result.scalars:
            self.accelerator.log(result.scalars, step=step)
        if not self.enabled or (not result.tables and not result.images):
            return
        import wandb

        artifacts = {}
        for key, table in result.tables.items():
            columns, rows = table
            artifacts[key] = wandb.Table(columns=columns, data=rows)
        for key, rows in result.images.items():
            artifacts[key] = [
                wandb.Image(Image.fromarray(array), caption=caption)
                for array, caption in rows
            ]
        if artifacts:
            if hasattr(self.accelerator, "log_artifacts"):
                self.accelerator.log_artifacts(artifacts)
            else:
                wandb.log(artifacts, step=step)


class ValidationRunner:
    def __init__(
        self,
        config,
        pipe,
        model,
        accelerator,
        output_dir,
        *,
        current_adapter="evaluation",
        has_base_adapter=True,
        task_prompt_path=None,
        task_namespace="eval",
        reusable_rewards=None,
    ):
        from .policies import PolicyContexts

        self.raw_config = config
        self.config = ValidationConfig.from_namespace(config)
        self.pipe = pipe
        self.model = model
        self.accelerator = accelerator
        self.backend = SD3GenerationBackend(pipe, accelerator, self.config)
        self.policies = PolicyContexts(model, current_adapter, has_base_adapter)
        self.cache = AtomicTensorCache(self.config.reference_cache_dir)
        self.sink = AcceleratorSink(accelerator, self.config.wandb_project)
        self.output_dir = Path(output_dir)
        self.task_prompt_path = Path(task_prompt_path) if task_prompt_path else None
        self.task_namespace = task_namespace
        self.reusable_rewards = dict(reusable_rewards or {})

    def _rng_context(self):
        class Context:
            def __enter__(inner):
                inner.python = random.getstate()
                inner.numpy = np.random.get_state()
                devices = []
                if self.accelerator.device.type == "cuda":
                    devices = [self.accelerator.device.index or 0]
                inner.torch = torch.random.fork_rng(devices=devices)
                inner.torch.__enter__()

            def __exit__(inner, kind, value, traceback):
                try:
                    return inner.torch.__exit__(kind, value, traceback)
                finally:
                    random.setstate(inner.python)
                    np.random.set_state(inner.numpy)

        return Context()

    def _gather_rows(self, local_rows):
        rows = gather_object(local_rows)
        return sorted(
            rows,
            key=lambda row: (
                float(row.get("guidance_scale", 0)),
                int(row.get("prompt_index", 0)),
                int(row.get("sample_index", 0)),
            ),
        )

    def _sample_group(self, prompt, seeds, guidance_scale, *, num_inference_steps=None):
        batches = []
        batch_size = self.config.distribution_batch_size
        for start in range(0, len(seeds), batch_size):
            chunk = seeds[start : start + batch_size]
            batches.append(
                self.backend.sample_latents(
                    [prompt] * len(chunk),
                    chunk,
                    guidance_scale,
                    num_inference_steps=num_inference_steps,
                )
            )
        latents = torch.cat(batches)
        if len(latents) != len(seeds):
            raise RuntimeError(
                f"Distribution sampler returned {len(latents)} images for {len(seeds)} seeds"
            )
        return latents

    def run_task(self, step, training_rewards=None):
        if self.task_prompt_path is None:
            return ValidationResult()
        prompts = load_prompts(self.task_prompt_path)
        if not prompts:
            raise ValueError(
                f"Validation prompt file is empty: {self.task_prompt_path}"
            )
        count = max(1, math.floor(len(prompts) * self.config.prompt_fraction))
        prompts = prompts[:count]
        local_indices = list(range(count))[
            self.accelerator.process_index :: self.accelerator.num_processes
        ]
        training_names = (
            list(getattr(self.raw_config, "rewards", ()))
            if training_rewards is None
            else list(training_rewards)
        )
        training_rewards = dict(training_rewards or {})
        reusable = dict(self.reusable_rewards)
        reusable.update(training_rewards)
        registry = MetricRegistry(
            self.accelerator.device, self.config.reward_batch_size, reusable=reusable
        )
        names = list(dict.fromkeys([*training_names, *self.config.task_rewards]))
        distributed = [
            registry.reward(name) for name in names if name not in MAIN_PROCESS_ONLY
        ]
        main_only_names = [name for name in names if name in MAIN_PROCESS_ONLY]
        local_rows = []
        try:
            for scale in self.config.guidance_scales:
                images_for_scale = []
                prompt_batch = []
                index_batch = []
                bs = self.config.actual_sampling_batch_size(scale)
                for start in range(0, len(local_indices), bs):
                    indices = local_indices[start : start + bs]
                    batch_prompts = [prompts[index] for index in indices]
                    seeds = [self.config.seed + index for index in indices]
                    # Task validation historically initializes noise with one
                    # device generator per prompt; keep that exact convention.
                    terminal = self.backend.sample_latents(
                        batch_prompts, seeds, scale, cpu_noise=False
                    )
                    images = self.backend.decode_fp32(
                        terminal, self.config.reward_batch_size
                    )
                    images_for_scale.append(images)
                    prompt_batch.extend(batch_prompts)
                    index_batch.extend(indices)
                if images_for_scale:
                    images = torch.cat(images_for_scale)
                else:
                    images = torch.empty((0, 3, 0, 0), device=self.accelerator.device)
                plugin_values = {}
                for plugin in distributed:
                    plugin_values.update(plugin.score(images, prompt_batch))
                # Keep uint8 images only for small artifacts and optional main-rank metrics.
                uint8 = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu()
                for offset, (index, prompt) in enumerate(
                    zip(index_batch, prompt_batch)
                ):
                    local_rows.append(
                        {
                            "guidance_scale": float(scale),
                            "prompt_index": index,
                            "prompt": prompt,
                            "scores": {
                                name: values[offset]
                                for name, values in plugin_values.items()
                            },
                            "image": uint8[offset]
                            if index < PREVIEW_PROMPTS or main_only_names
                            else torch.empty(0),
                        }
                    )
            rows = self._gather_rows(local_rows)
            if not self.accelerator.is_main_process:
                return ValidationResult()
            if main_only_names:
                main_registry = MetricRegistry(
                    self.accelerator.device, 1, reusable=reusable
                )
                for name in main_only_names:
                    plugin = main_registry.reward(name)
                    try:
                        for row in rows:
                            image = (
                                row["image"]
                                .unsqueeze(0)
                                .to(self.accelerator.device)
                                .float()
                                / 255
                            )
                            scores = plugin.score(image, [row["prompt"]])
                            row["scores"].update(
                                {key: value[0] for key, value in scores.items()}
                            )
                    finally:
                        plugin.close()
            result = ValidationResult()
            for scale in self.config.guidance_scales:
                selected = [
                    row for row in rows if row["guidance_scale"] == float(scale)
                ]
                keys = sorted({key for row in selected for key in row["scores"]})
                prefix = (
                    self.task_namespace
                    if len(self.config.guidance_scales) == 1
                    else f"{self.task_namespace}/cfg_{cfg_label(scale)}"
                )
                for key in keys:
                    values = torch.stack(
                        [
                            torch.as_tensor(row["scores"][key]).float()
                            for row in selected
                        ]
                    )
                    result.scalars[f"{prefix}/{key}"] = torch.nanmean(values).item()
                score_keys = [name for name in training_names if name in keys]
                if score_keys:
                    totals = [
                        sum(float(row["scores"][name]) for name in score_keys)
                        for row in selected
                    ]
                    result.scalars[f"{prefix}/rewards"] = float(np.mean(totals))
                if self.config.wandb_project:
                    key = (
                        "samples"
                        if len(self.config.guidance_scales) == 1
                        else f"samples/cfg_{cfg_label(scale)}"
                    )
                    result.images[key] = [
                        (
                            row["image"].permute(1, 2, 0).numpy(),
                            f"CFG={scale:g} {row['prompt'][:50]}",
                        )
                        for row in selected
                        if row["image"].numel()
                    ]
            return result
        finally:
            for plugin in distributed:
                plugin.close()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _suite_prompt_path(self, suite):
        if suite.source == "task_test":
            if self.task_prompt_path is None:
                raise ValueError(
                    f"Distribution suite {suite.namespace!r} requires a task test prompt path"
                )
            return self.task_prompt_path
        return suite.prompt_path

    def _suite_prompts(self, suite):
        path = self._suite_prompt_path(suite)
        prompts = select_distribution_prompts(
            path,
            suite.num_prompts,
            suite.prompt_seed,
            deduplicate=suite.deduplicate,
        )
        self._persist_suite_prompts(suite, path, prompts)
        return prompts

    def _persist_suite_prompts(self, suite, path, prompts):
        destination = (
            Path(self.output_dir) / "validation_prompts" / f"{suite.namespace}.json"
        )
        payload = {
            "namespace": suite.namespace,
            "source": suite.source,
            "prompt_path": str(path),
            "prompt_seed": suite.prompt_seed,
            "deduplicate": suite.deduplicate,
            "prompts": prompts,
        }
        encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        error = [None]
        if self.accelerator.is_main_process:
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                if destination.read_text() != encoded:
                    error[0] = (
                        f"Persisted validation prompt manifest disagrees with {suite.namespace}: "
                        f"{destination}"
                    )
            else:
                temporary = destination.with_name(
                    f".{destination.name}.part-{os.getpid()}"
                )
                temporary.write_text(encoded)
                os.replace(temporary, destination)
        self._broadcast(error)
        if error[0] is not None:
            raise RuntimeError(error[0])
        self.accelerator.wait_for_everyone()

    def _reference_metadata(self, prompts, suite, prompt_path):
        revision = getattr(getattr(self.pipe, "config", None), "_commit_hash", None)
        revision = revision or getattr(
            self.pipe.transformer.config, "_commit_hash", None
        )
        return {
            "schema": CACHE_SCHEMA,
            "kind": "base_latents",
            "model": self.config.model_name,
            "revision": revision,
            "transformer_config": dict(self.pipe.transformer.config),
            "suite": suite.namespace,
            "prompt_sha256": hashlib.sha256(Path(prompt_path).read_bytes()).hexdigest(),
            "prompts": prompts,
            "seeds": list(
                range(
                    self.config.distribution_seed_base,
                    self.config.distribution_seed_base
                    + self.config.distribution_samples_per_prompt,
                )
            ),
            "resolution": self.config.resolution,
            "scheduler": self.config.distribution_sampling_scheduler,
            "scheduler_source": self.backend.scheduler_fingerprint(),
            "steps": self.config.distribution_num_steps,
            "guidance_scales": list(self.config.distribution_guidance_scales),
        }

    def _broadcast(self, values):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.broadcast_object_list(values, src=0)

    def _reference_latents(self, prompts, suite):
        prompt_path = self._suite_prompt_path(suite)
        local_metadata = canonical_metadata(
            self._reference_metadata(prompts, suite, prompt_path)
        )
        shared_metadata = [local_metadata if self.accelerator.is_main_process else None]
        self._broadcast(shared_metadata)
        metadata = shared_metadata[0]
        local_key = content_key(local_metadata)
        shared_key = content_key(metadata)
        if local_key != shared_key:
            differences = metadata_differences(local_metadata, metadata)
            raise RuntimeError(
                "Validation cache metadata differs across distributed ranks: "
                f"rank={self.accelerator.process_index}, local_key={local_key}, "
                f"rank_0_key={shared_key}; differences={differences}"
            )
        path = self.cache.path("base-latents", metadata)
        lock = None
        cached = None
        if self.accelerator.is_main_process:
            lock = self.cache.lock(path)
            lock.__enter__()
            cached = self.cache.load(
                path,
                metadata,
                len(prompts) * len(self.config.distribution_guidance_scales),
            )
        state = [cached is not None]
        self._broadcast(state)
        try:
            if not state[0]:
                local_prompts = prompts[
                    self.accelerator.process_index :: self.accelerator.num_processes
                ]
                seeds = list(
                    range(
                        self.config.distribution_seed_base,
                        self.config.distribution_seed_base
                        + self.config.distribution_samples_per_prompt,
                    )
                )
                local = []
                with self.policies.base_policy():
                    for scale in self.config.distribution_guidance_scales:
                        for record in local_prompts:
                            terminal = self._sample_group(
                                record["prompt"],
                                seeds,
                                scale,
                                num_inference_steps=self.config.distribution_num_steps,
                            )
                            local.append(
                                {
                                    **record,
                                    "guidance_scale": float(scale),
                                    "latents": terminal.detach().cpu(),
                                }
                            )
                gathered = self._gather_rows(local)
                if self.accelerator.is_main_process:
                    self.cache.publish(path, metadata, gathered)
            self.accelerator.wait_for_everyone()
        finally:
            if lock is not None:
                lock.__exit__(None, None, None)
        records = self.cache.load(
            path,
            metadata,
            len(prompts) * len(self.config.distribution_guidance_scales),
            strict=True,
        )
        if records is None:
            raise RuntimeError(
                f"Base validation cache was not published correctly: {path}"
            )
        return records, content_key(metadata)

    def _score_distribution_groups(self, latent_rows, plugins, include_preview):
        result = []
        for row in latent_rows:
            if (
                not plugins
                and include_preview
                and row["prompt_index"] >= PREVIEW_PROMPTS
            ):
                result.append(
                    {key: value for key, value in row.items() if key != "latents"}
                    | {"values": {}, "preview": torch.empty(0, dtype=torch.uint8)}
                )
                continue
            latents = row["latents"].to(self.accelerator.device)
            images = self.backend.decode_fp32(
                latents, self.config.distribution_batch_size
            )
            values = {}
            prompts = [row["prompt"]] * len(images)
            for plugin in plugins:
                values.update(plugin.score(images, prompts))
            preview = torch.empty(0, dtype=torch.uint8)
            if include_preview and row["prompt_index"] < PREVIEW_PROMPTS:
                preview = (
                    (images[:PREVIEW_SAMPLES] * 255)
                    .round()
                    .clamp(0, 255)
                    .to(torch.uint8)
                    .cpu()
                )
            result.append(
                {key: value for key, value in row.items() if key != "latents"}
                | {"values": values, "preview": preview}
            )
        return result

    def _cached_reference_output(
        self,
        reference_latents,
        local_reference,
        reference_key,
        identity,
        plugins,
        include_preview,
    ):
        metadata = {
            "schema": CACHE_SCHEMA,
            "kind": "base_metric_features",
            "base_latents": reference_key,
            "metric": identity,
        }
        path = self.cache.path("base-features", metadata)
        records = None
        lock = None
        if self.accelerator.is_main_process:
            lock = self.cache.lock(path)
            lock.__enter__()
            records = self.cache.load(path, metadata, len(reference_latents))
        state = [records is not None]
        self._broadcast(state)
        try:
            if not state[0]:
                local = self._score_distribution_groups(
                    local_reference, plugins, include_preview
                )
                records = self._gather_rows(local)
                if self.accelerator.is_main_process:
                    self.cache.publish(path, metadata, records)
            self.accelerator.wait_for_everyone()
        finally:
            if lock is not None:
                lock.__exit__(None, None, None)
        records = self.cache.load(path, metadata, len(reference_latents), strict=True)
        if records is None:
            raise RuntimeError(f"Base feature cache was not published: {path}")
        return records

    def _reference_outputs(
        self, reference_latents, local_reference, reference_key, plugins
    ):
        # Start from the latent records rather than relying on the first metric
        # plugin to create rows.  This keeps image-only DrawBench evaluation
        # valid when rewards, DreamSim, and FID are all disabled.
        merged = {
            (row["guidance_scale"], row["prompt_index"]): {
                field: value for field, value in row.items() if field != "latents"
            }
            | {"values": {}, "preview": torch.empty(0, dtype=torch.uint8)}
            for row in reference_latents
        }
        for plugin in plugins:
            records = self._cached_reference_output(
                reference_latents,
                local_reference,
                reference_key,
                plugin.identity,
                [plugin],
                False,
            )
            for row in records:
                key = (row["guidance_scale"], row["prompt_index"])
                target = merged.setdefault(
                    key,
                    {
                        field: value
                        for field, value in row.items()
                        if field not in {"values", "preview"}
                    }
                    | {"values": {}, "preview": torch.empty(0, dtype=torch.uint8)},
                )
                target["values"].update(row["values"])
        if self.config.distribution_log_images:
            records = self._cached_reference_output(
                reference_latents,
                local_reference,
                reference_key,
                {
                    "kind": "preview",
                    "prompts": PREVIEW_PROMPTS,
                    "samples": PREVIEW_SAMPLES,
                },
                [],
                True,
            )
            for row in records:
                merged[(row["guidance_scale"], row["prompt_index"])]["preview"] = row[
                    "preview"
                ]
        return sorted(
            merged.values(),
            key=lambda row: (row["guidance_scale"], row["prompt_index"]),
        )

    def _validate_distribution_rows(self, rows, prompts, *, require_samples):
        expected = {
            (float(scale), prompt["prompt_index"])
            for scale in self.config.distribution_guidance_scales
            for prompt in prompts
        }
        observed = {(float(row["guidance_scale"]), row["prompt_index"]) for row in rows}
        if len(rows) != len(observed) or observed != expected:
            missing = sorted(expected - observed)[:8]
            extra = sorted(observed - expected)[:8]
            raise RuntimeError(
                "Distributed validation rows are incomplete or duplicated: "
                f"rows={len(rows)}, unique={len(observed)}, expected={len(expected)}, "
                f"missing={missing}, extra={extra}"
            )
        count = self.config.distribution_samples_per_prompt
        for row in rows:
            if "latents" in row and len(row["latents"]) != count:
                raise RuntimeError(
                    f"Generated {len(row['latents'])} latent samples for prompt "
                    f"{row['prompt_index']}; expected {count}"
                )
        if require_samples:
            for row in rows:
                values = row.get("values", {})
                for name, value in values.items():
                    if len(value) != count:
                        raise RuntimeError(
                            f"Metric {name} returned {len(value)} samples for prompt "
                            f"{row['prompt_index']}; expected {count}"
                        )

    def _run_distribution_suite(self, step, suite):
        prompts = self._suite_prompts(suite)
        local_prompts = prompts[
            self.accelerator.process_index :: self.accelerator.num_processes
        ]
        seeds = list(
            range(
                self.config.distribution_seed_base,
                self.config.distribution_seed_base
                + self.config.distribution_samples_per_prompt,
            )
        )
        registry = MetricRegistry(
            self.accelerator.device,
            self.config.distribution_batch_size,
            reusable=self.reusable_rewards,
        )
        plugins = []
        main_only_names = [
            name
            for name in self.config.distribution_rewards
            if name in MAIN_PROCESS_ONLY
        ]
        try:
            reference_latents, reference_key = self._reference_latents(prompts, suite)
            self._validate_distribution_rows(
                reference_latents, prompts, require_samples=False
            )
            reference_metadata = {
                f"{suite.namespace}/reference_cache_key": reference_key,
                "reference_cache_dir": str(self.config.reference_cache_dir),
            }
            if len(self.config.distribution_suites) == 1:
                reference_metadata["reference_cache_key"] = reference_key
            local_reference = [
                row
                for row in reference_latents
                if row["prompt_index"] % self.accelerator.num_processes
                == self.accelerator.process_index
            ]
            local_current_latents = []
            with self.policies.current_policy():
                for scale in self.config.distribution_guidance_scales:
                    for record in local_prompts:
                        terminal = self._sample_group(
                            record["prompt"],
                            seeds,
                            scale,
                            num_inference_steps=self.config.distribution_num_steps,
                        )
                        local_current_latents.append(
                            {
                                **record,
                                "guidance_scale": float(scale),
                                "latents": terminal.detach().cpu(),
                            }
                        )
            gathered_current_latents = (
                self._gather_rows(local_current_latents) if main_only_names else None
            )
            plugins.extend(registry.standard_distribution_plugins(self.config))
            base_rows = self._reference_outputs(
                reference_latents, local_reference, reference_key, plugins
            )
            local_current = self._score_distribution_groups(
                local_current_latents, plugins, self.config.distribution_log_images
            )
            current_rows = self._gather_rows(local_current)
            self._validate_distribution_rows(
                current_rows, prompts, require_samples=bool(plugins)
            )

            # DreamSim backends are deliberately constructed and released one at a
            # time.  The ensemble itself owns three encoders, so keeping all four
            # requested variants resident would create an unnecessary VRAM peak.
            for plugin in plugins:
                plugin.close()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            for type_name in self.config.distribution_dreamsim_types:
                plugin = registry.dreamsim(type_name)
                plugins.append(plugin)
                try:
                    base_extra = self._reference_outputs(
                        reference_latents, local_reference, reference_key, [plugin]
                    )
                    current_extra = self._gather_rows(
                        self._score_distribution_groups(
                            local_current_latents, [plugin], False
                        )
                    )
                    self._validate_distribution_rows(
                        current_extra, prompts, require_samples=True
                    )
                    self._merge_distribution_values(base_rows, base_extra)
                    self._merge_distribution_values(current_rows, current_extra)
                finally:
                    plugin.close()
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            main_only_plugins = []
            if main_only_names and self.accelerator.is_main_process:
                main_registry = MetricRegistry(self.accelerator.device, 1)
                main_only_plugins = [
                    main_registry.reward(name, aliases=True) for name in main_only_names
                ]
                try:
                    current_extra = self._score_distribution_groups(
                        gathered_current_latents, main_only_plugins, False
                    )
                    base_extra = self._score_main_only_reference(
                        reference_latents, reference_key, main_only_plugins
                    )
                    self._merge_distribution_values(current_rows, current_extra)
                    self._merge_distribution_values(base_rows, base_extra)
                except Exception:
                    for plugin in main_only_plugins:
                        plugin.close()
                    raise
                plugins.extend(main_only_plugins)
            if not self.accelerator.is_main_process:
                return ValidationResult()
            result = self._reduce_drawbench(
                step, suite, prompts, current_rows, base_rows, plugins
            )
            result.metadata.update(reference_metadata)
            return result
        finally:
            for plugin in plugins:
                plugin.close()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def run_drawbench(self, step):
        """Run every configured distribution suite (legacy name retained)."""
        result = ValidationResult()
        for suite in self.config.distribution_suites:
            with self.backend.evaluation_scheduler(
                self.config.distribution_sampling_scheduler
            ):
                result.merge(self._run_distribution_suite(step, suite))
        return result

    @staticmethod
    def _merge_distribution_values(target_rows, extra_rows):
        target = {
            (row["guidance_scale"], row["prompt_index"]): row for row in target_rows
        }
        for row in extra_rows:
            target[(row["guidance_scale"], row["prompt_index"])]["values"].update(
                row["values"]
            )

    def _score_main_only_reference(self, reference_latents, reference_key, plugins):
        identity = {"main_process_metrics": [plugin.identity for plugin in plugins]}
        metadata = {
            "schema": CACHE_SCHEMA,
            "kind": "base_metric_features",
            "base_latents": reference_key,
            "metric": identity,
        }
        path = self.cache.path("base-features", metadata)
        with self.cache.lock(path):
            records = self.cache.load(path, metadata, len(reference_latents))
            if records is None:
                records = self._score_distribution_groups(
                    reference_latents, plugins, False
                )
                self.cache.publish(path, metadata, records)
        return records

    def _reduce_drawbench(self, step, suite, prompts, current_rows, base_rows, plugins):
        current = {
            (row["guidance_scale"], row["prompt_index"]): row for row in current_rows
        }
        base = {(row["guidance_scale"], row["prompt_index"]): row for row in base_rows}
        result = ValidationResult()
        source_column = f"{suite.namespace}_source_index"
        columns = ["step", "cfg", "prompt_index", source_column, "prompt"]
        table_rows = []
        plugin_by_name = {plugin.name: plugin for plugin in plugins}
        for scale in self.config.distribution_guidance_scales:
            aggregated = {}
            per_prompt_rows = []
            for prompt in prompts:
                key = (float(scale), prompt["prompt_index"])
                generated = current[key]
                reference = base[key]
                values = {}
                for plugin_name, plugin in plugin_by_name.items():
                    generated_value = generated["values"][plugin_name]
                    reference_value = reference["values"][plugin_name]
                    if isinstance(plugin, FidMetric):
                        value = plugin.compare(generated_value, reference_value)
                        aggregated.setdefault("fid_to_base", []).append(value)
                        values["fid_to_base"] = value
                    elif isinstance(plugin, DreamSimMetric):
                        current_value = plugin.reduce_group(generated_value)
                        base_value = plugin.reduce_group(reference_value)
                        name = plugin.type_name
                        aggregated.setdefault(
                            f"dreamsim_diversity/{name}/current", []
                        ).append(current_value)
                        aggregated.setdefault(
                            f"dreamsim_diversity/{name}/base", []
                        ).append(base_value)
                        values[f"dreamsim_{name}_current"] = current_value
                        values[f"dreamsim_{name}_base"] = base_value
                    else:
                        current_values = generated_value.flatten().tolist()
                        base_values = reference_value.flatten().tolist()
                        aggregated.setdefault(
                            f"rewards/{plugin.name}/current", []
                        ).extend(current_values)
                        aggregated.setdefault(f"rewards/{plugin.name}/base", []).extend(
                            base_values
                        )
                        values[f"{plugin.name}_current"] = float(
                            np.mean(current_values)
                        )
                        values[f"{plugin.name}_base"] = float(np.mean(base_values))
                per_prompt_rows.append((prompt, values))
            prefix = f"{suite.namespace}/cfg_{cfg_label(scale)}"
            for name, values in aggregated.items():
                result.scalars[f"{prefix}/{name}"] = float(np.mean(values))
            for prompt, values in per_prompt_rows:
                for name in values:
                    if name not in columns:
                        columns.append(name)
                table_rows.append(
                    [
                        step,
                        float(scale),
                        prompt["prompt_index"],
                        prompt["source_index"],
                        prompt["prompt"],
                    ]
                    + [values.get(name, float("nan")) for name in columns[5:]]
                )
            if self.config.distribution_log_images:
                panels = []
                for prompt in prompts[:PREVIEW_PROMPTS]:
                    key = (float(scale), prompt["prompt_index"])
                    generated = current[key]["preview"]
                    reference = base[key]["preview"]
                    for sample in range(
                        min(len(generated), len(reference), PREVIEW_SAMPLES)
                    ):
                        paired = torch.cat(
                            (reference[sample], generated[sample]), dim=-1
                        )
                        panels.append(
                            (
                                paired.permute(1, 2, 0).numpy(),
                                f"base | current; CFG={scale:g}; seed={self.config.distribution_seed_base + sample}; "
                                f"{suite.namespace}[{prompt['source_index']}]: "
                                f"{prompt['prompt']}",
                            )
                        )
                result.images[f"{prefix}/samples"] = panels
        result.tables[f"{suite.namespace}/metrics_per_prompt"] = (
            columns,
            table_rows,
        )
        return result

    def _reduce_reference(self, suite, prompts, base_rows, plugins):
        """Reduce cached SD3.5 outputs without generating a current policy."""

        base = {(row["guidance_scale"], row["prompt_index"]): row for row in base_rows}
        result = ValidationResult()
        for scale in self.config.distribution_guidance_scales:
            aggregated = {}
            for prompt in prompts:
                reference = base[(float(scale), prompt["prompt_index"])]
                for plugin in plugins:
                    reference_value = reference["values"][plugin.name]
                    if isinstance(plugin, FidMetric):
                        aggregated.setdefault("fid_to_base", []).append(0.0)
                    elif isinstance(plugin, DreamSimMetric):
                        value = plugin.reduce_group(reference_value)
                        aggregated.setdefault(
                            f"dreamsim_diversity/{plugin.type_name}/base", []
                        ).append(value)
                    else:
                        aggregated.setdefault(f"rewards/{plugin.name}/base", []).extend(
                            reference_value.flatten().tolist()
                        )
            prefix = f"{suite.namespace}/cfg_{cfg_label(scale)}"
            for name, values in aggregated.items():
                result.scalars[f"{prefix}/{name}"] = float(np.mean(values))
        return result

    def run_reference_cache(self, step):
        """Populate and validate all reusable base-model cache entries."""

        started = time.perf_counter()
        result = ValidationResult()
        with self._rng_context():
            for suite in self.config.distribution_suites:
                prompts = self._suite_prompts(suite)
                registry = MetricRegistry(
                    self.accelerator.device,
                    self.config.distribution_batch_size,
                    reusable=self.reusable_rewards,
                )
                plugins = []
                with self.backend.evaluation_scheduler(
                    self.config.distribution_sampling_scheduler
                ):
                    reference_latents, reference_key = self._reference_latents(
                        prompts, suite
                    )
                    self._validate_distribution_rows(
                        reference_latents, prompts, require_samples=False
                    )
                    local_reference = [
                        row
                        for row in reference_latents
                        if row["prompt_index"] % self.accelerator.num_processes
                        == self.accelerator.process_index
                    ]
                    try:
                        plugins.extend(
                            registry.standard_distribution_plugins(self.config)
                        )
                        base_rows = self._reference_outputs(
                            reference_latents, local_reference, reference_key, plugins
                        )
                        for plugin in plugins:
                            plugin.close()
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        for type_name in self.config.distribution_dreamsim_types:
                            plugin = registry.dreamsim(type_name)
                            plugins.append(plugin)
                            try:
                                extra = self._reference_outputs(
                                    reference_latents,
                                    local_reference,
                                    reference_key,
                                    [plugin],
                                )
                                self._merge_distribution_values(base_rows, extra)
                            finally:
                                plugin.close()
                                gc.collect()
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                        if self.accelerator.is_main_process:
                            result.merge(
                                self._reduce_reference(
                                    suite, prompts, base_rows, plugins
                                )
                            )
                        result.metadata[f"{suite.namespace}/reference_cache_key"] = (
                            reference_key
                        )
                        if len(self.config.distribution_suites) == 1:
                            result.metadata["reference_cache_key"] = reference_key
                    finally:
                        for plugin in plugins:
                            plugin.close()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
        elapsed = reduce_max(self.accelerator, time.perf_counter() - started)
        result.metadata.update(
            {
                "reference_cache_dir": str(self.config.reference_cache_dir),
                "validation_seconds": elapsed,
            }
        )
        result.scalars["eval/validation_gpu_hours"] = (
            elapsed * self.accelerator.num_processes / 3600
        )
        return result

    def run(self, step, training_rewards=None, train_seconds=None):
        started = time.perf_counter()
        result = ValidationResult()
        with self._rng_context():
            if self.config.run_task_validation:
                with self.backend.evaluation_scheduler():
                    with self.policies.current_policy():
                        result.merge(self.run_task(step, training_rewards))
            result.merge(self.run_drawbench(step))
        elapsed = reduce_max(self.accelerator, time.perf_counter() - started)
        result.metadata["validation_seconds"] = elapsed
        result.scalars["eval/validation_gpu_hours"] = (
            elapsed * self.accelerator.num_processes / 3600
        )
        if train_seconds is not None:
            result.scalars["eval/gpu_hours"] = (
                float(train_seconds) * self.accelerator.num_processes / 3600
            )
        self.sink.write(result, step)
        return result
