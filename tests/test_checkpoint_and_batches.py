"""Recovery and distributed batch contracts, without model downloads or GPUs."""

import json
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from serf.config import Config
from serf.shared import checkpoint
from serf.shared.batch_sampler import PromptSeedBatchSampler


def test_prompt_seed_schedule_is_stable_across_topologies():
    for epoch in (0, 139):
        schedules = []
        for world_size in (4, 8):
            merged = []
            for rank in range(world_size):
                sampler = PromptSeedBatchSampler(400, 64, 1, world_size, rank, seed=7)
                sampler.set_epoch(epoch)
                local = next(iter(sampler))
                assert len(local) == 64 // world_size
                merged.extend(local)
            assert len({index for index, _ in merged}) == 64
            assert len({seeds[0] for _, seeds in merged}) == 64
            schedules.append(sorted(merged))
        assert schedules[0] == schedules[1]


def test_complete_checkpoint_restores_optimizer_weights_and_rng(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: [])
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", lambda state: None)
    monkeypatch.setattr(checkpoint, "save_policy_exports", lambda *args: None)
    from serf.validation import scorers

    monkeypatch.setattr(scorers, "_load_replicated_peft_adapter", nullcontext)
    accelerator = SimpleNamespace(
        is_main_process=True,
        process_index=0,
        num_processes=1,
        wait_for_everyone=lambda: None,
    )
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    expected = {key: value.clone() for key, value in model.state_dict().items()}
    config = Config("serf", "hpsv21")
    saved_rng = checkpoint.rng_state()
    path = checkpoint.save(
        tmp_path, 20, config, None, model, None, optimizer, accelerator, 12.5
    )
    with torch.no_grad():
        model.weight.zero_()
    completed, seconds, rng = checkpoint.load(
        path, config, model, None, optimizer, accelerator
    )
    assert completed == 20 and seconds == 12.5
    assert all(
        torch.equal(model.state_dict()[key], value) for key, value in expected.items()
    )
    assert all(state["step"] == 1 for state in optimizer.state.values())
    checkpoint.restore_rng(rng)
    assert torch.equal(torch.get_rng_state(), saved_rng["torch"])
    assert np.array_equal(np.random.get_state()[1], saved_rng["numpy"][1])
    assert (tmp_path / "latest").resolve() == path
    (path / "COMPLETE").unlink()
    with pytest.raises(ValueError, match="complete checkpoint"):
        checkpoint.load(path, config, model, None, optimizer, accelerator)


def test_bundled_prompt_splits_match_the_manifest():
    import hashlib
    from serf.config import ROOT

    records = json.loads((ROOT / "prompts/manifest.json").read_text())
    for record in records:
        data = (ROOT / record["path"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == record["sha256"]
    assert {record["path"]: record["unique"] for record in records} == {
        "prompts/drawbench/test.txt": 200,
        "prompts/hpsv2/test.txt": 400,
        "prompts/hpsv2/train.txt": 103665,
        "prompts/pickscore/test.txt": 2048,
        "prompts/pickscore/train.txt": 25432,
    }


def test_package_import_targets_survive_extraction():
    import ast
    import importlib.util
    from serf.config import ROOT

    for source in (ROOT / "serf").rglob("*.py"):
        package = ".".join(source.relative_to(ROOT).parts[:-1])
        for node in ast.walk(ast.parse(source.read_text())):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.level:
                name = importlib.util.resolve_name(
                    "." * node.level + (node.module or ""), package
                )
            else:
                name = node.module or ""
            if name.startswith("serf"):
                target = ROOT.joinpath(*name.split("."))
                assert (
                    target.with_suffix(".py").is_file()
                    or (target / "__init__.py").is_file()
                ), (source, name)


def test_validation_prompt_paths_do_not_depend_on_cwd(tmp_path, monkeypatch):
    from serf.validation.config import ValidationConfig, apply_validation_defaults
    from serf.config import ROOT

    monkeypatch.chdir(tmp_path)
    config = Config("serf", "hpsv21")
    apply_validation_defaults(config)
    validation = ValidationConfig.from_namespace(config)
    assert (
        validation.distribution_suites[0].prompt_path
        == ROOT / "prompts/drawbench/test.txt"
    )
    assert (
        config.to_dict()["distribution_eval_suites"][0]["prompt_path"]
        == "prompts/drawbench/test.txt"
    )
