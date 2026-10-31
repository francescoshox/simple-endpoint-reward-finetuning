from types import SimpleNamespace
import random
import numpy as np
import torch
from serf.config import Config
from serf.validation.config import apply_validation_defaults, ValidationConfig
from serf.validation.runner import ValidationRunner
from serf.validation.scorers import dreamsim_diversity


def test_paper_suites_and_metrics():
    c = Config("serf", "hpsv21")
    apply_validation_defaults(c)
    v = ValidationConfig.from_namespace(c)
    assert not v.run_task_validation
    assert [x.namespace for x in v.distribution_suites] == ["drawbench", "task_test"]
    assert all(x.num_prompts == 200 for x in v.distribution_suites)
    assert v.distribution_samples_per_prompt == 16
    assert v.distribution_guidance_scales == (4.5,) and v.distribution_num_steps == 20
    assert set(v.distribution_rewards) == {
        "PickScore",
        "HPSv2",
        "ImageReward",
        "CLIPScore",
    }


def test_validation_preserves_python_numpy_and_torch_rng():
    runner = object.__new__(ValidationRunner)
    runner.accelerator = SimpleNamespace(device=torch.device("cpu"))
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    before = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    with runner._rng_context():
        random.random()
        np.random.rand()
        torch.rand(4)
    assert random.getstate() == before[0]
    assert np.array_equal(np.random.get_state()[1], before[1][1])
    assert torch.equal(torch.get_rng_state(), before[2])


def test_diversity_is_pairwise_cosine_distance_not_feature_variance():
    x = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
    assert dreamsim_diversity(x) == 1
    assert dreamsim_diversity(torch.ones(2, 2)) < 1e-6
