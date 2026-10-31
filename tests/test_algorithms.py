"""Check algorithm distinctions and endpoint-gradient math without large models."""

from contextlib import nullcontext
import torch
from serf.config import Config
from serf.methods import serf, rsm, draft_lv
from serf.methods.adjoint_matching import core as am_core


class Accelerator:
    def autocast(self):
        return nullcontext()


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.4))
        self.enabled = True

    def set_adapter(self, name):
        pass

    def disable_adapter(self):
        model = self

        class C:
            def __enter__(self):
                model.enabled = False

            def __exit__(self, *args):
                model.enabled = True

        return C()

    def forward(
        self,
        hidden_states,
        timestep,
        encoder_hidden_states,
        pooled_projections,
        return_dict,
    ):
        scale = self.weight if self.enabled else self.weight.detach() * 0
        return (
            hidden_states * scale + encoder_hidden_states[:, 0, 0].view(-1, 1, 1, 1),
        )


def test_serf_and_rsm_have_separate_targets():
    reference = torch.zeros(4, 1, 2, 2)
    gradient = torch.ones_like(reference)
    sigma = torch.tensor([0.0, 0.25, 0.5, 0.75]).view(-1, 1, 1, 1)
    assert torch.equal(
        serf.target_velocity(reference, gradient, 2), torch.full_like(reference, -2)
    )
    assert torch.equal(
        rsm.target_velocity(reference, gradient, sigma, 2),
        -2 * (1 - sigma).expand_as(reference),
    )


def test_serf_noises_endpoint_but_rsm_requires_trajectory():
    c = Config("serf", "hpsv21")
    m = Model()
    x = torch.randn(4, 1, 2, 2)
    embeds = torch.ones(4, 1, 1)
    pooled = torch.ones(4, 1)
    g = torch.ones_like(x)
    t = torch.full((4,), 500.0)
    loss, _ = serf.loss(m, x, embeds, pooled, g, t, c, Accelerator())
    loss.backward()
    assert torch.isfinite(m.weight.grad) and m.weight.grad != 0
    c = Config("rsm", "hpsv21")
    try:
        rsm.loss(m, x, embeds, pooled, g, t, c, Accelerator())
    except ValueError:
        pass
    else:
        raise AssertionError("RSM accepted no trajectory")


def test_rsm_cfg_policy_gradient_detaches_unconditional():
    c = Config("rsm", "hpsv21")
    m = Model()
    x = torch.ones(4, 1, 2, 2)
    embeds = torch.ones(4, 1, 1)
    pooled = torch.ones(4, 1)
    t = torch.ones(4)
    v = rsm.first_order_model_velocity(
        m, x, t, embeds, pooled, c, Accelerator(), embeds * 0, pooled * 0
    )
    v.sum().backward()
    assert torch.allclose(m.weight.grad, torch.tensor(x.numel() * 4.5))


def test_am_terminal_adjoint_propagation():
    states = [torch.ones(1, 1, 2, 2) for _ in range(3)]
    progress = torch.tensor([0.0, 0.5, 1.0])
    h = torch.tensor([0.5, 0.5])
    terminal = torch.ones_like(states[0])
    result = am_core.lean_adjoint(
        states, progress, h, terminal, lambda x, t: x * 0, 0.1
    )
    expected = terminal
    for i in [1, 0]:
        _, kappa = am_core.coefficients(progress[i + 1], 0.1)
        expected = expected * (1 - h[i] * kappa)
    assert torch.allclose(result[0], expected)


def test_am_selects_four_unique_matching_states():
    indices = am_core.select_indices(10, generator=torch.Generator().manual_seed(0))
    assert len(set(indices.tolist())) == 4 and indices[-2:].tolist() == [8, 9]


def test_draft_path_average_preserves_scale():
    scores = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], requires_grad=True
    )
    loss = draft_lv._draft_lv_objective(scores, 1.0)
    loss.backward()
    assert torch.equal(scores.grad, torch.full_like(scores, -1 / 8))


def test_global_batch_independent_of_gpu_count():
    for method in ["serf", "rsm", "adjoint_matching", "vgg_flow", "draft_lv"]:
        for reward in ["hpsv21", "pickscore"]:
            c = Config(method, reward)
            for n in [4, 8]:
                batch = c.validate(n)
                assert batch["global_targets"] == 256
            assert c.eval_every_n_epochs == 20 and c.max_train_steps == 300
