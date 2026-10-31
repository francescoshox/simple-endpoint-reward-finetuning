from types import SimpleNamespace

import pytest
import torch
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

from serf.methods.adjoint_matching.preprocess import DeterministicPreprocess, install


@pytest.mark.parametrize("interpolation", [InterpolationMode.BILINEAR, InterpolationMode.BICUBIC])
def test_preprocessing_preserves_forward_and_input_gradient(interpolation):
    transform = T.Compose([
        T.Resize(12, interpolation=interpolation, antialias=True),
        T.CenterCrop(10),
        T.Normalize([.48, .46, .41], [.27, .26, .28]),
    ])
    images = torch.rand(1, 3, 24, 24, requires_grad=True)
    expected = transform(images)
    vector = torch.randn_like(expected)
    expected_gradient, = torch.autograd.grad(expected, images, vector)
    actual = DeterministicPreprocess(transform)(images)
    actual_gradient, = torch.autograd.grad(actual, images, vector)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=0)


@pytest.mark.parametrize("name,attribute", [("PickScoreDiff", "tform"), ("HPSv2Diff", "preprocess_val")])
def test_install_wraps_the_training_scorer(name, attribute):
    transform = lambda images: images * 2
    reward = SimpleNamespace(scorer=SimpleNamespace(**{attribute: transform}))
    install(reward, name)
    assert isinstance(getattr(reward.scorer, attribute), DeterministicPreprocess)
    with pytest.raises(ValueError, match="already wrapped"):
        install(reward, name)
