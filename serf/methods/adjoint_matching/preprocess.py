"""Keep reward preprocessing's GPU forward; compute its input VJP on CPU.

PyTorch has no deterministic CUDA backward for antialiased bilinear/bicubic
resize. The existing stateless tensor transform is reused on CPU in backward,
without changing the scorer, preprocessing parameters, or installed packages.
Only first derivatives are required for detached AM terminal targets.
"""
import torch


class _Preprocess(torch.autograd.Function):
    @staticmethod
    def forward(ctx, images, transform):
        ctx.transform = transform
        ctx.save_for_backward(images.detach())
        return transform(images)

    @staticmethod
    def backward(ctx, output_gradient):
        (images,) = ctx.saved_tensors
        with torch.enable_grad():
            inputs = images.detach().cpu().requires_grad_(True)
            outputs = ctx.transform(inputs)
            gradient, = torch.autograd.grad(outputs, inputs, output_gradient.detach().cpu())
        return gradient.to(device=images.device, dtype=images.dtype), None


class DeterministicPreprocess:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, images):
        return _Preprocess.apply(images, self.transform)


def install(reward, name):
    attribute = {'PickScoreDiff': 'tform', 'HPSv2Diff': 'preprocess_val'}[name]
    original = getattr(reward.scorer, attribute)
    if isinstance(original, DeterministicPreprocess):
        raise ValueError('Preprocessing is already wrapped')
    setattr(reward.scorer, attribute, DeterministicPreprocess(original))
