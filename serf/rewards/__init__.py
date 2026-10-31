"""Frozen differentiable rewards and paper evaluation scorers."""

from abc import ABC, abstractmethod
import torch
from torch import nn


class Reward(nn.Module, ABC):
    @abstractmethod
    def forward(self, images, prompts):
        """Score a batch of images.

        Args:
            images: float tensor in ``[0, 1]`` with shape ``(B, C, H, W)``.
            prompts: list of ``B`` text prompts.

        Returns:
            Float tensor of shape ``(B,)`` on ``images.device``.
        """


class ImageReward(Reward):
    def __init__(self):
        super().__init__()
        from .image_reward import rm_load

        self.rm = rm_load()
        self.rm.requires_grad_(False)

    def forward(self, images, prompts):
        images = [x for x in images]
        return self.rm.score_from_prompt_batched(prompts, images)


class ClipScore(Reward):
    def __init__(self):
        super().__init__()
        from .clip_scorer import ClipScorer

        self.scorer = ClipScorer(device="cpu")

    def forward(self, images, prompts):
        return self.scorer(images, prompts)


class ClipScoreDiff(Reward):
    """Gradient-preserving counterpart to :class:`ClipScore` (same scorer, ``no_grad=False``)."""

    def __init__(self):
        super().__init__()
        from .clip_scorer import ClipScorer

        self.scorer = ClipScorer(device="cpu", no_grad=False)

    def forward(self, images, prompts):
        return self.scorer(images, prompts)


class HPSv2(Reward):
    def __init__(self):
        super().__init__()
        from .hpsv2_scorer import HPSv2Scorer

        self.scorer = HPSv2Scorer(dtype=torch.float32, device="cpu")

    def forward(self, images, prompts):
        return self.scorer(images, prompts)


class HPSv2Diff(Reward):
    """Gradient-preserving counterpart to :class:`HPSv2` (same scorer, ``no_grad=False``)."""

    def __init__(self):
        super().__init__()
        from .hpsv2_scorer import HPSv2Scorer

        self.scorer = HPSv2Scorer(dtype=torch.float32, device="cpu", no_grad=False)

    def forward(self, images, prompts):
        return self.scorer(images, prompts)


class PickScore(Reward):
    """Raw-logit PickScore with the same tensor preprocessing as training."""

    def __init__(self):
        super().__init__()
        from .pickscore_scorer import PickScoreScorer

        self.scorer = PickScoreScorer(device="cpu", no_grad=True)

    def forward(self, images, prompts):
        return self.scorer(images, prompts)


class PickScoreDiff(Reward):
    """Gradient-preserving counterpart to :class:`PickScore`. See
    ``serf/rewards/pickscore_scorer.py``."""

    def __init__(self):
        super().__init__()
        from .pickscore_scorer import PickScoreScorer

        self.scorer = PickScoreScorer(device="cpu", no_grad=False)

    def forward(self, images, prompts):
        return self.scorer(images, prompts)
