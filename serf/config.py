"""Explicit paper recipes: one method and one reward, without config inheritance."""

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("serf", "rsm", "adjoint_matching", "vgg_flow", "draft_lv")
REWARDS = ("hpsv21", "pickscore")


@dataclass
class Config:
    method: str
    reward: str
    seed: int = 0
    model_name: str = "stabilityai/stable-diffusion-3.5-medium"
    model_revision: str = "b940f670f0eda2d07fbb75229e779da1ad11eb80"
    mixed_precision: str = "bf16"
    resolution: int = 512
    lora_rank: int = 32
    lora_alpha: int = 64
    lr: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.99
    weight_decay: float = 1e-4
    adam_epsilon: float = 1e-8
    max_grad_norm: float | None = None
    max_train_steps: int = 300
    num_epochs: int = 300
    num_prompts_per_epoch: int = 64
    num_samples_per_prompt: int = 1
    num_optim_steps_per_epoch: int = 1
    num_loss_targets_per_sample: int = 4
    sampling_batch_size: int = 4
    loss_batch_size: int = 1
    reward_batch_size: int = 1
    num_train_inference_steps: int = 10
    num_eval_inference_steps: int = 20
    train_guidance_scale: float = 4.5
    eval_guidance_scale: float = 4.5
    sampling_scheduler: str = "euler"
    eval_sampling_scheduler: str = "euler"
    eval_sampling_batch_size: int = 10
    eval_prompt_fraction: float = 1.0
    eval_every_n_epochs: int = 20
    checkpoint_every: int = 20
    use_ema: bool = True
    ema_decay: float = 0.9
    gradient_checkpointing: bool = False
    first_order_coeff: float = 36000.0
    first_order_loss_cfg: bool = False
    timestep_sampling: str = "weighted_low"
    trajectory_num_partitions: int = 3
    timestep_fraction: float = 0.3
    reward_scale: float = 200000.0
    training_reward_name: str = "HPSv2Diff"
    draft_reward_loss_scale: float = 1.0
    draft_lv_num_noised_samples: int = 3
    am_schedule_epsilon: float = 0.1
    am_loss_clip_factor: float = 1.6
    am_deterministic: bool = True
    coeff_terminal: float = 1000.0
    eta_mode: str = "quad"
    detach_dir: bool = True
    quantile_clipping: bool = True
    reward_masking: bool = False
    unet_reg_scale: float = 0.0
    value_lora_rank: int = 32
    value_lora_alpha: int = 32
    low_var_subsampling: bool = True
    trajectory_partition_mode: str = "main_near_equal"
    wandb_project: str | None = None
    wandb_entity: str | None = None
    wandb_run_name: str | None = None
    validation_reference_cache_dir: str | None = None

    def __post_init__(self):
        if self.method not in METHODS or self.reward not in REWARDS:
            raise ValueError("Choose a documented method and reward")
        self.prompts = "hpsv2" if self.reward == "hpsv21" else "pickscore"
        self.training_reward_name = (
            "HPSv2Diff" if self.reward == "hpsv21" else "PickScoreDiff"
        )
        self.rewards = [self.training_reward_name]
        self.validation_rewards = ["PickScore", "HPSv2", "ImageReward", "CLIPScore"]
        self.eval_guidance_scales = [4.5]
        self.distribution_eval_cfgs = [4.5]
        self.distribution_eval_num_prompts = 200
        self.distribution_eval_samples_per_prompt = 16
        self.distribution_eval_images_per_prompt = 16
        self.distribution_eval_prompt_seed = 20260529
        self.distribution_eval_seed_base = 0
        self.distribution_eval_dreamsim_types = ["ensemble"]
        self.distribution_eval_batch_size = 10
        self.distribution_eval_rewards = [
            "PickScore",
            "HPSv2",
            "ImageReward",
            "CLIPScore",
        ]
        self.distribution_eval_compute_fid = True
        self.distribution_eval_log_images = True
        self.distribution_eval_num_inference_steps = 20
        self.distribution_eval_sampling_scheduler = "euler"
        self.distribution_eval_suites = [
            dict(
                namespace="drawbench",
                source="file",
                prompt_path="prompts/drawbench/test.txt",
                num_prompts=200,
                deduplicate=True,
            ),
            dict(
                namespace="task_test",
                source="task_test",
                num_prompts=200,
                deduplicate=True,
            ),
        ]
        if self.method in ("serf", "rsm"):
            self.sampling_batch_size = 16
            self.loss_batch_size = 16
            self.reward_batch_size = 10
            if self.method == "serf":
                self.first_order_coeff = 36000.0 if self.reward == "hpsv21" else 1500.0
            else:
                self.first_order_coeff = 100000.0 if self.reward == "hpsv21" else 7000.0
                self.timestep_sampling = "trajectory_low_variance"
                self.first_order_loss_cfg = True
        elif self.method == "draft_lv":
            self.lr = 4e-5 if self.reward == "hpsv21" else 5e-5
        elif self.method == "adjoint_matching":
            self.reward_scale = 1000000.0 if self.reward == "hpsv21" else 25000.0
            self.max_grad_norm = 1.0
        elif self.method == "vgg_flow":
            self.reward_scale = 200000.0 if self.reward == "hpsv21" else 25000.0

    def to_dict(self):
        return vars(self).copy()

    def validate(self, world_size):
        if world_size < 1:
            raise ValueError("GPU count must be positive")
        if (
            self.num_prompts_per_epoch,
            self.num_samples_per_prompt,
            self.num_loss_targets_per_sample,
            self.num_optim_steps_per_epoch,
        ) != (64, 1, 4, 1):
            raise ValueError(
                "Paper recipes require 64 trajectories, four targets and one update"
            )
        if self.num_prompts_per_epoch % world_size:
            raise ValueError("64 global trajectories must divide evenly over GPUs")
        local = self.num_prompts_per_epoch // world_size
        if self.method in ("serf", "rsm") and local * 4 % self.loss_batch_size:
            raise ValueError("Local target count must divide by loss microbatch size")
        if self.eval_every_n_epochs <= 0 or self.max_train_steps <= 0:
            raise ValueError("Evaluation interval and update count must be positive")
        return dict(
            global_trajectories=64,
            targets_per_trajectory=4,
            global_targets=256,
            local_trajectories=local,
            world_size=world_size,
        )
