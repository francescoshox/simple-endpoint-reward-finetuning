"""Pinned upstream VGG-Flow dependency and configuration adapter."""

from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_REVISION = "e9748e71d9b2a9d1202effeb682b006a8c234ccf"


def upstream_config(config):
    return SimpleNamespace(
        training=SimpleNamespace(
            batch_size=config.loss_batch_size,
            gradient_accumulation_steps=1,
            coeff_terminal=config.coeff_terminal,
            reward_masking=config.reward_masking,
            use_jvp=False,
            detach_dir=config.detach_dir,
            quantile_clipping=config.quantile_clipping,
        ),
        model=SimpleNamespace(
            reward_scale=config.reward_scale,
            timestep_fraction=config.timestep_fraction,
            unet_reg_scale=config.unet_reg_scale,
            use_value_net=True,
            eta_mode=config.eta_mode,
            value_net_param="lora",
        ),
        sampling=SimpleNamespace(
            num_steps=config.num_train_inference_steps,
            low_var_subsampling=config.low_var_subsampling,
        ),
    )
