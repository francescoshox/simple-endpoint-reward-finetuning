"""SD3.5 policy backbone and CFG helpers shared by AM and DRaFT-LV."""

import os
import torch


def build_pipeline(config, device):
    from diffusers import FlowMatchEulerDiscreteScheduler, StableDiffusion3Pipeline
    from peft import LoraConfig
    from serf.shared.models import LORA_TARGET_MODULES

    cache_dir = os.environ.get("HF_HUB_CACHE")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        config.model_name,
        revision=config.model_revision,
        torch_dtype=torch.bfloat16,
        cache_dir=cache_dir,
        local_files_only=os.environ.get("HF_HUB_OFFLINE") == "1",
    )
    if getattr(config, "sampling_scheduler", "euler") != "euler":
        raise ValueError("The training grid requires sampling_scheduler=euler")
    pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)
    for module in (
        pipe.transformer,
        pipe.vae,
        pipe.text_encoder,
        pipe.text_encoder_2,
        pipe.text_encoder_3,
    ):
        module.requires_grad_(False)
    pipe.vae.to(device, dtype=torch.float32)
    for module in (
        pipe.text_encoder,
        pipe.text_encoder_2,
        pipe.text_encoder_3,
        pipe.transformer,
    ):
        module.to(device, dtype=torch.bfloat16)
    pipe.transformer.add_adapter(
        LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            init_lora_weights="gaussian",
            target_modules=LORA_TARGET_MODULES,
        ),
        adapter_name="default",
    )
    pipe.transformer.add_adapter(
        LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            init_lora_weights="gaussian",
            target_modules=LORA_TARGET_MODULES,
        ),
        adapter_name="evaluation",
    )
    pipe.transformer.set_adapter("default")
    for name, parameter in pipe.transformer.named_parameters():
        if "lora_" in name:
            parameter.data = parameter.data.float()
    pipe.set_progress_bar_config(disable=True)
    pipe._joint_attention_kwargs = None
    return pipe


def latent_to_image(pipe, latents):
    if hasattr(pipe.vae.config, "shift_factor"):
        latents = (
            latents / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor
        )
    else:
        latents = latents / pipe.vae.config.scaling_factor
    image = pipe.vae.decode(latents.float(), return_dict=False)[0]
    return ((image + 1) / 2).clamp(0, 1)


def _cfg_velocity(
    model,
    latents,
    timestep,
    prompt_embeds,
    pooled,
    negative_embeds,
    negative_pooled,
    scale,
):
    if float(scale) == 1.0:
        return model(
            hidden_states=latents,
            timestep=timestep.expand(latents.shape[0]),
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled,
            return_dict=False,
        )[0].float()
    if negative_embeds is None or negative_pooled is None:
        raise ValueError(
            "Negative conditioning is required when train_guidance_scale exceeds 1"
        )
    latent_input = torch.cat([latents, latents])
    embeddings = torch.cat([negative_embeds, prompt_embeds])
    pooled_embeddings = torch.cat([negative_pooled, pooled])
    predicted = model(
        hidden_states=latent_input,
        timestep=timestep.expand(latent_input.shape[0]),
        encoder_hidden_states=embeddings,
        pooled_projections=pooled_embeddings,
        return_dict=False,
    )[0].float()
    unconditional, conditional = predicted.chunk(2)
    return unconditional + scale * (conditional - unconditional)
