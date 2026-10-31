"""Local integration adapters for the pinned, unmodified VGGFlow implementation.

Algorithm losses come from upstream. Local adaptations cover near-equal state
partitions, common frozen rewards, bounded memory and distributed accounting.
"""

import copy
import os
import subprocess
import sys

import torch

from serf.upstream import ROOT, UPSTREAM_REVISION, upstream_config


def import_upstream():
    root = ROOT / "third_party" / "vggflow"
    if not (root / "lib/vggflow/algorithm.py").exists():
        raise RuntimeError(
            "Populate third_party/vggflow with python -m serf.prepare --vgg-flow"
        )
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != UPSTREAM_REVISION:
        raise RuntimeError(f"Expected VGGFlow {UPSTREAM_REVISION}, found {revision}")
    if subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain"], text=True
    ).strip():
        raise RuntimeError(
            "VGGFlow dependency must be clean; put adaptations in serf/methods/vgg_flow.py"
        )
    sys.path.insert(0, str(root))


class ModelHandle:
    """Expose upstream's .module interface without per-forward DDP hooks.

    VGGFlow uses several forwards/autograd.grad calls and switches adapters in
    one loss graph. Parameters are reduced explicitly ONCE after accumulation,
    avoiding DDP reducer interactions with those forwards and optional monitors.
    """

    def __init__(self, module):
        self.module = module

    def __call__(self, *args, **kwargs):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return self.module(*args, **kwargs)


def build_models(config, device):
    from peft import LoraConfig
    from diffusers import StableDiffusion3Pipeline
    from serf.shared.models import LORA_TARGET_MODULES

    cache_dir = os.environ.get("HF_HUB_CACHE")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        config.model_name,
        revision=config.model_revision,
        torch_dtype=torch.bfloat16,
        cache_dir=cache_dir,
        local_files_only=os.environ.get("HF_HUB_OFFLINE") == "1",
    )
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
    policy = pipe.transformer
    policy.add_adapter(
        LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            init_lora_weights="gaussian",
            target_modules=LORA_TARGET_MODULES,
        ),
        adapter_name="default",
    )
    policy.set_adapter("default")
    # Copy after adding the policy adapter, matching the upstream initialization.
    value = copy.deepcopy(policy)
    value.requires_grad_(False)
    value.add_adapter(
        LoraConfig(
            r=config.value_lora_rank,
            lora_alpha=config.value_lora_alpha,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        ),
        adapter_name="default_value",
    )
    value.set_adapter("default_value")
    value.proj_out.weight.requires_grad_(True)
    with torch.no_grad():
        value.proj_out.weight.mul_(1e-5)
    # Evaluation EMA belongs only to the policy; the value network remains an
    # exact copy of the upstream VGGFlow layout.
    policy.add_adapter(
        LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            init_lora_weights="gaussian",
            target_modules=LORA_TARGET_MODULES,
        ),
        adapter_name="evaluation",
    )
    policy.set_adapter("default")
    for model in (policy, value):
        for name, parameter in model.named_parameters():
            if parameter.requires_grad or "lora_" in name:
                parameter.data = parameter.data.float()
    # Adapter switching during backward recomputation must not change branches.
    # Upstream uses multiple adapter states in each graph; do not checkpoint it.
    if config.gradient_checkpointing:
        raise ValueError(
            "Gradient checkpointing is not supported with upstream adapter switching"
        )
    pipe.set_progress_bar_config(disable=True)
    pipe._joint_attention_kwargs = None
    pipe.scheduler.set_timesteps(config.num_train_inference_steps, device=device)
    return pipe, policy, value


class SharedTrainingReward(torch.nn.Module):
    """Adapt main's tensor reward interface to upstream VGGFlow's tuple API."""

    def __init__(self, scorer):
        super().__init__()
        self.scorer = scorer

    def forward(self, images, prompts, metadata=None):
        return self.scorer(images, prompts), {}


def make_algorithm(config, pipe, policy, value, reward):
    from lib.models.cfg_wrapper import CFGWrapper
    from lib.vggflow.value_network import ValueNetworkWrapper
    from lib.vggflow.reward_gradient import RewardGradientComputer
    from lib.vggflow.algorithm import VGGFlowAlgorithm

    upstream = upstream_config(config)

    class MicrobatchedRewardGradient(RewardGradientComputer):
        def compute_from_latent(
            self,
            latent_input,
            latent_requires_grad,
            prompts,
            prompt_metadata,
            rgrad_threshold,
            retain_graph=True,
        ):
            # Reward microbatching must include VAE backward, not just score
            # forward, otherwise all decoded graphs remain live at once.
            from lib.vggflow.reward_gradient import latent_to_image

            gradients, masks = [], []
            total = len(prompts)
            for start in range(0, total, config.reward_batch_size):
                stop = min(total, start + config.reward_batch_size)
                with torch.autocast("cuda", enabled=False):
                    images = latent_to_image(pipe, latent_input[start:stop].float())
                    scores, _ = reward(
                        images, prompts[start:stop], prompt_metadata[start:stop]
                    )
                    gradient = torch.autograd.grad(
                        scores.sum(),
                        latent_requires_grad,
                        retain_graph=retain_graph or stop < total,
                    )[0]
                gradients.append(gradient.detach().float())
                masks.append((scores.detach() >= 0).float())
            # Each chunk is a VJP wrt the complete latent batch.
            gradient = torch.stack(gradients).sum(0)
            norms = gradient.flatten(1).norm(dim=1)
            return (
                self._clip_gradient(gradient, rgrad_threshold),
                torch.cat(masks),
                norms,
            )

    algorithm = VGGFlowAlgorithm(
        ModelHandle(policy),
        ValueNetworkWrapper(ModelHandle(value), "lora", pipe),
        CFGWrapper(config.train_guidance_scale, True),
        MicrobatchedRewardGradient(reward, pipe, upstream),
        upstream,
        pipe,
    )
    return algorithm


def sample_epoch(config, pipe, policy, reward, prompts, seeds, device):
    from lib.diffusion.sample_trajectory_sd3 import sample_trajectory

    chunks, scores = [], []
    for start in range(0, len(prompts), config.sampling_batch_size):
        texts = prompts[start : start + config.sampling_batch_size]
        generators = [
            torch.Generator(device=device).manual_seed(seed)
            for seed in seeds[start : start + len(texts)]
        ]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            result = sample_trajectory(
                pipe,
                policy,
                prompt=texts,
                height=config.resolution,
                width=config.resolution,
                num_inference_steps=config.num_train_inference_steps,
                guidance_scale=config.train_guidance_scale,
                generator=generators,
                output_type="latent",
                return_output=config.unet_reg_scale > 0,
            )
        _, latents, _, embeds, pooled, neg, neg_pooled, outputs = result
        # Clone outside upstream inference_mode so these tensors can be saved
        # for backward later. Store trajectory and text embeddings on CPU.
        chunk = {
            "latents": torch.stack(latents, dim=1).detach().cpu().clone(),
            "prompt_embeds": embeds.detach().cpu().clone(),
            "pooled_prompt_embeds": pooled.detach().cpu().clone(),
            "negative_prompt_embeds": neg.detach().cpu().clone(),
            "negative_pooled_prompt_embeds": neg_pooled.detach().cpu().clone(),
            "timesteps": pipe.scheduler.timesteps.cpu().repeat(len(texts), 1),
            "step_index": torch.arange(config.num_train_inference_steps).repeat(
                len(texts), 1
            ),
        }
        if outputs is not None:
            chunk["unet_outputs"] = torch.stack(outputs, dim=1).detach().cpu().clone()
        from lib.vggflow.reward_gradient import latent_to_image

        for rstart in range(0, len(texts), config.reward_batch_size):
            rend = min(len(texts), rstart + config.reward_batch_size)
            with torch.no_grad():
                images = latent_to_image(pipe, latents[-1][rstart:rend].float())
                score, _ = reward(images, texts[rstart:rend], [{}] * (rend - rstart))
                scores.append(score.detach().cpu())
        chunks.append(chunk)
        del result, latents, outputs, images
    samples = {key: torch.cat([chunk[key] for chunk in chunks]) for key in chunks[0]}
    samples.update(prompts=prompts, prompt_metadata=[{} for _ in prompts])
    return samples, torch.cat(scores)


def select_near_equal_states(samples, num_steps, partitions):
    """Main's near-equal VGG-style partitions, including forced final and duplicates."""
    device = samples["timesteps"].device
    size, remainder = divmod(num_steps, partitions)
    sizes = [size + (i < remainder) for i in range(partitions)]
    trunks = list(torch.split(torch.arange(num_steps, device=device), sizes))
    order = list(reversed(range(partitions))) * max(sizes)
    perms = []
    for _ in range(len(samples["prompts"])):
        draws = [
            trunks[i][torch.randint(0, len(trunks[i]), (1,), device=device)]
            for i in order
        ]
        perms.append(
            torch.cat([torch.tensor([num_steps - 1], device=device), *draws]).clamp_min(
                1
            )[: partitions + 1]
        )
    indices = torch.stack(perms)
    rows = torch.arange(len(perms), device=device)[:, None]
    for key in ("timesteps", "latents", "step_index", "unet_outputs"):
        if key in samples:
            samples[key] = samples[key][rows, indices]
    return samples


def select_states(samples, config):
    """Delegate exact shuffling and state selection to the pinned trainer."""
    from lib.vggflow.trainer import VGGFlowTrainer

    selector = VGGFlowTrainer.__new__(VGGFlowTrainer)
    selector.config = upstream_config(config)
    samples = selector._shuffle_samples(samples, len(samples["prompts"]))
    count = selector._compute_num_train_timesteps(config.num_train_inference_steps)
    if getattr(config, "trajectory_partition_mode", "upstream") == "main_near_equal":
        samples = select_near_equal_states(
            samples,
            config.num_train_inference_steps,
            int(config.num_train_inference_steps * config.timestep_fraction),
        )
    elif config.timestep_fraction < 1:
        samples = selector._subsample_timesteps(
            samples,
            len(samples["prompts"]),
            config.num_train_inference_steps,
            count,
        )
    return samples, count


def loss_terms(algorithm, batch, step, pipe, config, threshold):
    xt = batch["latents"][:, step].detach()
    timestep = batch["timesteps"][:, step]
    sigma = pipe.scheduler.sigmas[batch["step_index"][:, step]].view(-1, 1, 1, 1)
    embeds = torch.cat([batch["negative_prompt_embeds"], batch["prompt_embeds"]])
    pooled = torch.cat(
        [batch["negative_pooled_prompt_embeds"], batch["pooled_prompt_embeds"]]
    )
    target, parts = algorithm.compute_velocity_target(
        xt,
        timestep,
        sigma,
        embeds,
        pooled,
        batch["prompts"],
        batch["prompt_metadata"],
        threshold,
    )
    diff = (
        (parts["velocity_current"] - target.detach()).float().square().mean((1, 2, 3))
    )
    velocity = (
        (diff * parts["reward_mask"]).sum() / (parts["reward_mask"].sum() + 1e-8)
        if config.reward_masking
        else diff.mean()
    )
    consistency, terminal = algorithm.compute_value_consistency_loss(
        xt,
        timestep,
        sigma,
        parts["value_correction"],
        parts["nabla_V"],
        parts["velocity_target_raw"],
        embeds,
        pooled,
        batch["prompts"],
        batch["prompt_metadata"],
        threshold,
    )
    terms = {
        "vgg_velocity": velocity,
        "vgg_consistency": consistency,
        "vgg_terminal": terminal * config.coeff_terminal,
    }
    if config.unet_reg_scale:
        terms["vgg_velocity_regularization"] = (
            config.unet_reg_scale
            * (parts["velocity_current"] - batch["unet_outputs"][:, step])
            .square()
            .mean()
        )
    return terms, parts["rgrad_norm"].detach()
