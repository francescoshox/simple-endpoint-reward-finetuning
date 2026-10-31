"""Generation backend isolated from validation suites and metrics."""

from __future__ import annotations

from contextlib import contextmanager

import torch


DTYPES = {"no": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


class SD3GenerationBackend:
    def __init__(self, pipe, accelerator, config):
        self.pipe = pipe
        self.accelerator = accelerator
        self.config = config

    @property
    def device(self):
        return self.accelerator.device

    def initial_latents(self, seeds):
        shape = (
            self.pipe.transformer.config.in_channels,
            self.config.resolution // self.pipe.vae_scale_factor,
            self.config.resolution // self.pipe.vae_scale_factor,
        )
        return torch.stack(
            [
                torch.randn(
                    shape,
                    generator=torch.Generator(device="cpu").manual_seed(int(seed)),
                )
                for seed in seeds
            ]
        )

    @contextmanager
    def evaluation_scheduler(self, requested=None):
        original = self.pipe.scheduler
        requested = self.config.sampling_scheduler if requested is None else requested
        if requested == "inherit":
            yield
            return
        try:
            from diffusers import FlowMatchEulerDiscreteScheduler

            if requested != "euler":
                raise ValueError("Paper evaluation uses the Euler scheduler")
            shift = getattr(
                original.config, "shift", getattr(original.config, "flow_shift", 1.0)
            )
            self.pipe.scheduler = FlowMatchEulerDiscreteScheduler(
                num_train_timesteps=original.config.num_train_timesteps,
                shift=float(shift),
            )
            yield
        finally:
            self.pipe.scheduler = original

    def sample_latents(
        self,
        prompts,
        seeds,
        guidance_scale,
        *,
        cpu_noise=True,
        num_inference_steps=None,
    ):
        if len(prompts) != len(seeds):
            raise ValueError("prompts and seeds must have equal length")
        if not prompts:
            return torch.empty(0, device=self.device)
        sampling_kwargs = {}
        if cpu_noise:
            sampling_kwargs["latents"] = self.initial_latents(seeds).to(
                self.device, dtype=DTYPES[self.config.mixed_precision]
            )
        else:
            sampling_kwargs["generator"] = [
                torch.Generator(device=self.device).manual_seed(int(seed))
                for seed in seeds
            ]
        with torch.no_grad(), self.accelerator.autocast():
            return self.pipe(
                list(prompts),
                height=self.config.resolution,
                width=self.config.resolution,
                num_inference_steps=(
                    self.config.num_steps
                    if num_inference_steps is None
                    else int(num_inference_steps)
                ),
                guidance_scale=float(guidance_scale),
                output_type="latent",
                return_dict=False,
                **sampling_kwargs,
            )[0]

    def decode_fp32(self, latents, batch_size=None):
        floating = [
            tensor
            for tensor in (*self.pipe.vae.parameters(), *self.pipe.vae.buffers())
            if tensor.is_floating_point()
        ]
        if floating and any(tensor.dtype != torch.float32 for tensor in floating):
            raise RuntimeError(
                "FP32 decoding requires all floating VAE state to be float32"
            )
        batch_size = int(batch_size or len(latents))
        outputs = []
        with torch.autocast(device_type=self.device.type, enabled=False):
            for start in range(0, len(latents), batch_size):
                batch = latents[start : start + batch_size].float()
                scaled = batch / self.pipe.vae.config.scaling_factor
                scaled += getattr(self.pipe.vae.config, "shift_factor", 0.0)
                decoded = self.pipe.vae.decode(scaled, return_dict=False)[0].float()
                outputs.append(((decoded + 1.0) / 2.0).clamp(0.0, 1.0))
        return (
            torch.cat(outputs)
            if outputs
            else latents.new_empty((0, 3, 0, 0), dtype=torch.float32)
        )

    def scheduler_fingerprint(self):
        scheduler = self.pipe.scheduler
        config = dict(scheduler.config)
        return {"class": type(scheduler).__name__, "config": config}
