# Locked environment

`uv.lock` resolves the complete runtime and development dependency graph. `.python-version` selects Python 3.11.15. `requirements-environment.txt` records the reference package freeze; `environment.json` records the reference platform and the comparison with the recreated environment.

```bash
uv sync --frozen
uv run pytest -q
```

The runtime uses the CUDA 13.0 PyTorch wheels on Linux ARM64 or x86-64. A compatible NVIDIA driver is required; the driver is supplied by the host and cannot be pinned by a Python lockfile. The original experiments used GH200 GPUs. Other architectures and kernels can change numerical results even with identical Python package versions.

No global `sitecustomize.py`, original research checkout or cluster-specific shell initialization is required. A narrow, scoped PEFT compatibility context handles the replicated DreamSim adapter/export path for the pinned Transformers version; it does not enable tensor-parallel model loading.

The lock was installed in a separate environment without modifying the original experiment environment. Python-level versions matched the reference environment. CPU tests and configuration checks can be run without requesting GPUs; a full GPU numerical reproduction check is a separate, potentially substantial experiment.

`hpsv2==1.2.0` omits its BPE vocabulary from the wheel. `python -m serf.prepare` restores only that data file from the pinned CLIP installation, verifying SHA-256 `924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a`. This required preparation step does not upgrade dependencies or change the tokenizer content used in the original experiments. `--check-only` verifies it without modifying the environment.
