# Simple Endpoint Reward Finetuning

Code for **Simple Endpoint Reward Finetuning (SERF)** and the SD3.5 Medium baselines used in our experiments: RSM, Adjoint Matching, VGG-Flow and DRaFT-LV. Each method has a separate algorithm module and one training script per reward, with shared model, reward and evaluation utilities.

## Install and prepare

Requires Git, uv, NVIDIA GPUs and a CUDA 13.0-compatible driver. The experiment environment is locked to Python 3.11.15 and PyTorch 2.12.1; Linux ARM64 and x86-64 dependencies are resolved in `uv.lock`. Reference hardware: NVIDIA GH200.

```bash
git clone https://github.com/francescoshox/simple-endpoint-reward-finetuning.git
cd simple-endpoint-reward-finetuning
uv sync --frozen
uv run python -m serf.prepare
```

Accept the [SD3.5 Medium model terms](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium) and authenticate with Hugging Face if required. Preparation runs on CPU and downloads the diffusion, reward and evaluation models, including the missing HPSv2 tokenizer vocabulary. Prompt files are included.

For VGG-Flow, use `uv run python -m serf.prepare --vgg-flow` to also fetch its pinned upstream source. **We do not redistribute VGG-Flow's algorithm source because that revision has no explicit license.**

Use `uv run python -m serf.prepare --check-only` to verify cached assets offline. Standard Hugging Face/Torch cache variables are respected; `SERF_ASSETS_DIR` selects additional model weights (default `~/.cache/serf/assets`) and `SERF_VALIDATION_CACHE` selects the frozen-base evaluation cache. Use the same cache locations for preparation and training.

## Train

Each recipe defaults to **four GPUs, 300 optimizer updates and validation every 20 updates**.

```bash
uv run bash scripts/hpsv21_serf.sh
uv run bash scripts/hpsv21_serf.sh --seed 1 --output runs/hpsv21-serf-s1 --wandb-project my-project
# Inspect the resolved configuration without GPUs or downloads:
uv run bash scripts/hpsv21_serf.sh --check-config
```

| Method | HPSv2.1 script | PickScore script | HPSv2.1 setting | PickScore setting |
|---|---|---|---:|---:|
| [SERF](serf/methods/serf.py) | `scripts/hpsv21_serf.sh` | `scripts/pickscore_serf.sh` | β = 36,000 | β = 1,500 |
| [RSM](serf/methods/rsm.py) | `scripts/hpsv21_rsm.sh` | `scripts/pickscore_rsm.sh` | β = 100,000 | β = 7,000 |
| [Adjoint Matching](serf/methods/adjoint_matching/) | `scripts/hpsv21_adjoint_matching.sh` | `scripts/pickscore_adjoint_matching.sh` | reward scale = 1,000,000 | reward scale = 25,000 |
| [VGG-Flow](serf/methods/vgg_flow.py) | `scripts/hpsv21_vgg_flow.sh` | `scripts/pickscore_vgg_flow.sh` | reward scale = 200,000 | reward scale = 25,000 |
| [DRaFT-LV](serf/methods/draft_lv.py) | `scripts/hpsv21_draft_lv.sh` | `scripts/pickscore_draft_lv.sh` | LR = 4e-5 | LR = 5e-5 |

Common settings: **64 trajectories × 4 targets/reward paths**, 512 × 512 images, 10 training steps, CFG 4.5, bf16, attention LoRA rank 32/alpha 64 and EMA decay 0.9. AM uses memoryless stochastic steps; the other methods use Euler.

Add `--gpus 8` for eight local GPUs; the global batch stays fixed. For multinode training, launch `python -m serf.train --method METHOD --reward REWARD` with your existing `torchrun` command. W&B logging is optional; configurations and `metrics.jsonl` are always saved locally.

## Evaluate and resume

Validation uses **20 Euler steps, CFG 4.5 and the policy EMA**. It generates 16 images for each of 200 task test-split prompts and 200 DrawBench prompts, reporting PickScore, HPSv2.1, ImageReward, CLIPScore, per-prompt FID and DreamSim ensemble diversity. Frozen-base images/features are cached. HPDv2 has overlapping train/test prompt text; see [implementation and split details](docs/implementation.md#prompts).

```bash
uv run bash scripts/hpsv21_serf.sh --output runs/hpsv21-serf-s0 --resume runs/hpsv21-serf-s0/latest
uv run torchrun --master_addr=127.0.0.1 --master_port=29501 --nproc_per_node=4 \
  -m serf.evaluate --checkpoint runs/hpsv21-serf-s0/step_300 --output runs/reevaluation
```

Checkpoints include live/EMA LoRA, optimizer, method-specific weights, counters and per-rank RNG state. Resume requires the same GPU count and recipe; interrupted validation is completed before training continues. EMA exports are in `step_N/evaluation/`. Standalone evaluation writes `metrics.json` and `validation_metadata.json`.

Table means require three seeds and the stated checkpoint-selection rule; a single final checkpoint is not a table mean. [Implementation details](docs/implementation.md) cover algorithms, metric definitions, seeding and source attribution.

## Tests and license

```bash
uv run pytest -q
```

CPU tests cover algorithm targets, CFG gradients, distributed batch accounting, checkpoint/RNG handling and evaluation. They do not establish full numerical reproduction.

Original source is [MIT licensed](LICENSE). Adapted research code retains [its MIT notice](third_party/first-order-ram-LICENSE); the ImageReward wrapper retains [Apache-2.0](third_party/ImageReward-LICENSE). Models, datasets and external VGG-Flow source retain their own terms. See [NOTICE](NOTICE) and [references](docs/implementation.md#sources).
