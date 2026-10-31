# Simple Endpoint Reward Finetuning

Minimal SD3.5 Medium post-training code for **SERF**, **Reward Score Matching (RSM)**, **Adjoint Matching**, **VGG-Flow**, and **DRaFT-LV**.

Each method has its own algorithm implementation. Model loading, reward scorers, distributed batch scheduling, checkpoints and evaluation are shared. The repository contains one complete training recipe per method for both HPSv2.1 and PickScore.

## Installation

The environment is locked to the experiment stack: **Python 3.11.15, PyTorch 2.12.1, CUDA 13.0**. Linux ARM64 and x86-64 dependencies are resolved in `uv.lock`; training requires NVIDIA GPUs and a compatible driver. The reference hardware is NVIDIA GH200.

```bash
git clone https://github.com/francescoshox/simple-endpoint-reward-finetuning.git
cd simple-endpoint-reward-finetuning
uv sync --frozen
```

Accept the [SD3.5 Medium model terms](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium) and authenticate with Hugging Face if access requires it. Then prepare the diffusion model, reward models, evaluation models and pinned VGG-Flow dependency:

```bash
uv run python -m serf.prepare
```

Standard Hugging Face and Torch cache variables are respected. Set `SERF_ASSETS_DIR` to choose the directory for additional reward and DreamSim weights; its default is `~/.cache/serf/assets`. `SERF_VALIDATION_CACHE` selects a reusable cache for frozen-base evaluation. Downloads are preparation steps, not part of training time.

Prompt-only copies of the experimental splits are included under `prompts/`. No preference images or labels are used for post-training. See [prompt provenance](prompts/README.md) and [source attribution](docs/provenance.md), including the separate terms for VGG-Flow.

## Train

Run a single recipe from the repository root. Each script launches four local GPU processes by default and performs **300 optimizer updates with validation every 20 updates**.

```bash
uv run bash scripts/hpsv21_serf.sh
# A second seed, a chosen output directory, and optional W&B logging:
uv run bash scripts/hpsv21_serf.sh --seed 1 --output runs/hpsv21-serf-s1 --wandb-project my-project
```

| Method | HPSv2.1 recipe | PickScore recipe | HPSv2.1 setting | PickScore setting |
|---|---|---|---:|---:|
| SERF | `scripts/hpsv21_serf.sh` | `scripts/pickscore_serf.sh` | β = 36,000 | β = 1,500 |
| RSM | `scripts/hpsv21_rsm.sh` | `scripts/pickscore_rsm.sh` | β = 100,000 | β = 7,000 |
| Adjoint Matching | `scripts/hpsv21_adjoint_matching.sh` | `scripts/pickscore_adjoint_matching.sh` | reward scale = 1,000,000 | reward scale = 25,000 |
| VGG-Flow | `scripts/hpsv21_vgg_flow.sh` | `scripts/pickscore_vgg_flow.sh` | reward scale = 200,000 | reward scale = 25,000 |
| DRaFT-LV | `scripts/hpsv21_draft_lv.sh` | `scripts/pickscore_draft_lv.sh` | LR = 4 × 10⁻⁵ | LR = 5 × 10⁻⁵ |

All recipes use 64 global trajectories and four targets or reward paths per trajectory. Training uses 10 steps and CFG 4.5. AM uses a stochastic memoryless sampler; the other methods use Euler. Evaluation always uses 20 Euler steps and CFG 4.5. LoRA has rank 32 and alpha 64; evaluation uses an EMA with decay 0.9.

For eight GPUs on one node, add `--gpus 8`. This keeps the global batch at 64 × 4. For a scheduler-managed or multinode launch, invoke `python -m serf.train` under your existing `torchrun` command; the entry point detects `LOCAL_RANK` and does not nest launchers. The repository does not depend on Slurm, cluster modules, private storage or a W&B relay.

Inspect the complete resolved configuration without a GPU or model downloads:

```bash
uv run bash scripts/hpsv21_rsm.sh --check-config
```

## Evaluate and resume

Each validation evaluates **200 held-out task prompts and 200 DrawBench prompts, with 16 images per prompt**: 6,400 current-policy images in total. It reports PickScore, HPSv2.1, ImageReward, CLIPScore, per-prompt FID to the frozen base, and DreamSim ensemble diversity. Base reference images/features are cached. See [the complete reproduction protocol](docs/reproduction.md).

`metrics.jsonl` and the resolved configuration are written to the run directory. Checkpoints contain the optimizer, live LoRA, evaluation EMA, method-specific mutable weights, counters and each rank's random state. Portable LoRA exports are included in `step_N/`, with EMA weights in `step_N/evaluation/`.

```bash
uv run bash scripts/hpsv21_serf.sh --output runs/hpsv21-serf-s0 --resume runs/hpsv21-serf-s0/latest

uv run torchrun --master_addr=127.0.0.1 --master_port=29501 --nproc_per_node=4   -m serf.evaluate --checkpoint runs/hpsv21-serf-s0/step_300 --output runs/reevaluation
```

A complete checkpoint resumes with the same GPU count and recipe. Random-state restoration preserves the training stream; changing hardware, GPU count, kernels or library versions can change numerical results. Interrupted `.partial` directories are never accepted as checkpoints. If validation was interrupted after a complete training checkpoint, resume finishes that validation before training continues. Standalone evaluation writes `metrics.json` and `validation_metadata.json`.

## Repository layout

```text
serf/methods/serf.py                 independent endpoint re-noising and SERF loss
serf/methods/rsm.py                  trajectory selection and RSM loss
serf/methods/adjoint_matching/       stochastic rollout, lean adjoint and matching
serf/methods/draft_lv.py             last-step differentiation and LV paths
serf/methods/vgg_flow.py             adapter to the pinned upstream algorithm
serf/shared/                       models, batches, EMA and checkpoints
serf/rewards/                      frozen differentiable reward scorers
serf/validation/                   common paper evaluation
scripts/                           ten complete recipes
```

The recipes reproduce individual experimental settings. Paper table entries aggregate three seeds and may select intermediate checkpoints near a stated target reward; a single default seed at update 300 is not the table mean. No reduced-validation or private correctness-run configuration is included.

## Tests

```bash
uv run pytest -q
```

These CPU tests check algorithm targets, CFG gradients, batch accounting, random-state handling and the evaluation protocol. End-to-end GPU reproduction is tracked separately; CPU tests do not establish agreement with the paper's numerical results.

## Acknowledgments and license

Adapted research code retains the original MIT notice in [third_party/first-order-ram-LICENSE](third_party/first-order-ram-LICENSE). See [NOTICE](NOTICE) and [provenance](docs/provenance.md) for algorithm references and the precise adaptations.

Our original source code is released under the [MIT license](LICENSE); the adapted ImageReward wrapper retains [Apache-2.0](third_party/ImageReward-LICENSE). Downloaded models, datasets and VGG-Flow retain their own terms. VGG-Flow is fetched at a fixed revision and excluded from this repository; its upstream snapshot has no explicit license file. No license for that external code is granted here.
