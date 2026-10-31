# Simple Endpoint Reward Finetuning

This repository contains the code for **Simple Endpoint Reward Finetuning (SERF)**, together with the baseline implementations and evaluation used in our experiments on Stable Diffusion 3.5 Medium.

SERF and each baseline have separate algorithm modules, with shared model, reward, checkpoint and evaluation utilities. We provide one complete training recipe per method for both HPSv2.1 and PickScore.

## Methods and source code

- **SERF:** our endpoint reward finetuning implementation in [`serf/methods/serf.py`](serf/methods/serf.py).
- **Reward Score Matching (RSM):** our SD3.5 trajectory-baseline implementation, following [Lee et al.](https://arxiv.org/abs/2604.17415). See the authors' [original code](https://github.com/jaylee2000/rsm) and our [implementation](serf/methods/rsm.py).
- **Adjoint Matching:** an SD3.5 adaptation of the [memoryless stochastic-control method](https://arxiv.org/abs/2409.08861), with SD3.5 integration informed by [FOCUS](https://github.com/ericbill21/FOCUS). See our [implementation](serf/methods/adjoint_matching/).
- **DRaFT-LV:** our SD3.5 flow-matching adaptation of [Directly Fine-Tuning Diffusion Models on Differentiable Rewards](https://arxiv.org/abs/2309.17400), using last-step differentiation and low-variance reward paths. See our [implementation](serf/methods/draft_lv.py).
- **VGG-Flow:** we provide an [integration wrapper](serf/methods/vgg_flow.py) for the [authors' implementation](https://github.com/lzzcd001/vggflow), described in [Value Gradient Guidance for Flow Matching Alignment](https://vggflow25.github.io/). **We do not vendor or redistribute VGG-Flow's algorithm source in this repository because the pinned upstream revision has no explicit license.** Its source is obtained separately from upstream and retains the upstream authors' rights; our MIT license does not cover it.

The local baseline implementations were extracted and adapted from our research codebase. Links above credit the original methods and the source code used as references; they do not imply that every baseline is an unchanged port of the authors' implementation. Exact code origins, retained notices and SD3.5 adaptations are documented in [provenance](docs/provenance.md) and [NOTICE](NOTICE).

## Installation

The environment is locked to the experiment stack: **Python 3.11.15, PyTorch 2.12.1, CUDA 13.0**. Linux ARM64 and x86-64 dependencies are resolved in `uv.lock`; training requires NVIDIA GPUs and a compatible driver. The reference hardware is NVIDIA GH200. Git and uv are required; VGG-Flow also needs Git on training workers to verify its pinned upstream checkout.

```bash
git clone https://github.com/francescoshox/simple-endpoint-reward-finetuning.git
cd simple-endpoint-reward-finetuning
uv sync --frozen
```

Accept the [SD3.5 Medium model terms](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium) and authenticate with Hugging Face if access requires it. Then prepare the diffusion model, reward models and evaluation models:

```bash
uv run python -m serf.prepare
```

For VGG-Flow, add `--vgg-flow` to the same preparation command to also obtain the separately distributed upstream source at the pinned revision:

```bash
uv run python -m serf.prepare --vgg-flow
```

Preparation downloads SD3.5 Medium, PickScore, HPSv2.1 and its CLIP backbone, CLIPScore, ImageReward and its tokenizer, DreamSim ensemble, and canonical FID weights. It runs on CPU and reuses cached downloads. Prompt splits are already included. Verify the prepared assets without downloads or GPUs with `uv run python -m serf.prepare --check-only`. Use the same cache locations during preparation and training, including on offline workers.

The pinned HPSv2 wheel omits a tokenizer vocabulary file. Preparation restores that single file from the locked CLIP package after checking its SHA-256; package versions remain unchanged.

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

Each validation evaluates **200 task test-split prompts and 200 DrawBench prompts, with 16 images per prompt**: 6,400 current-policy images in total. It reports PickScore, HPSv2.1, ImageReward, CLIPScore, per-prompt FID to the frozen base, and DreamSim ensemble diversity. Base reference images/features are cached. HPDv2 test-split prompts are not all unseen training text: 93 of the selected 200 appear in its training prompt pool; see [split details](prompts/README.md). See [the complete reproduction protocol](docs/reproduction.md).

`metrics.jsonl` and the resolved configuration are written to the run directory. Checkpoints contain the optimizer, live LoRA, evaluation EMA, method-specific mutable weights, counters and each rank's random state. Portable LoRA exports are included in `step_N/`, with EMA weights in `step_N/evaluation/`.

```bash
uv run bash scripts/hpsv21_serf.sh --output runs/hpsv21-serf-s0 --resume runs/hpsv21-serf-s0/latest

uv run torchrun --master_addr=127.0.0.1 --master_port=29501 --nproc_per_node=4 \
  -m serf.evaluate --checkpoint runs/hpsv21-serf-s0/step_300 --output runs/reevaluation
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
