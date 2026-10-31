# Implementation reference

## Training and seeding

The [resolved configuration](../serf/config.py) defines the recipes. Frozen components are the base transformer, text encoders, VAE and reward networks; VGG-Flow additionally trains a value model. AdamW uses LR 1e-4 (except DRaFT-LV), betas (0.9, 0.99), weight decay 1e-4 and epsilon 1e-8, with no LR scheduler. LoRA uses Gaussian initialization; trainable LoRA, VAE decoding and reward differentiation use float32. Gradients are averaged across ranks once per optimizer update; AM clips their global norm to 1. Euler uses SD3.5's shifted noise grid.

[Training](../serf/train.py) seeds model initialization with `seed`, then training randomness with `seed + rank` after setup. The global prompt/image-seed schedule is stable across GPU counts; rank-local randomness and numerical reductions can differ. Resume restores checkpoint RNG after initialization. The historical SERF/RSM trainer seeded after LoRA construction, so identical seed numbers do not recreate its original initialization or random stream.

The environment is recorded in [uv.lock](../uv.lock), [the reference freeze](../requirements-environment.txt) and [environment metadata](environment.json). Preparation repairs the pinned HPSv2 wheel's missing BPE vocabulary using a checksum-verified copy from CLIP. Available model revisions and the CLIP source commit are pinned; other external assets follow their libraries' download mechanisms.

## Methods

Here `t=0` denotes clean data and `t=1` noise. The release's SERF and RSM modules own separate losses and state construction.

- **SERF:** differentiate the generated endpoint reward once, then match conditional residual velocity at four independently re-noised endpoint states. The target is `v_ref − β ∇r`; timestep probabilities are proportional to `1 − t` on the 1,000-point SD3.5 grid. Rollouts use CFG 4.5.
- **RSM:** match at four retained Euler trajectory states with target `v_ref − β (1 − t) ∇r`. Matching uses CFG velocity with the unconditional policy branch detached. Selection forces the last pre-terminal state and samples from three reversed near-equal partitions, clamping index zero to one. Duplicates are retained; four entries need not be distinct states.
- **Adjoint Matching:** ten memoryless stochastic steps with epsilon 0.1, followed by the lean adjoint recursion. The terminal derivative uses a noiseless frozen-reference projection from the penultimate state. Match at `x8`, `x9` and two distinct states from `x0…x7`; clip individual squared-L2 losses at `1.6 × reward_scale²`. Deterministic reward preprocessing retains the GPU forward and computes the resize input gradient on CPU. The extra state VJPs make 64 × 4 accounting different from equal compute.
- **DRaFT-LV:** differentiate the final Euler step (`K=1`) and three independently re-noised copies of the detached endpoint. Average the four negative ReLU reward objectives at scale 1. Re-noising uses the flow interpolant at the final scheduler sigma; differentiable steps use CFG.
- **VGG-Flow:** the pinned upstream target, consistency and boundary losses use CFG velocity. A copy of the policy transformer gets value LoRA rank/alpha 32 and an output projection initialized at 1e-5 scale. Settings are `eta_mode=quad`, terminal coefficient 1,000, detached direction and reward-gradient norm cap 1, without reward masking or velocity regularization. Four-state selection follows RSM's near-equal partition adaptation. Common frozen reward wrappers replace upstream reward preprocessing; memory/distributed integration remains local.

These baselines preserve the experiment's SD3.5 adaptations. The AM target uses a terminal preference reward rather than FOCUS's attention running costs; DRaFT-LV uses a flow interpolant. They are not unchanged copies of the original authors' training programs.

## Evaluation

Every 20 updates, the EMA is evaluated on two 200-prompt suites (`task_test` and `drawbench`), with 16 images per prompt, CFG 4.5 and 20 Euler steps. Prompt seed: 20260529; image seed base: 0. Current and frozen-base policies share prompts/seeds. Validation restores training RNG and adapter state afterward.

PickScore, HPSv2.1, ImageReward and CLIPScore are mean image rewards. The historical `HPSv2` key uses `HPS_v2.1_compressed.pt`. FID compares 16 current/base images per prompt using 2,048-dimensional Inception features, then averages across prompts. DreamSim ensemble diversity is half the mean pairwise cosine distance between normalized embeddings, excluding self-pairs, averaged across prompts. No aesthetic or separate CLIP-diversity metric is included.

`train/gpu_hours` sums the slowest rank's training-update duration × GPU count / 3,600, excluding startup, checkpoints and validation. For matched-reward tables, choose one checkpoint per seed using the stated target and report all metrics from that checkpoint. State whether DrawBench uses that checkpoint or its own matching rule; aggregate three seeds with sample standard deviation.

## Prompts

Only text is bundled; no preference images or labels. [The manifest](../prompts/manifest.json) records exact hashes and counts.

| Dataset | Train | Test | Source |
|---|---:|---:|---|
| Pick-a-Pic | 25,432 | 2,048 | [Dataset](https://huggingface.co/datasets/yuvalkirstain/pickapic_v1), experimental prompt split |
| HPDv2 | 103,665 | 400 | [ymhao/HPDv2](https://huggingface.co/datasets/ymhao/HPDv2), revision `bb0f2563e451e11c3f376b2ba93615db00e042e1` |
| DrawBench | — | 200 unique | [Imagen](https://imagen.research.google/), experimental prompt file |

Validation deduplicates before selecting 200 prompts. HPDv2 has 158 test texts also present in training; 93 of the selected 200 overlap the training pool. A finite run samples only part of that pool. Pick-a-Pic has no exact stripped-text train/test overlap. Splits preserve the original experiments; dataset terms apply separately from the code license.

## Sources

Code was extracted from [first-order-ram](https://github.com/RomeoPassaro/first-order-ram), retaining its MIT notice. [The source manifest](source_manifest.json) records source revisions, original paths and pre-adaptation hashes, including the archived AM runtime. Local orchestration replaces inherited cluster/configuration machinery.

| Method | Reference | Original code/reference implementation |
|---|---|---|
| RSM | [Reward Score Matching](https://arxiv.org/abs/2604.17415) | [jaylee2000/rsm](https://github.com/jaylee2000/rsm) |
| AM | [Adjoint Matching](https://arxiv.org/abs/2409.08861) | SD3.5 integration informed by [FOCUS](https://github.com/ericbill21/FOCUS) |
| DRaFT-LV | [Directly Fine-Tuning Diffusion Models on Differentiable Rewards](https://arxiv.org/abs/2309.17400) | Local SD3.5 adaptation |
| VGG-Flow | [Value Gradient Guidance](https://vggflow25.github.io/) | [lzzcd001/vggflow](https://github.com/lzzcd001/vggflow), revision pinned in [upstream.py](../serf/upstream.py) |

VGG-Flow source is fetched separately and excluded from this repository: its pinned revision has no explicit license, and our MIT license grants no rights to that external code. The adapted ImageReward wrapper retains its Apache-2.0 license and modification notice; see [NOTICE](../NOTICE).

Models and evaluation: [SD3.5 Medium](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium), [PickScore](https://github.com/yuvalkirstain/PickScore), [HPSv2](https://github.com/tgxs002/HPSv2), [ImageReward](https://github.com/zai-org/ImageReward), [CLIP](https://github.com/openai/CLIP), [DreamSim](https://github.com/ssundaram21/dreamsim), [torch-fidelity](https://github.com/toshas/torch-fidelity). Their models and source retain their own terms.
