# Reproduction protocol

## Common configuration

- Stable Diffusion 3.5 Medium, pinned model revision `b940f670f0eda2d07fbb75229e779da1ad11eb80`; 512 × 512 images.
- Train only attention LoRA parameters: rank 32, alpha 64, Gaussian initialization. Freeze the base transformer, text encoders, VAE and reward networks. VGG-Flow additionally trains its value-gradient model.
- AdamW: LR 10⁻⁴ except tuned DRaFT-LV rates, β₁ = 0.9, β₂ = 0.99, weight decay 10⁻⁴, ε = 10⁻⁸. No LR scheduler. AM clips the global gradient norm to 1; the other recipes do not apply a global norm cap.
- 300 optimizer updates; 64 generated trajectories per update, four loss targets or reward paths per trajectory. Four GPUs receive 16 trajectories each; eight GPUs receive eight each. Microbatches affect memory, not the global batch. Gradients are accumulated locally and then averaged across ranks once per update.
- 10 training steps and CFG 4.5. Euler uses the SD3.5 scheduler's shifted noise grid. AM uses memoryless stochastic steps on that grid rather than deterministic Euler rollouts.
- bfloat16 model computation, float32 trainable LoRA, VAE decoding and reward differentiation. EMA decay 0.9, updated once after each optimizer update.
- The deterministic prompt/image-seed sampler constructs a global schedule before sharding it across ranks. Model initialization uses the shared seed; rank-local random operations use `seed + rank` after initialization. This keeps prompt/image-seed assignments stable when changing GPU count, but does not promise identical floating-point training across topologies.

## Method-specific behavior

**SERF.** Generate an endpoint and differentiate its reward once. Repeat the endpoint gradient across four independently sampled times/noises. Match the conditional residual velocity against the unweighted endpoint reward gradient. The four losses use independently re-noised endpoint states, not rollout states. Timestep sampling is biased toward low noise with weights proportional to `1 − t` on a 1,000-point SD3.5 grid. Rollouts use CFG 4.5; the matching loss uses the conditional velocity, preserving the paper recipe.

**RSM.** Retain the actual Euler trajectory. Use its endpoint reward gradient at four selected trajectory states, with target multiplier `1 − t`. The matching velocity uses CFG 4.5 with the unconditional policy branch detached, as in the original grid. The selector forces the final pre-terminal state and draws from three reversed near-equal partitions. Index zero is clamped to one. Duplicates are allowed and retained: 64 × 4 counts loss entries, not four guaranteed distinct states. This ten-step near-equal partition adaptation is documented rather than presented as an unmodified upstream selector.

**VGG-Flow.** Call the pinned upstream target, value-consistency and terminal-boundary losses. The value network copies the SD3.5 transformer after the policy adapter is attached, freezes the copied base, and trains a separate value LoRA (rank 32, alpha 32) plus the output projection initialized at 10⁻⁵ scale. `eta_mode=quad`, terminal coefficient 1,000, detached direction, the experiment's fixed reward-gradient norm cap of 1 (the global quantile is diagnostic), no reward masking and no velocity regularizer. Both rollout and loss use the CFG velocity. The four states use the same near-equal partition convention described for RSM. The frozen training rewards are the common `HPSv2Diff`/`PickScoreDiff` wrappers; this is an experimental adaptation, not the upstream raw-PickScore preprocessing. Reward/VAE microbatching changes memory use without changing the summed gradients.

**Adjoint Matching.** Ten memoryless stochastic rollout steps, regularization ε = 0.1, followed by the lean adjoint reverse recursion. The terminal reward derivative follows the original SD3.5 adaptation: differentiate the noiseless frozen-reference projection from the penultimate state. Match at `x8`, `x9`, and two distinct uniformly selected states from `x0 … x7`; `x10` is not a matching-loss entry. Use squared L2 matching loss, clip individual losses at `1.6 × reward_scale²`, and clip the global parameter-gradient norm to 1. This is the paper's ten-step SD3.5 baseline adaptation, not a claim to reproduce an untouched original AM configuration. Its extra state VJPs make 64 × 4 a batch-accounting convention, not equal compute to SERF.

**DRaFT-LV.** No gradients through the first nine Euler steps. Differentiate through the final step (`K=1`), plus three independent re-noised copies of the detached generated endpoint. Average the four reward terms at scale 1, using the experiment's negative mean ReLU reward objective. Re-noising follows the flow interpolant at the final scheduler sigma. CFG applies to the differentiable final steps as well as the prefix.

The ten recipes and method-specific scales are listed in the root README. No method is selected through a mode flag inside SERF's or RSM's loss.

## Evaluation

Validation runs after updates 20, 40, …, 300 and uses the policy EMA. Each run evaluates two suites:

1. `task_test`: 200 prompts from the held-out split of the corresponding training dataset, deduplicated and selected deterministically with prompt seed 20260529.
2. `drawbench`: the 200 unique DrawBench prompts, with the same deterministic selection rule.

Each prompt receives 16 fixed image seeds, with seed base 0, CFG 4.5 and 20 Euler steps. This is 3,200 current-policy images per suite, 6,400 total per evaluation. Frozen-base images and derived features use identical prompts/seeds and a versioned cache, so the first evaluation costs more than subsequent ones. Validation saves and restores the training RNG stream and adapter state.

Report the mean image reward for PickScore, HPSv2.1, ImageReward and CLIPScore. The historical metric key `HPSv2` refers to the **HPS v2.1 checkpoint** (`HPS_v2.1_compressed.pt`); HPDv2 is the dataset name. No aesthetic metric is reported.

FID is computed per prompt between the 16 fine-tuned and 16 frozen-base images using canonical 2,048-dimensional Inception features, then averaged over prompts. DreamSim diversity uses the ensemble embedding: half the mean pairwise cosine distance between normalized features, excluding self-pairs, averaged over prompts. It is not feature variance and does not include a separate CLIP-diversity score.

`train/gpu_hours` accumulates the slowest rank's training-update duration times the number of GPUs, divided by 3,600. It excludes model startup, checkpoint writes and validation. W&B is optional; local JSONL records remain available without it. Each optimizer update contributes one W&B row, with validation attached when due.

## Checkpoint selection and seeds

The public recipes run the full 300-update horizon and retain checkpoints every 20 updates. For tables that specify a target reward, select checkpoints using that table's stated matching rule and report the other metrics from the same checkpoint. Do not independently optimize each metric. A DrawBench comparison must identify whether it uses the task-selected checkpoint or its own explicitly stated matching rule.

Reported paper means and sample standard deviations require three seed runs and the original checkpoint-selection rule. The root recipes provide the training settings, not a guarantee that seed 0 at update 300 equals a three-seed table mean. Preserve resolved configuration, update number and validation prompt/seed metadata when reporting a result.

The private development checks that skip intermediate validation are intentionally outside this repository. They are not paper reproduction recipes.
