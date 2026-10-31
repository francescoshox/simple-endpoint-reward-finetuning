# Sources and attribution

The submitted manuscript is not bundled here. Publication links and a paper citation will be added when available; this repository does not claim that the paper has been accepted or published.

## Research implementation

The release is extracted from [RomeoPassaro/first-order-ram](https://github.com/RomeoPassaro/first-order-ram), retaining its MIT notice. The validated experiment branch snapshot is `6aadd1f1596e90cf86a7d0f94b1a4c533267bef1`. The newer SERF/RSM CFG-loss and near-equal trajectory-selection behavior was taken from main snapshot `5067e3832df0dae9a68030c672f88b44743807c6`. The AM runtime comes from the experiment's archived `adjoint-matching-20260921-220337` source snapshot.

[The source manifest](source_manifest.json) records original paths relative to these snapshots and SHA-256 hashes before release adaptations. The new package replaces inherited configuration/cluster orchestration with explicit recipes, a shared trainer, full checkpoints and portable launch commands. Source hashes are provenance, not claims that adapted files remain byte-identical.

SERF is the original experiment's **direct** method. RSM is its **trajectory** method. Their targets and state construction live in independent source modules. They share rollout, frozen reward differentiation and infrastructure.

## Baselines

- **Reward Score Matching (RSM):** Jeongjae Lee, Jinho Chang, Jeongsol Kim and Jong Chul Ye, [Reward Score Matching: Unifying Reward-based Fine-tuning for Flow and Diffusion Models](https://arxiv.org/abs/2604.17415). The authors' [official code](https://github.com/jaylee2000/rsm) provides the original implementation and MIT notice. Our trajectory baseline is the SD3.5 experiment implementation extracted from first-order-ram; it uses endpoint reward gradients, retained trajectory states and clean-time weighting. The reference link is attribution, not a claim that this file is an unchanged copy of their trainer.

- **DRaFT-LV:** Kevin Clark, Paul Vicol, Kevin Swersky and David J. Fleet, [Directly Fine-Tuning Diffusion Models on Differentiable Rewards](https://arxiv.org/abs/2309.17400), ICLR 2024. The implementation is the experiment code's SD3.5 flow-matching adaptation of `K=1` plus three low-variance paths, not an unmodified original Stable Diffusion 1.4 training program.
- **Adjoint Matching:** Carles Domingo-Enrich, Michal Drozdzal, Brian Karrer and Ricky T. Q. Chen, [Adjoint Matching: Fine-tuning Flow and Diffusion Generative Models with Memoryless Stochastic Optimal Control](https://arxiv.org/abs/2409.08861), ICLR 2025. The experiment implementation follows the memoryless dynamics and lean adjoint equations, with SD3.5 integration informed by [FOCUS](https://github.com/ericbill21/FOCUS). It uses a terminal preference reward; it does not use FOCUS's attention running costs. No FOCUS source tree is vendored or required. See the reproduction document for the ten-step discretization and terminal projection used in these experiments.
- **VGG-Flow:** Zhen Liu, Tim Z. Xiao, Carles Domingo-Enrich, Weiyang Liu and Dinghuai Zhang, [Value Gradient Guidance for Flow Matching Alignment](https://vggflow25.github.io/), NeurIPS 2025. Algorithm code is fetched from [lzzcd001/vggflow](https://github.com/lzzcd001/vggflow) at `e9748e71d9b2a9d1202effeb682b006a8c234ccf`. Our adapter pins that revision and checks that it is clean. The runtime retains the upstream velocity, consistency and boundary losses; SD3.5 value parametrization, common reward preprocessing, four-state near-equal selection and memory/distributed integration are documented adaptations.

The pinned VGG-Flow snapshot contains no explicit license file, and the upstream GitHub repository reports no detected license. It is downloaded separately into an ignored directory and is not covered by this repository's MIT license. Obtain the upstream authors' terms before redistributing their source. Attribution alone does not create a redistribution license; see [GitHub's licensing guidance](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository).

## Models and evaluation libraries

- [Stable Diffusion 3.5 Medium](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium), under its own model license.
- [PickScore](https://huggingface.co/yuvalkirstain/PickScore_v1) and its [official implementation](https://github.com/yuvalkirstain/PickScore).
- [HPSv2](https://github.com/tgxs002/HPSv2), using the **v2.1** checkpoint.
- [ImageReward](https://github.com/zai-org/ImageReward), [CLIP](https://github.com/openai/CLIP), [DreamSim](https://github.com/ssundaram21/dreamsim), and [torch-fidelity](https://github.com/toshas/torch-fidelity).

Model identifiers and available Hugging Face revisions are pinned in source. Package versions and the CLIP Git commit are pinned in `uv.lock`. External non-Hugging-Face weight downloads remain subject to the corresponding library's distribution mechanism.

The differentiable ImageReward wrapper includes adapted Apache-2.0 source (copyright 2023 Jiazheng Xu, Xiao Liu). Its original license and explicit modification notice are retained; it is not relicensed under MIT.
