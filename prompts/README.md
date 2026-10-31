# Prompt splits

These files preserve the prompt-only splits used by the experimental code. They contain no preference images or labels. Training generates fresh images from the prompts; this is not preference-pair training.

| Directory | Training prompts | Test-split prompts | Origin |
|---|---:|---:|---|
| `pickscore` | 25,432 | 2,048 | Pick-a-Pic prompt split from the original experiment repository |
| `hpsv2` | 103,665 | 400 | HPDv2 `train.json` / `test.json`, normalized and deduplicated in original order |
| `drawbench` | — | 200 unique | DrawBench evaluation file retained from the original experiment repository |

HPDv2 metadata is from [ymhao/HPDv2](https://huggingface.co/datasets/ymhao/HPDv2), revision `bb0f2563e451e11c3f376b2ba93615db00e042e1`. Pick-a-Pic is described by the [PickScore project](https://github.com/yuvalkirstain/PickScore) and [dataset](https://huggingface.co/datasets/yuvalkirstain/pickapic_v1). DrawBench is introduced in [Imagen](https://imagen.research.google/). The stored DrawBench file contains repeated prompts; validation deduplicates it before selecting the canonical 200.

Dataset terms apply independently of the source-code license. The experimental text files are preserved exactly; `manifest.json` records their byte hashes and line counts. Validation selects 200 test-split prompts and generates 16 images per selected prompt.

The HPDv2 files are separate dataset splits, but they are **not disjoint in prompt text**: 158 of the 400 unique test prompts also occur among the 103,665 training prompts. With the default evaluation prompt seed, 93 of the selected 200 prompts occur in that training pool. This does not establish which prompts a particular finite training run sampled. Pick-a-Pic has no exact stripped-text overlap between its bundled training and test splits. These files preserve the original experiments; removing overlaps would change the evaluation benchmark.
