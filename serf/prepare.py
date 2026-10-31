"""Prepare pinned model/reward assets and the upstream VGG-Flow dependency."""

import argparse, hashlib, shutil, subprocess, tempfile
from importlib.metadata import distribution
from pathlib import Path
from .config import Config, ROOT
from .upstream import UPSTREAM_REVISION
from .rewards.model_ids import (
    CLIP_L14_ID,
    CLIP_L14_REVISION,
    HPSV2_ID,
    HPSV2_REVISION,
    LAION_CLIP_H14_ID,
    LAION_CLIP_H14_REVISION,
    PICKSCORE_ID,
    PICKSCORE_REVISION,
)
from .rewards.paths import MODELS_DIR


HPS_VOCAB_SHA256 = "924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a"


def prepare_hps_vocab(check=False):
    """Restore one data file omitted from the pinned HPSv2 wheel.

    The locked CLIP package contains the byte-identical vocabulary. Only this
    verified data file is copied; no dependency source or version is changed.
    """
    target = Path(
        distribution("hpsv2").locate_file(
            "hpsv2/src/open_clip/bpe_simple_vocab_16e6.txt.gz"
        )
    )
    if target.is_file():
        if hashlib.sha256(target.read_bytes()).hexdigest() != HPS_VOCAB_SHA256:
            raise ValueError("Existing HPSv2 vocabulary has an unexpected SHA-256")
        return target
    if check:
        raise FileNotFoundError(
            f"Missing HPSv2 package data: {target}. Run python -m serf.prepare first."
        )
    source = Path(distribution("clip").locate_file("clip/bpe_simple_vocab_16e6.txt.gz"))
    data = source.read_bytes()
    if hashlib.sha256(data).hexdigest() != HPS_VOCAB_SHA256:
        raise ValueError("The locked CLIP vocabulary has an unexpected SHA-256")
    # The HPSv2 module reads this adjacent data file during import, before it
    # exposes a way to configure the tokenizer. Restore the wheel's omission.
    with target.open("xb") as output:
        output.write(data)
    print(f"Restored HPSv2 tokenizer data: {target}")
    return target


def prepare_vgg(check=False):
    root = ROOT / "third_party/vggflow"
    if not root.exists():
        if check:
            raise FileNotFoundError("Run python -m serf.prepare --vgg-flow")
        subprocess.run(
            ["git", "clone", "https://github.com/lzzcd001/vggflow.git", str(root)],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(root), "checkout", "--detach", UPSTREAM_REVISION],
            check=True,
        )
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != UPSTREAM_REVISION:
        raise ValueError("Unexpected VGG-Flow revision")



FID_WEIGHTS_SHA256 = "6726825d0af5f729cebd5821db510b11b1cfad8faad88a03f1befd49fb9129b2"


def prepare_fid_weights(check=False):
    """Fetch the canonical torch-fidelity checkpoint, verifying the paper asset."""
    from torch.hub import download_url_to_file
    from torch_fidelity.feature_extractor_inceptionv3 import URL_INCEPTION_V3
    from .validation.scorers import canonical_fid_weights_path, FID_WEIGHTS_FILENAME

    try:
        path = canonical_fid_weights_path()
    except FileNotFoundError:
        if check:
            raise
        path = MODELS_DIR / "torch/hub/checkpoints" / FID_WEIGHTS_FILENAME
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as file:
            temporary = Path(file.name)
        try:
            download_url_to_file(URL_INCEPTION_V3, str(temporary), hash_prefix=FID_WEIGHTS_SHA256)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for data in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(data)
    if digest.hexdigest() != FID_WEIGHTS_SHA256:
        raise ValueError(f"Canonical FID weights have an unexpected SHA-256: {path}")
    return path


def check_dreamsim_assets():
    """Verify the ensemble cache layout of the locked DreamSim version offline."""
    required = [
        "dino_vitb16_pretrain.pth",
        "clip_vitb16_pretrain.pth.tar",
        "open_clip_vitb16_pretrain.pth.tar",
        "ensemble_lora/adapter_config.json",
        "facebookresearch_dino_main/hubconf.py",
        "checkpoints/dino_vitbase16_pretrain.pth",
    ]
    adapters = [MODELS_DIR / "ensemble_lora" / name for name in
                ("adapter_model.safetensors", "adapter_model.bin")]
    missing = [str(MODELS_DIR / name) for name in required
               if not (MODELS_DIR / name).is_file() or not (MODELS_DIR / name).stat().st_size]
    if not any(path.is_file() and path.stat().st_size for path in adapters):
        missing.append(str(MODELS_DIR / "ensemble_lora/adapter_model.{safetensors,bin}"))
    if missing:
        raise FileNotFoundError("Missing DreamSim assets; run python -m serf.prepare: " + ", ".join(missing))


def prepare_assets(check=False):
    prepare_hps_vocab(check=check)
    from huggingface_hub import snapshot_download, hf_hub_download

    if not check:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
    c = Config("serf", "hpsv21")
    for name, revision in [
        (c.model_name, c.model_revision),
        (PICKSCORE_ID, PICKSCORE_REVISION),
        (LAION_CLIP_H14_ID, LAION_CLIP_H14_REVISION),
        (CLIP_L14_ID, CLIP_L14_REVISION),
    ]:
        snapshot_download(name, revision=revision, local_files_only=check)
    for repo, revision, name in [
        (HPSV2_ID, HPSV2_REVISION, "HPS_v2.1_compressed.pt"),
        (LAION_CLIP_H14_ID, LAION_CLIP_H14_REVISION, "open_clip_pytorch_model.bin"),
    ]:
        if check:
            if not (MODELS_DIR / name).is_file():
                raise FileNotFoundError(MODELS_DIR / name)
        else:
            src = hf_hub_download(repo, filename=name, revision=revision)
            if Path(src).resolve() != (MODELS_DIR / name).resolve():
                shutil.copyfile(src, MODELS_DIR / name)
    if check:
        required = ["ImageReward/ImageReward.pt", "ImageReward/med_config.json"]
        for name in required:
            if not (MODELS_DIR / name).is_file():
                raise FileNotFoundError(MODELS_DIR / name)
        from transformers import BertTokenizer

        BertTokenizer.from_pretrained("bert-base-uncased", local_files_only=True)
        check_dreamsim_assets()
        prepare_fid_weights(check=True)
    else:
        from .rewards.image_reward import rm_load
        from .validation.scorers import DreamSimScorer, CanonicalFidScorer

        del_model = rm_load()
        del del_model
        scorer = DreamSimScorer("ensemble", MODELS_DIR, 10, "cpu")
        del scorer
        check_dreamsim_assets()
        prepare_fid_weights()
        scorer = CanonicalFidScorer(batch_size=10, device="cpu")
        del scorer
    print("Pinned SD3.5 and paper scorer assets ready")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--vgg-flow", action="store_true", help="Also fetch the optional pinned VGG-Flow source")
    p.add_argument("--check-only", action="store_true")
    a = p.parse_args()
    prepare_assets(a.check_only)
    if a.vgg_flow:
        prepare_vgg(a.check_only)


if __name__ == "__main__":
    main()
