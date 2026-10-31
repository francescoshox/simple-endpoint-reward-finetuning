"""Verify package-data preparation without changing an installed environment."""

from types import SimpleNamespace
import hashlib
import pytest
from serf import prepare


def fixture_packages(tmp_path, monkeypatch, data=b"tokenizer-data"):
    source = tmp_path / "clip/clip/bpe_simple_vocab_16e6.txt.gz"
    target = tmp_path / "hps/hpsv2/src/open_clip/bpe_simple_vocab_16e6.txt.gz"
    source.parent.mkdir(parents=True)
    target.parent.mkdir(parents=True)
    source.write_bytes(data)

    def distribution(name):
        root = tmp_path / ("clip" if name == "clip" else "hps")
        return SimpleNamespace(locate_file=lambda name: root / name)

    monkeypatch.setattr(prepare, "distribution", distribution)
    monkeypatch.setattr(prepare, "HPS_VOCAB_SHA256", hashlib.sha256(data).hexdigest())
    return source, target


def test_hps_data_check_is_read_only_and_repair_preserves_bytes(tmp_path, monkeypatch):
    source, target = fixture_packages(tmp_path, monkeypatch)
    with pytest.raises(FileNotFoundError, match="Missing HPSv2"):
        prepare.prepare_hps_vocab(check=True)
    assert not target.exists()
    assert prepare.prepare_hps_vocab() == target
    assert target.read_bytes() == source.read_bytes()
    assert prepare.prepare_hps_vocab(check=True) == target


def test_hps_data_rejects_unexpected_vocabulary(tmp_path, monkeypatch):
    source, target = fixture_packages(tmp_path, monkeypatch)
    source.write_bytes(b"wrong-vocabulary")
    with pytest.raises(ValueError, match="unexpected SHA-256"):
        prepare.prepare_hps_vocab()
    assert not target.exists()


def test_fid_preparation_downloads_missing_weights_and_checks_them(tmp_path, monkeypatch):
    import torch
    from serf.validation import scorers
    data = b"canonical-fid-fixture"
    monkeypatch.setattr(prepare, "MODELS_DIR", tmp_path / "assets")
    monkeypatch.setattr(scorers, "MODELS_DIR", tmp_path / "assets")
    monkeypatch.setattr(torch.hub, "get_dir", lambda: str(tmp_path / "hub"))
    monkeypatch.setattr(prepare, "FID_WEIGHTS_SHA256", hashlib.sha256(data).hexdigest())
    calls = []
    def download(url, target, hash_prefix):
        calls.append((url, hash_prefix))
        from pathlib import Path
        Path(target).write_bytes(data)
    monkeypatch.setattr(torch.hub, "download_url_to_file", download)
    with pytest.raises(FileNotFoundError):
        prepare.prepare_fid_weights(check=True)
    assert not (tmp_path / "assets").exists()
    path = prepare.prepare_fid_weights()
    assert path.read_bytes() == data and len(calls) == 1
    assert prepare.prepare_fid_weights(check=True) == path
    assert len(calls) == 1
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="unexpected SHA-256"):
        prepare.prepare_fid_weights(check=True)


def test_dreamsim_check_rejects_incomplete_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare, "MODELS_DIR", tmp_path)
    required = ["dino_vitb16_pretrain.pth", "clip_vitb16_pretrain.pth.tar",
                "open_clip_vitb16_pretrain.pth.tar", "ensemble_lora/adapter_config.json",
                "facebookresearch_dino_main/hubconf.py", "checkpoints/dino_vitbase16_pretrain.pth"]
    for name in required:
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"fixture")
    with pytest.raises(FileNotFoundError, match="adapter_model"):
        prepare.check_dreamsim_assets()
    (tmp_path / "ensemble_lora/adapter_model.safetensors").write_bytes(b"fixture")
    prepare.check_dreamsim_assets()
    (tmp_path / "dino_vitb16_pretrain.pth").write_bytes(b"")
    with pytest.raises(FileNotFoundError, match="dino_vitb16_pretrain"):
        prepare.check_dreamsim_assets()
