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
