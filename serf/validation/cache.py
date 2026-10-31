"""Content-addressed tensor cache with inter-job locking and atomic publication."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import pickle
import time
from collections.abc import Mapping
from contextlib import contextmanager
from enum import Enum
from pathlib import Path

import numpy as np
import torch


CACHE_SCHEMA = 2


def canonical_metadata(value):
    """Return a deterministic, torch-serializable cache-metadata value.

    Diffusers configs contain sets such as ``_use_default_values``.  Their
    string representation depends on Python's per-process hash seed, so using
    ``default=str`` while building a content key gives every distributed rank
    a different cache path.  Normalize all supported metadata types before
    hashing *and* before storing/comparing cache records.
    """

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        # Scheduler configs legitimately use infinities for clipping bounds
        # (for example DPMSolverMultistepScheduler.lambda_min_clipped=-inf).
        # JSON's native Infinity spelling is non-standard, so encode these
        # values explicitly to keep keys portable and deterministic.
        if np.isnan(value):
            return {"__ram_non_finite_float__": "nan"}
        if np.isposinf(value):
            return {"__ram_non_finite_float__": "+inf"}
        if np.isneginf(value):
            return {"__ram_non_finite_float__": "-inf"}
        return value
    if isinstance(value, np.generic):
        return canonical_metadata(value.item())
    if isinstance(value, Enum):
        return canonical_metadata(value.value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (torch.dtype, torch.device)):
        return str(value)
    if isinstance(value, Mapping):
        normalized = {}
        for key, item in value.items():
            if not isinstance(key, (str, int, float, bool)):
                raise TypeError(
                    "Cache metadata mapping keys must be scalar JSON keys, "
                    f"got {type(key).__name__}"
                )
            normalized_key = str(key)
            normalized_item = canonical_metadata(item)
            # Diffusers may expose this set-like bookkeeping field as a list
            # built from a set. Preserve order for ordinary lists, while
            # sorting this specifically unordered field across processes.
            if normalized_key == "_use_default_values" and isinstance(
                normalized_item, list
            ):
                normalized_item = sorted(
                    normalized_item,
                    key=lambda entry: json.dumps(
                        entry, sort_keys=True, separators=(",", ":")
                    ),
                )
            normalized[normalized_key] = normalized_item
        return normalized
    if isinstance(value, (list, tuple)):
        return [canonical_metadata(item) for item in value]
    if isinstance(value, (set, frozenset)):
        normalized = [canonical_metadata(item) for item in value]
        return sorted(
            normalized,
            key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")),
        )
    if isinstance(value, np.ndarray):
        return canonical_metadata(value.tolist())
    raise TypeError(f"Unsupported cache metadata type: {type(value).__name__}")


def content_key(metadata):
    metadata = canonical_metadata(metadata)
    encoded = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


class AtomicTensorCache:
    def __init__(self, root):
        self.root = Path(root)

    def path(self, namespace, metadata):
        return self.root / namespace / f"{content_key(metadata)}.pt"

    @contextmanager
    def lock(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = path.with_suffix(path.suffix + ".lock")
        with lock_path.open("a+b") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def load(path, metadata, expected_records=None, *, strict=False):
        metadata = canonical_metadata(metadata)
        if not path.is_file():
            return None
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except (
            EOFError,
            OSError,
            RuntimeError,
            ValueError,
            TypeError,
            pickle.UnpicklingError,
        ) as error:
            if strict:
                raise RuntimeError(
                    f"Malformed validation cache {path}: {error}"
                ) from error
            return None
        if not isinstance(payload, dict):
            if strict:
                raise RuntimeError(
                    f"Malformed validation cache {path}: payload is not a dictionary"
                )
            return None
        if payload.get("schema") != CACHE_SCHEMA:
            if strict:
                raise RuntimeError(
                    f"Validation cache schema mismatch for {path}: "
                    f"expected {CACHE_SCHEMA}, got {payload.get('schema')!r}"
                )
            return None
        if payload.get("metadata") != metadata:
            if strict:
                raise RuntimeError(f"Validation cache metadata mismatch for {path}")
            return None
        records = payload.get("records")
        if not isinstance(records, list):
            if strict:
                raise RuntimeError(
                    f"Malformed validation cache {path}: records is not a list"
                )
            return None
        if expected_records is not None and len(records) != expected_records:
            if strict:
                raise RuntimeError(
                    f"Validation cache record-count mismatch for {path}: "
                    f"expected {expected_records}, got {len(records)}"
                )
            return None
        return records

    @staticmethod
    def publish(path, metadata, records):
        metadata = canonical_metadata(metadata)
        temporary = path.with_name(f".{path.name}.part-{os.getpid()}-{time.time_ns()}")
        torch.save(
            {"schema": CACHE_SCHEMA, "metadata": metadata, "records": records},
            temporary,
        )
        # Verify before exposing the entry to other jobs.
        payload = torch.load(temporary, map_location="cpu", weights_only=True)
        if payload.get("metadata") != metadata or len(
            payload.get("records", ())
        ) != len(records):
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f"Validation cache verification failed for {path}")
        os.replace(temporary, path)
