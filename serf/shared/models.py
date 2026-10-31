"""Model helpers adapted from first-order-ram (MIT; see NOTICE)."""

import json
from pathlib import Path
import torch

LORA_TARGET_MODULES = [
    "attn.add_k_proj",
    "attn.add_q_proj",
    "attn.add_v_proj",
    "attn.to_add_out",
    "attn.to_k",
    "attn.to_out.0",
    "attn.to_q",
    "attn.to_v",
]
DTYPE_MAP = {"bf16": torch.bfloat16, "fp16": torch.float16, "no": torch.float32}


def load_prompts(path):
    path = Path(path)
    if not path.exists() and path.suffix == ".txt":
        json_path = path.with_suffix(".json")
        if json_path.exists():
            path = json_path

    with path.open() as stream:
        if path.suffix == ".json":
            records = json.load(stream)
            if not isinstance(records, list) or not all(
                isinstance(record, dict) and isinstance(record.get("text"), str)
                for record in records
            ):
                raise ValueError(
                    f"Expected a JSON list of objects with a text field in {path}"
                )
            return [
                record["text"].strip() for record in records if record["text"].strip()
            ]
        return [line.strip() for line in stream if line.strip()]
