"""Shared fixed-decay evaluation EMA support for LoRA policies."""

from pathlib import Path


EMA_DECAY = 0.9
EMA_ADAPTER = "evaluation"
EMA_EXPORT_DIR = "evaluation"
LORA_WEIGHTS_NAME = "pytorch_lora_weights.safetensors"


def adapter_host(model):
    if hasattr(model, "set_adapter"):
        return model
    if hasattr(model, "module") and hasattr(model.module, "set_adapter"):
        return model.module
    raise AttributeError("Model does not expose PEFT adapter controls")


def collect_policy_params(model):
    """Return live and EMA LoRA parameters, leaving the live adapter active."""
    host = adapter_host(model)
    host.set_adapter("default")
    live = [parameter for parameter in host.parameters() if parameter.requires_grad]
    host.set_adapter(EMA_ADAPTER)
    evaluation = [
        parameter for parameter in host.parameters() if parameter.requires_grad
    ]
    host.set_adapter("default")
    if not live or len(live) != len(evaluation):
        raise RuntimeError(
            "Live and evaluation LoRA adapters do not have matching parameters"
        )
    if any(
        source.shape != target.shape
        for source, target in zip(live, evaluation, strict=True)
    ):
        raise RuntimeError("Live and evaluation LoRA adapter shapes do not match")
    return live, evaluation


def initialize_evaluation_ema(live, evaluation):
    for source, target in zip(live, evaluation, strict=True):
        target.data.copy_(source.detach().data)


def update_evaluation_ema(live, evaluation):
    """Apply the EAM evaluation EMA once after a successful optimizer update."""
    for source, target in zip(live, evaluation, strict=True):
        target.data.mul_(EMA_DECAY).add_(source.detach().data, alpha=1.0 - EMA_DECAY)


def evaluation_artifact(checkpoint):
    return Path(checkpoint) / EMA_EXPORT_DIR / LORA_WEIGHTS_NAME


def require_evaluation_artifact(checkpoint):
    artifact = evaluation_artifact(checkpoint)
    if not artifact.is_file():
        raise ValueError(
            f"Checkpoint is missing required evaluation EMA weights: {artifact}"
        )
    return artifact


def save_policy_exports(pipe, model, checkpoint):
    """Export the live policy at the root and its EMA in evaluation/."""
    from peft.utils import get_peft_model_state_dict

    checkpoint = Path(checkpoint)
    pipe.save_lora_weights(
        str(checkpoint),
        transformer_lora_layers=get_peft_model_state_dict(
            model, adapter_name="default"
        ),
        safe_serialization=True,
    )
    pipe.save_lora_weights(
        str(checkpoint / EMA_EXPORT_DIR),
        transformer_lora_layers=get_peft_model_state_dict(
            model, adapter_name=EMA_ADAPTER
        ),
        safe_serialization=True,
    )


def load_policy_export(model, artifact, adapter_name):
    """Load one portable Diffusers LoRA export into a named PEFT adapter."""
    from peft.utils import set_peft_model_state_dict
    from safetensors.torch import load_file

    state = load_file(str(artifact))
    state = {key.removeprefix("transformer."): value for key, value in state.items()}
    loaded = set_peft_model_state_dict(
        adapter_host(model), state, adapter_name=adapter_name
    )
    missing_target = [
        key
        for key in loaded.missing_keys
        if "lora_" in key and f".{adapter_name}." in key
    ]
    if loaded.unexpected_keys or missing_target:
        raise ValueError(
            f"Policy export does not match the configured {adapter_name!r} LoRA"
        )
