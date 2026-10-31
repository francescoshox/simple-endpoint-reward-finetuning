"""Failure-safe policy and base-model selection."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext


def _adapter_host(model):
    candidate = model.module if hasattr(model, "module") else model
    if hasattr(candidate, "set_adapter") and (
        hasattr(candidate, "disable_adapter") or hasattr(candidate, "disable_adapters")
    ):
        return candidate
    raise AttributeError("Transformer does not expose supported adapter controls")


def _active_adapter(host):
    value = getattr(host, "active_adapter", None)
    if callable(value):
        value = value()
    if value is None:
        value = getattr(host, "active_adapters", None)
        if callable(value):
            value = value()
    return value


def _adapters_enabled(host):
    explicit = getattr(host, "enabled", None)
    if isinstance(explicit, bool):
        return explicit
    states = []
    for module in host.modules():
        disabled = getattr(module, "disable_adapters", None)
        if isinstance(disabled, bool):
            states.append(not disabled)
    if states and any(state != states[0] for state in states):
        raise RuntimeError("Policy adapter layers have inconsistent enabled states")
    return states[0] if states else None


def _set_adapters_enabled(host, enabled):
    """Toggle adapters through PEFT or Diffusers without API-name collisions.

    ``PeftModel`` delegates unknown attributes to its wrapped Diffusers model,
    so calling ``host.enable_adapters()`` can accidentally reach Diffusers'
    loader API. That API sees no directly loaded adapter and raises even though
    the PEFT wrapper has valid adapters. Prefer the wrapper's BaseTuner controls
    when present, and use Diffusers controls only for an unwrapped host.
    """

    base_model = getattr(host, "base_model", None)
    if base_model is not None:
        method = "enable_adapter_layers" if enabled else "disable_adapter_layers"
        toggle = getattr(base_model, method, None)
        if callable(toggle):
            toggle()
            return
    method = "enable_adapters" if enabled else "disable_adapters"
    toggle = getattr(host, method, None)
    if not callable(toggle):
        raise AttributeError(f"Transformer does not expose {method} controls")
    toggle()


class PolicyContexts:
    def __init__(self, model, current_adapter=None, has_base_adapter=True):
        self.model = model
        self.current_adapter = current_adapter
        self.has_base_adapter = has_base_adapter

    @contextmanager
    def _state(self):
        training = self.model.training
        host = _adapter_host(self.model) if self.has_base_adapter else None
        active = _active_adapter(host) if host is not None else None
        adapters_enabled = _adapters_enabled(host) if host is not None else None
        self.model.eval()
        try:
            yield host
        finally:
            if host is not None:
                if active is not None:
                    host.set_adapter(active)
                if adapters_enabled is not None:
                    _set_adapters_enabled(host, adapters_enabled)
            self.model.train(training)

    @contextmanager
    def current_policy(self):
        with self._state() as host:
            if host is not None:
                _set_adapters_enabled(host, True)
                if self.current_adapter is not None:
                    host.set_adapter(self.current_adapter)
            yield

    @contextmanager
    def base_policy(self):
        with self._state() as host:
            if host is None:
                with nullcontext():
                    yield
            elif hasattr(host, "disable_adapter"):
                with host.disable_adapter():
                    yield
            else:
                _set_adapters_enabled(host, False)
                yield
