"""Memory-bounded mean gradient reduction across GPUs."""

import torch


def synchronize_gradients(parameters, accelerator):
    # One reduction phase per optimizer update. Buckets avoid one collective
    # per parameter and cap temporary memory. Includes locally unused params.
    bucket, size = [], 0

    def flush():
        if not bucket:
            return
        flat = torch.cat(
            [
                (p.grad if p.grad is not None else torch.zeros_like(p)).flatten()
                for p in bucket
            ]
        )
        flat = accelerator.reduce(flat, reduction="mean")
        offset = 0
        for parameter in bucket:
            n = parameter.numel()
            parameter.grad = flat[offset : offset + n].view_as(parameter).clone()
            offset += n
        bucket.clear()

    for parameter in parameters:
        if size + parameter.numel() > 4_000_000:
            flush()
            size = 0
        bucket.append(parameter)
        size += parameter.numel()
    flush()
