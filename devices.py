"""One source of truth for the device and dtype a run uses.

Every class that needs to know where tensors live used to carry its own
`if args["device"] == "mps": ... else: cpu` block. Six of them lived in
networks.py alone, and because none of them knew about "cuda", a CUDA run
silently placed the nets on the CPU while the data sat on the GPU.

Imports only torch, so any module can use it.
"""

import torch


def resolve_device(args):
    """Map ``args["device"]`` to the ``(DTYPE, DEVICE)`` pair a run should use.

    CUDA deliberately keeps float64, matching the CPU path, so GPU results stay
    numerically comparable with CPU results. MPS has to be float32 because
    Metal has no float64 support at all.
    """
    name = (args.get("device") or "cpu")
    name = name.lower() if isinstance(name, str) else "cpu"

    if name == "mps":
        return torch.float32, torch.device("mps")
    if name == "cuda":
        return torch.float64, torch.device("cuda")
    return torch.float64, torch.device("cpu")


def move_value(value, device, dtype):
    """Move one attribute value onto ``device``.

    Floating tensors are also cast to ``dtype``. Integer and bool tensors move
    device-only, so index tensors and the `long` heuristic-lambda tiers keep
    their dtype. Dicts are walked, which is what ``opt_targets`` needs; lists
    and tuples of tensors are walked too. Anything else is returned untouched.
    """
    if torch.is_tensor(value):
        if value.is_floating_point():
            return value.to(device=device, dtype=dtype)
        return value.to(device=device)
    if isinstance(value, dict):
        return {k: move_value(v, device, dtype) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        moved = [move_value(v, device, dtype) for v in value]
        return type(value)(moved) if isinstance(value, tuple) else moved
    return value


def move_attrs(obj, device, dtype):
    """Move every tensor attribute of a plain (non-nn.Module) object in place.

    Walks ``vars(obj)`` rather than a hand-written list of attribute names.
    The hand-written approach is what let ``to_mps()`` drift: it covered 7 of
    the problem set's 13 tensor attributes, leaving the rest on the CPU.
    """
    for name, value in list(vars(obj).items()):
        setattr(obj, name, move_value(value, device, dtype))
    return obj
