"""One source of truth for the device and dtype a run uses.

Every class that needs to know where tensors live used to carry its own
`if args["device"] == "mps": ... else: cpu` block. Six of them lived in
networks.py alone, and because none of them knew about "cuda", a CUDA run
silently placed the nets on the CPU while the data sat on the GPU.

Selection is by name: "cpu", "cuda", "mps", or "auto" to detect. An
unrecognised or unavailable name raises rather than falling back to the CPU.

Imports only torch, so any module can use it.
"""

import torch

KNOWN_DEVICES = ("auto", "cpu", "cuda", "mps")

#! Each device's precision. CUDA keeps float64 so GPU results stay numerically
#! comparable with the CPU results every experiment so far has produced. MPS
#! has no choice: Metal does not support float64 at all.
DEVICE_DTYPES = {
    "cpu": torch.float64,
    "cuda": torch.float64,
    "mps": torch.float32,
}


def is_available(name):
    if name == "cpu":
        return True
    if name == "cuda":
        return torch.cuda.is_available()
    if name == "mps":
        return getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    return False


def available_devices():
    return tuple(name for name in ("cpu", "cuda", "mps") if is_available(name))


def detect_device():
    """Pick a device automatically: CUDA when present, otherwise CPU.

    MPS is deliberately NOT auto-selected even on a Mac that supports it.
    Metal forces float32, so choosing it here would silently change results
    away from the float64 the CPU and CUDA paths use. Ask for it explicitly
    (--device mps) when you want it.
    """
    return "cuda" if is_available("cuda") else "cpu"


def resolve_device_name(name):
    """Normalise a device string to a concrete, available device name.

    Raises ValueError for an unrecognised name and RuntimeError for one that is
    recognised but not present, rather than quietly falling back to the CPU: a
    silent fallback means a GPU job that holds an allocation, runs on the CPU
    and still exits successfully.
    """
    if name is None:
        name = "auto"
    if not isinstance(name, str):
        raise ValueError(
            f"device must be a string, got {type(name).__name__}. "
            f"Valid values: {', '.join(KNOWN_DEVICES)}."
        )

    name = name.strip().lower()
    if name not in KNOWN_DEVICES:
        raise ValueError(
            f'Unknown device "{name}". Valid values: {", ".join(KNOWN_DEVICES)}. '
            f'Use "auto" to pick CUDA when it is present and CPU otherwise.'
        )

    if name == "auto":
        return detect_device()

    if not is_available(name):
        raise RuntimeError(
            f'Device "{name}" was requested but is not available on this machine. '
            f'Available: {", ".join(available_devices())}. '
            f'Use --device auto to select one automatically.'
        )
    return name


def resolve_device(args):
    """Map ``args["device"]`` to the ``(DTYPE, DEVICE)`` pair a run should use."""
    name = resolve_device_name(args.get("device"))
    return DEVICE_DTYPES[name], torch.device(name)


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
