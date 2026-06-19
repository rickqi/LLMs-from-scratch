"""Device selection helper for torch-directml (AMD GPU via DirectML on Windows).

DirectML is the practical way to train PyTorch models on AMD Radeon GPUs inside
WSL2 / native Windows when ROCm is unavailable (e.g. the GPU is not on the ROCm
WSL support matrix). It routes compute through DirectX 12, so it works on any
DX12 GPU regardless of whether ROCm officially supports that SKU.

Picks a discrete AMD Radeon adapter when present, falls back to the first
available DirectML device otherwise.

Usage:
    from dml_device import pick_device
    dev, name = pick_device()
    model = MyModel().to(dev)

Run directly to list adapters:
    python dml_device.py
"""
from __future__ import annotations

import torch_directml

# Preference order: most powerful first. Substring match, case-insensitive.
# Discrete RX cards rank above integrated "M Graphics" iGPUs.
PREFERRED_KEYWORDS = (
    "7900", "7800", "7700", "7600", "7500",
    "RX 9", "RX 7", "RX", "Radeon RX", "Radeon Pro",
    "780M", "760M",
)


def list_devices() -> list[tuple[int, str]]:
    n = torch_directml.device_count()
    return [(i, torch_directml.device_name(i)) for i in range(n)]


def pick_device(verbose: bool = True):
    """Return (torch.device, name) for the preferred AMD GPU.

    Raises RuntimeError if no DirectML adapter is available.
    """
    devices = list_devices()
    if not devices:
        raise RuntimeError("No DirectML device found. torch-directml sees 0 adapters.")

    chosen_idx = None
    for kw in PREFERRED_KEYWORDS:
        for idx, name in devices:
            if kw.lower() in name.lower():
                chosen_idx = idx
                break
        if chosen_idx is not None:
            break

    if chosen_idx is None:
        chosen_idx = devices[0][0]

    chosen_name = torch_directml.device_name(chosen_idx)
    dev = torch_directml.device(chosen_idx)

    if verbose:
        print("[dml_device] DirectML adapters:")
        for idx, name in devices:
            mark = "  <-- selected" if idx == chosen_idx else ""
            print(f"  [{idx}] {name}{mark}")
        print(f"[dml_device] picked: {chosen_name}  ({dev})")
    return dev, chosen_name


if __name__ == "__main__":
    pick_device()
