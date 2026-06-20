"""
paths.py — cross-platform default cache/checkpoint locations.

Production targets a dedicated NVMe mount at /mnt/nvme (fast local disk on
the GPU node — see checkpoint.py for why that matters on spot preemption).
That path is Linux-specific and doesn't exist, or isn't writable, on macOS
or Windows dev machines. Resolve a real default here so the loader doesn't
crash on those platforms, while leaving production behavior unchanged
wherever /mnt/nvme is actually available.

Always set HF_CACHE_DIR / CHECKPOINT_DIR explicitly wherever checkpoint
persistence across node restarts actually matters — this fallback is for
local development convenience only.
"""

import os
from pathlib import Path

_NVME_ROOT = Path("/mnt/nvme")


def default_cache_root() -> Path:
    if _NVME_ROOT.parent.exists() and os.access(_NVME_ROOT.parent, os.W_OK):
        return _NVME_ROOT
    return Path.home() / ".cache" / "gpu-orchestrator"


HF_CACHE_DIR_DEFAULT = str(default_cache_root() / "hf_cache")
CHECKPOINT_DIR_DEFAULT = str(default_cache_root() / "checkpoints")