"""
loader/checkpoint.py — Spot Instance Restart Manifest

This got simpler when the loader moved to vLLM. The old version serialized
the full model state_dict to NVMe on a preemption notice, so a replacement
node could restore fine-tuned weights without a full HF download. Under
vLLM that approach doesn't hold up:

  - vLLM shards and re-packs weights into its own internal layout
    (tensor-parallel workers, fused QKV projections, etc). There is no
    single clean state_dict to torch.save() the way there was with a plain
    `AutoModelForCausalLM` instance.
  - We aren't fine-tuning weights in place — every model here is served
    read-only from its HF checkpoint (base, AWQ, or GPTQ). There is nothing
    to preserve except which model was running.
  - The existing weight-affinity mechanism (see agent/src/main.rs
    model_weight_affinity and the bin-packer's affinity heuristic) already
    gets most of the win a checkpoint restore was chasing: the scheduler
    prefers a node that already has the repo's weights on local NVMe cache,
    so re-launching a vLLM engine there skips the HF download and only pays
    engine warmup.

So /checkpoint now just persists a small manifest — enough for a
replacement node to relaunch the same engine (repo_id, quantization, slice)
without re-deriving it from scratch — and skips weight serialization
entirely. This is an honest scope-down, not a stand-in for the old
behavior: if you need to preserve fine-tuned weights that only exist in
GPU memory, this file will not do that for you.
"""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

_MANIFEST_FILE = "manifest.json"


@dataclass
class RestartManifest:
    model_name: str
    repo_id: str
    quantization: Optional[str]
    slice_id: Optional[str]
    gpu_memory_utilization: float
    max_model_len: Optional[int]
    dtype: str
    saved_at: float


class CheckpointManager:
    """Persists/reads RestartManifests. No model weights touch this class."""

    def __init__(self, base_dir: str) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save(self, manifest: RestartManifest) -> Path:
        """
        Write the manifest atomically (temp dir + rename) so a reader never
        sees a half-written file if the process is killed mid-write — same
        commit pattern the old weight-serializing version used, just applied
        to a few hundred bytes of JSON instead of gigabytes of tensors.
        """
        target = self.base_dir / _manifest_name(manifest.model_name)
        tmp_dir = Path(tempfile.mkdtemp(dir=self.base_dir, prefix=".tmp_ckpt_"))
        try:
            with open(tmp_dir / _MANIFEST_FILE, "w") as f:
                json.dump(asdict(manifest), f, indent=2)
            if target.exists():
                shutil.rmtree(target)
            tmp_dir.rename(target)
            log.info("restart manifest committed: %s -> %s", manifest.model_name, target)
            return target
        except Exception:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise

    def restore(self, model_name: str) -> RestartManifest:
        target = self.base_dir / _manifest_name(model_name)
        manifest_path = target / _MANIFEST_FILE
        if not manifest_path.exists():
            raise FileNotFoundError(f"no restart manifest for {model_name!r} in {self.base_dir}")
        with open(manifest_path) as f:
            data = json.load(f)
        return RestartManifest(**data)

    def list_manifests(self) -> list[dict]:
        results = []
        for entry in sorted(self.base_dir.iterdir()):
            manifest_path = entry / _MANIFEST_FILE
            if manifest_path.exists():
                with open(manifest_path) as f:
                    results.append(json.load(f))
        return results

    def delete(self, model_name: str) -> bool:
        target = self.base_dir / _manifest_name(model_name)
        if target.exists():
            shutil.rmtree(target)
            return True
        return False


def _manifest_name(model_name: str) -> str:
    return model_name.replace("/", "--").replace(":", "_")
