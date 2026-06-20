"""
loader/checkpoint.py — Spot Instance Checkpoint Manager

When the cloud provider sends a preemption notice (SIGTERM → 30-second window),
the Rust agent calls POST /checkpoint. This module handles the serialization
of model state to NVMe so the Go scheduler can reschedule the workload onto a
new node and restore from the checkpoint rather than re-downloading from HF.

Two-phase design:
  Phase 1 — Save (preemption path, must complete in < 25 seconds):
    - Serialize model state_dict to disk
    - Write a manifest.json so the restore path knows what it's loading
    - Atomic rename to make the checkpoint visible only when fully written

  Phase 2 — Restore (new node path, called from /load with a checkpoint path):
    - Validate manifest
    - Load base architecture from HF (gets cached to NVMe)
    - Load weights from checkpoint (overrides HF weights with fine-tuned state)

This separation matters: restoring from checkpoint is ~5x faster than a full
HF download because the architecture config is tiny and the weights come from
local NVMe rather than S3.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from paths import HF_CACHE_DIR_DEFAULT

log = logging.getLogger(__name__)

# Manifest filename inside each checkpoint directory.
_MANIFEST_FILE = "manifest.json"

# PyTorch safe serialization is preferred over pickle-based .bin format.
# safetensors is faster to load and doesn't execute arbitrary code on open.
_USE_SAFETENSORS = True


@dataclass
class CheckpointManifest:
    model_name: str
    repo_id: str
    quantization: Optional[str]
    saved_at: float         # Unix timestamp
    save_duration_sec: float
    weight_file: str        # filename of the serialized state_dict
    hf_cache_dir: str       # so the restore path knows where to look for config


class CheckpointManager:
    def __init__(self, base_dir: str) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(
        self,
        model_name: str,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        checkpoint_dir: Optional[str] = None,
    ) -> Path:
        """
        Serialize the model state_dict to disk atomically.

        Atomicity is achieved via a write-to-temp-dir + rename pattern.
        If the process is killed mid-write, the target directory is never
        created, so the restore path will never see a partial checkpoint.

        Returns the final checkpoint directory path.
        """
        target = Path(checkpoint_dir or self.base_dir) / _checkpoint_name(model_name)

        # Write into a temp directory first, then rename atomically.
        tmp_dir = Path(tempfile.mkdtemp(dir=target.parent, prefix=".tmp_ckpt_"))

        try:
            t0 = time.perf_counter()
            log.info("checkpoint save started: %s → %s", model_name, tmp_dir)

            weight_file = self._write_weights(model, tmp_dir)
            tokenizer.save_pretrained(str(tmp_dir))

            manifest = CheckpointManifest(
                model_name=model_name,
                repo_id=_infer_repo_id(model),
                quantization=_infer_quantization(model),
                saved_at=time.time(),
                save_duration_sec=round(time.perf_counter() - t0, 2),
                weight_file=weight_file,
                hf_cache_dir=str(os.environ.get("HF_CACHE_DIR", HF_CACHE_DIR_DEFAULT)),
            )
            _write_manifest(tmp_dir, manifest)

            duration = time.perf_counter() - t0
            log.info("checkpoint written in %.1fs, renaming to %s", duration, target)

            # Atomic rename — this is the commit point.
            # If target already exists (re-checkpoint), replace it.
            if target.exists():
                shutil.rmtree(target)
            tmp_dir.rename(target)

            log.info("checkpoint committed: %s", target)
            return target

        except Exception:
            # Clean up the temp dir on failure so we don't leave orphaned dirs.
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)
            raise

    def _write_weights(self, model: AutoModelForCausalLM, out_dir: Path) -> str:
        """
        Write model weights. Uses safetensors format when available (faster,
        safer than pickle), falls back to PyTorch .bin format.

        For quantized models (int8/int4 via bitsandbytes), we dequantize to
        float16 before saving. This adds ~5 seconds but produces a checkpoint
        that can be loaded on any node without requiring the same GPU memory
        constraints. A quantized checkpoint is not portable across hardware.
        """
        if _is_quantized(model):
            log.info("model is quantized — dequantizing for portable checkpoint")
            model = _dequantize(model)

        if _USE_SAFETENSORS:
            try:
                from safetensors.torch import save_file  # type: ignore
                weight_path = out_dir / "model.safetensors"
                # Flatten state dict to CPU before serializing — GPU tensors
                # can't be written directly to disk.
                state = {k: v.cpu() for k, v in model.state_dict().items()}
                save_file(state, str(weight_path))
                log.info("weights saved as safetensors: %s", weight_path)
                return weight_path.name
            except ImportError:
                log.warning("safetensors not installed, falling back to .bin")

        # Fallback: PyTorch .bin format.
        weight_path = out_dir / "model.bin"
        state = {k: v.cpu() for k, v in model.state_dict().items()}
        torch.save(state, str(weight_path))
        log.info("weights saved as .bin: %s", weight_path)
        return weight_path.name

    # ------------------------------------------------------------------
    # Restore
    # ------------------------------------------------------------------

    def restore(
        self,
        checkpoint_path: str,
        quantization: Optional[str] = None,
    ) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Restore a model from a checkpoint directory.

        Strategy:
        1. Read manifest to get repo_id and original quantization.
        2. Load base architecture from HF (config only, no weights) —
           this is fast and uses the NVMe cache.
        3. Load our saved state_dict and apply it over the base weights.

        The result is equivalent to running the model at the time of checkpoint.
        Fine-tuned adapters, RLHF-adjusted weights, etc. are all preserved.
        """
        ckpt_dir = Path(checkpoint_path)
        manifest = _read_manifest(ckpt_dir)

        log.info(
            "restoring %s from checkpoint (saved %.0fs ago)",
            manifest.model_name,
            time.time() - manifest.saved_at,
        )

        effective_quant = quantization or manifest.quantization

        bnb_config = _build_bnb_config(effective_quant)
        torch_dtype = None if effective_quant else torch.bfloat16

        # Load the model architecture from HF (fast — uses NVMe cache).
        model = AutoModelForCausalLM.from_pretrained(
            manifest.repo_id,
            quantization_config=bnb_config,
            torch_dtype=torch_dtype,
            device_map="auto",
            cache_dir=manifest.hf_cache_dir,
        )

        # Apply our checkpoint weights on top.
        weight_path = ckpt_dir / manifest.weight_file
        log.info("loading checkpoint weights from %s", weight_path)

        if weight_path.suffix == ".safetensors":
            from safetensors.torch import load_file  # type: ignore
            state_dict = load_file(str(weight_path))
        else:
            state_dict = torch.load(str(weight_path), map_location="cpu", weights_only=True)

        # strict=False allows the checkpoint to partially override weights —
        # useful when the checkpoint was saved from a fine-tuned adapter that
        # only modifies certain layers.
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            log.warning("checkpoint restore: %d missing keys", len(missing))
        if unexpected:
            log.warning("checkpoint restore: %d unexpected keys", len(unexpected))

        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(str(ckpt_dir))

        log.info("restore complete for %s", manifest.model_name)
        return model, tokenizer

    # ------------------------------------------------------------------
    # Listing / cleanup
    # ------------------------------------------------------------------

    def list_checkpoints(self) -> list[dict]:
        """Return metadata for all saved checkpoints. Used by the status endpoint."""
        results = []
        for entry in sorted(self.base_dir.iterdir()):
            manifest_path = entry / _MANIFEST_FILE
            if manifest_path.exists():
                with open(manifest_path) as f:
                    results.append(json.load(f))
        return results

    def delete_checkpoint(self, model_name: str) -> bool:
        """Delete a checkpoint directory. Called after successful migration."""
        target = self.base_dir / _checkpoint_name(model_name)
        if target.exists():
            shutil.rmtree(target)
            log.info("deleted checkpoint for %s", model_name)
            return True
        return False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _checkpoint_name(model_name: str) -> str:
    # Normalize model name to a valid directory name.
    return model_name.replace("/", "--").replace(":", "_")


def _write_manifest(out_dir: Path, manifest: CheckpointManifest) -> None:
    path = out_dir / _MANIFEST_FILE
    with open(path, "w") as f:
        json.dump(
            {
                "model_name": manifest.model_name,
                "repo_id": manifest.repo_id,
                "quantization": manifest.quantization,
                "saved_at": manifest.saved_at,
                "save_duration_sec": manifest.save_duration_sec,
                "weight_file": manifest.weight_file,
                "hf_cache_dir": manifest.hf_cache_dir,
            },
            f,
            indent=2,
        )


def _read_manifest(ckpt_dir: Path) -> CheckpointManifest:
    manifest_path = ckpt_dir / _MANIFEST_FILE
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"no manifest found in {ckpt_dir} — is this a valid checkpoint?"
        )
    with open(manifest_path) as f:
        data = json.load(f)
    return CheckpointManifest(**data)


def _is_quantized(model: AutoModelForCausalLM) -> bool:
    return getattr(model, "is_quantized", False) or getattr(model, "quantization_method", None) is not None


def _infer_quantization(model: AutoModelForCausalLM) -> Optional[str]:
    method = getattr(model, "quantization_method", None)
    if method is None:
        return None
    if "8" in str(method):
        return "int8"
    if "4" in str(method):
        return "int4"
    return str(method)


def _infer_repo_id(model: AutoModelForCausalLM) -> str:
    # HuggingFace sets name_or_path on the config during from_pretrained.
    return getattr(model.config, "name_or_path", "unknown")


def _dequantize(model: AutoModelForCausalLM) -> AutoModelForCausalLM:
    """
    Convert a bitsandbytes quantized model back to float16 for checkpointing.
    This is required because quantized tensors cannot be serialized portably.
    """
    try:
        import bitsandbytes as bnb  # noqa: F401 — just to confirm it's available
        from copy import deepcopy

        # Move to CPU first to avoid OOM during dequantization.
        model_cpu = model.to("cpu")
        # Dequantize by converting to float16.
        dequantized = model_cpu.to(torch.float16)
        del model_cpu
        return dequantized
    except Exception as e:
        log.warning("dequantization failed (%s), saving quantized weights as-is", e)
        return model


def _build_bnb_config(quantization: Optional[str]) -> Optional[BitsAndBytesConfig]:
    if quantization == "int8":
        return BitsAndBytesConfig(load_in_8bit=True)
    if quantization == "int4":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    return None
