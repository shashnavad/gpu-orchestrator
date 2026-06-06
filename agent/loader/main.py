"""
loader/main.py — Model Loader

The Python sidecar that owns all weight manipulation. The Rust agent is the
orchestrator; this process is the muscle.

MIG change: LoadRequest now carries slice_id and slice_vram_cap_mib.
When slice_vram_cap_mib is set, _assert_vram_headroom enforces the slice
budget rather than the whole-GPU free VRAM. This is what prevents a model
scheduled on a 2g.20gb slice from consuming memory outside its hardware-
isolated partition — the Python process itself never leaves the slice's
cgroup, but we add a soft guard here to fail fast before the OOM killer does.

Everything else is identical to the original.
"""

from __future__ import annotations

import gc
import logging
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from checkpoint import CheckpointManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [loader] %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

LOADER_PORT = int(os.environ.get("LOADER_PORT", "8001"))
HF_CACHE_DIR = os.environ.get("HF_CACHE_DIR", "/mnt/nvme/hf_cache")
CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", "/mnt/nvme/checkpoints")
VRAM_HEADROOM_MIB = int(os.environ.get("VRAM_HEADROOM_MIB", "2048"))


@dataclass
class LoadedModel:
    model_name: str
    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    vram_used_mib: int
    loaded_at: float = field(default_factory=time.time)
    quantization: Optional[str] = None
    slice_id: Optional[str] = None  # which MIG slice this model lives on


class ModelStore:
    def __init__(self) -> None:
        self._models: dict[str, LoadedModel] = {}

    def add(self, entry: LoadedModel) -> None:
        self._models[entry.model_name] = entry

    def remove(self, model_name: str) -> Optional[LoadedModel]:
        return self._models.pop(model_name, None)

    def get(self, model_name: str) -> Optional[LoadedModel]:
        return self._models.get(model_name)

    def names(self) -> list[str]:
        return list(self._models.keys())

    def total_vram_mib(self) -> int:
        return sum(m.vram_used_mib for m in self._models.values())

    def vram_used_on_slice(self, slice_id: str) -> int:
        """Sum of VRAM used by all models currently assigned to this MIG slice."""
        return sum(
            m.vram_used_mib for m in self._models.values()
            if m.slice_id == slice_id
        )


store = ModelStore()
checkpoint_manager = CheckpointManager(base_dir=CHECKPOINT_DIR)


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("loader ready — device=%s  cache=%s", _device(), HF_CACHE_DIR)
    yield
    log.info("loader shutting down, evicting all models from GPU")
    for name in list(store.names()):
        _do_evict(name)


app = FastAPI(title="gpu-orchestrator model loader", lifespan=lifespan)


class LoadRequest(BaseModel):
    model_name: str
    repo_id: Optional[str] = None
    quantization: Optional[str] = None
    # MIG fields — both must be present or both absent.
    slice_id: Optional[str] = None          # e.g. "0/0/0"
    slice_vram_cap_mib: Optional[int] = None # VRAM budget for this slice


class LoadResponse(BaseModel):
    model_name: str
    vram_used_mib: int
    load_duration_sec: float
    quantization: Optional[str]
    affinity_cache_hit: bool
    slice_id: Optional[str] = None


class EvictRequest(BaseModel):
    model_name: str


class CheckpointRequest(BaseModel):
    model_name: str
    checkpoint_dir: Optional[str] = None


class StatusResponse(BaseModel):
    loaded: list[str]
    vram_used_mib: int
    vram_total_mib: int
    vram_free_mib: int


@app.post("/load", response_model=LoadResponse)
def load_model(req: LoadRequest) -> LoadResponse:
    if store.get(req.model_name):
        entry = store.get(req.model_name)
        log.info("model %s already loaded, skipping", req.model_name)
        return LoadResponse(
            model_name=req.model_name,
            vram_used_mib=entry.vram_used_mib,
            load_duration_sec=0.0,
            quantization=entry.quantization,
            affinity_cache_hit=True,
            slice_id=entry.slice_id,
        )

    repo_id = req.repo_id or req.model_name

    # Guard: enforce slice budget if this is a MIG-targeted load.
    # When slice_vram_cap_mib is set, we check headroom within the slice;
    # otherwise we fall back to the whole-GPU guard.
    _assert_vram_headroom(
        model_name=req.model_name,
        slice_id=req.slice_id,
        slice_vram_cap_mib=req.slice_vram_cap_mib,
    )

    t0 = time.perf_counter()
    affinity_hit = _weights_cached_locally(repo_id)

    log.info(
        "loading %s (repo=%s, quant=%s, affinity_hit=%s, slice=%s)",
        req.model_name, repo_id, req.quantization, affinity_hit, req.slice_id,
    )

    try:
        model, tokenizer = _load_from_hf(repo_id, req.quantization)
    except Exception as exc:
        log.exception("failed to load %s", req.model_name)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    duration = time.perf_counter() - t0
    vram_mib = _measure_model_vram(model)

    entry = LoadedModel(
        model_name=req.model_name,
        model=model,
        tokenizer=tokenizer,
        vram_used_mib=vram_mib,
        quantization=req.quantization,
        slice_id=req.slice_id,
    )
    store.add(entry)

    log.info(
        "loaded %s in %.1fs — VRAM: %d MiB (slice=%s)",
        req.model_name, duration, vram_mib, req.slice_id,
    )

    return LoadResponse(
        model_name=req.model_name,
        vram_used_mib=vram_mib,
        load_duration_sec=round(duration, 2),
        quantization=req.quantization,
        affinity_cache_hit=affinity_hit,
        slice_id=req.slice_id,
    )


@app.post("/evict")
def evict_model(req: EvictRequest) -> dict:
    if not store.get(req.model_name):
        return {"status": "not_loaded", "model_name": req.model_name}
    freed = _do_evict(req.model_name)
    return {"status": "evicted", "model_name": req.model_name, "freed_mib": freed}


@app.post("/checkpoint")
def checkpoint_model(req: CheckpointRequest) -> dict:
    entry = store.get(req.model_name)
    if not entry:
        raise HTTPException(
            status_code=404,
            detail=f"model {req.model_name!r} not loaded — cannot checkpoint",
        )
    target_dir = req.checkpoint_dir or CHECKPOINT_DIR
    log.info("checkpointing %s → %s", req.model_name, target_dir)
    try:
        path = checkpoint_manager.save(
            model_name=req.model_name,
            model=entry.model,
            tokenizer=entry.tokenizer,
            checkpoint_dir=target_dir,
        )
    except Exception as exc:
        log.exception("checkpoint failed for %s", req.model_name)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"status": "checkpointed", "model_name": req.model_name, "path": str(path)}


@app.get("/status", response_model=StatusResponse)
def status() -> StatusResponse:
    total, free = _vram_total_free_mib()
    return StatusResponse(
        loaded=store.names(),
        vram_used_mib=store.total_vram_mib(),
        vram_total_mib=total,
        vram_free_mib=free,
    )


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_from_hf(
    repo_id: str,
    quantization: Optional[str],
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    device = _device()

    bnb_config = None
    torch_dtype = torch.bfloat16

    if quantization == "int8":
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        torch_dtype = None
    elif quantization == "int4":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        torch_dtype = None

    model = AutoModelForCausalLM.from_pretrained(
        repo_id,
        quantization_config=bnb_config,
        torch_dtype=torch_dtype,
        device_map="auto",
        cache_dir=HF_CACHE_DIR,
        trust_remote_code=False,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        repo_id,
        cache_dir=HF_CACHE_DIR,
        trust_remote_code=False,
    )
    return model, tokenizer


def _do_evict(model_name: str) -> int:
    entry = store.remove(model_name)
    if not entry:
        return 0
    freed = entry.vram_used_mib
    del entry.model
    del entry.tokenizer
    del entry
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    log.info("evicted %s — freed ~%d MiB", model_name, freed)
    return freed


def _weights_cached_locally(repo_id: str) -> bool:
    cache_key = "models--" + repo_id.replace("/", "--")
    cache_path = os.path.join(HF_CACHE_DIR, cache_key, "snapshots")
    return os.path.isdir(cache_path)


def _assert_vram_headroom(
    model_name: str,
    slice_id: Optional[str] = None,
    slice_vram_cap_mib: Optional[int] = None,
) -> None:
    """
    Refuse the load if there is insufficient VRAM headroom.

    For MIG-targeted loads (slice_id + slice_vram_cap_mib both set):
      free = slice_vram_cap - vram_already_used_on_that_slice
      This enforces the per-slice budget independently of other slices.

    For non-MIG loads:
      free = whole-GPU free VRAM (original behavior, unchanged).

    Both paths require at least VRAM_HEADROOM_MIB free to proceed.
    """
    if not torch.cuda.is_available():
        return  # CPU path — no VRAM limit

    if slice_id is not None and slice_vram_cap_mib is not None:
        # Slice-scoped guard: how much of this slice's budget is still free?
        used_on_slice = store.vram_used_on_slice(slice_id)
        free = slice_vram_cap_mib - used_on_slice
        context = f"slice {slice_id} (cap={slice_vram_cap_mib} MiB, used={used_on_slice} MiB)"
    else:
        _, free = _vram_total_free_mib()
        context = "full GPU"

    if free < VRAM_HEADROOM_MIB:
        raise HTTPException(
            status_code=507,
            detail=(
                f"insufficient VRAM to load {model_name!r} on {context}: "
                f"{free} MiB free, need at least {VRAM_HEADROOM_MIB} MiB headroom"
            ),
        )


def _measure_model_vram(model: AutoModelForCausalLM) -> int:
    if not torch.cuda.is_available():
        param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        return param_bytes // (1024 * 1024)
    allocated_bytes = torch.cuda.memory_allocated()
    return allocated_bytes // (1024 * 1024)


def _vram_total_free_mib() -> tuple[int, int]:
    if not torch.cuda.is_available():
        return 0, 0
    props = torch.cuda.get_device_properties(0)
    total = props.total_memory // (1024 * 1024)
    used = torch.cuda.memory_allocated() // (1024 * 1024)
    return total, total - used


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GPU Orchestrator Model Loader")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    serve_parser = subparsers.add_parser("serve", help="Run as HTTP server")
    serve_parser.add_argument("--port", type=int, default=LOADER_PORT)

    load_parser = subparsers.add_parser("load", help="One-shot model load")
    load_parser.add_argument("--model", required=True)
    load_parser.add_argument("--repo-id")
    load_parser.add_argument("--quantization", choices=["int8", "int4"], default=None)
    load_parser.add_argument("--slice-id", default=None)
    load_parser.add_argument("--slice-vram-cap-mib", type=int, default=None)

    args = parser.parse_args()

    if args.mode == "serve":
        uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")
    elif args.mode == "load":
        req = LoadRequest(
            model_name=args.model,
            repo_id=args.repo_id,
            quantization=args.quantization,
            slice_id=args.slice_id,
            slice_vram_cap_mib=args.slice_vram_cap_mib,
        )
        result = load_model(req)
        print(result.model_dump_json(indent=2))
