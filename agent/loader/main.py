"""
loader/main.py — Model Loader

The Python sidecar that owns all weight manipulation. The Rust agent is the
orchestrator; this process is the muscle. It runs as a long-lived HTTP server
so the Rust agent can call it without subprocess spawn overhead on the hot path.

Contract with the Rust agent (src/main.rs):
  POST /load      { "model_name": str, "quantization": str|null }
  POST /evict     { "model_name": str }
  POST /checkpoint { "model_name": str, "checkpoint_dir": str }
  GET  /status    → { "loaded": [...], "vram_used_mib": int }

Quantization options: "int8", "int4", None (full precision).
Using bitsandbytes for quantization — it runs on the same GPU without
a separate compilation step.
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


# ---------------------------------------------------------------------------
# Config — driven by environment variables so the Rust agent can set them
# without rebuilding the image.
# ---------------------------------------------------------------------------

LOADER_PORT = int(os.environ.get("LOADER_PORT", "8001"))
HF_CACHE_DIR = os.environ.get("HF_CACHE_DIR", "/mnt/nvme/hf_cache")
CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", "/mnt/nvme/checkpoints")
# VRAM headroom to keep free (MiB). Loader refuses loads that would breach this.
VRAM_HEADROOM_MIB = int(os.environ.get("VRAM_HEADROOM_MIB", "2048"))


# ---------------------------------------------------------------------------
# Internal state
# ---------------------------------------------------------------------------

@dataclass
class LoadedModel:
    model_name: str
    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    vram_used_mib: int
    loaded_at: float = field(default_factory=time.time)
    quantization: Optional[str] = None  # "int8" | "int4" | None


class ModelStore:
    """Thread-safe registry of models currently loaded in GPU memory."""

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


store = ModelStore()
checkpoint_manager = CheckpointManager(base_dir=CHECKPOINT_DIR)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("loader ready — device=%s  cache=%s", _device(), HF_CACHE_DIR)
    yield
    log.info("loader shutting down, evicting all models from GPU")
    for name in list(store.names()):
        _do_evict(name)


app = FastAPI(title="gpu-orchestrator model loader", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class LoadRequest(BaseModel):
    model_name: str
    # HuggingFace repo ID, e.g. "microsoft/Phi-3-mini-4k-instruct"
    # If None, model_name is used directly as the repo ID.
    repo_id: Optional[str] = None
    quantization: Optional[str] = None  # "int8" | "int4" | None


class LoadResponse(BaseModel):
    model_name: str
    vram_used_mib: int
    load_duration_sec: float
    quantization: Optional[str]
    affinity_cache_hit: bool   # True if weights were already on NVMe


class EvictRequest(BaseModel):
    model_name: str


class CheckpointRequest(BaseModel):
    model_name: str
    checkpoint_dir: Optional[str] = None  # overrides CHECKPOINT_DIR env var


class StatusResponse(BaseModel):
    loaded: list[str]
    vram_used_mib: int
    vram_total_mib: int
    vram_free_mib: int


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post("/load", response_model=LoadResponse)
def load_model(req: LoadRequest) -> LoadResponse:
    """
    Load a model's weights into GPU memory.

    Weight affinity logic: HuggingFace's snapshot_download caches weights to
    HF_CACHE_DIR (on local NVMe). If the weights are already there, from_pretrained
    skips the network entirely — this is the affinity cache hit that the Go
    scheduler's ModelCache tracks.

    Quantization: int8 and int4 use bitsandbytes. Both cut VRAM roughly in half
    (int8) or to a quarter (int4) at the cost of some precision. The Rust agent
    passes through the quantization field from the Go scheduler's ScheduleRequest.
    """
    if store.get(req.model_name):
        entry = store.get(req.model_name)
        log.info("model %s already loaded, skipping", req.model_name)
        return LoadResponse(
            model_name=req.model_name,
            vram_used_mib=entry.vram_used_mib,
            load_duration_sec=0.0,
            quantization=entry.quantization,
            affinity_cache_hit=True,
        )

    repo_id = req.repo_id or req.model_name
    _assert_vram_headroom(req.model_name)

    t0 = time.perf_counter()
    affinity_hit = _weights_cached_locally(repo_id)

    log.info(
        "loading %s (repo=%s, quant=%s, affinity_hit=%s)",
        req.model_name, repo_id, req.quantization, affinity_hit,
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
    )
    store.add(entry)

    log.info(
        "loaded %s in %.1fs — VRAM: %d MiB",
        req.model_name, duration, vram_mib,
    )

    return LoadResponse(
        model_name=req.model_name,
        vram_used_mib=vram_mib,
        load_duration_sec=round(duration, 2),
        quantization=req.quantization,
        affinity_cache_hit=affinity_hit,
    )


@app.post("/evict")
def evict_model(req: EvictRequest) -> dict:
    """
    Evict a model from GPU memory but leave weights on NVMe.

    Critically, we do NOT delete the HuggingFace cache directory. Weights
    staying on NVMe is the affinity that the Go scheduler uses to prefer this
    node for future placements of the same model.
    """
    if not store.get(req.model_name):
        return {"status": "not_loaded", "model_name": req.model_name}

    freed = _do_evict(req.model_name)
    return {"status": "evicted", "model_name": req.model_name, "freed_mib": freed}


@app.post("/checkpoint")
def checkpoint_model(req: CheckpointRequest) -> dict:
    """
    Checkpoint a model's state to disk for spot instance preemption recovery.
    Delegates to checkpoint.py which handles the PyTorch state_dict serialization.
    """
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
    """
    Returns current loader state. The Rust agent polls this to update the
    model_weight_affinity field in its heartbeat payload to Go.
    """
    total, free = _vram_total_free_mib()
    return StatusResponse(
        loaded=store.names(),
        vram_used_mib=store.total_vram_mib(),
        vram_total_mib=total,
        vram_free_mib=free,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_from_hf(
    repo_id: str,
    quantization: Optional[str],
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load weights from HuggingFace (or local NVMe cache if already downloaded).

    Quantization is handled via BitsAndBytesConfig:
    - int8:  ~2x VRAM reduction, minimal accuracy loss — good for inference
    - int4:  ~4x VRAM reduction, small accuracy hit — good for bin-packing
             multiple small models on one GPU (MIG slices)
    - None:  full precision (bfloat16 on H100) — best for P0 production routes
    """
    device = _device()

    bnb_config = None
    torch_dtype = torch.bfloat16  # H100 native dtype

    if quantization == "int8":
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        torch_dtype = None  # bitsandbytes manages dtype internally

    elif quantization == "int4":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",          # NormalFloat4 — best quality/compression
            bnb_4bit_use_double_quant=True,      # nested quantization for extra MiB savings
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        torch_dtype = None

    model = AutoModelForCausalLM.from_pretrained(
        repo_id,
        quantization_config=bnb_config,
        torch_dtype=torch_dtype,
        device_map="auto",          # lets HF decide multi-GPU tensor placement
        cache_dir=HF_CACHE_DIR,     # NVMe cache — this is the affinity source
        trust_remote_code=False,
    )
    model.eval()  # inference mode — disables dropout, batch norm tracking

    tokenizer = AutoTokenizer.from_pretrained(
        repo_id,
        cache_dir=HF_CACHE_DIR,
        trust_remote_code=False,
    )

    return model, tokenizer


def _do_evict(model_name: str) -> int:
    """
    Remove a model from GPU memory. Returns MiB freed.

    The three-step eviction is critical:
    1. Delete the Python reference (model object)
    2. Empty the CUDA cache (PyTorch holds a memory pool even after del)
    3. Call gc.collect() to catch any cyclic references

    Skipping step 2 means the VRAM shows as freed in Python but the CUDA
    driver still reports it as allocated — the Rust agent would send a wrong
    used_vram_mib in its heartbeat.
    """
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
    """
    Check if HuggingFace has already downloaded this model to NVMe.
    HF stores weights under {cache_dir}/models--{org}--{model} using
    a snapshots directory structure.
    """
    # HF transforms "microsoft/Phi-3-mini-4k-instruct" → "models--microsoft--Phi-3-mini-4k-instruct"
    cache_key = "models--" + repo_id.replace("/", "--")
    cache_path = os.path.join(HF_CACHE_DIR, cache_key, "snapshots")
    return os.path.isdir(cache_path)


def _assert_vram_headroom(model_name: str) -> None:
    """
    Refuse the load if we don't have enough free VRAM headroom.
    This is a safety net — the Go scheduler already accounts for VRAM, but
    the loader adds a local guard to prevent OOM crashes.
    """
    if not torch.cuda.is_available():
        return  # CPU path — no VRAM limit
    _, free = _vram_total_free_mib()
    if free < VRAM_HEADROOM_MIB:
        raise HTTPException(
            status_code=507,
            detail=(
                f"insufficient VRAM to load {model_name!r}: "
                f"{free} MiB free, need at least {VRAM_HEADROOM_MIB} MiB headroom"
            ),
        )


def _measure_model_vram(model: AutoModelForCausalLM) -> int:
    """
    Measure actual VRAM consumed by a model after loading.
    Uses PyTorch's memory_allocated rather than memory_reserved — reserved
    includes the cache pool which isn't useful for scheduling decisions.
    """
    if not torch.cuda.is_available():
        # Estimate from parameter count — rough but useful for CPU dev.
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


# ---------------------------------------------------------------------------
# CLI entrypoint — used by the Rust agent subprocess call and direct invocation
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GPU Orchestrator Model Loader")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # Server mode: long-lived HTTP server (production path)
    serve_parser = subparsers.add_parser("serve", help="Run as HTTP server")
    serve_parser.add_argument("--port", type=int, default=LOADER_PORT)

    # One-shot mode: load a single model and exit (called by Rust subprocess stub)
    load_parser = subparsers.add_parser("load", help="One-shot model load")
    load_parser.add_argument("--model", required=True)
    load_parser.add_argument("--repo-id")
    load_parser.add_argument("--quantization", choices=["int8", "int4"], default=None)

    args = parser.parse_args()

    if args.mode == "serve":
        uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")

    elif args.mode == "load":
        # One-shot: used by the Rust agent's subprocess call in prewarm_model()
        # when the loader HTTP server isn't running yet.
        import asyncio
        req = LoadRequest(
            model_name=args.model,
            repo_id=args.repo_id,
            quantization=args.quantization,
        )
        result = load_model(req)
        print(result.model_dump_json(indent=2))
