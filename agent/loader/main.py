"""
loader/main.py — Model Loader / vLLM Serving Sidecar

The Rust agent is the orchestrator; this process is the muscle — except the
muscle is now vLLM's engine, not a hand-rolled transformers.generate() loop.
The Go scheduler decides *where* a model runs (bin-packing, MIG slice,
eviction at 85% VRAM); this process is what actually stands up something
capable of serving it and answering real generate requests.

MIG: LoadRequest carries slice_id and slice_vram_cap_mib. When present, the
requested vLLM engine's gpu_memory_utilization is scaled to that slice's
share of the physical card instead of vLLM's own whole-device default, and
_assert_vram_headroom enforces the slice budget (soft guard on top of the
hardware isolation MIG itself provides) before we let vLLM try to allocate.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from checkpoint import CheckpointManager, RestartManifest
from paths import CHECKPOINT_DIR_DEFAULT, HF_CACHE_DIR_DEFAULT
from vllm_engine import VLLM_MOCK, EngineHandle, EngineLoadError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [loader] %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

LOADER_PORT = int(os.environ.get("LOADER_PORT", "8001"))
HF_CACHE_DIR = os.environ.get("HF_CACHE_DIR", HF_CACHE_DIR_DEFAULT)
CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", CHECKPOINT_DIR_DEFAULT)
VRAM_HEADROOM_MIB = int(os.environ.get("VRAM_HEADROOM_MIB", "2048"))
DEFAULT_GPU_MEMORY_UTILIZATION = float(os.environ.get("DEFAULT_GPU_MEMORY_UTILIZATION", "0.85"))
DEFAULT_DTYPE = os.environ.get("VLLM_DTYPE", "bfloat16")

# Model loading is expensive enough (weight download + vLLM engine init +
# CUDA graph/kernel warmup) that we run it off the event loop, but not so
# concurrent that two loads should ever race for the same GPU's VRAM.
import anyio


@dataclass
class Entry:
    model_name: str
    engine: EngineHandle
    slice_id: Optional[str] = None
    loaded_at: float = field(default_factory=time.time)


class ModelStore:
    def __init__(self) -> None:
        self._models: dict[str, Entry] = {}

    def add(self, entry: Entry) -> None:
        self._models[entry.model_name] = entry

    def remove(self, model_name: str) -> Optional[Entry]:
        return self._models.pop(model_name, None)

    def get(self, model_name: str) -> Optional[Entry]:
        return self._models.get(model_name)

    def names(self) -> list[str]:
        return list(self._models.keys())

    def total_vram_mib(self) -> int:
        return sum(e.engine.vram_used_mib for e in self._models.values())

    def vram_used_on_slice(self, slice_id: str) -> int:
        return sum(
            e.engine.vram_used_mib for e in self._models.values()
            if e.slice_id == slice_id
        )


store = ModelStore()
checkpoint_manager = CheckpointManager(base_dir=CHECKPOINT_DIR)
_load_lock = anyio.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info(
        "loader ready — backend=%s cache=%s",
        "VLLM_MOCK (synthetic)" if VLLM_MOCK else "vllm",
        HF_CACHE_DIR,
    )
    yield
    log.info("loader shutting down, evicting all engines")
    for name in list(store.names()):
        _do_evict(name)


app = FastAPI(title="gpu-orchestrator vLLM loader", lifespan=lifespan)


class LoadRequest(BaseModel):
    model_name: str
    repo_id: Optional[str] = None
    quantization: Optional[str] = None  # vLLM-native: None | "awq" | "gptq" | "fp8" | "squeezellm"
    max_model_len: Optional[int] = None
    dtype: Optional[str] = None
    # MIG fields — both must be present or both absent.
    slice_id: Optional[str] = None
    slice_vram_cap_mib: Optional[int] = None


class LoadResponse(BaseModel):
    model_name: str
    vram_used_mib: int
    load_duration_sec: float
    quantization: Optional[str]
    backend: str
    slice_id: Optional[str] = None
    # Kept for the Rust agent's LoadResponse struct, which deserializes this
    # as a required field — see agent/src/main.rs. True when repo_id was
    # already present in HF_CACHE_DIR before this call (skips the download,
    # not the vLLM engine warmup).
    affinity_cache_hit: bool = False


class EvictRequest(BaseModel):
    model_name: str


class GenerateRequest(BaseModel):
    model_name: str
    prompt: str
    max_tokens: int = 128
    temperature: float = 0.7


class GenerateResponse(BaseModel):
    model_name: str
    text: str
    prompt_tokens: int
    completion_tokens: int
    tokens_per_sec: float
    backend: str


class CheckpointRequest(BaseModel):
    model_name: str


class StatusResponse(BaseModel):
    loaded: list[str]
    vram_used_mib: int
    vram_total_mib: int
    vram_free_mib: int
    backend: str


@app.post("/load", response_model=LoadResponse)
async def load_model(req: LoadRequest) -> LoadResponse:
    existing = store.get(req.model_name)
    if existing:
        log.info("model %s already has a live engine, skipping", req.model_name)
        return LoadResponse(
            model_name=req.model_name,
            vram_used_mib=existing.engine.vram_used_mib,
            load_duration_sec=0.0,
            quantization=existing.engine.quantization,
            backend="vllm-mock" if VLLM_MOCK else "vllm",
            slice_id=existing.slice_id,
            affinity_cache_hit=True,
        )

    repo_id = req.repo_id or req.model_name
    affinity_hit = _weights_cached_locally(repo_id)

    _assert_vram_headroom(
        model_name=req.model_name,
        slice_id=req.slice_id,
        slice_vram_cap_mib=req.slice_vram_cap_mib,
    )

    gpu_memory_utilization = _resolve_gpu_memory_utilization(req.slice_id, req.slice_vram_cap_mib)
    dtype = req.dtype or DEFAULT_DTYPE

    log.info(
        "loading %s via vLLM (repo=%s quant=%s slice=%s gpu_mem_util=%.2f)",
        req.model_name, repo_id, req.quantization, req.slice_id, gpu_memory_utilization,
    )

    engine = EngineHandle(
        model_name=req.model_name,
        repo_id=repo_id,
        quantization=req.quantization,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=req.max_model_len,
        dtype=dtype,
    )

    # Serialize engine init: two concurrent vLLM engine constructions racing
    # for the same physical GPU's free VRAM is exactly the failure mode
    # _assert_vram_headroom is trying to prevent, and the check-then-load is
    # only safe if "load" can't interleave with another "load".
    async with _load_lock:
        try:
            await anyio.to_thread.run_sync(engine.load)
        except EngineLoadError as exc:
            log.exception("vLLM engine failed to load %s", req.model_name)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    store.add(Entry(model_name=req.model_name, engine=engine, slice_id=req.slice_id))

    log.info(
        "loaded %s in %.1fs — VRAM: %d MiB (slice=%s)",
        req.model_name, engine.load_duration_sec, engine.vram_used_mib, req.slice_id,
    )

    return LoadResponse(
        model_name=req.model_name,
        vram_used_mib=engine.vram_used_mib,
        load_duration_sec=round(engine.load_duration_sec, 2),
        quantization=req.quantization,
        backend="vllm-mock" if VLLM_MOCK else "vllm",
        slice_id=req.slice_id,
        affinity_cache_hit=affinity_hit,
    )


def _weights_cached_locally(repo_id: str) -> bool:
    if VLLM_MOCK:
        return False
    cache_key = "models--" + repo_id.replace("/", "--")
    cache_path = os.path.join(HF_CACHE_DIR, cache_key, "snapshots")
    return os.path.isdir(cache_path)


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest) -> GenerateResponse:
    entry = store.get(req.model_name)
    if not entry:
        raise HTTPException(
            status_code=404,
            detail=f"model {req.model_name!r} has no live engine — call /load first",
        )
    try:
        result = await anyio.to_thread.run_sync(
            lambda: entry.engine.generate(req.prompt, req.max_tokens, req.temperature)
        )
    except EngineLoadError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return GenerateResponse(
        model_name=req.model_name,
        text=result.text,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        tokens_per_sec=round(result.tokens_per_sec, 1),
        backend="vllm-mock" if VLLM_MOCK else "vllm",
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
            detail=f"model {req.model_name!r} not loaded — cannot write restart manifest",
        )
    manifest = RestartManifest(
        model_name=entry.model_name,
        repo_id=entry.engine.repo_id,
        quantization=entry.engine.quantization,
        slice_id=entry.slice_id,
        gpu_memory_utilization=entry.engine.gpu_memory_utilization,
        max_model_len=entry.engine.max_model_len,
        dtype=entry.engine.dtype,
        saved_at=time.time(),
    )
    path = checkpoint_manager.save(manifest)
    log.info("restart manifest written for %s -> %s", req.model_name, path)
    return {"status": "checkpointed", "model_name": req.model_name, "path": str(path)}


@app.get("/status", response_model=StatusResponse)
def status() -> StatusResponse:
    total, free = _vram_total_free_mib()
    return StatusResponse(
        loaded=store.names(),
        vram_used_mib=store.total_vram_mib(),
        vram_total_mib=total,
        vram_free_mib=free,
        backend="vllm-mock" if VLLM_MOCK else "vllm",
    )


def _do_evict(model_name: str) -> int:
    entry = store.remove(model_name)
    if not entry:
        return 0
    freed = entry.engine.vram_used_mib
    entry.engine.unload()
    log.info("evicted %s — freed ~%d MiB", model_name, freed)
    return freed


def _resolve_gpu_memory_utilization(
    slice_id: Optional[str], slice_vram_cap_mib: Optional[int]
) -> float:
    """
    vLLM's gpu_memory_utilization is a fraction of the *visible* device's
    total memory. Under real MIG, the slice is its own CUDA device with its
    own total, so 0.85 there already means 85% of the slice — no scaling
    needed. In the demo (single simulated card, no real MIG device split),
    we approximate the slice's share of the whole card so mock/dev runs
    still produce believable numbers when someone reads the logs.
    """
    if slice_id is None or slice_vram_cap_mib is None:
        return DEFAULT_GPU_MEMORY_UTILIZATION

    total, _ = _vram_total_free_mib()
    if total <= 0:
        return DEFAULT_GPU_MEMORY_UTILIZATION

    fraction = slice_vram_cap_mib / total
    # Leave a little headroom inside the slice itself for vLLM's own
    # activation/KV-cache bookkeeping; don't hand it the slice's full cap.
    return max(0.05, min(fraction * 0.9, 0.95))


def _assert_vram_headroom(
    model_name: str,
    slice_id: Optional[str] = None,
    slice_vram_cap_mib: Optional[int] = None,
) -> None:
    """
    Refuse the load if there is insufficient VRAM headroom, before we ever
    ask vLLM to allocate anything.

    MIG-targeted loads (slice_id + slice_vram_cap_mib both set):
      free = slice_vram_cap - vram_already_used_on_that_slice
    Non-MIG loads:
      free = whole-GPU free VRAM
    """
    if VLLM_MOCK:
        return  # mock mode never touches real VRAM

    try:
        import torch
    except ImportError:
        return
    if not torch.cuda.is_available():
        return

    if slice_id is not None and slice_vram_cap_mib is not None:
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


def _vram_total_free_mib() -> tuple[int, int]:
    if VLLM_MOCK:
        return 81_920, 81_920 - store.total_vram_mib()  # matches node-002's simulated H100
    try:
        import torch
    except ImportError:
        return 0, 0
    if not torch.cuda.is_available():
        return 0, 0
    props = torch.cuda.get_device_properties(0)
    total = props.total_memory // (1024 * 1024)
    used = torch.cuda.memory_allocated() // (1024 * 1024)
    return total, total - used


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=LOADER_PORT, log_level="info")
