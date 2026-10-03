"""
loader/vllm_engine.py — vLLM serving backend

This is the piece that actually changed: the loader used to call
`AutoModelForCausalLM.from_pretrained(...)` and hand-loop generation itself.
It now hands the model to vLLM's `LLM` engine, so placement, continuous
batching, and KV-cache management (PagedAttention) all come from vLLM —
the thing the scheduler's bin-packing and eviction thresholds are actually
sizing against once inference traffic is real.

One EngineHandle == one vLLM engine == one model. The scheduler's slice
model maps directly onto this: a MIG-scoped load gets `gpu_memory_utilization`
sized to that slice's share of the physical card, so multiple EngineHandles
can coexist on one GPU when node/agent.rs reports MIG slices.

VLLM_MOCK=true swaps in a synthetic engine with the identical interface, so
the scheduler -> agent -> loader loop is demoable on a laptop with no GPU and
without pulling multi-GB model weights. This mirrors the Rust agent's
existing `mock` cargo feature (agent/src/main.rs) — same idea: log lines and
API responses say "mock" plainly, nothing pretends to be a real generation.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger(__name__)

VLLM_MOCK = os.environ.get("VLLM_MOCK", "false").lower() == "true"

# vLLM's own quantization methods. Not the same set bitsandbytes offered
# (int8/int4) — those applied to raw HF `from_pretrained` calls we no longer
# make. AWQ/GPTQ/FP8 checkpoints are pre-quantized on disk; vLLM reads the
# quant config out of the repo itself, so this just gets passed through.
SUPPORTED_QUANTIZATIONS = {None, "awq", "gptq", "fp8", "squeezellm"}

# Synthetic VRAM footprints for the mock demo models (see agent/src/main.rs
# MIG_MODELS / PLAIN_MODELS) so mock-mode scheduling decisions stay
# realistic even though no weights are actually loaded.
_MOCK_VRAM_TABLE_MIB = {
    "phi-3-mini": 3_800,
    "llama-3-8b": 8_192,
    "mistral-7b": 7_168,
    "llama-3-70b": 35_840,
    "codellama-13b": 13_312,
}


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    tokens_per_sec: float


class EngineLoadError(RuntimeError):
    pass


class EngineHandle:
    """One vLLM engine instance, serving exactly one model."""

    def __init__(
        self,
        model_name: str,
        repo_id: str,
        quantization: Optional[str],
        gpu_memory_utilization: float,
        max_model_len: Optional[int],
        dtype: str,
    ) -> None:
        if quantization not in SUPPORTED_QUANTIZATIONS:
            raise EngineLoadError(
                f"unsupported quantization {quantization!r}; vLLM expects one "
                f"of {sorted(q for q in SUPPORTED_QUANTIZATIONS if q)} "
                "(pre-quantized checkpoints), not a bitsandbytes mode"
            )
        self.model_name = model_name
        self.repo_id = repo_id
        self.quantization = quantization
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.dtype = dtype
        self._llm = None
        self._vram_used_mib = 0
        self._load_duration_sec = 0.0

    # -- load -----------------------------------------------------------

    def load(self) -> None:
        if VLLM_MOCK:
            self._load_mock()
        else:
            self._load_real()

    def _load_real(self) -> None:
        import torch
        from vllm import LLM

        t0 = time.perf_counter()
        before_mib = _cuda_allocated_mib()

        try:
            self._llm = LLM(
                model=self.repo_id,
                quantization=self.quantization,
                dtype=self.dtype,
                gpu_memory_utilization=self.gpu_memory_utilization,
                max_model_len=self.max_model_len,
                trust_remote_code=False,
                # Skip CUDA graph capture. Costs some steady-state throughput
                # but removes a slow, memory-hungry warmup step that isn't
                # worth it for a scheduler demo that loads/evicts models
                # repeatedly rather than serving one model for hours.
                enforce_eager=True,
            )
        except Exception as exc:  # vLLM raises a mix of its own + torch errors
            raise EngineLoadError(str(exc)) from exc

        after_mib = _cuda_allocated_mib()
        self._vram_used_mib = max(after_mib - before_mib, 0)
        self._load_duration_sec = time.perf_counter() - t0
        log.info(
            "vLLM engine up: %s (repo=%s, quant=%s) in %.1fs — %d MiB",
            self.model_name, self.repo_id, self.quantization,
            self._load_duration_sec, self._vram_used_mib,
        )

    def _load_mock(self) -> None:
        t0 = time.perf_counter()
        time.sleep(0.05)  # keep the demo's timing believable, not instant
        self._vram_used_mib = _mock_vram_for(self.repo_id)
        self._load_duration_sec = time.perf_counter() - t0
        log.info(
            "[mock] vLLM engine 'up': %s (repo=%s) — %d MiB (synthetic, no weights loaded)",
            self.model_name, self.repo_id, self._vram_used_mib,
        )

    # -- generate ---------------------------------------------------------

    def generate(self, prompt: str, max_tokens: int, temperature: float) -> GenerationResult:
        if VLLM_MOCK:
            return self._generate_mock(prompt, max_tokens)
        return self._generate_real(prompt, max_tokens, temperature)

    def _generate_real(self, prompt: str, max_tokens: int, temperature: float) -> GenerationResult:
        from vllm import SamplingParams

        if self._llm is None:
            raise EngineLoadError(f"{self.model_name} engine not loaded")

        params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
        t0 = time.perf_counter()
        outputs = self._llm.generate([prompt], params, use_tqdm=False)
        elapsed = max(time.perf_counter() - t0, 1e-6)

        completion = outputs[0].outputs[0]
        completion_tokens = len(completion.token_ids)
        prompt_tokens = len(outputs[0].prompt_token_ids)
        return GenerationResult(
            text=completion.text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            tokens_per_sec=completion_tokens / elapsed,
        )

    def _generate_mock(self, prompt: str, max_tokens: int) -> GenerationResult:
        t0 = time.perf_counter()
        time.sleep(0.02)
        words = prompt.split()
        tail = " ".join(words[-8:]) if words else "(empty prompt)"
        text = f"[mock:{self.model_name}] ...{tail} — synthetic continuation, budget {max_tokens} tok"
        completion_tokens = min(max_tokens, max(len(text.split()), 1))
        elapsed = max(time.perf_counter() - t0, 1e-6)
        return GenerationResult(
            text=text,
            prompt_tokens=len(words),
            completion_tokens=completion_tokens,
            tokens_per_sec=completion_tokens / elapsed,
        )

    # -- unload -----------------------------------------------------------

    def unload(self) -> None:
        if self._llm is not None:
            del self._llm
            self._llm = None
        if not VLLM_MOCK:
            import gc
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

    @property
    def vram_used_mib(self) -> int:
        return self._vram_used_mib

    @property
    def load_duration_sec(self) -> float:
        return self._load_duration_sec


def _cuda_allocated_mib() -> int:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() // (1024 * 1024)
    except ImportError:
        pass
    return 0


def _mock_vram_for(repo_id: str) -> int:
    key = repo_id.rsplit("/", 1)[-1].lower().replace("-", "").replace("_", "")
    for name, vram in _MOCK_VRAM_TABLE_MIB.items():
        if name.replace("-", "") in key:
            return vram
    return 6_000  # generic footprint for repos outside the demo's model set
