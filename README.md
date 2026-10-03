# GPU Orchestration & Cost Optimization

GPU compute is the new oil. Teams are over-provisioning expensive GPU hardware because they lack software that can orchestrate these high-value assets efficiently. This project focuses on building that software layer for smarter placement, better utilization, and lower cost.

The system is designed with **Golang, Rust, and Python**:
- **Go** for high-level scheduling, reconciliation, and traffic analysis.
- **Rust** for node-level sidecar/agent behavior and fast systems interaction.
- **Python + vLLM** as the actual serving engine the scheduler is placing and load-balancing.

## Project Structure

```text
gpu-orchestrator/
├── scheduler/          ← Go
│   ├── main.go
│   ├── go.mod
│   ├── registry/
│   │   └── node_registry.go
│   ├── scheduler/
│   │   └── bin_packer.go
│   ├── reconciler/
│   │   └── reconciler.go
│   ├── admission/
│   │   ├── gateway.go
│   │   └── queue.go
│   ├── cache/
│   │   └── model_cache.go
│   └── traffic/
│       └── traffic_analyzer.go
└── agent/              ← Rust + Python loader
    ├── Cargo.toml
    ├── src/
    │   └── main.rs
    └── loader/
        ├── main.py            ← FastAPI sidecar: /load, /generate, /evict, /checkpoint, /status
        ├── vllm_engine.py     ← the actual serving engine (vLLM), + VLLM_MOCK demo backend
        ├── checkpoint.py      ← restart manifests (see Design Decision 11)
        └── requirements.txt
```

## Design Decisions

1. **Centralized scheduling paradigm** to make globally informed placement decisions.
2. **Dynamic bin-packer** that packs by default, then evicts/spreads models when VRAM spikes above **85%**. On MIG-enabled nodes, placement decisions are made at slice granularity rather than whole-GPU level.
3. **Model weight affinity** to prefer nodes that already have large model artifacts (for example, a 50GB model) and avoid cold downloads.
4. **Reconciliation loop** to continuously compare observed state and desired state.
5. **Local-first GPU simulation in Rust** without owning GPUs:
   - Use conditional compilation.
   - Add a `mock` feature in `Cargo.toml`.
   - In mock mode, return synthetic telemetry instead of calling NVIDIA drivers. Two simulated nodes: node-001 is a MIG-partitioned H100 with 3x `2g.20gb` slices running `phi-3-mini`, `llama-3-8b`, and `mistral-7b`; node-002 is a plain H100 running `llama-3-70b` and `codellama-13b`. VRAM drifts at ±5% of each model's real footprint per tick rather than random noise, so affinity and scheduling decisions are meaningful.
   - Set `MIG_NODE=true` to boot the agent as node-001; omit or set `false` for node-002.
   - Outcome: build 100% of Go logic and ~80% of Rust logic at near-zero infrastructure cost.
6. **High-level scheduler-managed model cache** for placement-aware prewarm and reuse decisions.
7. **Sidecar container pattern** to coordinate the Go control plane and Rust node agent.
8. **Actual vs desired state healing**:
   - Production divergence happens from node failures and network blips.
   - Reconciler runs every **500ms** and listens for heartbeats via `select`.
   - It compares incoming state against internal `NodeRegistry`. On MIG nodes, desired state and drift detection operate at slice granularity (`nodeID/gpuID/sliceID`).
9. **Predictive cold-start reduction**:
   - A Go `TrafficAnalyzer` tracks request frequency over a rolling 5-minute window.
   - It emits `PREWARM` signals so the Rust agent loads weights before the first hit.
10. **vLLM as the actual serving engine**, not a hand-rolled `transformers.generate()` loop:
   - `/load` hands the model to vLLM's `LLM` engine — continuous batching and PagedAttention KV-cache management come from vLLM, not from this codebase.
   - `/generate` runs real inference through that engine and returns `tokens_per_sec`, so placement decisions can eventually be judged against real throughput, not just VRAM bookkeeping.
   - `quantization` is vLLM-native (`awq` / `gptq` / `fp8`, read off the checkpoint's own quant config) rather than a bitsandbytes mode applied to a raw HF load.
   - `VLLM_MOCK=true` swaps in a synthetic engine with the identical interface — same idea as the Rust agent's `mock` feature, for demoing the full loop without a GPU or multi-GB downloads. Every mock response says so explicitly (`backend: "vllm-mock"`, `[mock]` log lines); nothing pretends to be a real generation.
   - MIG slices map onto vLLM's `gpu_memory_utilization`: a load scoped to a slice gets that fraction of the card instead of vLLM's whole-device default (see `_resolve_gpu_memory_utilization` in `main.py`).
11. **Persistent FastAPI sidecar** on `:8001` (not subprocess-per-call):
   - Rust calls `POST /load`, `POST /generate`, `POST /evict`, `POST /checkpoint` over localhost HTTP.
   - `/checkpoint` was rewritten alongside the vLLM move: vLLM re-packs weights into its own internal layout, so there's no single clean `state_dict` to serialize the way there was with a raw `AutoModelForCausalLM`, and every model here is served read-only from its HF checkpoint anyway — nothing is being fine-tuned in place. `/checkpoint` now persists a small restart manifest (repo_id, quantization, slice, engine args) instead of gigabytes of tensors; a replacement node relaunches the same engine from the manifest and leans on weight affinity (Design Decision 3) for a fast local-cache reload rather than restoring serialized weights. See the module docstring in `checkpoint.py` for the full reasoning — this is a scope-down, not a like-for-like swap, and it will not preserve weights that only exist in GPU memory.
12. **MIG (Multi-Instance GPU) fractionalization** to run multiple models on a single A100/H100:
   - `GPUNode` carries `MIGEnabled` and `MIGSlices`, each with isolated `TotalVRAMMiB` and `UsedVRAMMiB`.
   - The bin-packer expands each MIG node into per-slice candidates and selects the tightest-fit slice (bin-pack) or most-free slice (spread), using the same 85% threshold.
   - Non-MIG nodes use a synthetic full-GPU slice so both code paths share one scheduler implementation.
   - The Python loader enforces per-slice VRAM budgets via `slice_vram_cap_mib` on each `/load` call, failing fast before the OOM killer acts.
   - Hardware isolation is provided by the MIG partition itself; the scheduler and loader add a soft guard layer on top.
13. **Admission backpressure with weighted fair queueing**:
   - `POST /schedule` attempts immediate placement via the bin-packer first.
   - On a capacity miss, the request is queued by priority class (mapped from the existing P0/P1/P2) instead of dropped — a 70/20/10 (High/Medium/Low) weighted round-robin so batch jobs can't starve production traffic out of retries.
   - The handler long-polls the queued request up to a configurable timeout. A full per-class queue or an expired wait returns `429 Too Many Requests` with `Retry-After`.
   - A background loop retries queued requests against the live registry on a fixed interval as capacity frees up.
14. **Scheduler decisions actually reach the agent now.** The reconciler already computed `PREWARM`/`EVICT` actions, but `handleActions` in `main.go` only logged them — nothing ever called the Rust agent's `/command` endpoint, so a placement decision never resulted in a real `/load` call. Fixed by having each agent report its own callback address (`agent_addr`) on every heartbeat; `dispatchToAgent` in `main.go` now POSTs the command to that address when the reconciler emits an action. This is what makes vLLM integration (Decision 10) something the scheduler actually exercises end to end, rather than something only reachable by curling the loader directly.

## Commands to Use and Start

### Prerequisites

- Go 1.25+
- Rust (stable toolchain) + Cargo
- Python 3.10+ (for the loader/vLLM sidecar)
- A CUDA GPU + real `vllm` install for real inference — **or** just set `VLLM_MOCK=true` and skip straight to the demo below with none of that.

### 1) Clone and enter

```bash
git clone https://github.com/shashnavad/gpu-orchestrator.git
cd gpu-orchestrator
```

### 2) Run the Go scheduler

```bash
cd scheduler
go mod tidy
go run .
```

### 3) Run the Rust agent

In a second terminal:

```bash
cd agent
cargo run
```

Run node-001 in mock mode (MIG H100, `phi-3-mini` / `llama-3-8b` / `mistral-7b`):

```bash
cd agent
MIG_NODE=true cargo run --features mock
```

In a third terminal, run node-002 (plain H100, `llama-3-70b` / `codellama-13b`):

```bash
cd agent
MIG_NODE=false NODE_ID=node-002 LISTEN_ADDR=0.0.0.0:9091 cargo run --features mock
```

### 3b) Run the full two-node mock cluster with Docker Compose

The fastest way to see cross-node scheduling decisions. Starts the Go scheduler,
node-001 (MIG H100), and node-002 (plain H100) in one command:

```bash
docker compose up --build
```

Expected output on the scheduler terminal:

```
[gpu] node-001/gpu-0 slice=0/0/0   allotted=20480 MiB  used= 3819 MiB  free=16661 MiB  (19%)  models: phi-3-mini
[gpu] node-001/gpu-0 slice=1/0/0   allotted=20480 MiB  used= 8354 MiB  free=12126 MiB  (41%)  models: llama-3-8b
[gpu] node-001/gpu-0 slice=2/0/0   allotted=20480 MiB  used= 7041 MiB  free=13439 MiB  (34%)  models: mistral-7b
[gpu] node-002/gpu-0  allotted=81920 MiB  used=49511 MiB  free=32409 MiB  (60%)  models: llama-3-70b, codellama-13b
```

At this point the mock agents are heartbeating and the scheduler knows their
`agent_addr`, but nothing has been scheduled yet — start the loader (step 4)
before step 3c if you want `PREWARM` to actually reach vLLM instead of
failing to connect.

### 3c) Test admission backpressure

With the mock cluster running, saturate a node and watch low-priority requests
queue or get throttled:

```bash
curl -i -X POST localhost:8888/schedule \
  -d '{"ModelName":"big-model","VRAMNeededMiB":40000,"Priority":2}'
```

`200` = placed immediately. `429` = every slice is full and the per-class
queue is also full — back off per `Retry-After`. On a `200`, watch the
scheduler log for a `[dispatch] PREWARM ... ok` line — that's the reconciler's
placement decision actually reaching the node's agent, which forwards it to
the loader (step 4).

A request that actually fits gets placed immediately instead of queuing:

```bash
curl -i -X POST localhost:8888/schedule \
  -d '{"ModelName":"phi-3-mini","VRAMNeededMiB":4000,"Priority":0}'
```

```
HTTP/1.1 200 OK
Content-Type: application/json

{"NodeID":"node-001","GPUID":"gpu-0","SliceID":"1/0/0","MIGEnabled":true,"AffinityHit":true}
```

`AffinityHit: true` means the bin-packer placed it on a slice that already had
`phi-3-mini`'s weights resident (Design Decision 3) rather than a cold node.
Watch the scheduler log right after this call for:

```
[dispatch] PREWARM phi-3-mini -> http://agent-node-001:9090 ok
```

That line is the reconciler's placement decision actually reaching the node's
agent (Design Decision 14) — without it, this `200` would be placement on
paper only, same as before the dispatch fix.

Note: `AffinityHit` depends on `phi-3-mini` already being loaded on node-001
at the moment you run this — which depends on timing relative to container
startup and whether you ran the backpressure example above first. Don't
read a `false` here as a bug; it just means this was a cold placement.

### 4) Start the Python vLLM loader

Install once:

```bash
cd agent/loader
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt  # heavy: pulls real vllm + torch
```

**No GPU, or just want to see the loop work?** Run in mock mode — the loader
answers `/load`, `/generate`, `/evict` with the same shapes a real vLLM
backend would, clearly labeled `[mock]` in logs and `"backend": "vllm-mock"`
in responses:

```bash
VLLM_MOCK=true python3 main.py
```

With a real GPU, drop `VLLM_MOCK` and run the same command — the first
`/load` for a given `repo_id` will download weights via Hugging Face and
stand up a real vLLM engine.

### 4b) Prove real inference is happening

With the loader running (mock or real) and at least one model loaded
(either via step 3c's admission flow, or directly for a quick check):

```bash
curl -s localhost:8001/load -X POST -H 'content-type: application/json' \
  -d '{"model_name":"phi-3-mini","repo_id":"microsoft/Phi-3-mini-4k-instruct"}'

curl -s localhost:8001/generate -X POST -H 'content-type: application/json' \
  -d '{"model_name":"phi-3-mini","prompt":"The GPU scheduler placed this model because","max_tokens":40}'
```

The response includes `tokens_per_sec` and which `backend` served it
(`vllm` or `vllm-mock`) — that field is the honest tell for whether you're
looking at a real generation or the demo path.

### 5) Run tests

```bash
cd scheduler
go test ./...
```

Run suites independently:

```bash
cd scheduler
go test ./tests/unit ./tests/integration ./tests/system
```

## Development Notes

- Keep the Go scheduler as the source of desired state.
- Treat the Rust agent as the source of observed node reality.
- Use reconciliation and prewarm signals to keep latency and cost controlled.
- On MIG nodes, pass the `SliceID` from `ScheduleDecision` into `SetDesired` so the reconciler diffs at slice granularity, not node level.
- Use `AddDesired`/`RemoveDesired` for incremental scheduling decisions. `SetDesired` replaces the entire desired list for a slice — fine for bulk/initial state, but it will silently evict any other model already placed on that slice if used per-request.
