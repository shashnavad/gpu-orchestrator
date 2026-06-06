# GPU Orchestration & Cost Optimization

GPU compute is the new oil. Teams are over-provisioning expensive GPU hardware because they lack software that can orchestrate these high-value assets efficiently. This project focuses on building that software layer for smarter placement, better utilization, and lower cost.

The system is designed with **Golang, Rust, and Python**:
- **Go** for high-level scheduling, reconciliation, and traffic analysis.
- **Rust** for node-level sidecar/agent behavior and fast systems interaction.
- **Python** for model loading workflows with PyTorch and Hugging Face.

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
│   ├── cache/
│   │   └── model_cache.go
│   └── traffic/
│       └── traffic_analyzer.go
└── agent/              ← Rust + Python loader
    ├── Cargo.toml
    ├── src/
    │   └── main.rs
    └── loader/
        ├── main.py
        └── checkpoint.py
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
10. **Python model loader** to integrate with PyTorch and Hugging Face for weight manipulation.
11. **Persistent FastAPI server** on `:8001` (not subprocess-per-call):
   - Rust calls `POST /load`, `POST /evict`, `POST /checkpoint` over localhost HTTP.
12. **MIG (Multi-Instance GPU) fractionalization** to run multiple models on a single A100/H100:
   - `GPUNode` carries `MIGEnabled` and `MIGSlices`, each with isolated `TotalVRAMMiB` and `UsedVRAMMiB`.
   - The bin-packer expands each MIG node into per-slice candidates and selects the tightest-fit slice (bin-pack) or most-free slice (spread), using the same 85% threshold.
   - Non-MIG nodes use a synthetic full-GPU slice so both code paths share one scheduler implementation.
   - The Python loader enforces per-slice VRAM budgets via `slice_vram_cap_mib` on each `/load` call, failing fast before the OOM killer acts.
   - Hardware isolation is provided by the MIG partition itself; the scheduler and loader add a soft guard layer on top.

## Commands to Use and Start

### Prerequisites

- Go 1.25+
- Rust (stable toolchain) + Cargo
- Python 3.10+ (for loader/API layer)

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

### 4) Start the Python FastAPI model loader

```bash
cd agent/loader
python3 main.py serve --port 8001
```

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
