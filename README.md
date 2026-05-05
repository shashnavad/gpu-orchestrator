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
└── agent/              ← Rust
    ├── Cargo.toml
    └── src/
        └── main.rs
```

## Design Decisions

1. **Centralized scheduling paradigm** to make globally informed placement decisions.
2. **Dynamic bin-packer** that packs by default, then evicts/spreads models when VRAM spikes above **85%**.
3. **Model weight affinity** to prefer nodes that already have large model artifacts (for example, a 50GB model) and avoid cold downloads.
4. **Reconciliation loop** to continuously compare observed state and desired state.
5. **Local-first GPU simulation in Rust** without owning GPUs:
   - Use conditional compilation.
   - Add a `mock` feature in `Cargo.toml`.
   - In mock mode, return random JSON telemetry instead of calling NVIDIA drivers.
   - Outcome: build 100% of Go logic and ~80% of Rust logic at near-zero infrastructure cost.
6. **High-level scheduler-managed model cache** for placement-aware prewarm and reuse decisions.
7. **Sidecar container pattern** to coordinate the Go control plane and Rust node agent.
8. **Actual vs desired state healing**:
   - Production divergence happens from node failures and network blips.
   - Reconciler runs every **500ms** and listens for heartbeats via `select`.
   - It compares incoming state against internal `NodeRegistry`.
9. **Predictive cold-start reduction**:
   - A Go `TrafficAnalyzer` tracks request frequency over a rolling 5-minute window.
   - It emits `PREWARM` signals so the Rust agent loads weights before the first hit.
10. **Python model loader** to integrate with PyTorch and Hugging Face for weight manipulation.
11. **Persistent FastAPI server** on `:8001` (not subprocess-per-call):
   - Rust calls `POST /load`, `POST /evict`, `POST /checkpoint` over localhost HTTP.

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

Run in mock mode (recommended locally without GPUs):

```bash
cd agent
cargo run --features mock
```

### 4) (Optional) Start the Python FastAPI model loader

```bash
# example only: adjust path when python service is added
cd python-loader
uvicorn app:app --host 127.0.0.1 --port 8001 --reload
```

## Development Notes

- Keep the Go scheduler as the source of desired state.
- Treat the Rust agent as the source of observed node reality.
- Use reconciliation and prewarm signals to keep latency and cost controlled.
