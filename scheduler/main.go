package main

import (
	"context"
	"log"
	"os"
	"os/signal"
	"syscall"

	"gpu-orchestrator/cache"
	"gpu-orchestrator/reconciler"
	"gpu-orchestrator/registry"
	schedulerpkg "gpu-orchestrator/scheduler"
	"gpu-orchestrator/traffic"
)

func main() {
	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()

	// --- Wire up core components ---

	// NodeRegistry: single source of truth for cluster state.
	// All components read from this; only the Reconciler writes to it.
	nodeReg := registry.NewNodeRegistry()

	// ModelCache: tracks which nodes have weight files on their NVMe.
	modelCache := cache.NewModelCache()
	_ = modelCache // used by the scheduler; wired below

	// BinPacker: the placement engine.
	bpScheduler := schedulerpkg.NewBinPacker(nodeReg)
	_ = bpScheduler // in a full implementation, exposed via HTTP handler

	// Channels — these are the nervous system connecting components.
	// Buffered so producers don't block on a slow consumer.
	heartbeatCh := make(chan reconciler.HeartbeatMsg, 256)
	actionCh := make(chan reconciler.ReconcileAction, 64)
	requestEventCh := make(chan traffic.RequestEvent, 1024)
	prewarmSignalCh := make(chan traffic.PrewarmSignal, 64)

	// Reconciler: anti-entropy loop, 500ms tick.
	rec := reconciler.NewReconciler(nodeReg, heartbeatCh, actionCh)

	// TrafficAnalyzer: rolling 5-minute window, emits PREWARM signals.
	analyzer := traffic.NewTrafficAnalyzer(requestEventCh, prewarmSignalCh)

	// --- Start goroutines ---

	go rec.Run(ctx)
	go analyzer.Run(ctx)

	// Action handler: consumes ReconcileActions and PrewarmSignals.
	// In a real deployment this would call the Rust agent's HTTP API.
	go handleActions(ctx, actionCh, prewarmSignalCh)

	// Heartbeat ingestion stub: in production, this comes from the HTTP server
	// where Rust agents POST their telemetry. Here we show the wiring.
	go ingestHeartbeats(ctx, heartbeatCh)

	log.Println("[main] gpu-orchestrator scheduler running")
	<-ctx.Done()
	log.Println("[main] shutdown complete")
}

// handleActions is the executor layer. It reads from both the Reconciler's
// action channel and the TrafficAnalyzer's prewarm channel and dispatches
// commands to the appropriate Rust agent.
//
// In production: each action results in an HTTP POST to the Rust agent
// running as a sidecar on the target node.
func handleActions(ctx context.Context, actions <-chan reconciler.ReconcileAction, prewarns <-chan traffic.PrewarmSignal) {
	for {
		select {
		case action, ok := <-actions:
			if !ok {
				return
			}
			switch action.Type {
			case reconciler.ActionEvict:
				log.Printf("[action] EVICT model=%s node=%s gpu=%s", action.ModelName, action.NodeID, action.GPUID)
				// TODO: POST /evict to Rust agent at node's IP
			case reconciler.ActionPrewarm:
				log.Printf("[action] PREWARM model=%s node=%s gpu=%s", action.ModelName, action.NodeID, action.GPUID)
				// TODO: POST /prewarm to Rust agent at node's IP
			case reconciler.ActionMarkDead:
				log.Printf("[action] MARK_DEAD node=%s gpu=%s — triggering rescheduling", action.NodeID, action.GPUID)
				// TODO: trigger scheduler to migrate workloads from this node
			}

		case sig, ok := <-prewarns:
			if !ok {
				return
			}
			log.Printf("[action] PREWARM (traffic-driven) model=%s reason=%s", sig.ModelName, sig.Reason)
			// TODO: select best node via cache.WarmNodes() and POST /prewarm

		case <-ctx.Done():
			return
		}
	}
}

// ingestHeartbeats is a placeholder for the HTTP handler that receives
// heartbeats from Rust agents. In production this is an http.HandleFunc.
// Here it simulates a single mock heartbeat to verify channel wiring.
func ingestHeartbeats(ctx context.Context, ch chan<- reconciler.HeartbeatMsg) {
	// Simulate one heartbeat from a mock Rust agent at startup.
	select {
	case ch <- reconciler.HeartbeatMsg{
		NodeID:       "node-001",
		GPUID:        "gpu-0",
		TotalVRAMMiB: 81920, // H100 80GB
		UsedVRAMMiB:  20480,
		LoadedModels: []string{"llama-3-8b"},
		ModelWeightAffinity: map[string]int64{
			"llama-3-8b":  8192,
			"phi-3-mini":  3800,
		},
	}:
		log.Println("[ingest] sent mock heartbeat for node-001/gpu-0")
	case <-ctx.Done():
	}
}
