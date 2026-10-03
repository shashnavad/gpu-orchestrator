package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"gpu-orchestrator/admission"
	"gpu-orchestrator/cache"
	"gpu-orchestrator/reconciler"
	"gpu-orchestrator/registry"
	schedulerpkg "gpu-orchestrator/scheduler"
	"gpu-orchestrator/traffic"
)

func main() {
	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()

	nodeReg := registry.NewNodeRegistry()
	modelCache := cache.NewModelCache()
	_ = modelCache
	bpScheduler := schedulerpkg.NewBinPacker(nodeReg)

	heartbeatCh := make(chan reconciler.HeartbeatMsg, 256)
	actionCh := make(chan reconciler.ReconcileAction, 64)
	requestEventCh := make(chan traffic.RequestEvent, 1024)
	prewarmSignalCh := make(chan traffic.PrewarmSignal, 64)

	rec := reconciler.NewReconciler(nodeReg, heartbeatCh, actionCh)
	analyzer := traffic.NewTrafficAnalyzer(requestEventCh, prewarmSignalCh)
	admissionQueue := admission.NewQueue(50) // capacity per WFQ class
	gateway := admission.NewGateway(bpScheduler, rec, admissionQueue, 5*time.Second)

	go rec.Run(ctx)
	go analyzer.Run(ctx)
	go handleActions(ctx, nodeReg, actionCh, prewarmSignalCh)
	go gateway.DrainLoop(ctx, 500*time.Millisecond)
	go serveHeartbeats(ctx, heartbeatCh, gateway)

	log.Println("[main] gpu-orchestrator scheduler running on :8080")
	<-ctx.Done()
	log.Println("[main] shutdown complete")
}

func serveHeartbeats(ctx context.Context, ch chan<- reconciler.HeartbeatMsg, gw *admission.Gateway) {
	mux := http.NewServeMux()
	mux.HandleFunc("/heartbeat", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		var msg reconciler.HeartbeatMsg
		if err := json.NewDecoder(r.Body).Decode(&msg); err != nil {
			http.Error(w, "bad request: "+err.Error(), http.StatusBadRequest)
			return
		}
		logHeartbeat(msg)
		select {
		case ch <- msg:
			w.WriteHeader(http.StatusNoContent)
		default:
			http.Error(w, "heartbeat channel full", http.StatusServiceUnavailable)
		}
	})

	mux.HandleFunc("/schedule", gw.HandleSchedule)
	srv := &http.Server{Addr: ":8080", Handler: mux}

	go func() {
		<-ctx.Done()
		if err := srv.Shutdown(context.Background()); err != nil {
			log.Printf("[heartbeat] shutdown error: %v", err)
		}
	}()

	log.Println("[heartbeat] listening on :8080")
	if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		log.Fatalf("[heartbeat] fatal: %v", err)
	}
}

// logHeartbeat prints a concise, human-readable summary of each incoming
// heartbeat. For MIG nodes it prints per-slice VRAM; for non-MIG nodes it
// prints node-level VRAM. Loaded models are shown inline so GPU allocation
// state is visible at a glance without reading raw JSON.
func logHeartbeat(msg reconciler.HeartbeatMsg) {
	if msg.MIGEnabled && len(msg.MIGSlices) > 0 {
		for _, s := range msg.MIGSlices {
			usedPct := 0.0
			if s.TotalVRAMMiB > 0 {
				usedPct = float64(s.UsedVRAMMiB) / float64(s.TotalVRAMMiB) * 100
			}
			models := "idle"
			if len(s.LoadedModels) > 0 {
				models = strings.Join(s.LoadedModels, ", ")
			}
			log.Printf(
				"[gpu] %s/%s slice=%-7s  allotted=%5d MiB  used=%5d MiB  free=%5d MiB  (%.0f%%)  models: %s",
				msg.NodeID, msg.GPUID, s.SliceID,
				s.TotalVRAMMiB, s.UsedVRAMMiB, s.TotalVRAMMiB-s.UsedVRAMMiB,
				usedPct, models,
			)
		}
	} else {
		usedPct := 0.0
		if msg.TotalVRAMMiB > 0 {
			usedPct = float64(msg.UsedVRAMMiB) / float64(msg.TotalVRAMMiB) * 100
		}
		models := "idle"
		if len(msg.LoadedModels) > 0 {
			models = strings.Join(msg.LoadedModels, ", ")
		}
		log.Printf(
			"[gpu] %s/%s  allotted=%5d MiB  used=%5d MiB  free=%5d MiB  (%.0f%%)  models: %s",
			msg.NodeID, msg.GPUID,
			msg.TotalVRAMMiB, msg.UsedVRAMMiB, msg.TotalVRAMMiB-msg.UsedVRAMMiB,
			usedPct, models,
		)
	}
}

// agentCommandClient is used to dispatch PREWARM/EVICT commands to the Rust
// agent's /command endpoint. A short timeout keeps one unreachable node from
// backing up the action loop; dispatch runs in its own goroutine per action
// anyway (see dispatchToAgent), but a hung request would still pin that
// goroutine and its connection indefinitely without this.
var agentCommandClient = &http.Client{Timeout: 5 * time.Second}

// agentCommand mirrors the Rust agent's AgentCommand struct (agent/src/main.rs).
type agentCommand struct {
	Action      string `json:"action"`
	ModelName   string `json:"model_name"`
	Quantization string `json:"quantization,omitempty"`
	SliceID     string `json:"slice_id,omitempty"`
}

// handleActions is where desired state actually becomes real state: it takes
// what the reconciler decided and dispatches it to the Rust agent that owns
// the target node, rather than only logging it. Without this, PREWARM/EVICT
// were purely observational — nothing ever told an agent to call the loader.
func handleActions(ctx context.Context, nodeReg *registry.NodeRegistry, actions <-chan reconciler.ReconcileAction, prewarns <-chan traffic.PrewarmSignal) {
	for {
		select {
		case action, ok := <-actions:
			if !ok {
				return
			}
			switch action.Type {
			case reconciler.ActionEvict:
				log.Println(formatAction("EVICT", action))
				dispatchToAgent(nodeReg, action, "EVICT")
			case reconciler.ActionPrewarm:
				log.Println(formatAction("PLACED", action))
				dispatchToAgent(nodeReg, action, "PREWARM")
			case reconciler.ActionMarkDead:
				log.Printf("[scheduler] NODE DOWN  %s/%s  -- workloads will be rescheduled", action.NodeID, action.GPUID)
			}
		case sig, ok := <-prewarns:
			if !ok {
				return
			}
			log.Printf("[scheduler] PREWARM  model=%-30s  reason: %s", sig.ModelName, sig.Reason)
		case <-ctx.Done():
			return
		}
	}
}

// dispatchToAgent sends the actual HTTP command to the node's Rust agent.
// Runs off the action-processing goroutine so one slow/unreachable agent
// can't stall reconciliation for every other node.
//
// Note: quantization isn't threaded through ScheduleRequest yet (see
// scheduler/scheduler/bin_packer.go), so every dispatch currently leaves it
// unset and the loader falls back to whatever quantization the repo itself
// ships pre-baked (vLLM reads AWQ/GPTQ config straight off the checkpoint).
// Per-request quantization selection is a reasonable next step, not done here.
func dispatchToAgent(nodeReg *registry.NodeRegistry, action reconciler.ReconcileAction, verb string) {
	node, ok := nodeReg.Get(action.NodeID, action.GPUID)
	if !ok || node.AgentAddr == "" {
		log.Printf("[dispatch] no known agent address for %s/%s, dropping %s %s",
			action.NodeID, action.GPUID, verb, action.ModelName)
		return
	}

	cmd := agentCommand{
		Action:    verb,
		ModelName: action.ModelName,
		SliceID:   action.SliceID,
	}
	body, err := json.Marshal(cmd)
	if err != nil {
		log.Printf("[dispatch] failed to encode command for %s: %v", action.ModelName, err)
		return
	}

	go func(addr string) {
		resp, err := agentCommandClient.Post(addr+"/command", "application/json", bytes.NewReader(body))
		if err != nil {
			log.Printf("[dispatch] %s %s -> %s failed: %v", verb, action.ModelName, addr, err)
			return
		}
		defer resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			log.Printf("[dispatch] %s %s -> %s returned %s", verb, action.ModelName, addr, resp.Status)
			return
		}
		// The agent's /command handler always answers 200 and puts the real
		// outcome in the body (see handle_command in agent/src/main.rs) — a
		// failed loader call still comes back as HTTP 200 {"status":"error"}.
		var result struct {
			Status string `json:"status"`
			Detail string `json:"detail"`
		}
		if err := json.NewDecoder(resp.Body).Decode(&result); err == nil && result.Status == "error" {
			log.Printf("[dispatch] %s %s -> %s reported error: %s", verb, action.ModelName, addr, result.Detail)
			return
		}
		log.Printf("[dispatch] %s %s -> %s ok", verb, action.ModelName, addr)
	}(node.AgentAddr)
}

// formatAction builds a readable log line for placement and eviction events.
// SliceID is omitted for non-MIG nodes to keep non-MIG output uncluttered.
func formatAction(verb string, action reconciler.ReconcileAction) string {
	location := fmt.Sprintf("%s/%s", action.NodeID, action.GPUID)
	if action.SliceID != "" {
		location = fmt.Sprintf("%s/%s slice=%s", action.NodeID, action.GPUID, action.SliceID)
	}
	return fmt.Sprintf("[scheduler] %-6s  model=%-30s  on %s", verb, action.ModelName, location)
}
