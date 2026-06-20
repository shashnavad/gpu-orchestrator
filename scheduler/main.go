package main

import (
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
	go handleActions(ctx, actionCh, prewarmSignalCh)
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

func handleActions(ctx context.Context, actions <-chan reconciler.ReconcileAction, prewarns <-chan traffic.PrewarmSignal) {
	for {
		select {
		case action, ok := <-actions:
			if !ok {
				return
			}
			switch action.Type {
			case reconciler.ActionEvict:
				log.Println(formatAction("EVICT", action))
			case reconciler.ActionPrewarm:
				log.Println(formatAction("PLACED", action))
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

// formatAction builds a readable log line for placement and eviction events.
// SliceID is omitted for non-MIG nodes to keep non-MIG output uncluttered.
func formatAction(verb string, action reconciler.ReconcileAction) string {
	location := fmt.Sprintf("%s/%s", action.NodeID, action.GPUID)
	if action.SliceID != "" {
		location = fmt.Sprintf("%s/%s slice=%s", action.NodeID, action.GPUID, action.SliceID)
	}
	return fmt.Sprintf("[scheduler] %-6s  model=%-30s  on %s", verb, action.ModelName, location)
}
