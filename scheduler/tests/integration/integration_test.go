package integration

import (
	"context"
	"testing"
	"time"

	"gpu-orchestrator/reconciler"
	"gpu-orchestrator/registry"
)

func TestReconcilerEmitsPrewarmForDesiredModelMissingInHeartbeat(t *testing.T) {
	reg := registry.NewNodeRegistry()
	heartbeats := make(chan reconciler.HeartbeatMsg, 1)
	actions := make(chan reconciler.ReconcileAction, 8)

	rec := reconciler.NewReconciler(reg, heartbeats, actions)
	rec.SetDesired("node-1", "gpu-0", []string{"model-a"})

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go rec.Run(ctx)

	heartbeats <- reconciler.HeartbeatMsg{
		NodeID:       "node-1",
		GPUID:        "gpu-0",
		TotalVRAMMiB: 1000,
		UsedVRAMMiB:  200,
		LoadedModels: []string{},
	}

	deadline := time.After(2 * time.Second)
	for {
		select {
		case act := <-actions:
			if act.Type == reconciler.ActionPrewarm && act.ModelName == "model-a" {
				return
			}
		case <-deadline:
			t.Fatal("timed out waiting for PREWARM reconcile action")
		}
	}
}

func TestReconcilerMarksStaleNodeDead(t *testing.T) {
	reg := registry.NewNodeRegistry()
	reg.Upsert(&registry.GPUNode{
		NodeID:        "node-stale",
		GPUID:         "gpu-0",
		TotalVRAMMiB:  1000,
		UsedVRAMMiB:   100,
		LastHeartbeat: time.Now().Add(-5 * time.Second),
		Healthy:       true,
	})

	heartbeats := make(chan reconciler.HeartbeatMsg)
	actions := make(chan reconciler.ReconcileAction, 8)
	rec := reconciler.NewReconciler(reg, heartbeats, actions)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go rec.Run(ctx)

	select {
	case act := <-actions:
		if act.Type != reconciler.ActionMarkDead || act.NodeID != "node-stale" {
			t.Fatalf("unexpected action: %+v", act)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for MARK_DEAD action")
	}
}
