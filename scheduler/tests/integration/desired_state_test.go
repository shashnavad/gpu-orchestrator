package integration

import (
	"context"
	"testing"
	"time"

	"gpu-orchestrator/reconciler"
	"gpu-orchestrator/registry"
)

// TestAddDesiredDoesNotEvictPreviouslyDesiredModel is the regression test for
// the overwrite bug: scheduling a second model onto a slice that already has
// a desired model must not cause the first to be evicted.
func TestAddDesiredDoesNotEvictPreviouslyDesiredModel(t *testing.T) {
	reg := registry.NewNodeRegistry()
	heartbeats := make(chan reconciler.HeartbeatMsg, 1)
	actions := make(chan reconciler.ReconcileAction, 8)

	rec := reconciler.NewReconciler(reg, heartbeats, actions)
	rec.AddDesired("node-1", "gpu-0", "", "model-a")
	rec.AddDesired("node-1", "gpu-0", "", "model-b") // second placement, same slice

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go rec.Run(ctx)

	heartbeats <- reconciler.HeartbeatMsg{
		NodeID:       "node-1",
		GPUID:        "gpu-0",
		TotalVRAMMiB: 1000,
		UsedVRAMMiB:  500,
		LoadedModels: []string{"model-a", "model-b"}, // both correctly placed
	}

	select {
	case act := <-actions:
		t.Fatalf("expected no action, both models are desired and present, got %+v", act)
	case <-time.After(700 * time.Millisecond): // > one 500ms reconcile tick
	}
}

// TestRemoveDesiredEvictsOnlyTheTargetedModel confirms RemoveDesired doesn't
// take a co-located model down with it.
func TestRemoveDesiredEvictsOnlyTheTargetedModel(t *testing.T) {
	reg := registry.NewNodeRegistry()
	heartbeats := make(chan reconciler.HeartbeatMsg, 1)
	actions := make(chan reconciler.ReconcileAction, 8)

	rec := reconciler.NewReconciler(reg, heartbeats, actions)
	rec.AddDesired("node-1", "gpu-0", "", "model-a")
	rec.AddDesired("node-1", "gpu-0", "", "model-b")
	rec.RemoveDesired("node-1", "gpu-0", "", "model-b")

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go rec.Run(ctx)

	heartbeats <- reconciler.HeartbeatMsg{
		NodeID:       "node-1",
		GPUID:        "gpu-0",
		TotalVRAMMiB: 1000,
		UsedVRAMMiB:  500,
		LoadedModels: []string{"model-a", "model-b"},
	}

	deadline := time.After(2 * time.Second)
	for {
		select {
		case act := <-actions:
			if act.Type == reconciler.ActionEvict && act.ModelName == "model-a" {
				t.Fatal("model-a should not be evicted, it's still desired")
			}
			if act.Type == reconciler.ActionEvict && act.ModelName == "model-b" {
				return // correct
			}
		case <-deadline:
			t.Fatal("timed out waiting for EVICT(model-b)")
		}
	}
}
