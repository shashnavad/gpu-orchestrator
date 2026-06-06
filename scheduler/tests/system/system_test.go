package system

import (
	"context"
	"testing"
	"time"

	"gpu-orchestrator/reconciler"
	"gpu-orchestrator/registry"
	schedulerpkg "gpu-orchestrator/scheduler"
)

func TestSchedulerToReconcilerEndToEnd(t *testing.T) {
	reg := registry.NewNodeRegistry()
	reg.Upsert(&registry.GPUNode{
		NodeID:              "node-a",
		GPUID:               "gpu-0",
		TotalVRAMMiB:        1000,
		UsedVRAMMiB:         100,
		ModelWeightAffinity: map[string]int64{"model-z": 4096},
	})

	packer := schedulerpkg.NewBinPacker(reg)
	decision, err := packer.Schedule(schedulerpkg.ScheduleRequest{
		ModelName:     "model-z",
		VRAMNeededMiB: 200,
		Priority:      schedulerpkg.P0Production,
	})
	if err != nil {
		t.Fatalf("scheduler failed: %v", err)
	}
	if decision.NodeID != "node-a" {
		t.Fatalf("expected placement on node-a, got %s", decision.NodeID)
	}

	heartbeats := make(chan reconciler.HeartbeatMsg, 1)
	actions := make(chan reconciler.ReconcileAction, 8)
	rec := reconciler.NewReconciler(reg, heartbeats, actions)
	// SliceID="" for non-MIG node — SetDesired now requires the slice argument.
	rec.SetDesired(decision.NodeID, decision.GPUID, decision.SliceID, []string{"model-z"})

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go rec.Run(ctx)

	heartbeats <- reconciler.HeartbeatMsg{
		NodeID:       "node-a",
		GPUID:        "gpu-0",
		TotalVRAMMiB: 1000,
		UsedVRAMMiB:  300,
		LoadedModels: []string{},
	}

	select {
	case act := <-actions:
		if act.Type != reconciler.ActionPrewarm || act.ModelName != "model-z" {
			t.Fatalf("expected PREWARM(model-z), got %+v", act)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for PREWARM action in end-to-end flow")
	}
}
