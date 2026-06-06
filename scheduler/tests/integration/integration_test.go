package integration

import (
	"context"
	"testing"
	"time"

	"gpu-orchestrator/reconciler"
	"gpu-orchestrator/registry"
)

// ── Existing tests (unchanged) ─────────────────────────────────────────────

func TestReconcilerEmitsPrewarmForDesiredModelMissingInHeartbeat(t *testing.T) {
	reg := registry.NewNodeRegistry()
	heartbeats := make(chan reconciler.HeartbeatMsg, 1)
	actions := make(chan reconciler.ReconcileAction, 8)

	rec := reconciler.NewReconciler(reg, heartbeats, actions)
	// SetDesired now takes sliceID; pass "" for non-MIG behavior.
	rec.SetDesired("node-1", "gpu-0", "", []string{"model-a"})

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

// ── MIG-specific integration tests ────────────────────────────────────────

// TestMIGReconcilerEmitsPrewarmWithSliceID verifies that when the desired state
// targets a specific MIG slice and the heartbeat reports that slice is empty,
// the reconciler emits a PREWARM action carrying the correct SliceID.
//
// This exercises the full path: heartbeat → applyHeartbeat (writes MIGSlices to
// registry) → reconcile tick → reconcileMIGNode → action channel.
func TestMIGReconcilerEmitsPrewarmWithSliceID(t *testing.T) {
	reg := registry.NewNodeRegistry()
	heartbeats := make(chan reconciler.HeartbeatMsg, 1)
	actions := make(chan reconciler.ReconcileAction, 8)

	rec := reconciler.NewReconciler(reg, heartbeats, actions)
	// Scheduler decided model-b should live on slice "1/0/0".
	rec.SetDesired("mig-node", "gpu-0", "1/0/0", []string{"model-b"})

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go rec.Run(ctx)

	// Heartbeat reports MIG-enabled node; slice "1/0/0" has no models loaded yet.
	heartbeats <- reconciler.HeartbeatMsg{
		NodeID:       "mig-node",
		GPUID:        "gpu-0",
		TotalVRAMMiB: 61_440,
		UsedVRAMMiB:  0,
		MIGEnabled:   true,
		MIGSlices: []registry.MIGSlice{
			{SliceID: "0/0/0", TotalVRAMMiB: 20_480, UsedVRAMMiB: 0, LoadedModels: []string{}, Healthy: true},
			{SliceID: "1/0/0", TotalVRAMMiB: 20_480, UsedVRAMMiB: 0, LoadedModels: []string{}, Healthy: true},
			{SliceID: "2/0/0", TotalVRAMMiB: 20_480, UsedVRAMMiB: 0, LoadedModels: []string{}, Healthy: true},
		},
	}

	deadline := time.After(2 * time.Second)
	for {
		select {
		case act := <-actions:
			if act.Type == reconciler.ActionPrewarm &&
				act.ModelName == "model-b" &&
				act.SliceID == "1/0/0" {
				return // correct action with correct slice
			}
		case <-deadline:
			t.Fatal("timed out waiting for PREWARM(model-b, slice=1/0/0)")
		}
	}
}

// TestMIGReconcilerEmitsEvictForUnwantedModelOnSlice verifies that a model
// running on a slice that the scheduler has NOT assigned it to gets evicted.
// This covers the drift-correction path for MIG nodes.
func TestMIGReconcilerEmitsEvictForUnwantedModelOnSlice(t *testing.T) {
	reg := registry.NewNodeRegistry()
	heartbeats := make(chan reconciler.HeartbeatMsg, 1)
	actions := make(chan reconciler.ReconcileAction, 8)

	rec := reconciler.NewReconciler(reg, heartbeats, actions)
	// Desired: slice "0/0/0" should have only "model-c". No desired state for slice "1/0/0".
	rec.SetDesired("mig-node", "gpu-0", "0/0/0", []string{"model-c"})

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go rec.Run(ctx)

	// Actual: slice "0/0/0" has model-c (correct) AND model-d (unwanted).
	heartbeats <- reconciler.HeartbeatMsg{
		NodeID:       "mig-node",
		GPUID:        "gpu-0",
		TotalVRAMMiB: 40_960,
		MIGEnabled:   true,
		MIGSlices: []registry.MIGSlice{
			{
				SliceID:      "0/0/0",
				TotalVRAMMiB: 20_480,
				UsedVRAMMiB:  12_000,
				LoadedModels: []string{"model-c", "model-d"}, // model-d is the intruder
				Healthy:      true,
			},
		},
	}

	deadline := time.After(2 * time.Second)
	for {
		select {
		case act := <-actions:
			if act.Type == reconciler.ActionEvict &&
				act.ModelName == "model-d" &&
				act.SliceID == "0/0/0" {
				return
			}
		case <-deadline:
			t.Fatal("timed out waiting for EVICT(model-d, slice=0/0/0)")
		}
	}
}

// TestMIGHeartbeatPropagatesSliceStateToRegistry confirms that after a MIG
// heartbeat is processed, the registry reflects the per-slice state accurately.
// Other components (e.g. BinPacker, TrafficAnalyzer) depend on this being correct.
func TestMIGHeartbeatPropagatesSliceStateToRegistry(t *testing.T) {
	reg := registry.NewNodeRegistry()
	heartbeats := make(chan reconciler.HeartbeatMsg, 1)
	actions := make(chan reconciler.ReconcileAction, 8)

	rec := reconciler.NewReconciler(reg, heartbeats, actions)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go rec.Run(ctx)

	heartbeats <- reconciler.HeartbeatMsg{
		NodeID:       "mig-node",
		GPUID:        "gpu-0",
		TotalVRAMMiB: 40_960,
		UsedVRAMMiB:  15_000,
		MIGEnabled:   true,
		MIGSlices: []registry.MIGSlice{
			{SliceID: "0/0/0", TotalVRAMMiB: 20_480, UsedVRAMMiB: 8_000, Healthy: true},
			{SliceID: "1/0/0", TotalVRAMMiB: 20_480, UsedVRAMMiB: 7_000, Healthy: true},
		},
	}

	// Give the reconciler time to process the heartbeat.
	time.Sleep(100 * time.Millisecond)

	node, ok := reg.Get("mig-node", "gpu-0")
	if !ok {
		t.Fatal("node not found in registry after heartbeat")
	}
	if !node.MIGEnabled {
		t.Fatal("expected MIGEnabled=true on registry node after heartbeat")
	}
	if len(node.MIGSlices) != 2 {
		t.Fatalf("expected 2 MIG slices in registry, got %d", len(node.MIGSlices))
	}
	// Verify slice-level VRAM was preserved.
	for _, s := range node.MIGSlices {
		switch s.SliceID {
		case "0/0/0":
			if s.UsedVRAMMiB != 8_000 {
				t.Errorf("slice 0/0/0: expected UsedVRAMMiB=8000, got %d", s.UsedVRAMMiB)
			}
		case "1/0/0":
			if s.UsedVRAMMiB != 7_000 {
				t.Errorf("slice 1/0/0: expected UsedVRAMMiB=7000, got %d", s.UsedVRAMMiB)
			}
		}
	}
}
