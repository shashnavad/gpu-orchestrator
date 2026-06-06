package unit

import (
	"testing"

	"gpu-orchestrator/cache"
	"gpu-orchestrator/registry"
	schedulerpkg "gpu-orchestrator/scheduler"
)

// ── Existing tests (unchanged, kept to verify no regression) ─────────────

func TestModelCacheRegisterEvictAndAffinity(t *testing.T) {
	c := cache.NewModelCache()
	c.Register("node-a", "gpu-0", "llama-3-8b", 8192, 180)

	warm := c.WarmNodes("llama-3-8b")
	if len(warm) != 1 {
		t.Fatalf("expected 1 warm node, got %d", len(warm))
	}
	if !c.ShouldPreferAffinity("llama-3-8b") {
		t.Fatalf("expected affinity preference for high download duration")
	}
	c.Evict("node-a", "gpu-0", "llama-3-8b")
	if got := len(c.WarmNodes("llama-3-8b")); got != 0 {
		t.Fatalf("expected model cache to be empty after eviction, got %d entries", got)
	}
}

func TestBinPackerPrefersAffinityInPackMode(t *testing.T) {
	reg := registry.NewNodeRegistry()
	reg.Upsert(&registry.GPUNode{
		NodeID:              "node-a",
		GPUID:               "gpu-0",
		TotalVRAMMiB:        1000,
		UsedVRAMMiB:         100,
		ModelWeightAffinity: map[string]int64{"model-x": 50},
	})
	reg.Upsert(&registry.GPUNode{
		NodeID:              "node-b",
		GPUID:               "gpu-0",
		TotalVRAMMiB:        1000,
		UsedVRAMMiB:         150,
		ModelWeightAffinity: map[string]int64{},
	})

	packer := schedulerpkg.NewBinPacker(reg)
	decision, err := packer.Schedule(schedulerpkg.ScheduleRequest{
		ModelName:     "model-x",
		VRAMNeededMiB: 100,
		Priority:      schedulerpkg.P0Production,
	})
	if err != nil {
		t.Fatalf("unexpected schedule error: %v", err)
	}
	if decision.NodeID != "node-a" || !decision.AffinityHit {
		t.Fatalf("expected affinity hit on node-a, got node=%s affinity=%v", decision.NodeID, decision.AffinityHit)
	}
}

func TestBinPackerSpreadsWhenThresholdExceeded(t *testing.T) {
	reg := registry.NewNodeRegistry()
	reg.Upsert(&registry.GPUNode{
		NodeID:              "hot-node",
		GPUID:               "gpu-0",
		TotalVRAMMiB:        1000,
		UsedVRAMMiB:         900,
		ModelWeightAffinity: map[string]int64{},
	})
	reg.Upsert(&registry.GPUNode{
		NodeID:              "cool-node",
		GPUID:               "gpu-0",
		TotalVRAMMiB:        1000,
		UsedVRAMMiB:         300,
		ModelWeightAffinity: map[string]int64{},
	})

	packer := schedulerpkg.NewBinPacker(reg)
	decision, err := packer.Schedule(schedulerpkg.ScheduleRequest{
		ModelName:     "model-y",
		VRAMNeededMiB: 50,
		Priority:      schedulerpkg.P0Production,
	})
	if err != nil {
		t.Fatalf("unexpected schedule error: %v", err)
	}
	if decision.NodeID != "cool-node" {
		t.Fatalf("expected spread mode to choose cool-node, got %s", decision.NodeID)
	}
}

// ── MIG-specific tests ─────────────────────────────────────────────────────

// TestMIGSliceSelectionBinPack verifies that on a MIG-enabled node with three
// 20 GiB slices the scheduler picks the tightest-fit slice (bin-pack mode),
// not the first or the largest.
//
// Setup: slice-0 has 18 GiB free, slice-1 has 12 GiB free, slice-2 has 8 GiB free.
// Request: 7 GiB. Tightest fit that still accommodates 7 GiB = slice-2 (8 GiB free).
func TestMIGSliceSelectionBinPack(t *testing.T) {
	reg := registry.NewNodeRegistry()
	reg.Upsert(&registry.GPUNode{
		NodeID:     "mig-node",
		GPUID:      "gpu-0",
		MIGEnabled: true,
		MIGSlices: []registry.MIGSlice{
			{SliceID: "0/0/0", Profile: registry.MIGProfile2g20gb, TotalVRAMMiB: 20_480, UsedVRAMMiB: 2_048, Healthy: true},
			{SliceID: "1/0/0", Profile: registry.MIGProfile2g20gb, TotalVRAMMiB: 20_480, UsedVRAMMiB: 8_192, Healthy: true},
			{SliceID: "2/0/0", Profile: registry.MIGProfile2g20gb, TotalVRAMMiB: 20_480, UsedVRAMMiB: 12_288, Healthy: true},
		},
		ModelWeightAffinity: map[string]int64{},
		TotalVRAMMiB:        61_440,
		UsedVRAMMiB:         22_528,
	})

	packer := schedulerpkg.NewBinPacker(reg)
	decision, err := packer.Schedule(schedulerpkg.ScheduleRequest{
		ModelName:     "phi-3-mini",
		VRAMNeededMiB: 7_168, // 7 GiB
		Priority:      schedulerpkg.P0Production,
	})
	if err != nil {
		t.Fatalf("schedule failed: %v", err)
	}
	if !decision.MIGEnabled {
		t.Fatalf("expected MIGEnabled=true in decision")
	}
	// slice-2 has 20480-12288=8192 MiB free — smallest free that fits 7168.
	if decision.SliceID != "2/0/0" {
		t.Fatalf("expected tightest-fit slice 2/0/0, got %s", decision.SliceID)
	}
}

// TestMIGSliceSelectionSpread verifies that when a slice exceeds the 85% eviction
// threshold the scheduler switches to spread mode and picks the most-free slice.
//
// Setup: slice-0 is at 90% (saturated), slice-1 is at 30% (most free).
// Spread mode should pick slice-1.
func TestMIGSliceSelectionSpread(t *testing.T) {
	const total = int64(20_480)
	reg := registry.NewNodeRegistry()
	reg.Upsert(&registry.GPUNode{
		NodeID:     "mig-node",
		GPUID:      "gpu-0",
		MIGEnabled: true,
		MIGSlices: []registry.MIGSlice{
			// 90% full — triggers spread mode.
			{SliceID: "0/0/0", Profile: registry.MIGProfile2g20gb, TotalVRAMMiB: total, UsedVRAMMiB: 18_432, Healthy: true},
			// 30% full — most free.
			{SliceID: "1/0/0", Profile: registry.MIGProfile2g20gb, TotalVRAMMiB: total, UsedVRAMMiB: 6_144, Healthy: true},
		},
		ModelWeightAffinity: map[string]int64{},
		TotalVRAMMiB:        total * 2,
		UsedVRAMMiB:         18_432 + 6_144,
	})

	packer := schedulerpkg.NewBinPacker(reg)
	decision, err := packer.Schedule(schedulerpkg.ScheduleRequest{
		ModelName:     "llama-3-8b",
		VRAMNeededMiB: 4_096,
		Priority:      schedulerpkg.P0Production,
	})
	if err != nil {
		t.Fatalf("schedule failed: %v", err)
	}
	if decision.SliceID != "1/0/0" {
		t.Fatalf("spread mode should pick most-free slice 1/0/0, got %s", decision.SliceID)
	}
}

// TestMIGSliceExhaustedReturnsError verifies that when every slice is too full
// to fit the request, Schedule returns an error rather than an invalid decision.
func TestMIGSliceExhaustedReturnsError(t *testing.T) {
	reg := registry.NewNodeRegistry()
	reg.Upsert(&registry.GPUNode{
		NodeID:     "full-mig-node",
		GPUID:      "gpu-0",
		MIGEnabled: true,
		MIGSlices: []registry.MIGSlice{
			{SliceID: "0/0/0", TotalVRAMMiB: 10_240, UsedVRAMMiB: 9_216, Healthy: true},
			{SliceID: "1/0/0", TotalVRAMMiB: 10_240, UsedVRAMMiB: 9_216, Healthy: true},
		},
		TotalVRAMMiB: 20_480,
		UsedVRAMMiB:  18_432,
	})

	packer := schedulerpkg.NewBinPacker(reg)
	_, err := packer.Schedule(schedulerpkg.ScheduleRequest{
		ModelName:     "llama-3-70b",
		VRAMNeededMiB: 40_960, // 40 GiB — impossible on any slice
		Priority:      schedulerpkg.P0Production,
	})
	if err == nil {
		t.Fatal("expected error when no slice has sufficient VRAM, got nil")
	}
}

// TestMIGUnhealthySliceSkipped verifies that a slice marked Healthy=false is
// never selected even if it would otherwise be the tightest fit.
func TestMIGUnhealthySliceSkipped(t *testing.T) {
	reg := registry.NewNodeRegistry()
	reg.Upsert(&registry.GPUNode{
		NodeID:     "mig-node",
		GPUID:      "gpu-0",
		MIGEnabled: true,
		MIGSlices: []registry.MIGSlice{
			// Best fit but unhealthy — must be skipped.
			{SliceID: "0/0/0", TotalVRAMMiB: 20_480, UsedVRAMMiB: 14_336, Healthy: false},
			// Only healthy slice.
			{SliceID: "1/0/0", TotalVRAMMiB: 20_480, UsedVRAMMiB: 2_048, Healthy: true},
		},
		TotalVRAMMiB: 40_960,
		UsedVRAMMiB:  16_384,
	})

	packer := schedulerpkg.NewBinPacker(reg)
	decision, err := packer.Schedule(schedulerpkg.ScheduleRequest{
		ModelName:     "model-z",
		VRAMNeededMiB: 4_096,
		Priority:      schedulerpkg.P0Production,
	})
	if err != nil {
		t.Fatalf("schedule failed: %v", err)
	}
	if decision.SliceID != "1/0/0" {
		t.Fatalf("expected healthy slice 1/0/0, got %s", decision.SliceID)
	}
}

// TestNonMIGNodeDecisionHasNoSliceID verifies the backward-compat contract:
// non-MIG nodes produce a ScheduleDecision with MIGEnabled=false and SliceID="".
func TestNonMIGNodeDecisionHasNoSliceID(t *testing.T) {
	reg := registry.NewNodeRegistry()
	reg.Upsert(&registry.GPUNode{
		NodeID:              "plain-node",
		GPUID:               "gpu-0",
		TotalVRAMMiB:        80_000,
		UsedVRAMMiB:         10_000,
		MIGEnabled:          false,
		ModelWeightAffinity: map[string]int64{},
	})

	packer := schedulerpkg.NewBinPacker(reg)
	decision, err := packer.Schedule(schedulerpkg.ScheduleRequest{
		ModelName:     "model-q",
		VRAMNeededMiB: 8_000,
		Priority:      schedulerpkg.P0Production,
	})
	if err != nil {
		t.Fatalf("schedule failed: %v", err)
	}
	if decision.MIGEnabled {
		t.Fatal("expected MIGEnabled=false for non-MIG node")
	}
	if decision.SliceID != "" {
		t.Fatalf("expected empty SliceID for non-MIG node, got %q", decision.SliceID)
	}
}

// TestMIGNodeFreeVRAMIsMaxAcrossSlices checks that GPUNode.FreeVRAM() returns
// the largest free slice — the relevant unit for scheduling feasibility checks.
func TestMIGNodeFreeVRAMIsMaxAcrossSlices(t *testing.T) {
	node := &registry.GPUNode{
		MIGEnabled: true,
		MIGSlices: []registry.MIGSlice{
			{TotalVRAMMiB: 20_480, UsedVRAMMiB: 18_000, Healthy: true},  // 2480 free
			{TotalVRAMMiB: 20_480, UsedVRAMMiB: 5_000, Healthy: true},   // 15480 free
			{TotalVRAMMiB: 20_480, UsedVRAMMiB: 10_000, Healthy: false},  // unhealthy, skip
		},
	}
	got := node.FreeVRAM()
	want := int64(20_480 - 5_000) // 15480
	if got != want {
		t.Fatalf("FreeVRAM() = %d, want %d", got, want)
	}
}
