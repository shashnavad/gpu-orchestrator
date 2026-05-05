package unit

import (
	"testing"

	"gpu-orchestrator/cache"
	"gpu-orchestrator/registry"
	schedulerpkg "gpu-orchestrator/scheduler"
)

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
		UsedVRAMMiB:         900, // 90% triggers spread mode
		ModelWeightAffinity: map[string]int64{},
	})
	reg.Upsert(&registry.GPUNode{
		NodeID:              "cool-node",
		GPUID:               "gpu-0",
		TotalVRAMMiB:        1000,
		UsedVRAMMiB:         300, // most free VRAM in spread mode
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
