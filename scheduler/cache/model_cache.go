package cache

import (
	"sync"
	"time"
)

// ModelCacheEntry records when model weights were last used on a node.
// The scheduler uses this to prefer nodes where weights are already warm.
type ModelCacheEntry struct {
	NodeID      string
	GPUID       string
	ModelName   string
	SizeMiB     int64
	LastUsed    time.Time
	// DownloadDurationSec is the measured time it took to download these weights.
	// Used in the affinity scoring: if downloading takes > AffinityWaitThreshold,
	// we strongly prefer the cached node even if it's more loaded.
	DownloadDurationSec float64
}

// AffinityWaitThresholdSec is the break-even point.
// If downloading weights would take longer than this, always prefer affinity.
// 120s = we'd rather wait up to 2 minutes on a hot GPU than do a cold download.
const AffinityWaitThresholdSec = 120.0

// ModelCache is the scheduler's awareness of which weights live on which node.
// It is updated by two sources:
//  1. Rust agent heartbeats (via Reconciler) — reports NVMe cache contents
//  2. Scheduler decisions — when the scheduler sends a PREWARM, it registers the intent
type ModelCache struct {
	mu      sync.RWMutex
	entries map[string][]*ModelCacheEntry // key = modelName
}

func NewModelCache() *ModelCache {
	return &ModelCache{
		entries: make(map[string][]*ModelCacheEntry),
	}
}

// Register records that a model's weights exist on a specific node's NVMe.
// Called when the Rust agent heartbeat reports a weight affinity update.
func (c *ModelCache) Register(nodeID, gpuID, modelName string, sizeMiB int64, downloadDur float64) {
	c.mu.Lock()
	defer c.mu.Unlock()

	key := modelName
	for _, e := range c.entries[key] {
		if e.NodeID == nodeID && e.GPUID == gpuID {
			// Update existing entry.
			e.LastUsed = time.Now()
			e.DownloadDurationSec = downloadDur
			return
		}
	}
	c.entries[key] = append(c.entries[key], &ModelCacheEntry{
		NodeID:              nodeID,
		GPUID:               gpuID,
		ModelName:           modelName,
		SizeMiB:             sizeMiB,
		LastUsed:            time.Now(),
		DownloadDurationSec: downloadDur,
	})
}

// Evict removes a model's cache record for a specific node.
// Called when the Reconciler confirms the weights have been freed from NVMe.
func (c *ModelCache) Evict(nodeID, gpuID, modelName string) {
	c.mu.Lock()
	defer c.mu.Unlock()

	entries := c.entries[modelName]
	filtered := entries[:0]
	for _, e := range entries {
		if e.NodeID != nodeID || e.GPUID != gpuID {
			filtered = append(filtered, e)
		}
	}
	c.entries[modelName] = filtered
}

// WarmNodes returns all nodes that already have the given model's weights cached.
// The scheduler calls this before making a placement decision.
func (c *ModelCache) WarmNodes(modelName string) []*ModelCacheEntry {
	c.mu.RLock()
	defer c.mu.RUnlock()

	result := make([]*ModelCacheEntry, len(c.entries[modelName]))
	copy(result, c.entries[modelName])
	return result
}

// ShouldPreferAffinity returns true when the download cost is high enough that
// we should wait for a cached node rather than use a cold one immediately.
// This encodes the key design decision: "waiting 2s for a hot GPU beats
// downloading 50GB from S3."
func (c *ModelCache) ShouldPreferAffinity(modelName string) bool {
	c.mu.RLock()
	defer c.mu.RUnlock()

	for _, e := range c.entries[modelName] {
		if e.DownloadDurationSec >= AffinityWaitThresholdSec {
			return true
		}
	}
	return false
}

// TouchNode updates the LastUsed timestamp for a model on a node.
// Called after a successful inference to keep LRU data fresh.
func (c *ModelCache) TouchNode(nodeID, gpuID, modelName string) {
	c.mu.Lock()
	defer c.mu.Unlock()

	for _, e := range c.entries[modelName] {
		if e.NodeID == nodeID && e.GPUID == gpuID {
			e.LastUsed = time.Now()
			return
		}
	}
}
