package registry

import (
	"sync"
	"time"
)

// GPUNode represents a single physical GPU within a node.
// VRAM is tracked in MiB. CUDACores is used for fractional-sharing decisions.
type GPUNode struct {
	NodeID        string
	GPUID         string // e.g. "gpu-0", "gpu-1"
	TotalVRAMMiB  int64
	UsedVRAMMiB   int64
	CUDACores     int
	NVLinkEnabled bool

	// ModelWeightAffinity tracks which model weight files are already on the
	// node's local NVMe cache. Key = model name, value = size in MiB.
	ModelWeightAffinity map[string]int64

	// LoadedModels are models currently resident in GPU memory.
	LoadedModels []string

	LastHeartbeat time.Time
	Healthy       bool
}

// FreeVRAM returns available VRAM headroom in MiB.
func (g *GPUNode) FreeVRAM() int64 {
	return g.TotalVRAMMiB - g.UsedVRAMMiB
}

// VRAMUtilizationPct returns current VRAM usage as a percentage.
func (g *GPUNode) VRAMUtilizationPct() float64 {
	if g.TotalVRAMMiB == 0 {
		return 0
	}
	return float64(g.UsedVRAMMiB) / float64(g.TotalVRAMMiB) * 100.0
}

// HasWeights returns true if the model weights are already cached locally.
func (g *GPUNode) HasWeights(modelName string) bool {
	_, ok := g.ModelWeightAffinity[modelName]
	return ok
}

// NodeRegistry is the single source of truth for cluster state.
// The Go scheduler is centralized — one NodeRegistry, one process.
type NodeRegistry struct {
	mu    sync.RWMutex
	nodes map[string]*GPUNode // key = NodeID+GPUID composite
}

func NewNodeRegistry() *NodeRegistry {
	return &NodeRegistry{
		nodes: make(map[string]*GPUNode),
	}
}

// Upsert inserts or updates a GPUNode. Called on every heartbeat from the Rust agent.
func (r *NodeRegistry) Upsert(node *GPUNode) {
	r.mu.Lock()
	defer r.mu.Unlock()
	key := node.NodeID + "/" + node.GPUID
	node.Healthy = true
	r.nodes[key] = node
}

// MarkUnhealthy flags a node as unhealthy when heartbeats stop.
func (r *NodeRegistry) MarkUnhealthy(nodeID, gpuID string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	key := nodeID + "/" + gpuID
	if n, ok := r.nodes[key]; ok {
		n.Healthy = false
	}
}

// All returns a snapshot of all healthy GPU nodes.
func (r *NodeRegistry) All() []*GPUNode {
	r.mu.RLock()
	defer r.mu.RUnlock()
	out := make([]*GPUNode, 0, len(r.nodes))
	for _, n := range r.nodes {
		if n.Healthy {
			out = append(out, n)
		}
	}
	return out
}

// Get returns a specific GPU node by composite key.
func (r *NodeRegistry) Get(nodeID, gpuID string) (*GPUNode, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	n, ok := r.nodes[nodeID+"/"+gpuID]
	return n, ok
}

// StaleNodes returns nodes whose last heartbeat is older than the threshold.
// The Reconciler uses this to detect dead agents.
func (r *NodeRegistry) StaleNodes(threshold time.Duration) []*GPUNode {
	r.mu.RLock()
	defer r.mu.RUnlock()
	var stale []*GPUNode
	cutoff := time.Now().Add(-threshold)
	for _, n := range r.nodes {
		if n.LastHeartbeat.Before(cutoff) {
			stale = append(stale, n)
		}
	}
	return stale
}
