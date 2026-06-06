package registry

import (
	"sync"
	"time"
)

// MIGProfile names the NVIDIA partition size.
// These match the nvidia-smi MIG profile naming convention exactly.
// A100-40GB supports up to 7x 1g.5gb slices.
// H100-80GB supports up to 7x 1g.10gb slices.
//
// The scheduler only needs to know VRAMMiB and ComputeFraction;
// the profile name is carried through for logging and agent commands.
type MIGProfile string

const (
	MIGProfile7g80gb MIGProfile = "7g.80gb" // full GPU — no partitioning
	MIGProfile4g40gb MIGProfile = "4g.40gb"
	MIGProfile3g40gb MIGProfile = "3g.40gb"
	MIGProfile2g20gb MIGProfile = "2g.20gb"
	MIGProfile1g10gb MIGProfile = "1g.10gb"
	MIGProfile1g5gb  MIGProfile = "1g.5gb"  // A100-40GB smallest slice
)

// MIGSlice represents a single GPU Instance on a MIG-enabled GPU.
// Each slice has isolated VRAM and a fraction of compute cores.
// Isolation is hardware-enforced: one model's OOM cannot spill into another.
//
// When MIG is disabled, the GPUNode has a single synthetic MIGSlice that
// covers the whole GPU (SliceID = "full", Profile = MIGProfile7g80gb).
// This lets the scheduler use one code path for both MIG and non-MIG nodes.
type MIGSlice struct {
	SliceID      string     `json:"slice_id"`
	Profile      MIGProfile `json:"profile"`
	TotalVRAMMiB int64      `json:"total_vram_mib"`
	UsedVRAMMiB  int64      `json:"used_vram_mib"`
	LoadedModels []string   `json:"loaded_models"`
	Healthy      bool       `json:"healthy"`
}

// FreeVRAM returns headroom remaining in this slice.
func (s *MIGSlice) FreeVRAM() int64 {
	return s.TotalVRAMMiB - s.UsedVRAMMiB
}

// VRAMUtilizationPct returns utilization as a percentage, safe against zero-total.
func (s *MIGSlice) VRAMUtilizationPct() float64 {
	if s.TotalVRAMMiB == 0 {
		return 0
	}
	return float64(s.UsedVRAMMiB) / float64(s.TotalVRAMMiB) * 100.0
}

// GPUNode represents a single physical GPU within a node.
// VRAM is tracked in MiB. CUDACores is used for fractional-sharing decisions.
type GPUNode struct {
	NodeID        string
	GPUID         string // e.g. "gpu-0", "gpu-1"
	TotalVRAMMiB  int64
	UsedVRAMMiB   int64
	CUDACores     int
	NVLinkEnabled bool
	MIGEnabled    bool

	// MIGSlices is non-empty only when MIGEnabled is true.
	// The scheduler reads this to make sub-GPU placement decisions.
	// When MIGEnabled is false, the scheduler falls back to node-level VRAM.
	MIGSlices []MIGSlice

	// ModelWeightAffinity tracks which model weight files are already on the
	// node's local NVMe cache. Key = model name, value = size in MiB.
	// Affinity is node-scoped (NVMe is shared across MIG slices on one host).
	ModelWeightAffinity map[string]int64

	// LoadedModels are models currently resident in GPU memory across all slices.
	LoadedModels []string

	LastHeartbeat time.Time
	Healthy       bool
}

// FreeVRAM returns the largest contiguous free VRAM available on this node.
// For MIG nodes, this is the maximum free VRAM across all healthy slices —
// because a single model goes onto one slice; it cannot span slices.
// For non-MIG nodes, this is node-level headroom as before.
func (g *GPUNode) FreeVRAM() int64 {
	if !g.MIGEnabled || len(g.MIGSlices) == 0 {
		return g.TotalVRAMMiB - g.UsedVRAMMiB
	}
	var max int64
	for i := range g.MIGSlices {
		if g.MIGSlices[i].Healthy {
			if free := g.MIGSlices[i].FreeVRAM(); free > max {
				max = free
			}
		}
	}
	return max
}

// VRAMUtilizationPct returns current VRAM usage as a percentage.
// For MIG nodes this is the average utilization across slices, which gives
// the bin-packer a meaningful signal for eviction threshold decisions.
func (g *GPUNode) VRAMUtilizationPct() float64 {
	if !g.MIGEnabled || len(g.MIGSlices) == 0 {
		if g.TotalVRAMMiB == 0 {
			return 0
		}
		return float64(g.UsedVRAMMiB) / float64(g.TotalVRAMMiB) * 100.0
	}
	var totalUsed, totalCapacity int64
	for i := range g.MIGSlices {
		if g.MIGSlices[i].Healthy {
			totalUsed += g.MIGSlices[i].UsedVRAMMiB
			totalCapacity += g.MIGSlices[i].TotalVRAMMiB
		}
	}
	if totalCapacity == 0 {
		return 0
	}
	return float64(totalUsed) / float64(totalCapacity) * 100.0
}

// HasWeights returns true if the model weights are already cached locally.
func (g *GPUNode) HasWeights(modelName string) bool {
	_, ok := g.ModelWeightAffinity[modelName]
	return ok
}

// BestSliceFor returns the MIG slice that is the tightest fit for neededMiB
// (bin-pack mode) or the most free slice (spread mode), depending on spreadMode.
// Returns nil if MIG is disabled or no slice can fit the request.
// The caller falls back to node-level logic when nil is returned.
func (g *GPUNode) BestSliceFor(neededMiB int64, spreadMode bool) *MIGSlice {
	if !g.MIGEnabled || len(g.MIGSlices) == 0 {
		return nil
	}

	var best *MIGSlice
	for i := range g.MIGSlices {
		s := &g.MIGSlices[i]
		if !s.Healthy || s.FreeVRAM() < neededMiB {
			continue
		}
		if best == nil {
			best = s
			continue
		}
		if spreadMode {
			// Spread: pick slice with most free VRAM.
			if s.FreeVRAM() > best.FreeVRAM() {
				best = s
			}
		} else {
			// Bin-pack: pick slice with least free VRAM that still fits.
			if s.FreeVRAM() < best.FreeVRAM() {
				best = s
			}
		}
	}
	return best
}

// SyntheticFullSlice returns a single MIGSlice covering the whole GPU.
// Used internally so non-MIG code paths always have a slice to reason about.
func (g *GPUNode) SyntheticFullSlice() MIGSlice {
	return MIGSlice{
		SliceID:      "full",
		Profile:      MIGProfile7g80gb,
		TotalVRAMMiB: g.TotalVRAMMiB,
		UsedVRAMMiB:  g.UsedVRAMMiB,
		LoadedModels: g.LoadedModels,
		Healthy:      g.Healthy,
	}
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

// StaleNodes returns healthy nodes whose last heartbeat is older than the threshold.
// Nodes already marked unhealthy are skipped — they have already been acted on
// and would otherwise re-fire MARK_DEAD on every reconcile tick until a fresh
// heartbeat arrives and Upsert restores them to Healthy=true.
func (r *NodeRegistry) StaleNodes(threshold time.Duration) []*GPUNode {
	r.mu.RLock()
	defer r.mu.RUnlock()
	var stale []*GPUNode
	cutoff := time.Now().Add(-threshold)
	for _, n := range r.nodes {
		if n.Healthy && n.LastHeartbeat.Before(cutoff) {
			stale = append(stale, n)
		}
	}
	return stale
}
