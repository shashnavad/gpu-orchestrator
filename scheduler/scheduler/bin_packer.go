package scheduler

import (
	"container/heap"
	"errors"
	"fmt"
	"sort"

	"gpu-orchestrator/registry"
)

// Priority levels for workloads.
const (
	P0Production  = 0 // Cannot be interrupted — live inference
	P1Development = 1 // Can be paused if a P0 request arrives
	P2BatchTrain  = 2 // Spot/idle only
)

// VRAMEvictionThresholdPct triggers spread-mode when any GPU or slice exceeds this.
const VRAMEvictionThresholdPct = 85.0

// ScheduleRequest is a request to place a model workload on a GPU or MIG slice.
type ScheduleRequest struct {
	ModelName     string
	VRAMNeededMiB int64
	Priority      int
}

// ScheduleDecision is the output of the scheduler.
// SliceID is non-empty only when the target node is MIG-enabled;
// the Rust agent uses it to send the nvidia-smi MIG placement command.
type ScheduleDecision struct {
	NodeID      string
	GPUID       string
	SliceID     string // "" for non-MIG nodes; e.g. "0/0/0" for MIG slices
	MIGEnabled  bool
	AffinityHit bool // true if the model weights were already cached on this node
}

// placementCandidate is an internal structure that pairs a node with the
// specific slice (or synthetic full-GPU slice) chosen for this request.
// Using a unified candidate struct means the sorting and selection logic
// does not branch on MIG vs non-MIG — it always operates on slices.
type placementCandidate struct {
	node    *registry.GPUNode
	slice   registry.MIGSlice // always populated (synthetic for non-MIG nodes)
	affHit  bool
}

// BinPacker is a centralized scheduler that applies bin-packing by default and
// switches to spread mode when VRAM utilization exceeds VRAMEvictionThresholdPct.
//
// MIG awareness: when a node reports MIGEnabled=true and has populated MIGSlices,
// the scheduler picks the best individual slice rather than treating the whole
// GPU as a monolith. This is the key change for multi-tenant fractionalization:
// a single H100 can run llama-3-70b on a 4g.40gb slice while phi-3-mini runs
// on a separate 1g.10gb slice simultaneously, with hardware-enforced isolation.
type BinPacker struct {
	registry  *registry.NodeRegistry
	taskQueue *priorityQueue
}

func NewBinPacker(reg *registry.NodeRegistry) *BinPacker {
	pq := &priorityQueue{}
	heap.Init(pq)
	return &BinPacker{
		registry:  reg,
		taskQueue: pq,
	}
}

// expandNode turns one GPUNode into a list of placementCandidates — one per
// MIG slice for MIG-enabled nodes, or one synthetic candidate for non-MIG nodes.
// Only slices with free VRAM >= neededMiB are included.
func (b *BinPacker) expandNode(n *registry.GPUNode, neededMiB int64) []placementCandidate {
	affHit := n.HasWeights("") // affinity is node-scoped; checked per-request below
	_ = affHit                 // computed per-slice below using the real model name

	if !n.MIGEnabled || len(n.MIGSlices) == 0 {
		// Non-MIG: single synthetic candidate representing the whole GPU.
		syn := n.SyntheticFullSlice()
		if syn.FreeVRAM() < neededMiB {
			return nil
		}
		return []placementCandidate{{
			node:   n,
			slice:  syn,
			affHit: false, // caller sets this; expandNode doesn't know model name
		}}
	}

	var out []placementCandidate
	for _, s := range n.MIGSlices {
		if !s.Healthy || s.FreeVRAM() < neededMiB {
			continue
		}
		out = append(out, placementCandidate{node: n, slice: s, affHit: false})
	}
	return out
}

// Schedule needs affinity per-model, so we re-compute it after expansion.
// Refactored Schedule to pass model name into expansion properly:
func (b *BinPacker) expandNodeForModel(n *registry.GPUNode, neededMiB int64, modelName string) []placementCandidate {
	affHit := n.HasWeights(modelName)

	if !n.MIGEnabled || len(n.MIGSlices) == 0 {
		syn := n.SyntheticFullSlice()
		if syn.FreeVRAM() < neededMiB {
			return nil
		}
		return []placementCandidate{{node: n, slice: syn, affHit: affHit}}
	}

	var out []placementCandidate
	for _, s := range n.MIGSlices {
		if !s.Healthy || s.FreeVRAM() < neededMiB {
			continue
		}
		out = append(out, placementCandidate{node: n, slice: s, affHit: affHit})
	}
	return out
}

// Schedule selects the best GPU node (and MIG slice, if applicable) for the request.
//
// Selection algorithm:
//  1. Expand each node into placement candidates — one per MIG slice for MIG
//     nodes, one synthetic candidate for non-MIG nodes.
//  2. Filter by free VRAM >= VRAMNeededMiB.
//  3. Separate affinity hits (NVMe already has model weights) from cold candidates.
//  4. If any candidate slice exceeds VRAMEvictionThresholdPct, switch to spread
//     mode (most-free slice); otherwise bin-pack (least-fit slice).
//  5. Return chosen node + slice as ScheduleDecision.
func (b *BinPacker) Schedule(req ScheduleRequest) (*ScheduleDecision, error) {
	nodes := b.registry.All()
	if len(nodes) == 0 {
		return nil, errors.New("no healthy GPU nodes in registry")
	}

	var candidates []placementCandidate
	for _, n := range nodes {
		for _, c := range b.expandNodeForModel(n, req.VRAMNeededMiB, req.ModelName) {
			candidates = append(candidates, c)
		}
	}
	if len(candidates) == 0 {
		return nil, fmt.Errorf(
			"no GPU slice has %d MiB free VRAM for model %s",
			req.VRAMNeededMiB, req.ModelName,
		)
	}

	var affinityHits, cold []placementCandidate
	for _, c := range candidates {
		if c.affHit {
			affinityHits = append(affinityHits, c)
		} else {
			cold = append(cold, c)
		}
	}
	pool := affinityHits
	if len(pool) == 0 {
		pool = cold
	}

	spreadMode := b.anySliceAboveThreshold(candidates)

	var chosen *placementCandidate
	if spreadMode {
		chosen = b.mostFreeSlice(pool)
	} else {
		chosen = b.leastFitSlice(pool, req.VRAMNeededMiB)
	}
	if chosen == nil {
		return nil, fmt.Errorf("scheduler could not select a slice for model %s", req.ModelName)
	}

	sliceID := ""
	if chosen.node.MIGEnabled {
		sliceID = chosen.slice.SliceID
	}

	return &ScheduleDecision{
		NodeID:      chosen.node.NodeID,
		GPUID:       chosen.node.GPUID,
		SliceID:     sliceID,
		MIGEnabled:  chosen.node.MIGEnabled,
		AffinityHit: len(affinityHits) > 0,
	}, nil
}

// anySliceAboveThreshold returns true if any candidate slice is over the eviction threshold.
func (b *BinPacker) anySliceAboveThreshold(candidates []placementCandidate) bool {
	for _, c := range candidates {
		if c.slice.VRAMUtilizationPct() > VRAMEvictionThresholdPct {
			return true
		}
	}
	return false
}

// leastFitSlice returns the candidate whose slice has the smallest free VRAM
// that still fits neededMiB (tightest-fit bin-packing).
func (b *BinPacker) leastFitSlice(candidates []placementCandidate, neededMiB int64) *placementCandidate {
	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].slice.FreeVRAM() < candidates[j].slice.FreeVRAM()
	})
	for i := range candidates {
		if candidates[i].slice.FreeVRAM() >= neededMiB {
			return &candidates[i]
		}
	}
	return nil
}

// mostFreeSlice returns the candidate whose slice has the most free VRAM (spread mode).
func (b *BinPacker) mostFreeSlice(candidates []placementCandidate) *placementCandidate {
	if len(candidates) == 0 {
		return nil
	}
	best := &candidates[0]
	for i := range candidates[1:] {
		if candidates[i+1].slice.FreeVRAM() > best.slice.FreeVRAM() {
			best = &candidates[i+1]
		}
	}
	return best
}

// anyAboveThreshold returns true if any node-level VRAM utilization is over threshold.
// Kept for callers that operate at node granularity (e.g. health dashboards).
func (b *BinPacker) anyAboveThreshold(nodes []*registry.GPUNode) bool {
	for _, n := range nodes {
		if n.VRAMUtilizationPct() > VRAMEvictionThresholdPct {
			return true
		}
	}
	return false
}

// --- Priority Queue for preemption ---

type task struct {
	ModelName string
	Priority  int
	VRAMMiB   int64
	NodeID    string
	GPUID     string
	SliceID   string // MIG slice, "" if non-MIG
	index     int
}

type priorityQueue []*task

func (pq priorityQueue) Len() int { return len(pq) }

func (pq priorityQueue) Less(i, j int) bool {
	return pq[i].Priority > pq[j].Priority
}

func (pq priorityQueue) Swap(i, j int) {
	pq[i], pq[j] = pq[j], pq[i]
	pq[i].index = i
	pq[j].index = j
}

func (pq *priorityQueue) Push(x any) {
	t := x.(*task)
	t.index = len(*pq)
	*pq = append(*pq, t)
}

func (pq *priorityQueue) Pop() any {
	old := *pq
	n := len(old)
	t := old[n-1]
	old[n-1] = nil
	*pq = old[:n-1]
	return t
}

func (b *BinPacker) Evict() (*task, error) {
	if b.taskQueue.Len() == 0 {
		return nil, errors.New("no tasks available to evict")
	}
	t := heap.Pop(b.taskQueue).(*task)
	if t.Priority == P0Production {
		heap.Push(b.taskQueue, t)
		return nil, errors.New("cannot evict a P0 production task")
	}
	return t, nil
}

func (b *BinPacker) Enqueue(t *task) {
	heap.Push(b.taskQueue, t)
}
