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

// VRAMEvictionThresholdPct triggers spread-mode when any GPU exceeds this.
const VRAMEvictionThresholdPct = 85.0

// ScheduleRequest is a request to place a model workload on a GPU.
type ScheduleRequest struct {
	ModelName    string
	VRAMNeededMiB int64
	Priority     int
}

// ScheduleDecision is the output of the scheduler.
type ScheduleDecision struct {
	NodeID      string
	GPUID       string
	AffinityHit bool // true if the model weights were already cached on this node
}

// BinPacker is a centralized scheduler that applies bin-packing by default and
// switches to spread mode when VRAM utilization exceeds VRAMEvictionThresholdPct.
//
// Design decision: centralized paradigm is correct for clusters < 1,000 GPUs.
// A single process with a read lock on NodeRegistry gives us global optimum
// placement without the race conditions of a shared-state approach.
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

// Schedule selects the best GPU node for the given request.
//
// Selection algorithm (in order of priority):
//  1. Affinity: prefer nodes that already have the model weights on NVMe.
//     Waiting 2s for a hot-but-busy GPU beats downloading 50GB from S3.
//  2. Bin-pack: among affinity candidates, choose the node with the LEAST
//     free VRAM that can still fit the workload (fill GPUs to 100% before
//     moving on, enabling scale-to-zero on idle nodes).
//  3. Eviction check: if any candidate node is above the VRAM eviction
//     threshold, switch to spread mode and pick the node with MOST free VRAM.
func (b *BinPacker) Schedule(req ScheduleRequest) (*ScheduleDecision, error) {
	nodes := b.registry.All()
	if len(nodes) == 0 {
		return nil, errors.New("no healthy GPU nodes in registry")
	}

	// Filter: only nodes with enough free VRAM.
	var candidates []*registry.GPUNode
	for _, n := range nodes {
		if n.FreeVRAM() >= req.VRAMNeededMiB {
			candidates = append(candidates, n)
		}
	}
	if len(candidates) == 0 {
		return nil, fmt.Errorf("no GPU has %d MiB free VRAM for model %s", req.VRAMNeededMiB, req.ModelName)
	}

	// Check if any candidate is in eviction territory.
	spreadMode := b.anyAboveThreshold(candidates)

	// Separate affinity hits from cold candidates.
	var affinityHits, coldNodes []*registry.GPUNode
	for _, n := range candidates {
		if n.HasWeights(req.ModelName) {
			affinityHits = append(affinityHits, n)
		} else {
			coldNodes = append(coldNodes, n)
		}
	}

	// Prefer affinity candidates; fall back to cold nodes.
	pool := affinityHits
	if len(pool) == 0 {
		pool = coldNodes
	}
	affinityHit := len(affinityHits) > 0

	var chosen *registry.GPUNode
	if spreadMode {
		// Spread: pick the node with the MOST free VRAM to reduce OOM risk.
		chosen = b.mostFree(pool)
	} else {
		// Bin-pack: pick the node with the LEAST free VRAM that still fits,
		// so we fill GPUs fully and can power down empty nodes.
		chosen = b.leastFitFree(pool, req.VRAMNeededMiB)
	}

	if chosen == nil {
		return nil, fmt.Errorf("scheduler could not select a node for model %s", req.ModelName)
	}

	return &ScheduleDecision{
		NodeID:      chosen.NodeID,
		GPUID:       chosen.GPUID,
		AffinityHit: affinityHit,
	}, nil
}

// anyAboveThreshold returns true if any node has VRAM usage above the eviction threshold.
// This is the trigger to switch from bin-pack to spread mode.
func (b *BinPacker) anyAboveThreshold(nodes []*registry.GPUNode) bool {
	for _, n := range nodes {
		if n.VRAMUtilizationPct() > VRAMEvictionThresholdPct {
			return true
		}
	}
	return false
}

// leastFitFree returns the node with the smallest free VRAM that still fits the request.
// This is the bin-packing "tightest fit" heuristic.
func (b *BinPacker) leastFitFree(nodes []*registry.GPUNode, neededMiB int64) *registry.GPUNode {
	sort.Slice(nodes, func(i, j int) bool {
		return nodes[i].FreeVRAM() < nodes[j].FreeVRAM()
	})
	for _, n := range nodes {
		if n.FreeVRAM() >= neededMiB {
			return n
		}
	}
	return nil
}

// mostFree returns the node with the most free VRAM (spread mode).
func (b *BinPacker) mostFree(nodes []*registry.GPUNode) *registry.GPUNode {
	if len(nodes) == 0 {
		return nil
	}
	best := nodes[0]
	for _, n := range nodes[1:] {
		if n.FreeVRAM() > best.FreeVRAM() {
			best = n
		}
	}
	return best
}

// --- Priority Queue for preemption ---
// When a P0 request arrives and the cluster is full, the scheduler must find
// the cheapest P1 or P2 task to evict. Go's container/heap is ideal here.

type task struct {
	ModelName string
	Priority  int
	VRAMMiB   int64
	NodeID    string
	GPUID     string
	index     int
}

type priorityQueue []*task

func (pq priorityQueue) Len() int { return len(pq) }

// Lower priority number = higher urgency (P0 beats P1 beats P2).
// For eviction we want to pop the LOWEST urgency (highest Priority number).
func (pq priorityQueue) Less(i, j int) bool {
	return pq[i].Priority > pq[j].Priority // max-heap on priority number
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

// Evict removes the lowest-priority task from the queue to make room for a P0 request.
// Returns the evicted task so the caller can send a SIGTERM to the Rust agent.
func (b *BinPacker) Evict() (*task, error) {
	if b.taskQueue.Len() == 0 {
		return nil, errors.New("no tasks available to evict")
	}
	t := heap.Pop(b.taskQueue).(*task)
	if t.Priority == P0Production {
		// Should never happen — re-push and refuse.
		heap.Push(b.taskQueue, t)
		return nil, errors.New("cannot evict a P0 production task")
	}
	return t, nil
}

// Enqueue adds a new task to the priority queue.
func (b *BinPacker) Enqueue(t *task) {
	heap.Push(b.taskQueue, t)
}
