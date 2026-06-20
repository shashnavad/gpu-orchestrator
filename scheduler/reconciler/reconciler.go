package reconciler

import (
	"context"
	"log"
	"sync"
	"time"

	"gpu-orchestrator/registry"
)

// HeartbeatMsg is the message the Rust agent sends every 500ms.
// It represents the *actual* state of a GPU as observed by nvidia-smi (or mock).
//
// MIG change: MIGEnabled and MIGSlices carry per-slice state from the agent.
// When MIGEnabled is false, the scheduler treats the node as a single slice.
type HeartbeatMsg struct {
	NodeID              string              `json:"node_id"`
	GPUID               string              `json:"gpu_id"`
	UsedVRAMMiB         int64               `json:"used_vram_mib"`
	TotalVRAMMiB        int64               `json:"total_vram_mib"`
	LoadedModels        []string            `json:"loaded_models"`
	NVLinkEnabled       bool                `json:"nvlink_enabled"`
	MIGEnabled          bool                `json:"mig_enabled"`
	MIGSlices           []registry.MIGSlice `json:"mig_slices"`
	ModelWeightAffinity map[string]int64    `json:"model_weight_affinity"`
}

// Action is an instruction the reconciler emits when desired ≠ actual state.
type ActionType string

const (
	ActionEvict    ActionType = "EVICT"
	ActionPrewarm  ActionType = "PREWARM"
	ActionMarkDead ActionType = "MARK_DEAD"
)

// ReconcileAction now carries SliceID so the action dispatcher knows
// which MIG slice to target when calling the Rust agent.
type ReconcileAction struct {
	Type      ActionType
	NodeID    string
	GPUID     string
	SliceID   string // "" for non-MIG nodes
	ModelName string
}

// desiredSliceState captures what models should be on a given slice.
type desiredSliceState struct {
	models []string
}

// Reconciler is the anti-entropy loop. Every 500ms it compares the
// Desired State (what the Go scheduler wants) against the Actual State
// (what the Rust agent's heartbeats report) and emits corrective actions.
type Reconciler struct {
	registry   *registry.NodeRegistry
	heartbeats <-chan HeartbeatMsg
	actions    chan<- ReconcileAction
	mu         sync.RWMutex
	// desiredState key is "nodeID/gpuID/sliceID".
	// For non-MIG nodes sliceID is "full".
	desiredState map[string]desiredSliceState
	staleTimeout time.Duration
}

func NewReconciler(
	reg *registry.NodeRegistry,
	heartbeats <-chan HeartbeatMsg,
	actions chan<- ReconcileAction,
) *Reconciler {
	return &Reconciler{
		registry:     reg,
		heartbeats:   heartbeats,
		actions:      actions,
		desiredState: make(map[string]desiredSliceState),
		staleTimeout: 2 * time.Second,
	}
}

// SetDesired replaces the full desired model list for a node/gpu/slice. This
// wipes any model not present in `models` — correct for bulk initialization,
// unsafe for incremental scheduling decisions where another model may already
// be desired on the same slice. Use AddDesired/RemoveDesired for those.ls []string) {
func (r *Reconciler) SetDesired(nodeID, gpuID, sliceID string, models []string) {
	key := sliceKey(nodeID, gpuID, sliceID)

	r.mu.Lock()
	defer r.mu.Unlock()
	r.desiredState[key] = desiredSliceState{models: append([]string(nil), models...)}
}

// AddDesired adds modelName to the desired set for a slice without disturbing
// any other model already desired there. Idempotent.
func (r *Reconciler) AddDesired(nodeID, gpuID, sliceID, modelName string) {
	key := sliceKey(nodeID, gpuID, sliceID)
	r.mu.Lock()
	defer r.mu.Unlock()
	state := r.desiredState[key]
	for _, m := range state.models {
		if m == modelName {
			return
		}
	}
	state.models = append(state.models, modelName)
	r.desiredState[key] = state
}

// RemoveDesired removes modelName from the desired set for a slice, leaving
// any other desired model untouched.
func (r *Reconciler) RemoveDesired(nodeID, gpuID, sliceID, modelName string) {
	key := sliceKey(nodeID, gpuID, sliceID)
	r.mu.Lock()
	defer r.mu.Unlock()
	state, ok := r.desiredState[key]
	if !ok {
		return
	}
	out := state.models[:0]
	for _, m := range state.models {
		if m != modelName {
			out = append(out, m)
		}
	}
	state.models = out
	r.desiredState[key] = state
}

// snapshotDesired returns a copy of the desired model set for a slice,
// safe to use after the lock is released.
func (r *Reconciler) snapshotDesired(key string) (map[string]bool, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	state, ok := r.desiredState[key]
	if !ok {
		return nil, false
	}
	return toSet(state.models), true
}

// Run starts the reconciliation loop. Blocks until ctx is cancelled.
func (r *Reconciler) Run(ctx context.Context) {
	ticker := time.NewTicker(500 * time.Millisecond)
	defer ticker.Stop()

	log.Println("[reconciler] started")

	for {
		select {
		case hb, ok := <-r.heartbeats:
			if !ok {
				log.Println("[reconciler] heartbeat channel closed, shutting down")
				return
			}
			r.applyHeartbeat(hb)

		case <-ticker.C:
			r.reconcile()

		case <-ctx.Done():
			log.Println("[reconciler] context cancelled, shutting down")
			return
		}
	}
}

// applyHeartbeat updates the NodeRegistry with actual state from the Rust agent.
func (r *Reconciler) applyHeartbeat(hb HeartbeatMsg) {
	node := &registry.GPUNode{
		NodeID:              hb.NodeID,
		GPUID:               hb.GPUID,
		UsedVRAMMiB:         hb.UsedVRAMMiB,
		TotalVRAMMiB:        hb.TotalVRAMMiB,
		LoadedModels:        hb.LoadedModels,
		NVLinkEnabled:       hb.NVLinkEnabled,
		MIGEnabled:          hb.MIGEnabled,
		MIGSlices:           hb.MIGSlices,
		ModelWeightAffinity: hb.ModelWeightAffinity,
		LastHeartbeat:       time.Now(),
	}
	r.registry.Upsert(node)
}

// reconcile diffs desired vs actual state, emitting corrective actions.
// For MIG nodes it diffs at slice granularity; for non-MIG nodes at node level.
func (r *Reconciler) reconcile() {
	stale := r.registry.StaleNodes(r.staleTimeout)
	for _, n := range stale {
		log.Printf("[reconciler] node %s/%s missed heartbeats, marking dead", n.NodeID, n.GPUID)
		r.registry.MarkUnhealthy(n.NodeID, n.GPUID)
		r.actions <- ReconcileAction{
			Type:   ActionMarkDead,
			NodeID: n.NodeID,
			GPUID:  n.GPUID,
		}
	}

	for _, node := range r.registry.All() {
		if node.MIGEnabled && len(node.MIGSlices) > 0 {
			r.reconcileMIGNode(node)
		} else {
			r.reconcileNonMIGNode(node)
		}
	}
}

// reconcileMIGNode diffs at slice granularity.
func (r *Reconciler) reconcileMIGNode(node *registry.GPUNode) {
	for _, slice := range node.MIGSlices {
		if !slice.Healthy {
			continue
		}
		key := sliceKey(node.NodeID, node.GPUID, slice.SliceID)
		desiredSet, hasDesired := r.snapshotDesired(key)
		if !hasDesired {
			continue
		}

		actualSet := toSet(slice.LoadedModels)

		for model := range desiredSet {
			if !actualSet[model] {
				log.Printf("[reconciler] drift: %s/%s slice=%s should have %s loaded",
					node.NodeID, node.GPUID, slice.SliceID, model)
				r.actions <- ReconcileAction{
					Type:      ActionPrewarm,
					NodeID:    node.NodeID,
					GPUID:     node.GPUID,
					SliceID:   slice.SliceID,
					ModelName: model,
				}
			}
		}
		for model := range actualSet {
			if !desiredSet[model] {
				log.Printf("[reconciler] drift: %s/%s slice=%s has unwanted %s",
					node.NodeID, node.GPUID, slice.SliceID, model)
				r.actions <- ReconcileAction{
					Type:      ActionEvict,
					NodeID:    node.NodeID,
					GPUID:     node.GPUID,
					SliceID:   slice.SliceID,
					ModelName: model,
				}
			}
		}
	}
}

// reconcileNonMIGNode is the original node-level diff, unchanged in semantics.
func (r *Reconciler) reconcileNonMIGNode(node *registry.GPUNode) {
	key := sliceKey(node.NodeID, node.GPUID, "full")
	desiredSet, hasDesired := r.snapshotDesired(key)
	if !hasDesired {
		return
	}

	actualSet := toSet(node.LoadedModels)

	for model := range desiredSet {
		if !actualSet[model] {
			log.Printf("[reconciler] drift: %s/%s should have %s loaded", node.NodeID, node.GPUID, model)
			r.actions <- ReconcileAction{
				Type:      ActionPrewarm,
				NodeID:    node.NodeID,
				GPUID:     node.GPUID,
				ModelName: model,
			}
		}
	}
	for model := range actualSet {
		if !desiredSet[model] {
			log.Printf("[reconciler] drift: %s/%s has unwanted %s", node.NodeID, node.GPUID, model)
			r.actions <- ReconcileAction{
				Type:      ActionEvict,
				NodeID:    node.NodeID,
				GPUID:     node.GPUID,
				ModelName: model,
			}
		}
	}
}

func sliceKey(nodeID, gpuID, sliceID string) string {
	if sliceID == "" {
		sliceID = "full"
	}
	return nodeID + "/" + gpuID + "/" + sliceID
}

func toSet(ss []string) map[string]bool {
	m := make(map[string]bool, len(ss))
	for _, s := range ss {
		m[s] = true
	}
	return m
}
