package reconciler

import (
	"context"
	"log"
	"time"

	"gpu-orchestrator/registry"
)

// HeartbeatMsg is the message the Rust agent sends every 500ms.
// It represents the *actual* state of a GPU as observed by nvidia-smi (or mock).
type HeartbeatMsg struct {
	NodeID        string
	GPUID         string
	UsedVRAMMiB   int64
	TotalVRAMMiB  int64
	LoadedModels  []string
	NVLinkEnabled bool
	// ModelWeightAffinity is updated incrementally: agent reports which weights
	// exist on its local NVMe cache.
	ModelWeightAffinity map[string]int64
}

// Action is an instruction the reconciler emits when desired ≠ actual state.
type ActionType string

const (
	ActionEvict   ActionType = "EVICT"
	ActionPrewarm ActionType = "PREWARM"
	ActionMarkDead ActionType = "MARK_DEAD"
)

type ReconcileAction struct {
	Type      ActionType
	NodeID    string
	GPUID     string
	ModelName string // relevant for EVICT and PREWARM
}

// Reconciler is the anti-entropy loop. Every 500ms it compares the
// Desired State (what the Go scheduler wants) against the Actual State
// (what the Rust agent's heartbeats report) and emits corrective actions.
//
// Design: the reconciler does NOT make scheduling decisions — that is the
// BinPacker's job. The reconciler only detects and corrects drift.
type Reconciler struct {
	registry     *registry.NodeRegistry
	heartbeats   <-chan HeartbeatMsg
	actions      chan<- ReconcileAction
	desiredState map[string][]string // nodeKey → list of model names that should be loaded
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
		desiredState: make(map[string][]string),
		staleTimeout: 2 * time.Second, // miss 4 heartbeats → node is dead
	}
}

// SetDesired tells the reconciler what models should be loaded on a given GPU.
// Called by the scheduler after a placement decision.
func (r *Reconciler) SetDesired(nodeID, gpuID string, models []string) {
	r.desiredState[nodeID+"/"+gpuID] = models
}

// Run starts the reconciliation loop. It blocks until ctx is cancelled.
// It uses a select statement to multiplex three inputs:
//  1. Incoming heartbeats from Rust agents (updates actual state)
//  2. A 500ms ticker (triggers the diff loop)
//  3. Context cancellation (clean shutdown)
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

// applyHeartbeat updates the NodeRegistry with the latest actual state from
// the Rust agent. This is the "write path" — heartbeats are the ground truth.
func (r *Reconciler) applyHeartbeat(hb HeartbeatMsg) {
	node := &registry.GPUNode{
		NodeID:              hb.NodeID,
		GPUID:               hb.GPUID,
		UsedVRAMMiB:         hb.UsedVRAMMiB,
		TotalVRAMMiB:        hb.TotalVRAMMiB,
		LoadedModels:        hb.LoadedModels,
		NVLinkEnabled:       hb.NVLinkEnabled,
		ModelWeightAffinity: hb.ModelWeightAffinity,
		LastHeartbeat:       time.Now(),
	}
	r.registry.Upsert(node)
}

// reconcile is the diff loop. It runs on every tick and compares:
//   - Desired state: what the scheduler decided should be loaded
//   - Actual state: what the Rust agent last reported
//
// Divergences produce ReconcileActions sent to the actions channel,
// which the API server (or a dedicated action handler) executes.
func (r *Reconciler) reconcile() {
	// 1. Check for stale nodes (missed heartbeats → treat as dead).
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

	// 2. Diff desired vs actual for each node.
	for _, node := range r.registry.All() {
		key := node.NodeID + "/" + node.GPUID
		desired, hasDesired := r.desiredState[key]
		if !hasDesired {
			continue
		}

		actualSet := toSet(node.LoadedModels)
		desiredSet := toSet(desired)

		// Models in desired but not in actual → need to be loaded (PREWARM).
		for model := range desiredSet {
			if !actualSet[model] {
				log.Printf("[reconciler] drift: %s should have %s loaded but doesn't", key, model)
				r.actions <- ReconcileAction{
					Type:      ActionPrewarm,
					NodeID:    node.NodeID,
					GPUID:     node.GPUID,
					ModelName: model,
				}
			}
		}

		// Models in actual but not in desired → should be evicted.
		for model := range actualSet {
			if !desiredSet[model] {
				log.Printf("[reconciler] drift: %s has %s loaded but shouldn't", key, model)
				r.actions <- ReconcileAction{
					Type:      ActionEvict,
					NodeID:    node.NodeID,
					GPUID:     node.GPUID,
					ModelName: model,
				}
			}
		}
	}
}

func toSet(ss []string) map[string]bool {
	m := make(map[string]bool, len(ss))
	for _, s := range ss {
		m[s] = true
	}
	return m
}
