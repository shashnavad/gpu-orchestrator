package admission

import (
	"context"
	"encoding/json"
	"net/http"
	"sync/atomic"
	"time"

	"gpu-orchestrator/reconciler"
	schedulerpkg "gpu-orchestrator/scheduler"
)

type pendingRequest struct {
	req       schedulerpkg.ScheduleRequest
	resultCh  chan placementResult
	cancelled atomic.Bool
}

type placementResult struct {
	decision *schedulerpkg.ScheduleDecision
	err      error
}

// Gateway is the admission-controlled front door to the BinPacker. A
// capacity miss queues the request by WFQ class instead of dropping it;
// DrainLoop retries placement as capacity frees.
type Gateway struct {
	packer      *schedulerpkg.BinPacker
	reconciler  *reconciler.Reconciler
	queue       *Queue
	waitTimeout time.Duration
}

func NewGateway(packer *schedulerpkg.BinPacker, rec *reconciler.Reconciler, queue *Queue, waitTimeout time.Duration) *Gateway {
	return &Gateway{packer: packer, reconciler: rec, queue: queue, waitTimeout: waitTimeout}
}

// classFor maps the existing P0/P1/P2 priority onto a WFQ class: P0
// production gets the 70% share, P1 development 20%, P2 batch 10%.
func classFor(p int) PriorityClass {
	switch p {
	case schedulerpkg.P0Production:
		return High
	case schedulerpkg.P1Development:
		return Medium
	default:
		return Low
	}
}

func (g *Gateway) HandleSchedule(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	var req schedulerpkg.ScheduleRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "bad request: "+err.Error(), http.StatusBadRequest)
		return
	}

	if decision, err := g.packer.Schedule(req); err == nil {
		g.commit(decision, req)
		writeDecision(w, http.StatusOK, decision)
		return
	}

	pr := &pendingRequest{req: req, resultCh: make(chan placementResult, 1)}
	if err := g.queue.Admit(classFor(req.Priority), pr); err != nil {
		w.Header().Set("Retry-After", "1")
		http.Error(w, err.Error(), http.StatusTooManyRequests)
		return
	}

	ctx, cancel := context.WithTimeout(r.Context(), g.waitTimeout)
	defer cancel()

	select {
	case res := <-pr.resultCh:
		if res.err != nil {
			http.Error(w, res.err.Error(), http.StatusTooManyRequests)
			return
		}
		g.commit(res.decision, req)
		writeDecision(w, http.StatusOK, res.decision)
	case <-ctx.Done():
		pr.cancelled.Store(true)
		w.Header().Set("Retry-After", "1")
		http.Error(w, "timed out waiting for GPU capacity", http.StatusTooManyRequests)
	}
}

func (g *Gateway) commit(decision *schedulerpkg.ScheduleDecision, req schedulerpkg.ScheduleRequest) {
	g.reconciler.AddDesired(decision.NodeID, decision.GPUID, decision.SliceID, req.ModelName)
}

func writeDecision(w http.ResponseWriter, status int, d *schedulerpkg.ScheduleDecision) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(d)
}

// DrainLoop retries queued requests in WFQ order on a fixed interval.
func (g *Gateway) DrainLoop(ctx context.Context, interval time.Duration) {
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			g.drainOnce()
		}
	}
}

func (g *Gateway) drainOnce() {
	attempts := g.queue.Len() // snapshot — each queued item gets one try per tick
	for i := 0; i < attempts; i++ {
		pr, class, ok := g.queue.Next()
		if !ok {
			return
		}
		if pr.cancelled.Load() {
			continue
		}
		decision, err := g.packer.Schedule(pr.req)
		if err != nil {
			_ = g.queue.Admit(class, pr) // still no room — back of its own bucket
			continue
		}
		pr.resultCh <- placementResult{decision: decision}
	}
}
