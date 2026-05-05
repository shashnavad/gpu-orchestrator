package traffic

import (
	"context"
	"log"
	"sync"
	"time"
)

// RequestEvent represents a single inference request arriving for a model.
type RequestEvent struct {
	ModelName string
	Timestamp time.Time
}

// PrewarmSignal is emitted by the TrafficAnalyzer when a model is trending
// toward a burst. The Reconciler turns this into a PREWARM action for the
// Rust agent, so weights are loaded before the first real request hits.
type PrewarmSignal struct {
	ModelName string
	Reason    string
}

// windowSize is the rolling window over which request frequency is measured.
const windowSize = 5 * time.Minute

// prewarmThresholdRPM is the requests-per-minute rate at which we start
// pre-warming. A model seeing > 5 RPM in the last 5 minutes is "heating up".
const prewarmThresholdRPM = 5.0

// TrafficAnalyzer monitors incoming request frequency per model over a rolling
// 5-minute window and emits PrewarmSignals when a model's rate exceeds the
// threshold.
//
// The design principle: it's far cheaper to pre-load a model that might not
// get a request than to let the first real request stall for 30-60 seconds
// waiting for weight download.
type TrafficAnalyzer struct {
	mu       sync.Mutex
	// events is a ring-buffer approximation: we store all events from the last
	// windowSize and prune on each analysis tick.
	events   map[string][]time.Time // modelName → timestamps of recent requests
	requests <-chan RequestEvent
	signals  chan<- PrewarmSignal
	// prewarmSent tracks models we've already signaled so we don't spam.
	prewarmSent map[string]time.Time
}

func NewTrafficAnalyzer(
	requests <-chan RequestEvent,
	signals chan<- PrewarmSignal,
) *TrafficAnalyzer {
	return &TrafficAnalyzer{
		events:      make(map[string][]time.Time),
		requests:    requests,
		signals:     signals,
		prewarmSent: make(map[string]time.Time),
	}
}

// Run starts the analyzer. It:
//  1. Ingests RequestEvents as they arrive (fast path, just appends timestamp)
//  2. Every 30 seconds, prunes old events and evaluates whether any model
//     exceeds the prewarm threshold
func (t *TrafficAnalyzer) Run(ctx context.Context) {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	log.Println("[traffic-analyzer] started, window=5m, threshold=5rpm")

	for {
		select {

		case evt, ok := <-t.requests:
			if !ok {
				return
			}
			t.recordEvent(evt)

		case <-ticker.C:
			t.analyze()

		case <-ctx.Done():
			log.Println("[traffic-analyzer] shutting down")
			return
		}
	}
}

// RecordRequest is the public entry point for the API server to log a request.
// It's non-blocking: drops the event if the channel is full to avoid back-pressure
// on the hot inference path.
func RecordRequest(ch chan<- RequestEvent, modelName string) {
	select {
	case ch <- RequestEvent{ModelName: modelName, Timestamp: time.Now()}:
	default:
		// Drop silently — telemetry must never block inference.
	}
}

func (t *TrafficAnalyzer) recordEvent(evt RequestEvent) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.events[evt.ModelName] = append(t.events[evt.ModelName], evt.Timestamp)
}

// analyze prunes stale events and emits prewarm signals for hot models.
func (t *TrafficAnalyzer) analyze() {
	t.mu.Lock()
	defer t.mu.Unlock()

	cutoff := time.Now().Add(-windowSize)

	for model, timestamps := range t.events {
		// Prune events older than the rolling window.
		fresh := timestamps[:0]
		for _, ts := range timestamps {
			if ts.After(cutoff) {
				fresh = append(fresh, ts)
			}
		}
		t.events[model] = fresh

		if len(fresh) == 0 {
			continue
		}

		// Calculate RPM over the window.
		rpm := float64(len(fresh)) / windowSize.Minutes()

		if rpm >= prewarmThresholdRPM {
			t.maybeSignalPrewarm(model, rpm)
		}
	}
}

// maybeSignalPrewarm emits a PrewarmSignal but rate-limits to once per window
// per model so we don't flood the Reconciler.
func (t *TrafficAnalyzer) maybeSignalPrewarm(modelName string, rpm float64) {
	lastSent, alreadySent := t.prewarmSent[modelName]
	if alreadySent && time.Since(lastSent) < windowSize {
		return // already signaled recently, skip
	}

	t.prewarmSent[modelName] = time.Now()

	sig := PrewarmSignal{
		ModelName: modelName,
		Reason:    "rate exceeded threshold",
	}

	select {
	case t.signals <- sig:
		log.Printf("[traffic-analyzer] PREWARM signal for %s (%.1f RPM)", modelName, rpm)
	default:
		log.Printf("[traffic-analyzer] signal channel full, dropping prewarm for %s", modelName)
	}
}
