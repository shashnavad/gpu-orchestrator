package admission

import (
	"testing"

	schedulerpkg "gpu-orchestrator/scheduler"
)

func newPending() *pendingRequest {
	return &pendingRequest{
		req:      schedulerpkg.ScheduleRequest{ModelName: "m"},
		resultCh: make(chan placementResult, 1),
	}
}

func TestWFQRatioOverTenDispatches(t *testing.T) {
	q := NewQueue(100)
	for i := 0; i < 20; i++ {
		_ = q.Admit(High, newPending())
		_ = q.Admit(Medium, newPending())
		_ = q.Admit(Low, newPending())
	}

	counts := map[PriorityClass]int{}
	for i := 0; i < 10; i++ {
		_, class, ok := q.Next()
		if !ok {
			t.Fatal("expected a request, queue reported empty")
		}
		counts[class]++
	}

	if counts[High] != 7 || counts[Medium] != 2 || counts[Low] != 1 {
		t.Fatalf("expected 7/2/1 split over 10 dispatches, got high=%d medium=%d low=%d",
			counts[High], counts[Medium], counts[Low])
	}
}

func TestWFQSkipsEmptyClassWithoutStalling(t *testing.T) {
	q := NewQueue(10)
	_ = q.Admit(Low, newPending()) // only Low has work
	_, class, ok := q.Next()
	if !ok || class != Low {
		t.Fatalf("expected Low served despite dominating High slots, got class=%v ok=%v", class, ok)
	}
}

func TestQueueRejectsOverCapacityPerClass(t *testing.T) {
	q := NewQueue(1)
	if err := q.Admit(Low, newPending()); err != nil {
		t.Fatalf("first admit should succeed: %v", err)
	}
	if err := q.Admit(Low, newPending()); err != ErrQueueFull {
		t.Fatalf("expected ErrQueueFull, got %v", err)
	}
	if err := q.Admit(High, newPending()); err != nil {
		t.Fatalf("High bucket should be unaffected by Low being full: %v", err)
	}
}
