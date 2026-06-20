package admission

import (
	"container/list"
	"errors"
	"sync"
)

// PriorityClass buckets queued requests for weighted fair queueing.
// Distinct from scheduler.Priority (P0/P1/P2), which governs preemption
// of already-placed work. PriorityClass only governs the rate at which
// different classes get retried while no slice has room.
type PriorityClass int

const (
	High PriorityClass = iota
	Medium
	Low
)

var ErrQueueFull = errors.New("admission queue full for this priority class")

// weightSequence encodes a 70/20/10 split as a 10-slot repeating cycle,
// so the ratio is exact and deterministic rather than drifting like a
// floating-point credit scheme would.
var weightSequence = [10]PriorityClass{
	High, High, High, High, High, High, High, Medium, Medium, Low,
}

// Queue is a bounded, per-class FIFO with weighted round-robin dequeue order.
type Queue struct {
	mu       sync.Mutex
	buckets  map[PriorityClass]*list.List
	capacity int
	cursor   int
}

func NewQueue(capacityPerClass int) *Queue {
	return &Queue{
		buckets: map[PriorityClass]*list.List{
			High: list.New(), Medium: list.New(), Low: list.New(),
		},
		capacity: capacityPerClass,
	}
}

func (q *Queue) Admit(class PriorityClass, pr *pendingRequest) error {
	q.mu.Lock()
	defer q.mu.Unlock()
	b := q.buckets[class]
	if b.Len() >= q.capacity {
		return ErrQueueFull
	}
	b.PushBack(pr)
	return nil
}

func (q *Queue) Next() (*pendingRequest, PriorityClass, bool) {
	q.mu.Lock()
	defer q.mu.Unlock()
	for i := 0; i < len(weightSequence); i++ {
		class := weightSequence[q.cursor]
		q.cursor = (q.cursor + 1) % len(weightSequence)
		if b := q.buckets[class]; b.Len() > 0 {
			el := b.Front()
			b.Remove(el)
			return el.Value.(*pendingRequest), class, true
		}
	}
	return nil, 0, false
}

func (q *Queue) Len() int {
	q.mu.Lock()
	defer q.mu.Unlock()
	return q.buckets[High].Len() + q.buckets[Medium].Len() + q.buckets[Low].Len()
}
