package usage

import (
	"context"
	"sync/atomic"
	"testing"
	"time"
)

type blockingPlugin struct {
	started   chan struct{}
	release   chan struct{}
	calls     atomic.Int32
}

func (b *blockingPlugin) HandleUsage(_ context.Context, _ Record) {
	b.calls.Add(1)
	closeOnce(b.started)
	<-b.release
}

type callCountingPlugin struct {
	calls atomic.Int32
}

func (p *callCountingPlugin) HandleUsage(_ context.Context, _ Record) {
	p.calls.Add(1)
}

type panicPlugin struct {
	calls atomic.Int32
}

func (p *panicPlugin) HandleUsage(_ context.Context, _ Record) {
	p.calls.Add(1)
	panic("intentional test panic")
}

func closeOnce(ch chan struct{}) {
	select {
	case <-ch:
	default:
		close(ch)
	}
}

func waitStarted(t *testing.T, ch chan struct{}) {
	t.Helper()
	select {
	case <-ch:
	case <-time.After(2 * time.Second):
		t.Fatalf("plugin did not start in time")
	}
}

func TestManagerNewManagerDefaultsBufferWhenNonPositive(t *testing.T) {
	for _, input := range []int{-1, 0} {
		m := NewManager(input)
		if m == nil {
			t.Fatalf("NewManager(%d) returned nil", input)
		}
		if m.cap != 512 {
			t.Fatalf("NewManager(%d) cap = %d, want 512", input, m.cap)
		}
	}
}

func TestManagerPublishDropsWhenBufferFull(t *testing.T) {
	m := NewManager(2)
	defer m.Stop()
	// slowPlugin blocks inside HandleUsage and is invoked fire-and-forget,
	// so the dispatcher itself is not blocked. With 4 Publish calls and a
	// cap of 2, at most 2 items can be in the queue at any time, but the
	// exact number of drops depends on whether the dispatcher has consumed
	// items before subsequent Publishes arrive. We assert that the dropped
	// count is in {0, 1, 2} (3 or 4 would indicate a broken cap check).
	slow := &slowPlugin{}
	m.Register(slow)
	m.Start(context.Background())

	for i := 0; i < 4; i++ {
		m.Publish(context.Background(), Record{Model: "x"})
	}

	// Wait for the dispatcher to drain everything that's still in the queue
	// (fire-and-forget means dispatch is non-blocking so this happens quickly).
	deadline := time.Now().Add(time.Second)
	for time.Now().Before(deadline) {
		m.mu.Lock()
		empty := len(m.queue) == 0
		m.mu.Unlock()
		if empty {
			break
		}
		time.Sleep(5 * time.Millisecond)
	}

	if dropped := m.Dropped(); dropped > 2 {
		t.Fatalf("Dropped() = %d, want <=2 (cap=2)", dropped)
	}
	if dropped := m.Dropped(); dropped < 0 {
		t.Fatalf("Dropped() = %d, want >=0", dropped)
	}
	// every published record was either queued+drained or dropped
	if total := int(slow.calls.Load()) + int(m.Dropped()); total != 4 {
		t.Fatalf("calls(%d) + dropped(%d) != 4", slow.calls.Load(), m.Dropped())
	}
}

// slowPlugin simulates a plugin whose HandleUsage takes a tiny but non-zero
// amount of time, allowing Publish to race the dispatcher.
type slowPlugin struct {
	calls atomic.Int32
}

func (s *slowPlugin) HandleUsage(_ context.Context, _ Record) {
	s.calls.Add(1)
	time.Sleep(time.Millisecond)
}

func TestManagerPublishOverflowsWhenDispatcherStalls(t *testing.T) {
	// When the queue reaches its cap, additional Publish calls are dropped
	// rather than growing the queue without bound. We saturate the queue
	// by directly filling it via the package's mu mutex; this avoids race
	// with the dispatcher draining it.
	m := NewManager(2)
	m.Register(&callCountingPlugin{})
	m.Start(context.Background())

	m.mu.Lock()
	m.queue = append(m.queue,
		queueItem{ctx: context.Background(), record: Record{Model: "a"}},
		queueItem{ctx: context.Background(), record: Record{Model: "b"}},
	)
	m.mu.Unlock()

	// Now Publish should drop because len(queue) == cap
	m.Publish(context.Background(), Record{Model: "c"})
	if dropped := m.Dropped(); dropped != 1 {
		t.Fatalf("Dropped() = %d, want 1", dropped)
	}

	m.Stop()
}

func TestManagerSafeInvokeRecoversPanic(t *testing.T) {
	m := NewManager(8)
	defer m.Stop()
	m.Register(&panicPlugin{})
	m.Start(context.Background())
	m.Publish(context.Background(), Record{Model: "x"})
	// wait for at least one dispatch
	deadline := time.Now().Add(time.Second)
	for time.Now().Before(deadline) {
		// panic is recovered inside safeInvoke, so we just ensure no test crash
		time.Sleep(20 * time.Millisecond)
	}
}

func TestManagerSafeInvokePluginTimeoutDoesNotBlockDispatcher(t *testing.T) {
	m := NewManager(8)
	defer m.Stop()

	blocking := &blockingPlugin{
		started: make(chan struct{}),
		release: make(chan struct{}),
	}
	fast := &callCountingPlugin{}

	m.Register(blocking)
	m.Register(fast)
	m.Start(context.Background())

	m.Publish(context.Background(), Record{Model: "slow"})
	waitStarted(t, blocking.started)

	m.Publish(context.Background(), Record{Model: "fast"})

	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) && fast.calls.Load() == 0 {
		time.Sleep(20 * time.Millisecond)
	}
	if fast.calls.Load() == 0 {
		t.Fatalf("expected fast plugin to be invoked even while blocking plugin is stalled")
	}

	close(blocking.release)
}

func TestManagerStopDrainsQueue(t *testing.T) {
	m := NewManager(8)
	m.Register(&callCountingPlugin{})
	m.Start(context.Background())
	for i := 0; i < 5; i++ {
		m.Publish(context.Background(), Record{Model: "m"})
	}
	// Give the worker a moment to dequeue and spawn plugin goroutines
	// (fire-and-forget). The queue itself may already be empty because
	// dispatch pops instantly.
	time.Sleep(50 * time.Millisecond)
	m.Stop()
	// After Stop, the worker exits as soon as the queue is empty.
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		m.mu.Lock()
		empty := len(m.queue) == 0
		m.mu.Unlock()
		if empty {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if !m.closed {
		t.Fatalf("manager not marked closed after Stop")
	}
	if len(m.queue) != 0 {
		t.Fatalf("queue not drained: len=%d", len(m.queue))
	}
}
