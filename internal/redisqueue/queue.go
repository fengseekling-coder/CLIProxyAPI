package redisqueue

import (
	"sync"
	"sync/atomic"
	"time"

	log "github.com/sirupsen/logrus"
)

const (
	defaultRetentionSeconds int64 = 60
	maxRetentionSeconds     int64 = 3600
	usageSubscriberBuffer         = 256
	errorSubscriberBuffer         = 256

	usageSupportRefreshPayload = `{"support_refresh":true}`
	usageRefreshPayload        = `{"refresh":true}`

	// maxPayloadBytes caps a single enqueue / subscriber delivery. Large
	// payloads (e.g. accidental image bodies, oversized response headers) are
	// dropped with a WARN log instead of growing the queue without bound.
	maxPayloadBytes = 1 << 20 // 1 MiB

	// shutdownPayload is delivered to every active subscriber right before
	// their channel is closed by SetEnabled(false). Subscribers can
	// distinguish "service is shutting down" from a network disconnect.
	shutdownPayload = `{"shutdown":true}`
)

type queueItem struct {
	enqueuedAt time.Time
	payload    []byte
}

type queue struct {
	mu               sync.Mutex
	items            []queueItem
	head             int
	subscribers      map[uint64]chan []byte
	nextSubscriberID uint64
}

var (
	enabled          atomic.Bool
	retentionSeconds atomic.Int64
	errorRetentionSeconds atomic.Int64
	global           queue
	errorGlobal      queue
)

func init() {
	retentionSeconds.Store(defaultRetentionSeconds)
	errorRetentionSeconds.Store(defaultRetentionSeconds)
}

// SetErrorRetentionSeconds mirrors SetRetentionSeconds but for the error
// channel. Errors are kept in a bounded buffer for at most this many seconds
// so they can still be observed by a late subscriber. Default 60s.
func SetErrorRetentionSeconds(value int) {
	normalized := int64(value)
	if normalized <= 0 {
		normalized = defaultRetentionSeconds
	} else if normalized > maxRetentionSeconds {
		normalized = maxRetentionSeconds
	}
	errorRetentionSeconds.Store(normalized)
}

func SetEnabled(value bool) {
	enabled.Store(value)
	if !value {
		global.clear()
		errorGlobal.clear()
	}
}

func Enabled() bool {
	return enabled.Load()
}

func SetRetentionSeconds(value int) {
	normalized := int64(value)
	if normalized <= 0 {
		normalized = defaultRetentionSeconds
	} else if normalized > maxRetentionSeconds {
		normalized = maxRetentionSeconds
	}
	retentionSeconds.Store(normalized)
}

func Enqueue(payload []byte) {
	if !Enabled() {
		return
	}
	if len(payload) == 0 {
		return
	}
	if len(payload) > maxPayloadBytes {
		return
	}
	if global.publishToSubscribers(payload) {
		return
	}
	global.enqueue(payload)
}

func EnqueueError(payload []byte) {
	if !Enabled() {
		return
	}
	if len(payload) == 0 || len(payload) > maxPayloadBytes {
		return
	}
	if errorGlobal.publishToSubscribers(payload) {
		return
	}
	errorGlobal.enqueueError(payload)
}

func PopOldest(count int) [][]byte {
	if !Enabled() {
		return nil
	}
	if count <= 0 {
		return nil
	}
	return global.popOldest(count)
}

func PopOldestErrors(count int) [][]byte {
	if !Enabled() {
		return nil
	}
	if count <= 0 {
		return nil
	}
	return errorGlobal.popOldestErrors(count)
}

func SubscribeUsage() (<-chan []byte, func()) {
	return global.subscribe(usageSubscriberBuffer, []byte(usageSupportRefreshPayload))
}

func SubscribeErrors() (<-chan []byte, func()) {
	return errorGlobal.subscribe(errorSubscriberBuffer, nil)
}

func NotifyUsageRefresh() {
	global.publishToSubscribers([]byte(usageRefreshPayload))
}

func (q *queue) clear() {
	q.mu.Lock()

	// Send a shutdown notice to active subscribers BEFORE closing their
	// channels so they can distinguish "service going away" from "network
	// disconnect" and avoid spurious reconnect storms.
	shutdown := []byte(shutdownPayload)
	subscribers := make([]chan []byte, 0, len(q.subscribers))
	for _, subscriber := range q.subscribers {
		select {
		case subscriber <- shutdown:
		default:
			// Subscriber channel is full; they will be closed below.
		}
		subscribers = append(subscribers, subscriber)
	}
	q.items = nil
	q.head = 0
	q.subscribers = nil
	q.mu.Unlock()

	for _, subscriber := range subscribers {
		close(subscriber)
	}
}

func (q *queue) enqueue(payload []byte) {
	now := time.Now()

	q.mu.Lock()
	defer q.mu.Unlock()

	q.pruneLocked(now)
	q.items = append(q.items, queueItem{
		enqueuedAt: now,
		payload:    append([]byte(nil), payload...),
	})
	q.maybeCompactLocked()
}

func (q *queue) publishToSubscribers(payload []byte) bool {
	q.mu.Lock()
	defer q.mu.Unlock()

	if len(q.subscribers) == 0 {
		return false
	}

	cloned := append([]byte(nil), payload...)
	for id, subscriber := range q.subscribers {
		select {
		case subscriber <- cloned:
		default:
			// Subscriber buffer is full. Close the channel so the consumer
			// sees a graceful disconnect; log so operators can correlate.
			delete(q.subscribers, id)
			close(subscriber)
			log.Warnf("redisqueue: dropping slow subscriber (buffer=%d) and closing its channel", cap(subscriber))
		}
	}

	return true
}

func (q *queue) subscribe(buffer int, initialPayload []byte) (<-chan []byte, func()) {
	subscriber := make(chan []byte, buffer)
	if len(initialPayload) > 0 {
		subscriber <- append([]byte(nil), initialPayload...)
	}

	q.mu.Lock()
	if q.subscribers == nil {
		q.subscribers = make(map[uint64]chan []byte)
	}
	q.nextSubscriberID++
	id := q.nextSubscriberID
	q.subscribers[id] = subscriber
	q.mu.Unlock()

	var once sync.Once
	unsubscribe := func() {
		once.Do(func() {
			q.unsubscribe(id)
		})
	}
	return subscriber, unsubscribe
}

func (q *queue) unsubscribe(id uint64) {
	q.mu.Lock()
	subscriber, ok := q.subscribers[id]
	if ok {
		delete(q.subscribers, id)
	}
	q.mu.Unlock()

	if ok {
		close(subscriber)
	}
}

func (q *queue) enqueueError(payload []byte) {
	now := time.Now()

	q.mu.Lock()
	defer q.mu.Unlock()

	q.pruneLockedWith(now, errorRetentionSeconds.Load())
	q.items = append(q.items, queueItem{
		enqueuedAt: now,
		payload:    append([]byte(nil), payload...),
	})
	q.maybeCompactLocked()
}

func (q *queue) popOldest(count int) [][]byte {
	return q.popOldestWith(count, retentionSeconds.Load())
}

func (q *queue) popOldestErrors(count int) [][]byte {
	return q.popOldestWith(count, errorRetentionSeconds.Load())
}

func (q *queue) popOldestWith(count int, windowSeconds int64) [][]byte {
	now := time.Now()

	q.mu.Lock()
	defer q.mu.Unlock()

	q.pruneLockedWith(now, windowSeconds)
	available := len(q.items) - q.head
	if available <= 0 {
		q.items = nil
		q.head = 0
		return nil
	}
	if count > available {
		count = available
	}

	out := make([][]byte, 0, count)
	for i := 0; i < count; i++ {
		item := q.items[q.head+i]
		out = append(out, item.payload)
	}
	q.head += count
	q.maybeCompactLocked()
	return out
}

func (q *queue) pruneLocked(now time.Time) {
	q.pruneLockedWith(now, retentionSeconds.Load())
}

func (q *queue) pruneLockedWith(now time.Time, windowSeconds int64) {
	if q.head >= len(q.items) {
		q.items = nil
		q.head = 0
		return
	}

	if windowSeconds <= 0 {
		windowSeconds = defaultRetentionSeconds
	}
	cutoff := now.Add(-time.Duration(windowSeconds) * time.Second)
	for q.head < len(q.items) && q.items[q.head].enqueuedAt.Before(cutoff) {
		q.head++
	}
}

func (q *queue) maybeCompactLocked() {
	if q.head == 0 {
		return
	}
	if q.head >= len(q.items) {
		q.items = nil
		q.head = 0
		return
	}
	if q.head < 1024 && q.head*2 < len(q.items) {
		return
	}
	q.items = append([]queueItem(nil), q.items[q.head:]...)
	q.head = 0
}
