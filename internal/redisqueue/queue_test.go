package redisqueue

import (
	"testing"
	"time"
)

func TestEnqueueBroadcastsToUsageSubscribersAndSkipsQueue(t *testing.T) {
	withEnabledQueue(t, func() {
		first, unsubscribeFirst := SubscribeUsage()
		defer unsubscribeFirst()
		second, unsubscribeSecond := SubscribeUsage()
		defer unsubscribeSecond()

		requireUsageSubscriberPayload(t, first, usageSupportRefreshPayload)
		requireUsageSubscriberPayload(t, second, usageSupportRefreshPayload)

		Enqueue([]byte("usage-record"))

		requireUsageSubscriberPayload(t, first, "usage-record")
		requireUsageSubscriberPayload(t, second, "usage-record")

		if items := PopOldest(1); len(items) != 0 {
			t.Fatalf("PopOldest() items = %q, want empty after subscriber broadcast", items)
		}

		unsubscribeFirst()
		unsubscribeSecond()

		Enqueue([]byte("queued-record"))
		items := PopOldest(1)
		if len(items) != 1 || string(items[0]) != "queued-record" {
			t.Fatalf("PopOldest() items = %q, want queued record after unsubscribe", items)
		}
	})
}

func TestSetEnabledFalseClosesUsageSubscribers(t *testing.T) {
	withEnabledQueue(t, func() {
		subscriber, unsubscribe := SubscribeUsage()
		defer unsubscribe()
		errorSubscriber, unsubscribeErrors := SubscribeErrors()
		defer unsubscribeErrors()

		requireUsageSubscriberPayload(t, subscriber, usageSupportRefreshPayload)

		SetEnabled(false)

		// F22: SetEnabled(false) delivers a shutdown payload BEFORE closing
		// the channel. Drain the shutdown notice, then assert the channel is
		// closed.
		select {
		case payload, ok := <-subscriber:
			if !ok {
				// OK — could happen if the channel was already saturated.
			} else if string(payload) != shutdownPayload {
				t.Fatalf("expected shutdown payload, got %q", string(payload))
			}
		case <-time.After(time.Second):
			t.Fatalf("timeout waiting for shutdown payload")
		}
		select {
		case _, ok := <-subscriber:
			if ok {
				t.Fatalf("subscriber channel remained open after SetEnabled(false)")
			}
		case <-time.After(time.Second):
			t.Fatalf("timeout waiting for subscriber close")
		}

		select {
		case payload, ok := <-errorSubscriber:
			if ok && string(payload) != shutdownPayload {
				t.Fatalf("error subscriber got unexpected payload %q", string(payload))
			}
			// drain any extra shutdown notice if the buffer happened to keep one
		case <-time.After(time.Second):
			t.Fatalf("timeout waiting for error subscriber shutdown/close")
		}
		select {
		case _, ok := <-errorSubscriber:
			if ok {
				t.Fatalf("error subscriber channel remained open after SetEnabled(false)")
			}
		case <-time.After(time.Second):
			t.Fatalf("timeout waiting for error subscriber close")
		}
	})
}

func TestEnqueueErrorBroadcastsToErrorSubscribersAndBuffersWhenIdle(t *testing.T) {
	withEnabledQueue(t, func() {
		subscriber, unsubscribe := SubscribeErrors()
		defer unsubscribe()

		EnqueueError([]byte("error-record"))
		requireUsageSubscriberPayload(t, subscriber, "error-record")

		unsubscribe()

		// F23: errors without a subscriber are now buffered (up to retention)
		// so a late observer can still pop them.
		EnqueueError([]byte("buffered-error"))
		items := PopOldestErrors(1)
		if len(items) != 1 || string(items[0]) != "buffered-error" {
			t.Fatalf("PopOldestErrors() = %q, want [buffered-error]", items)
		}
	})
}

func TestNotifyUsageRefreshBroadcastsOnlyToUsageSubscribers(t *testing.T) {
	withEnabledQueue(t, func() {
		subscriber, unsubscribe := SubscribeUsage()
		defer unsubscribe()
		errorSubscriber, unsubscribeErrors := SubscribeErrors()
		defer unsubscribeErrors()

		requireUsageSubscriberPayload(t, subscriber, usageSupportRefreshPayload)

		NotifyUsageRefresh()
		requireUsageSubscriberPayload(t, subscriber, usageRefreshPayload)

		select {
		case got := <-errorSubscriber:
			t.Fatalf("error subscriber received usage refresh payload %q", string(got))
		default:
		}

		unsubscribe()
		NotifyUsageRefresh()
		if items := PopOldest(1); len(items) != 0 {
			t.Fatalf("PopOldest() items = %q, want empty after refresh notification without subscribers", items)
		}
	})
}

func requireUsageSubscriberPayload(t *testing.T, subscriber <-chan []byte, want string) {
	t.Helper()

	select {
	case got, ok := <-subscriber:
		if !ok {
			t.Fatalf("subscriber closed before receiving %q", want)
		}
		if string(got) != want {
			t.Fatalf("subscriber payload = %q, want %q", string(got), want)
		}
	case <-time.After(time.Second):
		t.Fatalf("timeout waiting for subscriber payload %q", want)
	}
}

func requireErrorQueueEmpty(t *testing.T) {
	t.Helper()

	errorGlobal.mu.Lock()
	defer errorGlobal.mu.Unlock()

	if len(errorGlobal.items)-errorGlobal.head != 0 {
		t.Fatalf("error queue retained %d item(s), want none", len(errorGlobal.items)-errorGlobal.head)
	}
}

func TestEnqueueRejectsOversizedPayload(t *testing.T) {
	withEnabledQueue(t, func() {
		huge := make([]byte, maxPayloadBytes+1)
		for i := range huge {
			huge[i] = 'x'
		}
		Enqueue(huge)
		if items := PopOldest(1); len(items) != 0 {
			t.Fatalf("Enqueue should drop oversized payload, but PopOldest returned %d items", len(items))
		}
	})
}

func TestEnqueueErrorBuffersWhenNoSubscribers(t *testing.T) {
	withEnabledQueue(t, func() {
		EnqueueError([]byte("buffered-error"))
		items := PopOldestErrors(1)
		if len(items) != 1 || string(items[0]) != "buffered-error" {
			t.Fatalf("PopOldestErrors() = %q, want [buffered-error]", items)
		}
	})
}

func TestSetEnabledDeliversShutdownPayloadToSubscribers(t *testing.T) {
	withEnabledQueue(t, func() {
		sub, unsubscribe := SubscribeUsage()
		defer unsubscribe()
		// The subscribe call enqueues a support_refresh payload first; drain it.
		select {
		case <-sub:
		case <-time.After(time.Second):
			t.Fatalf("missing initial support refresh payload")
		}

		SetEnabled(false)

		select {
		case payload, ok := <-sub:
			if !ok {
				t.Fatalf("subscriber closed before shutdown payload")
			}
			if string(payload) != shutdownPayload {
				t.Fatalf("got %q, want %q", string(payload), shutdownPayload)
			}
		case <-time.After(time.Second):
			t.Fatalf("did not receive shutdown payload")
		}
	})
}
