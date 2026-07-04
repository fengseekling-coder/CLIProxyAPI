package tokenusage

import (
	"os"
	"path/filepath"
	"sync"
	"testing"
)

func newTempStore(t *testing.T) *Store {
	t.Helper()
	dir := t.TempDir()
	s, err := NewStore(dir)
	if err != nil {
		t.Fatalf("NewStore: %v", err)
	}
	return s
}

func int64Ptr(v int64) *int64 { return &v }

// --- NewStore / load / Snapshot --------------------------------------------------

func TestNewStore_EmptyOnFreshDir(t *testing.T) {
	s := newTempStore(t)
	got := s.Snapshot()
	if got.Version != schemaVersion {
		t.Errorf("version: want %d, got %d", schemaVersion, got.Version)
	}
	if got.Models == nil {
		t.Fatal("Models should be non-nil empty map, got nil")
	}
	if len(got.Models) != 0 {
		t.Errorf("want empty models, got %d entries", len(got.Models))
	}
	if _, err := os.Stat(s.Path()); err == nil {
		t.Errorf("path %q should not exist before first write", s.Path())
	}
}

func TestNewStore_EmptyDataDirRejected(t *testing.T) {
	if _, err := NewStore(""); err == nil {
		t.Fatal("NewStore(\"\") should error, got nil")
	}
}

func TestNewStore_RejectsEmptyDirAfterTrim(t *testing.T) {
	if _, err := NewStore("   "); err == nil {
		t.Fatal("NewStore(\"   \") should error, got nil")
	}
}

func TestNewStore_PersistsAcrossInstances(t *testing.T) {
	dir := t.TempDir()

	s1, err := NewStore(dir)
	if err != nil {
		t.Fatalf("NewStore #1: %v", err)
	}
	ts := int64(1700000000000)
	in := PersistedShape{
		Version: schemaVersion,
		Models: map[string]ModelState{
			"claude-opus-4.8": {
				Monthly: map[string]MonthlyBucket{
					"2026-07": {
						Input:    100,
						Output:   200,
						Total:    300,
						Requests: 5,
						Daily: map[string]DailyBucket{
							"2026-07-04": {Input: 100, Output: 200, Total: 300, Requests: 5},
						},
					},
				},
				LastUpdatedAt: &ts,
				Lifetime:      DailyBucket{Input: 100, Output: 200, Total: 300, Requests: 5},
			},
		},
	}
	if err := s1.Replace(in); err != nil {
		t.Fatalf("Replace: %v", err)
	}

	// Re-open: a fresh Store should see the same data on disk.
	s2, err := NewStore(dir)
	if err != nil {
		t.Fatalf("NewStore #2: %v", err)
	}
	got := s2.Snapshot()
	if len(got.Models) != 1 {
		t.Fatalf("want 1 model after reload, got %d", len(got.Models))
	}
	m, ok := got.Models["claude-opus-4.8"]
	if !ok {
		t.Fatal("claude-opus-4.8 not found in reloaded snapshot")
	}
	if m.Lifetime.Input != 100 || m.Lifetime.Output != 200 {
		t.Errorf("lifetime mismatch: %+v", m.Lifetime)
	}
	if m.Monthly["2026-07"].Daily["2026-07-04"].Requests != 5 {
		t.Errorf("daily bucket requests mismatch: %+v", m.Monthly["2026-07"].Daily["2026-07-04"])
	}
}

func TestNewStore_CorruptFileBackedUp(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, fileName)
	if err := os.WriteFile(path, []byte("{not-json"), 0o644); err != nil {
		t.Fatalf("seed corrupt file: %v", err)
	}

	s, err := NewStore(dir)
	if err != nil {
		t.Fatalf("NewStore should recover from corrupt file, got: %v", err)
	}
	if len(s.Snapshot().Models) != 0 {
		t.Errorf("corrupt file should give empty store, got %d models", len(s.Snapshot().Models))
	}
	// The corrupt file should have been renamed to *.corrupt-<ts>.
	entries, _ := os.ReadDir(dir)
	var foundBackup bool
	for _, e := range entries {
		if filepath.Ext(e.Name()) != "" && len(e.Name()) > len(".corrupt-") && e.Name()[:len("token_usage.json.corrupt-")] == "token_usage.json.corrupt-" {
			foundBackup = true
		}
	}
	if !foundBackup {
		t.Errorf("expected a .corrupt-<ts> backup, dir entries: %+v", entries)
	}
}

func TestNewStore_UnsupportedVersionRejected(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, fileName)
	if err := os.WriteFile(path, []byte(`{"version":999,"models":{}}`), 0o644); err != nil {
		t.Fatalf("seed: %v", err)
	}
	s, err := NewStore(dir)
	if err != nil {
		t.Fatalf("NewStore should recover from bad-version file, got: %v", err)
	}
	if got := s.Snapshot().Version; got != schemaVersion {
		t.Errorf("recovered store version: want %d, got %d", schemaVersion, got)
	}
}

// --- Snapshot is a copy, not a reference ---------------------------------------

func TestSnapshot_DoesNotMutateStore(t *testing.T) {
	s := newTempStore(t)
	if err := s.Replace(PersistedShape{Version: schemaVersion, Models: map[string]ModelState{
		"m1": {Lifetime: DailyBucket{Input: 10}, Monthly: map[string]MonthlyBucket{
			"2026-07": {Input: 10, Daily: map[string]DailyBucket{"2026-07-04": {Input: 10}}},
		}},
	}}); err != nil {
		t.Fatalf("Replace: %v", err)
	}

	snap := s.Snapshot()
	// Mutate the snapshot freely: pull values into addressable locals first
	// because Go map-index expressions aren't addressable.
	m := snap.Models["m1"]
	m.Lifetime.Input = 9999
	monthly := m.Monthly["2026-07"]
	daily := monthly.Daily["2026-07-04"]
	daily.Input = 9999
	monthly.Daily["2026-07-04"] = daily
	m.Monthly["2026-07"] = monthly
	snap.Models["m1"] = m
	snap.Models["rogue"] = ModelState{}

	// Reload from disk; the original should be unchanged.
	if got := s.Snapshot().Models["m1"].Lifetime.Input; got != 10 {
		t.Errorf("snapshot leaked into store lifetime: got %d, want 10", got)
	}
	if got := s.Snapshot().Models["m1"].Monthly["2026-07"].Daily["2026-07-04"].Input; got != 10 {
		t.Errorf("snapshot leaked into daily bucket: got %d, want 10", got)
	}
	if _, ok := s.Snapshot().Models["rogue"]; ok {
		t.Errorf("snapshot leaked a new key back into store")
	}
}

// --- Replace --------------------------------------------------------------------

func TestReplace_NilModelsBecomesEmptyMap(t *testing.T) {
	s := newTempStore(t)
	if err := s.Replace(PersistedShape{Version: schemaVersion, Models: nil}); err != nil {
		t.Fatalf("Replace: %v", err)
	}
	got := s.Snapshot()
	if got.Models == nil {
		t.Error("Replace should normalize nil Models to empty map")
	}
	if len(got.Models) != 0 {
		t.Errorf("want empty, got %d", len(got.Models))
	}
}

func TestReplace_ZeroVersionBecomesSchemaVersion(t *testing.T) {
	s := newTempStore(t)
	if err := s.Replace(PersistedShape{Version: 0, Models: map[string]ModelState{}}); err != nil {
		t.Fatalf("Replace: %v", err)
	}
	if got := s.Snapshot().Version; got != schemaVersion {
		t.Errorf("want version %d, got %d", schemaVersion, got)
	}
}

func TestReplace_NilStoreNoop(t *testing.T) {
	var s *Store
	if err := s.Replace(PersistedShape{}); err == nil {
		t.Error("nil store Replace should error")
	}
	if s.Snapshot().Version != schemaVersion {
		t.Error("nil store Snapshot should return safe default, not crash")
	}
}

func TestReplace_AtomicAcrossCrash(t *testing.T) {
	// Simulate a crash mid-write by leaving a .tmp file behind from a previous
	// process; Replace should still complete cleanly and the result on disk
	// should be the new state, not a half-written one.
	dir := t.TempDir()
	s, err := NewStore(dir)
	if err != nil {
		t.Fatalf("NewStore: %v", err)
	}
	// Leave a stale tmp file.
	tmp := s.Path() + ".tmp"
	if err := os.WriteFile(tmp, []byte("stale"), 0o644); err != nil {
		t.Fatalf("seed tmp: %v", err)
	}
	if err := s.Replace(PersistedShape{Version: schemaVersion, Models: map[string]ModelState{
		"after-crash": {Lifetime: DailyBucket{Input: 1}},
	}}); err != nil {
		t.Fatalf("Replace: %v", err)
	}
	if _, err := os.Stat(tmp); err == nil {
		t.Error("tmp file should have been renamed away")
	}
	if got := s.Snapshot().Models["after-crash"].Lifetime.Input; got != 1 {
		t.Errorf("post-replace lifetime: want 1, got %d", got)
	}
}

// --- Merge ---------------------------------------------------------------------

func TestMerge_AddsBuckets(t *testing.T) {
	s := newTempStore(t)
	if err := s.Replace(PersistedShape{Version: schemaVersion, Models: map[string]ModelState{
		"m": {
			Lifetime: DailyBucket{Input: 10, Output: 20, Total: 30, Requests: 1},
			Monthly: map[string]MonthlyBucket{
				"2026-07": {Input: 10, Output: 20, Total: 30, Requests: 1, Daily: map[string]DailyBucket{
					"2026-07-04": {Input: 10, Output: 20, Total: 30, Requests: 1},
				}},
			},
		},
	}}); err != nil {
		t.Fatalf("Replace: %v", err)
	}

	if err := s.Merge(PersistedShape{Version: schemaVersion, Models: map[string]ModelState{
		"m": {
			Lifetime: DailyBucket{Input: 5, Output: 5, Total: 10, Requests: 1},
			Monthly: map[string]MonthlyBucket{
				"2026-07": {Input: 5, Output: 5, Total: 10, Requests: 1, Daily: map[string]DailyBucket{
					"2026-07-05": {Input: 5, Output: 5, Total: 10, Requests: 1},
				}},
			},
		},
	}}); err != nil {
		t.Fatalf("Merge: %v", err)
	}

	got := s.Snapshot().Models["m"]
	if got.Lifetime.Input != 15 || got.Lifetime.Total != 40 || got.Lifetime.Requests != 2 {
		t.Errorf("lifetime after merge: %+v", got.Lifetime)
	}
	m := got.Monthly["2026-07"]
	if m.Input != 15 || m.Total != 40 || m.Requests != 2 {
		t.Errorf("monthly after merge: %+v", m)
	}
	if m.Daily["2026-07-04"].Input != 10 {
		t.Errorf("existing daily bucket should be unchanged: %+v", m.Daily["2026-07-04"])
	}
	if m.Daily["2026-07-05"].Input != 5 {
		t.Errorf("new daily bucket missing: %+v", m.Daily["2026-07-05"])
	}
}

func TestMerge_NewModel(t *testing.T) {
	s := newTempStore(t)
	if err := s.Merge(PersistedShape{Version: schemaVersion, Models: map[string]ModelState{
		"new-model": {Lifetime: DailyBucket{Input: 7}},
	}}); err != nil {
		t.Fatalf("Merge: %v", err)
	}
	if got := s.Snapshot().Models["new-model"].Lifetime.Input; got != 7 {
		t.Errorf("new model merge: got %d, want 7", got)
	}
}

func TestMerge_NewMonthOnExistingModel(t *testing.T) {
	s := newTempStore(t)
	if err := s.Replace(PersistedShape{Version: schemaVersion, Models: map[string]ModelState{
		"m": {Monthly: map[string]MonthlyBucket{"2026-06": {Input: 1}}},
	}}); err != nil {
		t.Fatalf("Replace: %v", err)
	}
	if err := s.Merge(PersistedShape{Version: schemaVersion, Models: map[string]ModelState{
		"m": {Monthly: map[string]MonthlyBucket{"2026-07": {Input: 2}}},
	}}); err != nil {
		t.Fatalf("Merge: %v", err)
	}
	got := s.Snapshot().Models["m"].Monthly
	if got["2026-06"].Input != 1 || got["2026-07"].Input != 2 {
		t.Errorf("cross-month merge failed: %+v", got)
	}
}

func TestMerge_NilIncomingNoop(t *testing.T) {
	s := newTempStore(t)
	if err := s.Replace(PersistedShape{Version: schemaVersion, Models: map[string]ModelState{
		"m": {Lifetime: DailyBucket{Input: 4}},
	}}); err != nil {
		t.Fatalf("Replace: %v", err)
	}
	if err := s.Merge(PersistedShape{Version: schemaVersion, Models: nil}); err != nil {
		t.Fatalf("Merge nil-models should be a noop error, got: %v", err)
	}
	if got := s.Snapshot().Models["m"].Lifetime.Input; got != 4 {
		t.Errorf("nil-models merge should not touch store, got %d", got)
	}
}

func TestMerge_DoesNotMutateIncoming(t *testing.T) {
	s := newTempStore(t)
	incoming := PersistedShape{Version: schemaVersion, Models: map[string]ModelState{
		"m": {Lifetime: DailyBucket{Input: 9}, Monthly: map[string]MonthlyBucket{
			"2026-07": {Input: 9, Daily: map[string]DailyBucket{"2026-07-04": {Input: 9}}},
		}},
	}}
	if err := s.Merge(incoming); err != nil {
		t.Fatalf("Merge: %v", err)
	}
	// Caller's struct should be untouched (in particular, its Daily map
	// shouldn't have been aliased into the store).
	if incoming.Models["m"].Lifetime.Input != 9 {
		t.Errorf("incoming lifetime was mutated: %+v", incoming.Models["m"].Lifetime)
	}
	if incoming.Models["m"].Monthly["2026-07"].Daily["2026-07-04"].Input != 9 {
		t.Errorf("incoming daily bucket was mutated: %+v", incoming.Models["m"].Monthly["2026-07"].Daily["2026-07-04"])
	}
}

func TestMerge_LastUpdatedAtPicksMax(t *testing.T) {
	older := int64(1000)
	newer := int64(2000)
	s := newTempStore(t)
	if err := s.Replace(PersistedShape{Version: schemaVersion, Models: map[string]ModelState{
		"m": {LastUpdatedAt: &older},
	}}); err != nil {
		t.Fatalf("Replace: %v", err)
	}
	if err := s.Merge(PersistedShape{Version: schemaVersion, Models: map[string]ModelState{
		"m": {LastUpdatedAt: &newer},
	}}); err != nil {
		t.Fatalf("Merge: %v", err)
	}
	got := s.Snapshot().Models["m"].LastUpdatedAt
	if got == nil || *got != newer {
		t.Errorf("LastUpdatedAt should pick newer (2000), got %v", got)
	}
}

func TestMerge_LastUpdatedAtNilHandled(t *testing.T) {
	ts := int64(1234)
	s := newTempStore(t)
	if err := s.Replace(PersistedShape{Version: schemaVersion, Models: map[string]ModelState{
		"m": {LastUpdatedAt: &ts},
	}}); err != nil {
		t.Fatalf("Replace: %v", err)
	}
	if err := s.Merge(PersistedShape{Version: schemaVersion, Models: map[string]ModelState{
		"m": {LastUpdatedAt: nil},
	}}); err != nil {
		t.Fatalf("Merge: %v", err)
	}
	if got := s.Snapshot().Models["m"].LastUpdatedAt; got == nil || *got != ts {
		t.Errorf("nil incoming LastUpdatedAt should keep existing %d, got %v", ts, got)
	}
}

// --- Concurrency ---------------------------------------------------------------

func TestStore_ConcurrentWritesAreSafe(t *testing.T) {
	s := newTempStore(t)
	const writers = 8
	const iters = 50
	var wg sync.WaitGroup
	wg.Add(writers)
	for w := 0; w < writers; w++ {
		go func(id int) {
			defer wg.Done()
			for i := 0; i < iters; i++ {
				key := []string{"a", "b", "c", "d"}[id%4]
				in := PersistedShape{Version: schemaVersion, Models: map[string]ModelState{
					key: {Lifetime: DailyBucket{Input: 1, Output: 1, Total: 2, Requests: 1}},
				}}
				if err := s.Merge(in); err != nil {
					t.Errorf("Merge: %v", err)
					return
				}
			}
		}(w)
	}
	wg.Wait()

	// Each writer contributed iters increments of (1,1,2,1); expected totals
	// per key are iters * writers / 4 of those, scaled by the 4-way split.
	got := s.Snapshot().Models
	if len(got) != 4 {
		t.Fatalf("want 4 models after merge, got %d", len(got))
	}
	for _, m := range got {
		// Lifetime.Input should be exactly 100 (1 * 50 * 2 writers per key, but
		// because writers are split 2/2/2/2 across 4 keys for 8 writers, each
		// key gets 2 writers × 50 iters = 100 increments of 1).
		if m.Lifetime.Input != 100 {
			t.Errorf("model %+v: input %d, want 100", m.Lifetime, m.Lifetime.Input)
		}
	}
}

func TestStore_ConcurrentReadWrite(t *testing.T) {
	s := newTempStore(t)
	if err := s.Replace(PersistedShape{Version: schemaVersion, Models: map[string]ModelState{
		"seed": {Lifetime: DailyBucket{Input: 0}},
	}}); err != nil {
		t.Fatalf("Replace: %v", err)
	}

	stop := make(chan struct{})
	var wg sync.WaitGroup
	wg.Add(2)

	go func() {
		defer wg.Done()
		for {
			select {
			case <-stop:
				return
			default:
			}
			_ = s.Merge(PersistedShape{Version: schemaVersion, Models: map[string]ModelState{
				"seed": {Lifetime: DailyBucket{Input: 1}},
			}})
		}
	}()
	go func() {
		defer wg.Done()
		for i := 0; i < 200; i++ {
			_ = s.Snapshot()
		}
		close(stop)
	}()
	wg.Wait()

	if got := s.Snapshot().Models["seed"].Lifetime.Input; got <= 0 {
		t.Errorf("expected some Input > 0 after concurrent writes, got %d", got)
	}
}

// --- mergeMonthly / mergeModelState unit tests ---------------------------------

func TestMergeMonthly_AddsScalarAndMergesDaily(t *testing.T) {
	a := MonthlyBucket{Input: 1, Output: 2, Total: 3, Requests: 4, Daily: map[string]DailyBucket{
		"2026-07-04": {Input: 1, Output: 2, Total: 3, Requests: 4},
	}}
	b := MonthlyBucket{Input: 10, Output: 20, Total: 30, Requests: 40, Daily: map[string]DailyBucket{
		"2026-07-04": {Input: 10, Output: 20, Total: 30, Requests: 40},
		"2026-07-05": {Input: 100, Output: 200, Total: 300, Requests: 400},
	}}
	got := mergeMonthly(a, b)
	if got.Input != 11 || got.Output != 22 || got.Total != 33 || got.Requests != 44 {
		t.Errorf("scalar sum: %+v", got)
	}
	if got.Daily["2026-07-04"].Input != 11 {
		t.Errorf("overlapping daily merge: %+v", got.Daily["2026-07-04"])
	}
	if got.Daily["2026-07-05"].Input != 100 {
		t.Errorf("new daily merge: %+v", got.Daily["2026-07-05"])
	}
}

func TestMergeModelState_PicksNewerLastUpdatedAt(t *testing.T) {
	older := int64(100)
	newer := int64(200)
	merged := mergeModelState(
		ModelState{LastUpdatedAt: &older, Lifetime: DailyBucket{Input: 1}},
		ModelState{LastUpdatedAt: &newer, Lifetime: DailyBucket{Input: 2}},
	)
	if merged.LastUpdatedAt == nil || *merged.LastUpdatedAt != newer {
		t.Errorf("want newer ts, got %v", merged.LastUpdatedAt)
	}
	if merged.Lifetime.Input != 3 {
		t.Errorf("lifetime should sum, got %d", merged.Lifetime.Input)
	}
}

func TestMergeModelState_NilLastUpdatedAtOnBothSides(t *testing.T) {
	merged := mergeModelState(
		ModelState{Lifetime: DailyBucket{Input: 1}},
		ModelState{Lifetime: DailyBucket{Input: 2}},
	)
	if merged.LastUpdatedAt != nil {
		t.Errorf("both nil should stay nil, got %v", merged.LastUpdatedAt)
	}
	if merged.Lifetime.Input != 3 {
		t.Errorf("lifetime should still sum, got %d", merged.Lifetime.Input)
	}
}