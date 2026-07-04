// Package tokenusage persists per-model token usage to a project-local
// JSON file so that usage statistics survive browser-localStorage wipes,
// incognito sessions, and machine migrations. The store is the source of
// truth on the server side; the frontend polls `/v0/management/usage-summary`
// to hydrate its in-memory model and falls back to localStorage only when
// the server has no record (e.g. first-ever run on a fresh install).
//
// File format (1:1 with the frontend `useTokenUsageStore` PersistedShape
// minus the `seenIds` field, which only the frontend cares about):
//
//	{
//	  "version": 1,
//	  "models": {
//	    "<modelKey>": {
//	      "monthly": {
//	        "YYYY-MM": {
//	          "input": ..., "output": ..., "reasoning": ..., "cached": ...,
//	          "total": ..., "requests": ...,
//	          "daily": {
//	            "YYYY-MM-DD": { ...same shape... }
//	          }
//	        }
//	      },
//	      "lastUpdatedAt": 1700000000000,
//	      "lifetime": { ...TokenBreakdown... }
//	    }
//	  }
//	}
package tokenusage

import (
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/router-for-me/CLIProxyAPI/v7/internal/util"
	log "github.com/sirupsen/logrus"
)

const (
	schemaVersion = 1
	// fileName is the on-disk JSON file inside the project data dir.
	fileName = "token_usage.json"
)

// DailyBucket / MonthlyBucket / ModelState mirror the frontend TypeScript
// types. Field names use snake_case so the JSON shape matches what the
// frontend already serializes from localStorage — that lets a one-shot
// migration POST from the frontend drop straight in without reshaping.
type DailyBucket struct {
	Input     int64 `json:"input"`
	Output    int64 `json:"output"`
	Reasoning int64 `json:"reasoning"`
	Cached    int64 `json:"cached"`
	Total     int64 `json:"total"`
	Requests  int64 `json:"requests"`
}

type MonthlyBucket struct {
	Input     int64                 `json:"input"`
	Output    int64                 `json:"output"`
	Reasoning int64                 `json:"reasoning"`
	Cached    int64                 `json:"cached"`
	Total     int64                 `json:"total"`
	Requests  int64                 `json:"requests"`
	Daily     map[string]DailyBucket `json:"daily"`
}

type ModelState struct {
	Monthly       map[string]MonthlyBucket `json:"monthly"`
	LastUpdatedAt *int64                   `json:"lastUpdatedAt"`
	Lifetime      DailyBucket              `json:"lifetime"`
}

type PersistedShape struct {
	Version int                     `json:"version"`
	Models  map[string]ModelState   `json:"models"`
}

// Store is the on-disk usage aggregator. Safe for concurrent use.
type Store struct {
	mu   sync.RWMutex
	path string
	data PersistedShape
}

var (
	// singletonStore is lazily created on first access via Default().
	// Tests can construct their own Store via NewStore.
	singletonStore *Store
	singletonOnce  sync.Once
)

// ResolveDataDir returns the directory used for project-local data files.
// Mirrors the logic in logging.ResolveLogDirectory: prefer WRITABLE_PATH,
// fall back to "data" relative to the current working directory (which is
// the project root under launchd).
func ResolveDataDir() string {
	if base := util.WritablePath(); base != "" {
		return filepath.Join(base, "data")
	}
	return "data"
}

// NewStore creates a Store backed by `<dataDir>/token_usage.json`.
// If the file does not exist the store starts empty. If it exists but is
// corrupt the corrupt file is renamed to `<file>.corrupt-<unix-ts>` and a
// fresh store is returned — better to lose one snapshot than to crash the
// management plane and lock the user out of usage stats.
func NewStore(dataDir string) (*Store, error) {
	if strings.TrimSpace(dataDir) == "" {
		return nil, errors.New("tokenusage: dataDir is empty")
	}
	if err := os.MkdirAll(dataDir, 0o755); err != nil {
		return nil, fmt.Errorf("tokenusage: mkdir %s: %w", dataDir, err)
	}
	s := &Store{
		path: filepath.Join(dataDir, fileName),
		data: PersistedShape{Version: schemaVersion, Models: map[string]ModelState{}},
	}
	if err := s.load(); err != nil {
		// Defensive: backup the bad file and start fresh.
		_ = os.Rename(s.path, fmt.Sprintf("%s.corrupt-%d", s.path, time.Now().Unix()))
		log.Warnf("tokenusage: corrupt usage file, starting fresh: %v", err)
	}
	return s, nil
}

// Default returns the process-wide singleton store, creating it on first use.
func Default() *Store {
	singletonOnce.Do(func() {
		s, err := NewStore(ResolveDataDir())
		if err != nil {
			log.Errorf("tokenusage: failed to initialize default store: %v", err)
			// Fall back to an in-memory empty store so the management API
			// still responds instead of 500ing every poll.
			singletonStore = &Store{data: PersistedShape{Version: schemaVersion, Models: map[string]ModelState{}}}
			return
		}
		singletonStore = s
	})
	return singletonStore
}

// Path returns the on-disk file path. Useful for management GET responses.
func (s *Store) Path() string {
	if s == nil {
		return ""
	}
	return s.path
}

func (s *Store) load() error {
	raw, err := os.ReadFile(s.path)
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			return nil
		}
		return fmt.Errorf("read: %w", err)
	}
	var parsed PersistedShape
	if err := json.Unmarshal(raw, &parsed); err != nil {
		return fmt.Errorf("unmarshal: %w", err)
	}
	if parsed.Version != schemaVersion {
		return fmt.Errorf("unsupported version %d", parsed.Version)
	}
	if parsed.Models == nil {
		parsed.Models = map[string]ModelState{}
	}
	s.data = parsed
	return nil
}

// snapshot returns a deep-enough copy of the current state to hand back to
// HTTP callers without holding the lock during JSON marshaling.
func (s *Store) snapshot() PersistedShape {
	s.mu.RLock()
	defer s.mu.RUnlock()
	out := PersistedShape{Version: s.data.Version, Models: make(map[string]ModelState, len(s.data.Models))}
	for k, v := range s.data.Models {
		out.Models[k] = cloneModelState(v)
	}
	return out
}

// Snapshot returns a copy of the full persisted state. Used by GET endpoint.
func (s *Store) Snapshot() PersistedShape {
	if s == nil {
		return PersistedShape{Version: schemaVersion, Models: map[string]ModelState{}}
	}
	return s.snapshot()
}

// Replace overwrites the entire persisted state atomically. Used by the
// one-shot migration endpoint that drains a frontend localStorage payload
// into the file. The caller-supplied shape must already be valid; we still
// guard against nil maps.
func (s *Store) Replace(next PersistedShape) error {
	if s == nil {
		return errors.New("tokenusage: nil store")
	}
	if next.Version == 0 {
		next.Version = schemaVersion
	}
	if next.Models == nil {
		next.Models = map[string]ModelState{}
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.data = next
	return s.persistLocked()
}

// Merge folds `incoming` models into the persisted state, summing buckets
// field-by-field. Models missing on either side are treated as zero.
// `incoming` is never mutated.
func (s *Store) Merge(incoming PersistedShape) error {
	if s == nil {
		return errors.New("tokenusage: nil store")
	}
	if incoming.Models == nil {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	for key, inModel := range incoming.Models {
		existing := s.data.Models[key]
		merged := mergeModelState(existing, inModel)
		if s.data.Models == nil {
			s.data.Models = map[string]ModelState{}
		}
		s.data.Models[key] = merged
	}
	return s.persistLocked()
}

// persistLocked writes the current state to disk atomically (write to
// tmp file, fsync, rename). Caller must hold s.mu.
func (s *Store) persistLocked() error {
	if s.path == "" {
		return nil
	}
	tmp := s.path + ".tmp"
	raw, err := json.MarshalIndent(s.data, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal: %w", err)
	}
	if err := os.WriteFile(tmp, raw, 0o644); err != nil {
		return fmt.Errorf("write tmp: %w", err)
	}
	if err := os.Rename(tmp, s.path); err != nil {
		return fmt.Errorf("rename: %w", err)
	}
	return nil
}

func cloneModelState(m ModelState) ModelState {
	out := ModelState{
		Monthly:       make(map[string]MonthlyBucket, len(m.Monthly)),
		LastUpdatedAt: m.LastUpdatedAt,
		Lifetime:      m.Lifetime,
	}
	for k, v := range m.Monthly {
		daily := make(map[string]DailyBucket, len(v.Daily))
		for dk, dv := range v.Daily {
			daily[dk] = dv
		}
		v.Daily = daily
		out.Monthly[k] = v
	}
	return out
}

func mergeModelState(a, b ModelState) ModelState {
	merged := ModelState{
		Monthly: make(map[string]MonthlyBucket),
		Lifetime: addDaily(a.Lifetime, b.Lifetime),
	}
	switch {
	case a.LastUpdatedAt == nil && b.LastUpdatedAt == nil:
		merged.LastUpdatedAt = nil
	case a.LastUpdatedAt == nil:
		merged.LastUpdatedAt = b.LastUpdatedAt
	case b.LastUpdatedAt == nil:
		merged.LastUpdatedAt = a.LastUpdatedAt
	default:
		if *a.LastUpdatedAt > *b.LastUpdatedAt {
			merged.LastUpdatedAt = a.LastUpdatedAt
		} else {
			merged.LastUpdatedAt = b.LastUpdatedAt
		}
	}
	for month, aBucket := range a.Monthly {
		merged.Monthly[month] = mergeMonthly(aBucket, MonthlyBucket{Daily: map[string]DailyBucket{}})
	}
	for month, bBucket := range b.Monthly {
		if existing, ok := merged.Monthly[month]; ok {
			merged.Monthly[month] = mergeMonthly(existing, bBucket)
		} else {
			merged.Monthly[month] = cloneMonthly(bBucket)
		}
	}
	return merged
}

func mergeMonthly(a, b MonthlyBucket) MonthlyBucket {
	merged := addBucketsToMonthly(a, b)
	merged.Daily = make(map[string]DailyBucket, len(a.Daily)+len(b.Daily))
	for k, v := range a.Daily {
		merged.Daily[k] = v
	}
	for k, v := range b.Daily {
		if existing, ok := merged.Daily[k]; ok {
			merged.Daily[k] = addDaily(existing, v)
		} else {
			merged.Daily[k] = v
		}
	}
	return merged
}

func cloneMonthly(m MonthlyBucket) MonthlyBucket {
	out := m
	out.Daily = make(map[string]DailyBucket, len(m.Daily))
	for k, v := range m.Daily {
		out.Daily[k] = v
	}
	return out
}

// addBucketsToMonthly sums the scalar fields of two monthly buckets.
// (Daily buckets are merged separately by mergeMonthly to avoid double-
// counting.)
func addBucketsToMonthly(a, b MonthlyBucket) MonthlyBucket {
	return MonthlyBucket{
		Input:     a.Input + b.Input,
		Output:    a.Output + b.Output,
		Reasoning: a.Reasoning + b.Reasoning,
		Cached:    a.Cached + b.Cached,
		Total:     a.Total + b.Total,
		Requests:  a.Requests + b.Requests,
	}
}

func addDaily(a, b DailyBucket) DailyBucket {
	return DailyBucket{
		Input:     a.Input + b.Input,
		Output:    a.Output + b.Output,
		Reasoning: a.Reasoning + b.Reasoning,
		Cached:    a.Cached + b.Cached,
		Total:     a.Total + b.Total,
		Requests:  a.Requests + b.Requests,
	}
}