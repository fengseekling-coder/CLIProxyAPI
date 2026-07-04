package management

import (
	"encoding/json"
	"io"
	"net/http"

	"github.com/gin-gonic/gin"
	"github.com/router-for-me/CLIProxyAPI/v7/internal/tokenusage"
)

// GetUsageSummary returns the entire persisted token-usage state. The
// frontend calls this once on store start (and again whenever the user
// hits "Refresh" on the Models page) to hydrate its in-memory model from
// disk-backed truth instead of from per-browser localStorage.
//
// Response: { "version": 1, "models": { ... } }
//
// Returns an empty `models` map on first-ever run rather than 404 — the
// frontend treats "no data yet" as a normal state.
func (h *Handler) GetUsageSummary(c *gin.Context) {
	if h == nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "handler not initialized"})
		return
	}
	store := tokenusage.Default()
	c.JSON(http.StatusOK, store.Snapshot())
}

// PutUsageSummary replaces the entire persisted token-usage state. Used by
// the one-shot migration: when a frontend store discovers it has data in
// localStorage but the backend is empty (or explicitly tells the client
// to migrate), the client POSTs its localStorage dump here.
//
// The backend treats this as idempotent — same payload twice yields the
// same on-disk state. The frontend must clear its localStorage after a
// successful 2xx to avoid re-uploading.
func (h *Handler) PutUsageSummary(c *gin.Context) {
	if h == nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "handler not initialized"})
		return
	}
	body, err := io.ReadAll(c.Request.Body)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "read body: " + err.Error()})
		return
	}
	var payload tokenusage.PersistedShape
	if err := json.Unmarshal(body, &payload); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid JSON: " + err.Error()})
		return
	}
	store := tokenusage.Default()
	if err := store.Replace(payload); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "persist: " + err.Error()})
		return
	}
	c.JSON(http.StatusOK, gin.H{"status": "ok", "path": store.Path()})
}

// PostUsageSummary merges the request payload into the persisted state by
// summing matching model/month/day buckets. Used after every successful
// poll: the frontend sends only the records it just learned about and the
// backend folds them in. Merging (rather than replacing) means a freshly-
// opened tab can write its deltas without clobbering data written by a
// sibling tab or by an earlier browser session that already migrated.
//
// If the body is empty or `models` is missing the request is a no-op.
func (h *Handler) PostUsageSummary(c *gin.Context) {
	if h == nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "handler not initialized"})
		return
	}
	body, err := io.ReadAll(c.Request.Body)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "read body: " + err.Error()})
		return
	}
	if len(body) == 0 {
		c.JSON(http.StatusOK, gin.H{"status": "noop"})
		return
	}
	var payload tokenusage.PersistedShape
	if err := json.Unmarshal(body, &payload); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid JSON: " + err.Error()})
		return
	}
	store := tokenusage.Default()
	if err := store.Merge(payload); err != nil {
		// A persist failure should be visible to the caller so the frontend
		// can retry rather than silently lose data.
		c.JSON(http.StatusInternalServerError, gin.H{"error": "persist: " + err.Error()})
		return
	}
	c.JSON(http.StatusOK, gin.H{"status": "ok"})
}