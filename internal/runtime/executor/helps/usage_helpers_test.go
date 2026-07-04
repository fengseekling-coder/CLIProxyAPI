package helps

import (
	"context"
	"io"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/router-for-me/CLIProxyAPI/v7/sdk/cliproxy/usage"
)

func TestParseOpenAIUsageChatCompletions(t *testing.T) {
	data := []byte(`{"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3,"prompt_tokens_details":{"cached_tokens":4},"completion_tokens_details":{"reasoning_tokens":5}}}`)
	detail := ParseOpenAIUsage(data)
	if detail.InputTokens != 1 {
		t.Fatalf("input tokens = %d, want %d", detail.InputTokens, 1)
	}
	if detail.OutputTokens != 2 {
		t.Fatalf("output tokens = %d, want %d", detail.OutputTokens, 2)
	}
	if detail.TotalTokens != 3 {
		t.Fatalf("total tokens = %d, want %d", detail.TotalTokens, 3)
	}
	if detail.CachedTokens != 4 {
		t.Fatalf("cached tokens = %d, want %d", detail.CachedTokens, 4)
	}
	if detail.ReasoningTokens != 5 {
		t.Fatalf("reasoning tokens = %d, want %d", detail.ReasoningTokens, 5)
	}
}

func TestParseOpenAIUsageResponses(t *testing.T) {
	data := []byte(`{"usage":{"input_tokens":10,"output_tokens":20,"total_tokens":30,"input_tokens_details":{"cached_tokens":7},"output_tokens_details":{"reasoning_tokens":9}}}`)
	detail := ParseOpenAIUsage(data)
	if detail.InputTokens != 10 {
		t.Fatalf("input tokens = %d, want %d", detail.InputTokens, 10)
	}
	if detail.OutputTokens != 20 {
		t.Fatalf("output tokens = %d, want %d", detail.OutputTokens, 20)
	}
	if detail.TotalTokens != 30 {
		t.Fatalf("total tokens = %d, want %d", detail.TotalTokens, 30)
	}
	if detail.CachedTokens != 7 {
		t.Fatalf("cached tokens = %d, want %d", detail.CachedTokens, 7)
	}
	if detail.ReasoningTokens != 9 {
		t.Fatalf("reasoning tokens = %d, want %d", detail.ReasoningTokens, 9)
	}
}

func TestParseOpenAIUsageIgnoresNullUsage(t *testing.T) {
	data := []byte(`{"usage":null}`)
	detail := ParseOpenAIUsage(data)
	if detail != (usage.Detail{}) {
		t.Fatalf("detail = %+v, want zero detail", detail)
	}
}

func TestParseOpenAIStreamUsageIgnoresNullUsage(t *testing.T) {
	line := []byte(`data: {"id":"chunk_1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":null}],"usage":null}`)
	if detail, ok := ParseOpenAIStreamUsage(line); ok {
		t.Fatalf("ParseOpenAIStreamUsage() = (%+v, true), want false for null usage", detail)
	}
}

func TestParseOpenAIStreamUsageResponsesFields(t *testing.T) {
	line := []byte(`data: {"id":"chunk_1","object":"chat.completion.chunk","choices":[],"usage":{"input_tokens":8,"output_tokens":5,"total_tokens":13,"input_tokens_details":{"cached_tokens":3},"output_tokens_details":{"reasoning_tokens":2}}}`)
	detail, ok := ParseOpenAIStreamUsage(line)
	if !ok {
		t.Fatal("ParseOpenAIStreamUsage() ok = false, want true")
	}
	if detail.InputTokens != 8 {
		t.Fatalf("input tokens = %d, want %d", detail.InputTokens, 8)
	}
	if detail.OutputTokens != 5 {
		t.Fatalf("output tokens = %d, want %d", detail.OutputTokens, 5)
	}
	if detail.TotalTokens != 13 {
		t.Fatalf("total tokens = %d, want %d", detail.TotalTokens, 13)
	}
	if detail.CachedTokens != 3 {
		t.Fatalf("cached tokens = %d, want %d", detail.CachedTokens, 3)
	}
	if detail.ReasoningTokens != 2 {
		t.Fatalf("reasoning tokens = %d, want %d", detail.ReasoningTokens, 2)
	}
}

func TestParseClaudeUsageIncludesCacheTokensInTotal(t *testing.T) {
	data := []byte(`{"usage":{"input_tokens":3085,"output_tokens":253,"cache_read_input_tokens":7,"cache_creation_input_tokens":19514}}`)
	detail := ParseClaudeUsage(data)
	if detail.InputTokens != 3085 {
		t.Fatalf("input tokens = %d, want %d", detail.InputTokens, 3085)
	}
	if detail.OutputTokens != 253 {
		t.Fatalf("output tokens = %d, want %d", detail.OutputTokens, 253)
	}
	if detail.CacheReadTokens != 7 {
		t.Fatalf("cache read tokens = %d, want %d", detail.CacheReadTokens, 7)
	}
	if detail.CacheCreationTokens != 19514 {
		t.Fatalf("cache creation tokens = %d, want %d", detail.CacheCreationTokens, 19514)
	}
	if detail.CachedTokens != 7 {
		t.Fatalf("cached tokens = %d, want %d", detail.CachedTokens, 7)
	}
	if detail.TotalTokens != 22859 {
		t.Fatalf("total tokens = %d, want %d", detail.TotalTokens, 22859)
	}
}

func TestParseClaudeUsageCacheCreationGoesToTotalButNotCachedTokens(t *testing.T) {
	// F3 clarified the semantic: CachedTokens reflects cache HITS only.
	// CacheCreationTokens is its own bucket and still counts toward Total.
	data := []byte(`{"usage":{"input_tokens":3085,"output_tokens":253,"cache_creation_input_tokens":19514}}`)
	detail := ParseClaudeUsage(data)
	if detail.CachedTokens != 0 {
		t.Fatalf("cached tokens = %d, want 0 (cache_creation is not a cache hit)", detail.CachedTokens)
	}
	if detail.CacheCreationTokens != 19514 {
		t.Fatalf("cache creation tokens = %d, want %d", detail.CacheCreationTokens, 19514)
	}
	// Total = input + output + cache_creation (no cache_read in this payload)
	wantTotal := int64(3085 + 253 + 19514)
	if detail.TotalTokens != wantTotal {
		t.Fatalf("total tokens = %d, want %d", detail.TotalTokens, wantTotal)
	}
}

func TestNormalizeUsageDetailTotalIncludesCacheTokens(t *testing.T) {
	// F1 regression: when a record comes in with TotalTokens=0 but carries
	// cache tokens, the fallback should include cache so downstream sums
	// remain accurate.
	detail := normalizeUsageDetailTotal(usage.Detail{
		InputTokens:         1,
		CacheReadTokens:     2,
		CacheCreationTokens: 3,
	})
	if detail.TotalTokens != 6 {
		t.Fatalf("total tokens = %d, want 6 (input+cache_read+cache_creation)", detail.TotalTokens)
	}
}

func TestParseGeminiUsageReturnsFalseWhenMissingUsageMetadata(t *testing.T) {
	// F15: callers now check the bool to avoid publishing 0-token ghost rows.
	data := []byte(`{"candidates":[{"content":{"parts":[{"text":"hi"}]}}]}`)
	if _, ok := ParseGeminiUsage(data); ok {
		t.Fatalf("ParseGeminiUsage should return ok=false when usageMetadata is missing")
	}
}

func TestParseGeminiUsageReturnsFalseWhenAllZeroTokens(t *testing.T) {
	data := []byte(`{"usageMetadata":{"promptTokenCount":0,"candidatesTokenCount":0,"totalTokenCount":0}}`)
	if _, ok := ParseGeminiUsage(data); ok {
		t.Fatalf("ParseGeminiUsage should return ok=false when usageMetadata has no billable tokens")
	}
}

func TestParseGeminiUsageReturnsTrueWithRealTokens(t *testing.T) {
	data := []byte(`{"usageMetadata":{"promptTokenCount":3,"candidatesTokenCount":7,"totalTokenCount":10}}`)
	detail, ok := ParseGeminiUsage(data)
	if !ok {
		t.Fatalf("ParseGeminiUsage returned ok=false on a valid payload")
	}
	if detail.InputTokens != 3 || detail.OutputTokens != 7 || detail.TotalTokens != 10 {
		t.Fatalf("ParseGeminiUsage detail = %+v, want input=3, output=7, total=10", detail)
	}
}

func TestParseAntigravityUsageHonorsBool(t *testing.T) {
	if _, ok := ParseAntigravityUsage([]byte(`{"response":{"candidates":[]}}`)); ok {
		t.Fatalf("ParseAntigravityUsage should return ok=false when usageMetadata is missing")
	}
	detail, ok := ParseAntigravityUsage([]byte(`{"response":{"usageMetadata":{"promptTokenCount":4,"candidatesTokenCount":5,"totalTokenCount":9}}}`))
	if !ok {
		t.Fatalf("ParseAntigravityUsage returned ok=false on a valid payload")
	}
	if detail.TotalTokens != 9 {
		t.Fatalf("detail.TotalTokens = %d, want 9", detail.TotalTokens)
	}
}

func TestBoundedStopMapHonorsTTLAndCap(t *testing.T) {
	b := newBoundedStopMap(2)
	b.remember("trace-a")
	b.remember("trace-b")
	b.remember("trace-c") // evicts oldest
	if b.consume("trace-a") {
		t.Fatalf("trace-a should have been evicted by cap")
	}
	if !b.consume("trace-b") {
		t.Fatalf("trace-b should still be present")
	}
	if !b.consume("trace-c") {
		t.Fatalf("trace-c should still be present")
	}
}

func TestBoundedStopMapKeysAreHashed(t *testing.T) {
	b := newBoundedStopMap(8)
	b.remember("trace-X")
	b.mu.Lock()
	defer b.mu.Unlock()
	if _, ok := b.entries["trace-X"]; ok {
		t.Fatalf("traceID should be hashed, not stored verbatim")
	}
}

func TestHasNonZeroTokenUsageTreatsCacheReadAndCacheCreation(t *testing.T) {
	if !hasNonZeroTokenUsage(usage.Detail{CacheReadTokens: 1}) {
		t.Fatalf("CacheReadTokens should make the detail non-zero")
	}
	if !hasNonZeroTokenUsage(usage.Detail{CacheCreationTokens: 1}) {
		t.Fatalf("CacheCreationTokens should make the detail non-zero")
	}
	if hasNonZeroTokenUsage(usage.Detail{}) {
		t.Fatalf("all-zero detail should be treated as zero")
	}
}

func TestUsageReporterBuildRecordIncludesLatency(t *testing.T) {
	reporter := &UsageReporter{
		provider:    "openai",
		model:       "gpt-5.4",
		requestedAt: time.Now().Add(-1500 * time.Millisecond),
	}

	record := reporter.buildRecord(usage.Detail{TotalTokens: 3}, false)
	if record.Latency < time.Second {
		t.Fatalf("latency = %v, want >= 1s", record.Latency)
	}
	if record.Latency > 3*time.Second {
		t.Fatalf("latency = %v, want <= 3s", record.Latency)
	}
}

func TestUsageReporterTrackHTTPClientStartsTTFTBeforeRoundTrip(t *testing.T) {
	delay := 40 * time.Millisecond
	reporter := NewUsageReporter(context.Background(), "openai", "gpt-5.4", nil)
	client := reporter.TrackHTTPClient(&http.Client{
		Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
			time.Sleep(delay)
			return &http.Response{
				StatusCode: http.StatusOK,
				Status:     "200 OK",
				Header:     make(http.Header),
				Body:       io.NopCloser(strings.NewReader("ok")),
				Request:    req,
			}, nil
		}),
	})

	req, errNewRequest := http.NewRequestWithContext(context.Background(), http.MethodPost, "https://example.invalid/v1/chat/completions", strings.NewReader("{}"))
	if errNewRequest != nil {
		t.Fatalf("NewRequestWithContext() error = %v", errNewRequest)
	}
	resp, errDo := client.Do(req)
	if errDo != nil {
		t.Fatalf("Do() error = %v", errDo)
	}
	if _, errRead := io.ReadAll(resp.Body); errRead != nil {
		t.Fatalf("ReadAll() error = %v", errRead)
	}
	if errClose := resp.Body.Close(); errClose != nil {
		t.Fatalf("response body close error = %v", errClose)
	}
	if got := reporter.ttftDuration(); got < delay {
		t.Fatalf("ttft = %v, want >= %v", got, delay)
	}
}

func TestUsageReporterBuildRecordIncludesRequestedModelAlias(t *testing.T) {
	ctx := usage.WithRequestedModelAlias(context.Background(), "client-gpt")
	reporter := NewUsageReporter(ctx, "openai", "gpt-5.4", nil)

	record := reporter.buildRecord(usage.Detail{TotalTokens: 3}, false)
	if record.Model != "gpt-5.4" {
		t.Fatalf("model = %q, want %q", record.Model, "gpt-5.4")
	}
	if record.Alias != "client-gpt" {
		t.Fatalf("alias = %q, want %q", record.Alias, "client-gpt")
	}
}

func TestNewExecutorUsageReporterIncludesExecutorType(t *testing.T) {
	reporter := NewExecutorUsageReporter(context.Background(), &TestUsageExecutor{}, "gpt-5.4", nil)

	record := reporter.buildRecord(usage.Detail{TotalTokens: 3}, false)
	if record.Provider != "test-provider" {
		t.Fatalf("provider = %q, want %q", record.Provider, "test-provider")
	}
	if record.ExecutorType != "TestUsageExecutor" {
		t.Fatalf("executor type = %q, want %q", record.ExecutorType, "TestUsageExecutor")
	}
}

func TestUsageReporterBuildRecordIncludesReasoningEffort(t *testing.T) {
	ctx := usage.WithReasoningEffort(context.Background(), "medium")
	reporter := NewUsageReporter(ctx, "openai", "gpt-5.4", nil)

	record := reporter.buildRecord(usage.Detail{TotalTokens: 3}, false)
	if record.ReasoningEffort != "medium" {
		t.Fatalf("reasoning effort = %q, want %q", record.ReasoningEffort, "medium")
	}
}

func TestUsageReporterBuildRecordIncludesServiceTier(t *testing.T) {
	ctx := usage.WithServiceTier(context.Background(), "priority")
	reporter := NewUsageReporter(ctx, "openai", "gpt-5.4", nil)

	record := reporter.buildRecord(usage.Detail{TotalTokens: 3}, false)
	if record.ServiceTier != "priority" {
		t.Fatalf("service tier = %q, want %q", record.ServiceTier, "priority")
	}
}

func TestUsageReporterSetTranslatedReasoningEffortUpdatesServiceTier(t *testing.T) {
	reporter := NewUsageReporter(context.Background(), "openai", "gpt-5.4", nil)

	reporter.SetTranslatedReasoningEffort([]byte(`{"service_tier":"priority"}`), "openai")

	record := reporter.buildRecord(usage.Detail{TotalTokens: 3}, false)
	if record.ServiceTier != "priority" {
		t.Fatalf("service tier = %q, want %q", record.ServiceTier, "priority")
	}
}

func TestUsageReporterSetTranslatedReasoningEffortDefaultsServiceTierWhenRemoved(t *testing.T) {
	ctx := usage.WithServiceTier(context.Background(), "priority")
	reporter := NewUsageReporter(ctx, "openai", "gpt-5.4", nil)

	reporter.SetTranslatedReasoningEffort([]byte(`{"model":"gpt-5.4"}`), "openai")

	record := reporter.buildRecord(usage.Detail{TotalTokens: 3}, false)
	if record.ServiceTier != usage.DefaultServiceTier {
		t.Fatalf("service tier = %q, want %q", record.ServiceTier, usage.DefaultServiceTier)
	}
}

func TestUsageReporterBuildAdditionalModelRecordSkipsZeroTokens(t *testing.T) {
	reporter := &UsageReporter{
		provider:    "codex",
		model:       "gpt-5.4",
		requestedAt: time.Now(),
	}

	if _, ok := reporter.buildAdditionalModelRecord("gpt-image-2", usage.Detail{}); ok {
		t.Fatalf("expected all-zero token usage to be skipped")
	}
	if _, ok := reporter.buildAdditionalModelRecord("gpt-image-2", usage.Detail{InputTokens: 2}); !ok {
		t.Fatalf("expected non-zero input token usage to be recorded")
	}
	if _, ok := reporter.buildAdditionalModelRecord("gpt-image-2", usage.Detail{CachedTokens: 2}); !ok {
		t.Fatalf("expected non-zero cached token usage to be recorded")
	}
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(req *http.Request) (*http.Response, error) {
	return f(req)
}

type TestUsageExecutor struct{}

func (TestUsageExecutor) Identifier() string {
	return "test-provider"
}
