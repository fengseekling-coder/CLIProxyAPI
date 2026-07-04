package helps

import (
	"bytes"
	"context"
	"crypto/sha1"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"net/http"
	"reflect"
	"strings"
	"sync"
	"time"

	"github.com/gin-gonic/gin"
	internallogging "github.com/router-for-me/CLIProxyAPI/v7/internal/logging"
	"github.com/router-for-me/CLIProxyAPI/v7/internal/thinking"
	cliproxyauth "github.com/router-for-me/CLIProxyAPI/v7/sdk/cliproxy/auth"
	"github.com/router-for-me/CLIProxyAPI/v7/sdk/cliproxy/usage"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

type UsageReporter struct {
	provider     string
	executorType string
	model        string
	alias        string
	authID       string
	authIndex    string
	authType     string
	apiKey       string
	source       string
	reasoning    string
	serviceTier  string
	requestedAt  time.Time
	ttftMu       sync.RWMutex
	ttft         time.Duration
	ttftStart    time.Time
	ttftSet      bool

	// stateMu guards published / failure flags. The first publish sets
	// `published=true` and the detail it observed; a later failure that arrives
	// after a non-zero token publish upgrades the failure flag and emits a
	// second record so downstream dashboards don't silently lose the failure.
	// See publishWithOutcome + ensurePublishedForOutcome for the actual logic.
	stateMu        sync.Mutex
	published      bool
	publishedDetail usage.Detail
	publishedFailed bool
	publishedFail   usage.Failure

	// reasoningCharsMu guards reasoningChars.
	//
	// Some upstreams (GLM-5, DeepSeek-R1, Qwen3-Thinking, …) emit the
	// model's internal "thinking" tokens through a separate
	// `choices[].delta.reasoning_content` channel and report a flat
	// `completion_tokens` total that does not break out the reasoning
	// portion. `usage.completion_tokens_details.reasoning_tokens` is
	// therefore always 0 on those providers. To keep the per-model
	// "本月 TOKEN" column in the Management Center accurate, every SSE
	// chunk that carries a non-empty `reasoning_content` delta calls
	// AccumulateReasoning with the byte length of the delta; Publish()
	// folds the running total into Detail.ReasoningTokens via the
	// charsToTokens heuristic (~4 ASCII chars / token — close enough
	// for both CJK and Latin, since this only drives UI aggregates).
	reasoningCharsMu sync.Mutex
	reasoningChars   int
}

type usageExecutor interface {
	Identifier() string
}

func NewExecutorUsageReporter(ctx context.Context, executor usageExecutor, model string, auth *cliproxyauth.Auth) *UsageReporter {
	provider := ""
	if executor != nil {
		provider = executor.Identifier()
	}
	reporter := NewUsageReporter(ctx, provider, model, auth)
	reporter.executorType = ExecutorTypeName(executor)
	return reporter
}

func NewUsageReporter(ctx context.Context, provider, model string, auth *cliproxyauth.Auth) *UsageReporter {
	apiKey := APIKeyFromContext(ctx)
	alias := usage.RequestedModelAliasFromContext(ctx)
	if alias == "" {
		alias = model
	}
	reporter := &UsageReporter{
		provider:    provider,
		model:       model,
		alias:       strings.TrimSpace(alias),
		requestedAt: time.Now(),
		apiKey:      apiKey,
		source:      resolveUsageSource(auth, apiKey),
		authType:    resolveUsageAuthType(auth),
		reasoning:   usage.ReasoningEffortFromContext(ctx),
		serviceTier: usage.ServiceTierFromContext(ctx),
	}
	if auth != nil {
		reporter.authID = auth.ID
		reporter.authIndex = auth.EnsureIndex()
	}
	return reporter
}

func ExecutorTypeName(executor any) string {
	if executor == nil {
		return ""
	}
	executorType := reflect.TypeOf(executor)
	for executorType.Kind() == reflect.Pointer {
		executorType = executorType.Elem()
	}
	name := strings.TrimSpace(executorType.Name())
	if name == "" || name == "struct {}" {
		// Anonymous funcs / closures / unexported types fall back to the
		// provider identifier so downstream dashboards still get a stable
		// label. If the executor implements a static Provider() method, prefer
		// that.
		if p, ok := executor.(interface{ Provider() string }); ok {
			if v := strings.TrimSpace(p.Provider()); v != "" {
				return "anonymous(" + v + ")"
			}
		}
		return "anonymous"
	}
	return name
}

func (r *UsageReporter) Publish(ctx context.Context, detail usage.Detail) {
	r.publishWithOutcome(ctx, detail, false, usage.Failure{})
}

// PublishOpenAIChatUsage parses an OpenAI-style non-streamed usage payload and
// publishes it. If the payload lacks any token fields, it falls back to
// EnsurePublished so the request is still recorded as a (zero-token) row.
func (r *UsageReporter) PublishOpenAIChatUsage(ctx context.Context, body []byte) {
	if detail := ParseOpenAIUsage(body); hasNonZeroTokenUsage(detail) {
		r.Publish(ctx, detail)
		return
	}
	r.EnsurePublished(ctx)
}

// PublishGeminiUsage parses a Gemini-family usage payload (aistudio /
// antigravity / vertex / plain gemini) and publishes it. Falls back to
// EnsurePublished when the payload carries no usageMetadata.
func (r *UsageReporter) PublishGeminiUsage(ctx context.Context, body []byte) {
	if detail, ok := ParseGeminiUsage(body); ok {
		r.Publish(ctx, detail)
		return
	}
	if detail, ok := ParseAntigravityUsage(body); ok {
		r.Publish(ctx, detail)
		return
	}
	r.EnsurePublished(ctx)
}

// PublishCodexUsage parses a codex response.completed usage payload and
// publishes it. Falls back to EnsurePublished when the payload carries no
// usage.
func (r *UsageReporter) PublishCodexUsage(ctx context.Context, body []byte) {
	if detail, ok := ParseCodexUsage(body); ok {
		r.Publish(ctx, detail)
		return
	}
	r.EnsurePublished(ctx)
}

func (r *UsageReporter) PublishAdditionalModel(ctx context.Context, model string, detail usage.Detail) {
	record, ok := r.buildAdditionalModelRecord(model, detail)
	if !ok {
		return
	}
	r.publishRecord(ctx, record)
}

func (r *UsageReporter) SetTranslatedReasoningEffort(payload []byte, format string) {
	if r == nil {
		return
	}
	r.reasoning = thinking.ExtractTranslatedReasoningEffort(payload, format)
	r.serviceTier = extractServiceTierFromPayload(payload)
}

func (r *UsageReporter) TrackHTTPClient(client *http.Client) *http.Client {
	if r == nil || client == nil {
		return client
	}
	tracked := *client
	transport := tracked.Transport
	if transport == nil {
		transport = http.DefaultTransport
	}
	tracked.Transport = usageTTFTRoundTripper{
		base:     transport,
		reporter: r,
	}
	return &tracked
}

func (r *UsageReporter) ObserveResponse(resp *http.Response) {
	if r == nil || resp == nil || resp.Body == nil {
		return
	}
	r.StartResponseTTFT()
	resp.Body = &usageTTFTReadCloser{
		ReadCloser: resp.Body,
		mark: func() {
			r.MarkFirstResponseByte()
		},
	}
}

func (r *UsageReporter) StartResponseTTFT() {
	if r == nil {
		return
	}
	r.ttftMu.Lock()
	if !r.ttftSet && r.ttftStart.IsZero() {
		r.ttftStart = time.Now()
	}
	r.ttftMu.Unlock()
}

func (r *UsageReporter) MarkFirstResponseByte() {
	if r == nil {
		return
	}
	r.ttftMu.Lock()
	if r.ttftSet {
		r.ttftMu.Unlock()
		return
	}
	start := r.ttftStart
	r.ttftStart = time.Time{}
	r.ttftMu.Unlock()
	if start.IsZero() {
		return
	}
	r.setTTFT(time.Since(start))
}

func (r *UsageReporter) buildAdditionalModelRecord(model string, detail usage.Detail) (usage.Record, bool) {
	if r == nil {
		return usage.Record{}, false
	}
	model = strings.TrimSpace(model)
	if model == "" {
		return usage.Record{}, false
	}
	detail = normalizeUsageDetailTotal(detail)
	if !hasNonZeroTokenUsage(detail) {
		return usage.Record{}, false
	}
	return r.buildRecordForModel(model, detail, false, usage.Failure{}), true
}

func (r *UsageReporter) PublishFailure(ctx context.Context, errs ...error) {
	r.publishWithOutcome(ctx, usage.Detail{}, true, failFromErrors(errs...))
}

func (r *UsageReporter) TrackFailure(ctx context.Context, errPtr *error) {
	if r == nil || errPtr == nil {
		return
	}
	if *errPtr != nil {
		r.PublishFailure(ctx, *errPtr)
	}
}

func (r *UsageReporter) publishWithOutcome(ctx context.Context, detail usage.Detail, failed bool, fail usage.Failure) {
	if r == nil {
		return
	}
	detail = r.foldReasoningTokens(detail)
	detail = normalizeUsageDetailTotal(detail)

	r.stateMu.Lock()
	if !r.published {
		r.published = true
		r.publishedDetail = detail
		r.publishedFailed = failed
		r.publishedFail = fail
		r.stateMu.Unlock()
		r.publishRecord(ctx, r.buildRecord(detail, failed, fail))
		return
	}
	alreadyFailed := r.publishedFailed
	hasTokens := hasNonZeroTokenUsage(detail)
	r.stateMu.Unlock()

	if failed && !alreadyFailed && hasTokens {
		r.markFailureAndRepublish(ctx, fail)
	}
}

// AccumulateReasoning records the byte length of a single
// `choices[].delta.reasoning_content` SSE chunk. Streams from thinking
// models (GLM-5, DeepSeek-R1, Qwen3-Thinking, …) funnel their internal
// chain-of-thought through this field, but the upstream `usage` chunk
// reports `reasoning_tokens: 0` because the platform bills the reasoning
// pass as free. We sum the bytes here so Publish() can fold an estimated
// reasoning token count into Detail.ReasoningTokens — without this, every
// thinking-model request shows up in the Management Center as a content
// row of zero tokens even when several kB of reasoning fired.
//
// Safe to call concurrently; the underlying counter is mutex-guarded and
// intentionally never reset (it's monotonically accumulated between
// Publish calls; Publish reads and folds it exactly once).
func (r *UsageReporter) AccumulateReasoning(deltaBytes int) {
	if r == nil || deltaBytes <= 0 {
		return
	}
	r.reasoningCharsMu.Lock()
	r.reasoningChars += deltaBytes
	r.reasoningCharsMu.Unlock()
}

// foldReasoningTokens adds the accumulated reasoning-content byte count
// (estimated at ~4 bytes per token, matching OpenAI's published
// rule-of-thumb for English text) to detail.ReasoningTokens so that the
// row in the Models page reflects the real cost of running a thinking
// model. The estimate is intentionally conservative — CJK runs a little
// denser than 4 bytes/token, so the displayed value is a slight
// under-count for Chinese thinking content, but still much closer to the
// truth than the upstream-reported zero.
//
// When reasoning was absent from the upstream payload (think GLM-5 /
// DeepSeek-R1 / Qwen3-Thinking — they all report
// `usage.completion_tokens_details.reasoning_tokens: 0` while charging
// reasoning_content as free), the upstream `usage.total_tokens` also
// omits the reasoning portion — so we also recompute detail.TotalTokens
// from scratch via computeTokenTotal. Otherwise the Models page's 本月
// TOKEN column would understate the real spend by the entire reasoning
// bucket (e.g. for the GLM-5.2 "黑洞信息悖论" probe earlier today the
// upstream total was 2246 but the actual token spend was 2246 + 1148 =
// 3394). The same logic would be a no-op for upstreams that already
// count reasoning in their total_tokens (OpenAI, Anthropic, Gemini);
// computeTokenTotal is idempotent when the upstream accounting already
// double-counts.
func (r *UsageReporter) foldReasoningTokens(detail usage.Detail) usage.Detail {
	if r == nil {
		return detail
	}
	r.reasoningCharsMu.Lock()
	chars := r.reasoningChars
	r.reasoningChars = 0
	r.reasoningCharsMu.Unlock()
	if chars <= 0 {
		return detail
	}
	const charsPerToken = 4
	estimated := int64((chars + charsPerToken - 1) / charsPerToken)
	if estimated <= 0 {
		return detail
	}
	detail.ReasoningTokens += estimated
	detail.TotalTokens = computeTokenTotal(detail)
	return detail
}

// markFailureAndRepublish promotes a previously-successful token record into a
// failed record. Sinks must dedupe on (model, requested_at) for this to remain
// idempotent; the redis queue plugin already keeps records append-only.
func (r *UsageReporter) markFailureAndRepublish(ctx context.Context, fail usage.Failure) {
	if r == nil {
		return
	}
	r.stateMu.Lock()
	if r.publishedFailed {
		r.stateMu.Unlock()
		return
	}
	r.publishedFailed = true
	r.publishedFail = fail
	detail := r.publishedDetail
	r.stateMu.Unlock()
	r.publishRecord(ctx, r.buildRecord(detail, true, fail))
}

func normalizeUsageDetailTotal(detail usage.Detail) usage.Detail {
	if detail.TotalTokens == 0 {
		if total := computeTokenTotal(detail); total > 0 {
			detail.TotalTokens = total
		}
	}
	return detail
}

// computeTokenTotal derives a TotalTokens value from a detail by summing every
// contributing bucket (input, output, reasoning, cache). It is used as a
// fallback when the upstream payload omits total_tokens. CacheReadTokens is
// included because prompt caching is an actual charge on providers like
// Anthropic; CacheCreationTokens likewise on Anthropic + Vertex.
func computeTokenTotal(detail usage.Detail) int64 {
	return detail.InputTokens + detail.OutputTokens + detail.ReasoningTokens +
		detail.CacheReadTokens + detail.CacheCreationTokens
}

func hasNonZeroTokenUsage(detail usage.Detail) bool {
	return detail.InputTokens != 0 ||
		detail.OutputTokens != 0 ||
		detail.ReasoningTokens != 0 ||
		detail.CachedTokens != 0 ||
		detail.CacheReadTokens != 0 ||
		detail.CacheCreationTokens != 0 ||
		detail.TotalTokens != 0
}

// ensurePublished guarantees that a usage record is emitted exactly once.
// It is safe to call multiple times; only the first call wins. This is used to
// keep request counting accurate even when upstream responses do not include
// any usage fields (tokens), especially for streaming paths.
//
// If the first observation carries zero tokens (typical for stream paths where
// no final usage chunk was seen) AND no failure was raised, the record is
// still emitted with zero tokens so dashboards see "request happened" — but a
// subsequent PublishFailure call will upgrade that record to failed=true
// rather than appending a second zero-token success row.
func (r *UsageReporter) EnsurePublished(ctx context.Context) {
	if r == nil {
		return
	}
	r.stateMu.Lock()
	if r.published {
		r.stateMu.Unlock()
		return
	}
	r.published = true
	r.publishedFailed = false
	r.stateMu.Unlock()
	r.publishRecord(ctx, r.buildRecord(usage.Detail{}, false, usage.Failure{}))
}

func (r *UsageReporter) publishRecord(ctx context.Context, record usage.Record) {
	record.ResponseHeaders = internallogging.GetResponseHeaders(ctx)
	usage.PublishRecord(ctx, record)
}

func (r *UsageReporter) buildRecord(detail usage.Detail, failed bool, failures ...usage.Failure) usage.Record {
	var fail usage.Failure
	if len(failures) > 0 {
		fail = failures[0]
	}
	if r == nil {
		return usage.Record{Detail: detail, Failed: failed, Fail: fail}
	}
	return r.buildRecordForModel(r.model, detail, failed, fail)
}

func (r *UsageReporter) buildRecordForModel(model string, detail usage.Detail, failed bool, fail usage.Failure) usage.Record {
	if r == nil {
		return usage.Record{Model: model, Detail: detail, Failed: failed, Fail: fail}
	}
	return usage.Record{
		Provider:        r.provider,
		ExecutorType:    r.executorType,
		Model:           model,
		Alias:           r.alias,
		Source:          r.source,
		APIKey:          r.apiKey,
		AuthID:          r.authID,
		AuthIndex:       r.authIndex,
		AuthType:        r.authType,
		ReasoningEffort: r.reasoning,
		ServiceTier:     r.serviceTier,
		RequestedAt:     r.requestedAt,
		Latency:         r.latency(),
		TTFT:            r.ttftDuration(),
		Failed:          failed,
		Fail:            fail,
		Detail:          detail,
	}
}

func extractServiceTierFromPayload(payload []byte) string {
	if len(payload) == 0 {
		return usage.DefaultServiceTier
	}
	for _, path := range []string{"service_tier", "request.service_tier", "response.service_tier"} {
		serviceTier := strings.TrimSpace(gjson.GetBytes(payload, path).String())
		if serviceTier != "" {
			return serviceTier
		}
	}
	return usage.DefaultServiceTier
}

func failFromErrors(errs ...error) usage.Failure {
	for _, err := range errs {
		if err == nil {
			continue
		}
		fail := usage.Failure{
			Body: strings.TrimSpace(err.Error()),
		}
		var se interface{ StatusCode() int }
		if errors.As(err, &se) && se != nil {
			fail.StatusCode = se.StatusCode()
		}
		return fail
	}
	return usage.Failure{}
}

func (r *UsageReporter) latency() time.Duration {
	if r == nil || r.requestedAt.IsZero() {
		return 0
	}
	latency := time.Since(r.requestedAt)
	if latency < 0 {
		return 0
	}
	return latency
}

func (r *UsageReporter) setTTFT(ttft time.Duration) {
	if r == nil {
		return
	}
	if ttft < 0 {
		ttft = 0
	}
	r.ttftMu.Lock()
	if r.ttftSet {
		r.ttftMu.Unlock()
		return
	}
	r.ttft = ttft
	r.ttftSet = true
	r.ttftStart = time.Time{}
	r.ttftMu.Unlock()
}

func (r *UsageReporter) ttftDuration() time.Duration {
	if r == nil {
		return 0
	}
	r.ttftMu.RLock()
	defer r.ttftMu.RUnlock()
	return r.ttft
}

type usageTTFTRoundTripper struct {
	base     http.RoundTripper
	reporter *UsageReporter
}

func (t usageTTFTRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	t.reporter.StartResponseTTFT()
	resp, errRoundTrip := t.base.RoundTrip(req)
	if errRoundTrip != nil {
		return resp, errRoundTrip
	}
	t.reporter.ObserveResponse(resp)
	return resp, nil
}

type usageTTFTReadCloser struct {
	io.ReadCloser
	once sync.Once
	mark func()
}

func (r *usageTTFTReadCloser) Read(p []byte) (int, error) {
	if r == nil || r.ReadCloser == nil {
		return 0, io.ErrClosedPipe
	}
	n, errRead := r.ReadCloser.Read(p)
	if n > 0 && r.mark != nil {
		r.once.Do(r.mark)
	}
	return n, errRead
}

func APIKeyFromContext(ctx context.Context) string {
	if ctx == nil {
		return ""
	}
	ginCtx, ok := ctx.Value("gin").(*gin.Context)
	if !ok || ginCtx == nil {
		return ""
	}
	if v, exists := ginCtx.Get("userApiKey"); exists {
		switch value := v.(type) {
		case string:
			return value
		case fmt.Stringer:
			return value.String()
		default:
			return fmt.Sprintf("%v", value)
		}
	}
	return ""
}

func resolveUsageSource(auth *cliproxyauth.Auth, ctxAPIKey string) string {
	if auth != nil {
		provider := strings.TrimSpace(auth.Provider)
		if strings.EqualFold(provider, "vertex") {
			if auth.Metadata != nil {
				if projectID, ok := auth.Metadata["project_id"].(string); ok {
					if trimmed := strings.TrimSpace(projectID); trimmed != "" {
						return trimmed
					}
				}
				if project, ok := auth.Metadata["project"].(string); ok {
					if trimmed := strings.TrimSpace(project); trimmed != "" {
						return trimmed
					}
				}
			}
		}
		if _, value := auth.AccountInfo(); value != "" {
			return strings.TrimSpace(value)
		}
		if auth.Metadata != nil {
			if email, ok := auth.Metadata["email"].(string); ok {
				if trimmed := strings.TrimSpace(email); trimmed != "" {
					return trimmed
				}
			}
		}
		if auth.Attributes != nil {
			if key := strings.TrimSpace(auth.Attributes["api_key"]); key != "" {
				return key
			}
		}
	}
	if trimmed := strings.TrimSpace(ctxAPIKey); trimmed != "" {
		return trimmed
	}
	return ""
}

func resolveUsageAuthType(auth *cliproxyauth.Auth) string {
	if auth == nil {
		return ""
	}
	return auth.AuthKind()
}

func ParseCodexUsage(data []byte) (usage.Detail, bool) {
	usageNode := gjson.ParseBytes(data).Get("response.usage")
	if !hasOpenAIStyleUsageTokenFields(usageNode) {
		return usage.Detail{}, false
	}
	return parseOpenAIStyleUsageNode(usageNode), true
}

func ParseCodexImageToolUsage(data []byte) (usage.Detail, bool) {
	usageNode := gjson.ParseBytes(data).Get("response.tool_usage.image_gen")
	if !hasOpenAIStyleUsageTokenFields(usageNode) {
		return usage.Detail{}, false
	}
	return parseOpenAIStyleUsageNode(usageNode), true
}

func ParseOpenAIUsage(data []byte) usage.Detail {
	usageNode := gjson.ParseBytes(data).Get("usage")
	if !hasOpenAIStyleUsageTokenFields(usageNode) {
		return usage.Detail{}
	}
	return parseOpenAIStyleUsageNode(usageNode)
}

func hasOpenAIStyleUsageTokenFields(usageNode gjson.Result) bool {
	if !usageNode.Exists() || !usageNode.IsObject() {
		return false
	}
	return usageNode.Get("prompt_tokens").Exists() ||
		usageNode.Get("input_tokens").Exists() ||
		usageNode.Get("completion_tokens").Exists() ||
		usageNode.Get("output_tokens").Exists() ||
		usageNode.Get("total_tokens").Exists() ||
		usageNode.Get("prompt_tokens_details.cached_tokens").Exists() ||
		usageNode.Get("input_tokens_details.cached_tokens").Exists() ||
		usageNode.Get("completion_tokens_details.reasoning_tokens").Exists() ||
		usageNode.Get("output_tokens_details.reasoning_tokens").Exists()
}

func parseOpenAIStyleUsageNode(usageNode gjson.Result) usage.Detail {
	inputNode := usageNode.Get("prompt_tokens")
	if !inputNode.Exists() {
		inputNode = usageNode.Get("input_tokens")
	}
	outputNode := usageNode.Get("completion_tokens")
	if !outputNode.Exists() {
		outputNode = usageNode.Get("output_tokens")
	}
	detail := usage.Detail{
		InputTokens:  inputNode.Int(),
		OutputTokens: outputNode.Int(),
		TotalTokens:  usageNode.Get("total_tokens").Int(),
	}
	cached := usageNode.Get("prompt_tokens_details.cached_tokens")
	if !cached.Exists() {
		cached = usageNode.Get("input_tokens_details.cached_tokens")
	}
	if cached.Exists() {
		detail.CachedTokens = cached.Int()
	}
	reasoning := usageNode.Get("completion_tokens_details.reasoning_tokens")
	if !reasoning.Exists() {
		reasoning = usageNode.Get("output_tokens_details.reasoning_tokens")
	}
	if reasoning.Exists() {
		detail.ReasoningTokens = reasoning.Int()
	}
	return detail
}

func ParseOpenAIStreamUsage(line []byte) (usage.Detail, bool) {
	payload := jsonPayload(line)
	if len(payload) == 0 || !gjson.ValidBytes(payload) {
		return usage.Detail{}, false
	}
	usageNode := gjson.GetBytes(payload, "usage")
	if !hasOpenAIStyleUsageTokenFields(usageNode) {
		return usage.Detail{}, false
	}
	return parseOpenAIStyleUsageNode(usageNode), true
}

func ParseClaudeUsage(data []byte) usage.Detail {
	usageNode := gjson.ParseBytes(data).Get("usage")
	if !usageNode.Exists() {
		return usage.Detail{}
	}
	return parseClaudeUsageNode(usageNode)
}

func ParseClaudeStreamUsage(line []byte) (usage.Detail, bool) {
	payload := jsonPayload(line)
	if len(payload) == 0 || !gjson.ValidBytes(payload) {
		return usage.Detail{}, false
	}
	usageNode := gjson.GetBytes(payload, "usage")
	if !usageNode.Exists() {
		return usage.Detail{}, false
	}
	return parseClaudeUsageNode(usageNode), true
}

func parseClaudeUsageNode(usageNode gjson.Result) usage.Detail {
	cacheReadTokens := usageNode.Get("cache_read_input_tokens").Int()
	cacheCreationTokens := usageNode.Get("cache_creation_input_tokens").Int()
	detail := usage.Detail{
		InputTokens:         usageNode.Get("input_tokens").Int(),
		OutputTokens:        usageNode.Get("output_tokens").Int(),
		CacheReadTokens:     cacheReadTokens,
		CacheCreationTokens: cacheCreationTokens,
	}
	// CachedTokens is kept for backward compatibility (older reporters and the
	// redisqueue JSON wire format still rely on it). Semantically it reflects
	// "cache read" hits only — CacheCreation is intentionally excluded so that
	// downstream dashboards do not confuse a fresh cache write with a cache hit.
	detail.CachedTokens = cacheReadTokens
	detail.TotalTokens = computeTokenTotal(detail)
	return detail
}

func parseGeminiFamilyUsageDetail(node gjson.Result) usage.Detail {
	detail := usage.Detail{
		InputTokens:     node.Get("promptTokenCount").Int(),
		OutputTokens:    node.Get("candidatesTokenCount").Int(),
		ReasoningTokens: node.Get("thoughtsTokenCount").Int(),
		TotalTokens:     node.Get("totalTokenCount").Int(),
		CachedTokens:    node.Get("cachedContentTokenCount").Int(),
	}
	if detail.TotalTokens == 0 {
		detail.TotalTokens = computeTokenTotal(detail)
	}
	return detail
}

func ParseGeminiUsage(data []byte) (usage.Detail, bool) {
	usageNode := gjson.ParseBytes(data)
	node := usageNode.Get("usageMetadata")
	if !node.Exists() {
		node = usageNode.Get("usage_metadata")
	}
	if !node.Exists() {
		return usage.Detail{}, false
	}
	detail := parseGeminiFamilyUsageDetail(node)
	if !hasNonZeroTokenUsage(detail) {
		return usage.Detail{}, false
	}
	return detail, true
}

func ParseGeminiStreamUsage(line []byte) (usage.Detail, bool) {
	payload := jsonPayload(line)
	if len(payload) == 0 || !gjson.ValidBytes(payload) {
		return usage.Detail{}, false
	}
	node := gjson.GetBytes(payload, "usageMetadata")
	if !node.Exists() {
		node = gjson.GetBytes(payload, "usage_metadata")
	}
	if !node.Exists() {
		return usage.Detail{}, false
	}
	return parseGeminiFamilyUsageDetail(node), true
}

func firstExistingUsageNode(root gjson.Result, paths ...string) gjson.Result {
	for _, path := range paths {
		node := root.Get(path)
		if node.Exists() {
			return node
		}
	}
	return gjson.Result{}
}

func ParseAntigravityUsage(data []byte) (usage.Detail, bool) {
	usageNode := gjson.ParseBytes(data)
	node := usageNode.Get("response.usageMetadata")
	if !node.Exists() {
		node = usageNode.Get("usageMetadata")
	}
	if !node.Exists() {
		node = usageNode.Get("usage_metadata")
	}
	if !node.Exists() {
		return usage.Detail{}, false
	}
	detail := parseGeminiFamilyUsageDetail(node)
	if !hasNonZeroTokenUsage(detail) {
		return usage.Detail{}, false
	}
	return detail, true
}

func ParseAntigravityStreamUsage(line []byte) (usage.Detail, bool) {
	payload := jsonPayload(line)
	if len(payload) == 0 || !gjson.ValidBytes(payload) {
		return usage.Detail{}, false
	}
	node := gjson.GetBytes(payload, "response.usageMetadata")
	if !node.Exists() {
		node = gjson.GetBytes(payload, "usageMetadata")
	}
	if !node.Exists() {
		node = gjson.GetBytes(payload, "usage_metadata")
	}
	if !node.Exists() {
		return usage.Detail{}, false
	}
	return parseGeminiFamilyUsageDetail(node), true
}

// stopChunkWithoutUsage correlates stream "stop" chunks that arrive without
// usageMetadata with the next chunk that DOES carry usageMetadata so we can
// leave both untouched (terminal chunks are not stripped). The map is bounded
// by both a hard cap and a TTL: traceIDs are stored as a short hex hash and
// dropped after 10 minutes, so memory cannot grow unbounded under hostile or
// chatty upstream workloads.
const (
	stopChunkTTL          = 10 * time.Minute
	stopChunkMaxEntries   = 4096
	stopChunkHashByteLen  = 8
)

var stopChunkWithoutUsage = newBoundedStopMap(stopChunkMaxEntries)

func rememberStopWithoutUsage(traceID string) {
	stopChunkWithoutUsage.remember(traceID)
}

// FilterSSEUsageMetadata removes usageMetadata from SSE events that are not
// terminal (finishReason != "stop"). Stop chunks are left untouched. This
// function is shared between aistudio and antigravity executors.
func FilterSSEUsageMetadata(payload []byte) []byte {
	if len(payload) == 0 {
		return payload
	}

	lines := bytes.Split(payload, []byte("\n"))
	modified := false
	foundData := false
	for idx, line := range lines {
		trimmed := bytes.TrimSpace(line)
		if len(trimmed) == 0 || !bytes.HasPrefix(trimmed, []byte("data:")) {
			continue
		}
		foundData = true
		dataIdx := bytes.Index(line, []byte("data:"))
		if dataIdx < 0 {
			continue
		}
		rawJSON := bytes.TrimSpace(line[dataIdx+5:])
		traceID := gjson.GetBytes(rawJSON, "traceId").String()
		if isStopChunkWithoutUsage(rawJSON) && traceID != "" {
			rememberStopWithoutUsage(traceID)
			continue
		}
		if traceID != "" {
			if ok := stopChunkWithoutUsage.consume(traceID); ok && hasUsageMetadata(rawJSON) {
				continue
			}
		}

		cleaned, changed := StripUsageMetadataFromJSON(rawJSON)
		if !changed {
			continue
		}
		var rebuilt []byte
		rebuilt = append(rebuilt, line[:dataIdx]...)
		rebuilt = append(rebuilt, []byte("data:")...)
		if len(cleaned) > 0 {
			rebuilt = append(rebuilt, ' ')
			rebuilt = append(rebuilt, cleaned...)
		}
		lines[idx] = rebuilt
		modified = true
	}
	if !modified {
		if !foundData {
			// Handle payloads that are raw JSON without SSE data: prefix.
			trimmed := bytes.TrimSpace(payload)
			cleaned, changed := StripUsageMetadataFromJSON(trimmed)
			if !changed {
				return payload
			}
			return cleaned
		}
		return payload
	}
	return bytes.Join(lines, []byte("\n"))
}

// StripUsageMetadataFromJSON drops usageMetadata unless finishReason is present (terminal).
// It handles both formats:
// - Aistudio: candidates.0.finishReason
// - Antigravity: response.candidates.0.finishReason
func StripUsageMetadataFromJSON(rawJSON []byte) ([]byte, bool) {
	jsonBytes := bytes.TrimSpace(rawJSON)
	if len(jsonBytes) == 0 || !gjson.ValidBytes(jsonBytes) {
		return rawJSON, false
	}

	// Check for finishReason in both aistudio and antigravity formats
	finishReason := gjson.GetBytes(jsonBytes, "candidates.0.finishReason")
	if !finishReason.Exists() {
		finishReason = gjson.GetBytes(jsonBytes, "response.candidates.0.finishReason")
	}
	terminalReason := finishReason.Exists() && strings.TrimSpace(finishReason.String()) != ""

	usageMetadata := gjson.GetBytes(jsonBytes, "usageMetadata")
	if !usageMetadata.Exists() {
		usageMetadata = gjson.GetBytes(jsonBytes, "response.usageMetadata")
	}

	// Terminal chunk: keep as-is.
	if terminalReason {
		return rawJSON, false
	}

	// Nothing to strip
	if !usageMetadata.Exists() {
		return rawJSON, false
	}

	// Hide usageMetadata from downstream clients by renaming it to
	// cpaUsageMetadata. This is a two-step rename (SetRawBytes → DeleteBytes)
	// because sjson has no atomic "rename" primitive; doing it this way
	// preserves nested array / object structure intact. The renamed field
	// keeps the upstream value around for our own FilterSSEUsageMetadata
	// correlation (see stopChunkWithoutUsage).
	cleaned := jsonBytes
	var changed bool

	if usageMetadata = gjson.GetBytes(cleaned, "usageMetadata"); usageMetadata.Exists() {
		cleaned, _ = sjson.SetRawBytes(cleaned, "cpaUsageMetadata", []byte(usageMetadata.Raw))
		cleaned, _ = sjson.DeleteBytes(cleaned, "usageMetadata")
		changed = true
	}

	if usageMetadata = gjson.GetBytes(cleaned, "response.usageMetadata"); usageMetadata.Exists() {
		cleaned, _ = sjson.SetRawBytes(cleaned, "response.cpaUsageMetadata", []byte(usageMetadata.Raw))
		cleaned, _ = sjson.DeleteBytes(cleaned, "response.usageMetadata")
		changed = true
	}

	return cleaned, changed
}

func hasUsageMetadata(jsonBytes []byte) bool {
	if len(jsonBytes) == 0 || !gjson.ValidBytes(jsonBytes) {
		return false
	}
	if gjson.GetBytes(jsonBytes, "usageMetadata").Exists() {
		return true
	}
	if gjson.GetBytes(jsonBytes, "response.usageMetadata").Exists() {
		return true
	}
	return false
}

func isStopChunkWithoutUsage(jsonBytes []byte) bool {
	if len(jsonBytes) == 0 || !gjson.ValidBytes(jsonBytes) {
		return false
	}
	finishReason := gjson.GetBytes(jsonBytes, "candidates.0.finishReason")
	if !finishReason.Exists() {
		finishReason = gjson.GetBytes(jsonBytes, "response.candidates.0.finishReason")
	}
	trimmed := strings.TrimSpace(finishReason.String())
	if !finishReason.Exists() || trimmed == "" {
		return false
	}
	return !hasUsageMetadata(jsonBytes)
}

func JSONPayload(line []byte) []byte {
	return jsonPayload(line)
}

func jsonPayload(line []byte) []byte {
	trimmed := bytes.TrimSpace(line)
	if len(trimmed) == 0 {
		return nil
	}
	if bytes.Equal(trimmed, []byte("[DONE]")) {
		return nil
	}
	if bytes.HasPrefix(trimmed, []byte("event:")) {
		return nil
	}
	if bytes.HasPrefix(trimmed, []byte("data:")) {
		trimmed = bytes.TrimSpace(trimmed[len("data:"):])
	}
	if len(trimmed) == 0 || trimmed[0] != '{' {
		return nil
	}
	return trimmed
}

// boundedStopMap caps the number of "stop chunk without usage" entries and
// expires them after a TTL. Keys are stored as a short hex digest of the
// original traceID so attackers cannot blow up memory by sending large
// traceIDs. The implementation is intentionally simple — a single mutex around
// a map + a slice for FIFO eviction — because the workload is "remember an ID
// briefly, then forget".
type boundedStopMap struct {
	mu      sync.Mutex
	cap     int
	ttl     time.Duration
	entries map[string]time.Time
	order   []string
}

func newBoundedStopMap(cap int) *boundedStopMap {
	return &boundedStopMap{
		cap:     cap,
		ttl:     stopChunkTTL,
		entries: make(map[string]time.Time, cap),
		order:   make([]string, 0, cap),
	}
}

func (b *boundedStopMap) remember(traceID string) {
	if b == nil {
		return
	}
	key := hashTraceID(traceID)
	if key == "" {
		return
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	now := time.Now()
	if existing, ok := b.entries[key]; ok {
		b.entries[key] = now
		_ = existing
		return
	}
	if len(b.entries) >= b.cap {
		// FIFO evict the oldest entry to keep the cap.
		old := b.order[0]
		b.order = b.order[1:]
		delete(b.entries, old)
	}
	b.entries[key] = now
	b.order = append(b.order, key)
}

// consume returns true and removes the entry if it was remembered within TTL.
// Used when a follow-up chunk carries usageMetadata and the previously-remembered
// stop chunk should be paired with it (so we do not strip the usage chunk).
func (b *boundedStopMap) consume(traceID string) bool {
	if b == nil {
		return false
	}
	key := hashTraceID(traceID)
	if key == "" {
		return false
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	stored, ok := b.entries[key]
	if !ok {
		return false
	}
	if time.Since(stored) > b.ttl {
		delete(b.entries, key)
		return false
	}
	delete(b.entries, key)
	// Remove from order slice (linear scan is fine; cap is small).
	for i, v := range b.order {
		if v == key {
			b.order = append(b.order[:i], b.order[i+1:]...)
			break
		}
	}
	return true
}

// hashTraceID returns a short hex digest of the traceID. We deliberately do
// not store the full ID to prevent hostile upstream payloads from inflating
// the bounded map's effective memory cost.
func hashTraceID(traceID string) string {
	if traceID == "" {
		return ""
	}
	sum := sha1.Sum([]byte(traceID))
	return hex.EncodeToString(sum[:])[:stopChunkHashByteLen*2]
}
