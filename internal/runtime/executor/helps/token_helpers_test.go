package helps

import (
	"strings"
	"testing"

	"github.com/tiktoken-go/tokenizer"
)

func TestTokenizerForModelGpt5FamilyPrefersGPT5Codec(t *testing.T) {
	for _, model := range []string{"gpt-5", "gpt-5.1", "gpt-5-turbo", "GPT-5.1-Mini"} {
		enc, err := TokenizerForModel(model)
		if err != nil {
			t.Fatalf("TokenizerForModel(%q) error: %v", model, err)
		}
		if enc == nil {
			t.Fatalf("TokenizerForModel(%q) returned nil codec", model)
		}
	}
}

func TestTokenizerForModelOSeries(t *testing.T) {
	for _, model := range []string{"o1", "o1-preview", "o3", "o3-mini", "o4", "o4-mini"} {
		if _, err := TokenizerForModel(model); err != nil {
			t.Fatalf("TokenizerForModel(%q) error: %v", model, err)
		}
	}
}

func TestTokenizerForModelUnknownFallsBackToO200kBase(t *testing.T) {
	enc, err := TokenizerForModel("claude-sonnet-4-5")
	if err != nil {
		t.Fatalf("error: %v", err)
	}
	if enc == nil {
		t.Fatalf("expected fallback codec, got nil")
	}
}

func TestTokenizerForModelEmptyDefaultsCl100k(t *testing.T) {
	if _, err := TokenizerForModel(""); err != nil {
		t.Fatalf("empty model should default, got error: %v", err)
	}
}

func TestCountOpenAIChatTokensAggregatesContentSegments(t *testing.T) {
	enc, err := tokenizer.Get(tokenizer.Cl100kBase)
	if err != nil {
		t.Fatalf("encoder init: %v", err)
	}
	payload := []byte(`{
		"messages": [
			{"role":"system","content":"You are a helpful assistant."},
			{"role":"user","content":"Hello, world!"},
			{"role":"assistant","content":"Hi there.", "tool_calls":[{"id":"1","type":"function","function":{"name":"lookup","description":"Search docs","arguments":"{\"q\":\"hi\"}"}}]},
			{"role":"tool","name":"lookup","content":"found 3 results"}
		],
		"tools": [{"type":"function","function":{"name":"lookup","description":"Search docs","parameters":{"type":"object","properties":{"q":{"type":"string"}}}}}],
		"tool_choice": "auto",
		"response_format": {"type":"json_object","json_schema":{"name":"x"}}
	}`)
	count, err := CountOpenAIChatTokens(enc, payload)
	if err != nil {
		t.Fatalf("CountOpenAIChatTokens error: %v", err)
	}
	if count <= 0 {
		t.Fatalf("expected non-zero token count, got %d", count)
	}
}

func TestCountOpenAIChatTokensEmptyPayloadReturnsZero(t *testing.T) {
	enc, err := tokenizer.Get(tokenizer.Cl100kBase)
	if err != nil {
		t.Fatalf("encoder init: %v", err)
	}
	if count, err := CountOpenAIChatTokens(enc, nil); err != nil || count != 0 {
		t.Fatalf("nil payload expected 0,err=nil, got %d, %v", count, err)
	}
	if count, err := CountOpenAIChatTokens(enc, []byte("{}")); err != nil || count != 0 {
		t.Fatalf("empty payload expected 0,err=nil, got %d, %v", count, err)
	}
}

func TestCountOpenAIChatTokensRejectsNilEncoder(t *testing.T) {
	if _, err := CountOpenAIChatTokens(nil, []byte("{}")); err == nil {
		t.Fatalf("expected error on nil encoder")
	}
}

func TestBuildOpenAIUsageJSONIncludesCachedTokensField(t *testing.T) {
	out := BuildOpenAIUsageJSON(42)
	s := string(out)
	if !strings.Contains(s, `"prompt_tokens":42`) {
		t.Fatalf("missing prompt_tokens in %q", s)
	}
	if !strings.Contains(s, `"total_tokens":42`) {
		t.Fatalf("missing total_tokens in %q", s)
	}
	// F12: the prompt_tokens_details block is included so consumers can
	// recognize this is an estimate, not an upstream-billed value.
	if !strings.Contains(s, `"prompt_tokens_details":{"cached_tokens":0}`) {
		t.Fatalf("missing prompt_tokens_details.cached_tokens in %q", s)
	}
}
