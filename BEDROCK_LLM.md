# AWS Bedrock LLM Provider Proposal

This document proposes adding AWS Bedrock as an LLM provider for the neo4j-graphrag-python package, complementing the existing BedrockEmbeddings implementation.

---

## Design Decisions

> **Last Updated:** January 2026

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Default Model | Claude Sonnet 4.5 | Latest balanced model |
| Model Focus | Claude 4.x only | Keep simple, expand later |
| Multimodal | Text-only | Simplicity first |
| Streaming | Deferred | Not needed for initial release |
| Model Lifecycle | No warnings | Keep implementation simple |
---

## Implementation Status

| Phase | Status | Notes |
|-------|--------|-------|
| Phase 1: Core Implementation | ✅ Complete | `src/neo4j_graphrag/llm/bedrock_llm.py` |
| Phase 2: Unit Tests | ✅ Complete | `tests/unit/llm/test_bedrock_llm.py` (28 tests) |
| Phase 3: Integration | ✅ Complete | Export added to `src/neo4j_graphrag/llm/__init__.py` |

---

## Executive Summary

AWS Bedrock provides access to foundation models from multiple providers (Anthropic Claude, Amazon Titan, Meta Llama, Mistral, and others) through a unified API. Adding BedrockLLM support enables AWS-centric organizations to use neo4j-graphrag features like SimpleKGPipeline for entity extraction, Text2CypherRetriever for natural language queries, and GraphRAG pipelines without leaving the AWS ecosystem.

This proposal builds on the successfully implemented BedrockEmbeddings provider and uses the same dependency (boto3) and authentication patterns.

---

## Current State Analysis

### Supported LLM Providers

The project now supports eight LLM providers:

| Provider | Class | Default Model |
|----------|-------|---------------|
| OpenAI | `OpenAILLM` | gpt-4o |
| Azure OpenAI | `AzureOpenAILLM` | (required parameter) |
| Anthropic | `AnthropicLLM` | claude-3-5-sonnet |
| **AWS Bedrock** | **`BedrockLLM`** | **claude-sonnet-4-5** |
| Google Vertex AI | `VertexAILLM` | gemini-1.5-flash-001 |
| Cohere | `CohereLLM` | command-r |
| Mistral AI | `MistralAILLM` | mistral-small-latest |
| Ollama | `OllamaLLM` | (required parameter) |

### ✅ Gap Closed: AWS Bedrock LLM Support Added

BedrockLLM has been implemented to complement BedrockEmbeddings. Users can now use Bedrock-hosted models (like Claude via Bedrock) for entity extraction, Cypher generation, and other LLM-dependent features.

This enables complete AWS-native GraphRAG workflows using both BedrockEmbeddings and BedrockLLM together.

### Architecture Pattern

All LLM providers follow a consistent plugin pattern:

1. Inherit from LLMInterfaceV2 (the current interface, replacing deprecated LLMInterface)
2. Implement required invoke and ainvoke methods for both sync and async operations
3. Support optional tool/function calling via invoke_with_tools and ainvoke_with_tools
4. Support optional rate limiting via RateLimitHandler
5. Use lazy imports for optional dependencies
6. Wrap provider-specific errors in LLMGenerationError

---

## Why Bedrock for LLM Access

### Unified AWS Billing and Governance

Organizations using AWS benefit from consolidated billing, IAM-based access control, and compliance with existing AWS governance policies. Using Bedrock means no separate API keys or vendor relationships to manage.

### The Converse API Advantage

AWS Bedrock provides the Converse API, a unified interface that works consistently across all Bedrock-hosted models. This means:

- Same request/response format for Claude, Llama, Mistral, Titan, and other models
- Built-in tool/function calling support with consistent schema
- Automatic retry handling (up to 5 retries by default)
- Streaming support via ConverseStream
- No need to handle provider-specific request formats

The Converse API is the recommended approach for all Bedrock LLM interactions as of 2025.

### Model Flexibility

With a single BedrockLLM implementation, users can access:

- Anthropic Claude (3.5 Sonnet, 3 Haiku, 3 Opus, Sonnet 4, etc.)
- Amazon Titan Text models
- Meta Llama models
- Mistral AI models
- AI21 Jamba models
- Cohere Command models

Users simply change the model_id parameter to switch between models.

---

## Supported Models

### Recommended Default: Claude 3.5 Sonnet v2

- Model ID: `anthropic.claude-3-5-sonnet-20241022-v2:0`
- Context Window: 200,000 tokens
- Strengths: Excellent at structured extraction, code generation, and following complex instructions
- Access: Direct on-demand invocation (no inference profile required)
- Note: Anthropic models require a one-time usage form acceptance in the AWS console

### Latest Models (January 2026)

The following models are now available on Bedrock with full Converse API and tool use support:

**Anthropic Claude 4.x Series:**
| Model | Model ID | Notes |
|-------|----------|-------|
| Claude Opus 4.5 | `anthropic.claude-opus-4-5-20251101-v1:0` | Most capable, vision support |
| Claude Sonnet 4.5 | `anthropic.claude-sonnet-4-5-20250929-v1:0` | Balanced performance/cost |
| Claude Haiku 4.5 | `anthropic.claude-haiku-4-5-20251001-v1:0` | Fast, cost-effective |
| Claude Opus 4.1 | `anthropic.claude-opus-4-1-20250805-v1:0` | Vision support |
| Claude Sonnet 4 | `anthropic.claude-sonnet-4-20250514-v1:0` | Vision support |

**Amazon Nova Series (AWS-native):**
| Model | Model ID | Notes |
|-------|----------|-------|
| Nova Premier | `amazon.nova-premier-v1:0` | Most capable, vision + video |
| Nova Pro | `amazon.nova-pro-v1:0` | Balanced, vision + video |
| Nova Lite | `amazon.nova-lite-v1:0` | Fast, vision + video |
| Nova Micro | `amazon.nova-micro-v1:0` | Text only, lowest cost |

**Meta Llama 4 Series:**
| Model | Model ID | Notes |
|-------|----------|-------|
| Llama 4 Maverick 17B | `meta.llama4-maverick-17b-instruct-v1:0` | Vision support, tool use |
| Llama 4 Scout 17B | `meta.llama4-scout-17b-instruct-v1:0` | Vision support, tool use |
| Llama 3.3 70B | `meta.llama3-3-70b-instruct-v1:0` | Text only |

**Mistral AI Latest:**
| Model | Model ID | Notes |
|-------|----------|-------|
| Mistral Large 3 | `mistral.mistral-large-3-675b-instruct` | Vision support |
| Pixtral Large | `mistral.pixtral-large-2502-v1:0` | Vision-focused |
| Magistral Small | `mistral.magistral-small-2509` | Vision support |

### Legacy Models (Still Supported)

Users can still specify older Bedrock models via the model_id parameter:

| Model Family | Example Model ID | Use Case |
|--------------|------------------|----------|
| Claude 3.5 Haiku | `anthropic.claude-3-5-haiku-20241022-v1:0` | Fast, cost-effective |
| Claude 3 Opus | `anthropic.claude-3-opus-20240229-v1:0` | Complex reasoning (legacy) |
| Amazon Titan | `amazon.titan-text-premier-v1:0` | AWS-native option |
| Meta Llama 3.1 405B | `meta.llama3-1-405b-instruct-v1:0` | Large open-weight |
| Mistral Large 2 | `mistral.mistral-large-2407-v1:0` | European provider option |

---

## Proposed Implementation

### BedrockLLM Class

A class for LLM text generation following the existing provider pattern and using the Converse API.

**Constructor Parameters:**

- model_id (string): The Bedrock model identifier. Defaults to Claude 3.5 Sonnet v2.
- region_name (string, optional): AWS region. Falls back to AWS_REGION or AWS_DEFAULT_REGION environment variable.
- inference_profile_id (string, optional): Inference profile ARN for cross-region inference. When provided, used instead of model_id. Required for some newer models like Claude Sonnet 4.5.
- client (optional): A pre-configured boto3 bedrock-runtime client. If provided, region_name is ignored.
- model_params (dict, optional): Additional parameters passed to the model (temperature, max_tokens, etc.)
- rate_limit_handler (optional): Custom rate limit handler
- Additional kwargs: Passed to boto3.client() if client is not provided

**Required Methods:**

- invoke: Send messages to the LLM and retrieve a response (synchronous)
- ainvoke: Same as invoke but asynchronous

**Optional Methods (Tool Calling):**

- invoke_with_tools: Send messages with tool definitions and retrieve tool call response
- ainvoke_with_tools: Same as invoke_with_tools but asynchronous

The Converse API natively supports tool calling, so implementing these methods is straightforward.

### Inference Profiles

Some newer Bedrock models (like Claude Sonnet 4.5) require inference profiles for access. The implementation should support both direct model invocation and inference profile usage, following the same pattern as BedrockEmbeddings.

### Configuration Pattern

Follow the same pattern as BedrockEmbeddings and other providers. boto3 handles credential resolution automatically through its default chain:

- Environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
- Shared credentials file (~/.aws/credentials)
- IAM roles (when running on AWS infrastructure)
- AWS SSO / Identity Center

No special credential handling is needed in the implementation.

### Dependency Management

Uses the existing bedrock optional dependency group already defined for BedrockEmbeddings:

- Package: boto3
- Version: >=1.35.0 (already specified)
- Installation: pip install "neo4j-graphrag[bedrock]"

No changes to pyproject.toml dependencies are needed.

### File Structure

New files:
- src/neo4j_graphrag/llm/bedrock_llm.py (implementation)
- tests/unit/llm/test_bedrock_llm.py (unit tests)

Modified files:
- src/neo4j_graphrag/llm/__init__.py (add export)

---

## Error Handling

Map Bedrock-specific errors to the project's exception hierarchy:

| Bedrock Error | neo4j-graphrag Exception |
|---------------|--------------------------|
| ThrottlingException | Trigger rate limit retry |
| ValidationException | LLMGenerationError |
| ModelNotReadyException | LLMGenerationError |
| AccessDeniedException | LLMGenerationError |
| ServiceQuotaExceededException | Trigger rate limit retry |
| ModelTimeoutException | LLMGenerationError |
| ModelStreamErrorException | LLMGenerationError |

Rate limit detection should look for common patterns in error messages: "throttling", "rate", "429", "quota exceeded".

---

## Design Decisions

### Use the Converse API

The implementation should use the Converse API (bedrock-runtime converse operation) rather than the older invoke_model operation. The Converse API provides:

- Unified message format across all models
- Native tool calling support
- Consistent error handling
- Streaming support (for future enhancement)

### Support Both Interface Versions

For backward compatibility, the implementation should work with both the deprecated LLMInterface (string-based input) and the current LLMInterfaceV2 (message list input). This matches how other providers handle the transition.

### Implement Tool Calling

Unlike some providers that raise NotImplementedError for tool calling, BedrockLLM should implement invoke_with_tools and ainvoke_with_tools. The Converse API has native tool support, making this straightforward.

Tool calling enables advanced use cases like:
- Structured entity extraction with guaranteed schema
- Agent-based graph navigation
- Multi-step reasoning with tool use

### Synchronous and Asynchronous Support

Implement both sync (invoke) and async (ainvoke) methods. boto3 is synchronous, so the async methods will need to run the sync code in an executor, following the pattern used by other providers.

### No Streaming in Initial Implementation

While the Converse API supports streaming via ConverseStream, the initial implementation should focus on non-streaming responses to keep complexity low. Streaming can be added in a future enhancement.

---

## Implementation Plan

### Phase 1: Core Implementation

- Create src/neo4j_graphrag/llm/bedrock_llm.py
- Implement BedrockLLM class inheriting from LLMInterfaceV2
- Support model_id, region_name, inference_profile_id parameters
- Implement invoke method using Converse API
- Implement ainvoke method using executor for async support
- Add error handling and rate limit detection
- Implement invoke_with_tools and ainvoke_with_tools for tool calling

### Phase 2: Unit Tests

- Create tests/unit/llm/test_bedrock_llm.py
- Test missing dependency error
- Test successful text generation (mocked)
- Test message formatting (user, assistant, system roles)
- Test error handling (validation, access denied, throttling)
- Test rate limit retry behavior
- Test inference profile usage
- Test tool calling with mock tool definitions

### Phase 3: Integration

- Update src/neo4j_graphrag/llm/__init__.py with export
- Add integration tests that require AWS credentials (skip if not available)
- Update documentation with usage examples
- Test with SimpleKGPipeline for entity extraction
- Test with Text2CypherRetriever for query generation

---

## Testing Strategy

### Unit Tests

- Mock boto3 client responses
- Test successful text generation
- Test error handling (validation errors, access denied, throttling)
- Test rate limit retry behavior
- Test inference profile parameter
- Test tool calling flow

### Integration Tests (End-to-End)

- Require AWS credentials (skip if not available)
- Test against actual Bedrock API
- Verify response format matches LLMResponse schema
- Test with SimpleKGPipeline for entity extraction
- Test with Text2CypherRetriever for Cypher generation

---

## Use Cases Enabled

Adding BedrockLLM enables AWS users to leverage neo4j-graphrag for:

### Entity Extraction with SimpleKGPipeline

Extract entities and relationships from documents using Claude via Bedrock, storing results in Neo4j.

### Natural Language to Cypher with Text2CypherRetriever

Convert natural language questions to Cypher queries using Bedrock-hosted models.

### GraphRAG Pipelines

Build complete retrieval-augmented generation pipelines using Bedrock for both embeddings (BedrockEmbeddings) and text generation (BedrockLLM).

### Agent-Based Graph Navigation

Use tool calling capabilities to build agents that can navigate and query Neo4j graphs.

---

## Relationship to BedrockEmbeddings

BedrockLLM complements the existing BedrockEmbeddings implementation:

| Component | Purpose | Bedrock Operation |
|-----------|---------|-------------------|
| BedrockEmbeddings | Vector embeddings for similarity search | invoke_model (Titan Embeddings) |
| BedrockLLM | Text generation for extraction/reasoning | converse (Claude, Titan Text, etc.) |

Together, these two providers enable complete GraphRAG workflows using only AWS Bedrock services.

---

## References

- Amazon Bedrock Converse API Documentation: docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html
- Bedrock Supported Models: docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html
- Bedrock Tool Use: docs.aws.amazon.com/bedrock/latest/userguide/tool-use.html
- Boto3 Bedrock Runtime converse: boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock-runtime/client/converse.html
- Claude on Amazon Bedrock: docs.anthropic.com/en/api/claude-on-amazon-bedrock
- Bedrock Inference Profiles: docs.aws.amazon.com/bedrock/latest/userguide/inference-profiles.html
