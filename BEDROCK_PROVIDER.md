# AWS Bedrock Embeddings Provider Proposal

This document proposes adding AWS Bedrock as an embedding provider for the neo4j-graphrag-python package.

---

## Implementation Status

| Phase | Status | Notes |
|-------|--------|-------|
| Phase 1: Core Implementation | Complete | `src/neo4j_graphrag/embeddings/bedrock.py` |
| Phase 2: Unit Tests | Complete | `tests/unit/embeddings/test_bedrock_embedder.py` (10 tests) |
| Phase 3: Integration | Complete | Exports and `pyproject.toml` updated |

---

## Executive Summary

AWS Bedrock is a fully managed service providing access to foundation models through a unified API. Adding Bedrock support expands the project's reach to AWS-centric organizations and provides access to Amazon Titan embedding models.

---

## Current State Analysis

### Supported Embedding Providers

The project currently supports seven embedding providers:

| Provider | Class | Default Model |
|----------|-------|---------------|
| OpenAI | `OpenAIEmbeddings` | text-embedding-ada-002 |
| Azure OpenAI | `AzureOpenAIEmbeddings` | text-embedding-ada-002 |
| Google Vertex AI | `VertexAIEmbeddings` | text-embedding-004 |
| Cohere | `CohereEmbeddings` | (required parameter) |
| Mistral AI | `MistralAIEmbeddings` | mistral-embed |
| Ollama | `OllamaEmbeddings` | (required parameter) |
| Sentence Transformers | `SentenceTransformerEmbeddings` | all-MiniLM-L6-v2 |

### Architecture Pattern

All embedding providers follow a consistent plugin pattern:

1. Inherit from the abstract `Embedder` base class
2. Implement the required `embed_query(text: str) -> list[float]` method (synchronous only)
3. Support optional rate limiting via `RateLimitHandler`
4. Use lazy imports for optional dependencies
5. Wrap provider-specific errors in `EmbeddingsGenerationError`

### Configuration Pattern

Existing providers follow a consistent configuration approach:

1. **Constructor kwargs**: Parameters passed to the constructor are forwarded to the underlying SDK client
2. **Environment variable fallback**: The underlying SDK handles environment variables automatically
3. **No .env file loading**: Providers do not load .env files directly; users handle this externally if needed

---

## Supported Model

### Amazon Titan Text Embeddings V2 (Default)

- **Model ID**: `amazon.titan-embed-text-v2:0`
- **Dimensions**: 1024 (default), also supports 256 and 384
- **Max Input**: 8,192 tokens
- **Languages**: 100+ languages
- **Use Case**: General-purpose text embeddings for semantic search and RAG

This is the only model supported in the initial implementation. Multimodal models are out of scope.

---

## Proposed Implementation

### BedrockEmbeddings Class

A simple class for text embeddings following the existing provider pattern:

**Constructor Parameters**:
- `model_id` (str): The Bedrock model identifier (default: `amazon.titan-embed-text-v2:0`)
- `region_name` (str, optional): AWS region (falls back to `AWS_REGION` or `AWS_DEFAULT_REGION` env var)
- `inference_profile_id` (str, optional): Inference profile ARN for cross-region inference
- `rate_limit_handler` (optional): Custom rate limit handler
- `**kwargs`: Additional boto3 client configuration passed directly to `boto3.client()`

**Key Method**:
- `embed_query(text: str) -> list[float]`: Generate embedding for input text

### Inference Profiles

Bedrock supports inference profiles for cross-region inference and workload management. When `inference_profile_id` is provided, it will be used instead of `model_id` in the `invoke_model` call.

```python
# Without inference profile (direct model invocation)
embedder = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v2:0",
    region_name="us-east-1"
)

# With inference profile (cross-region or managed throughput)
embedder = BedrockEmbeddings(
    inference_profile_id="arn:aws:bedrock:us-east-1:123456789:inference-profile/my-profile",
    region_name="us-east-1"
)
```

### Configuration Pattern (Matching Existing Providers)

Follow the same pattern as other providers - boto3 handles credential resolution automatically:

```python
# Basic usage - boto3 reads AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY from environment
embedder = BedrockEmbeddings()

# Explicit region
embedder = BedrockEmbeddings(region_name="us-east-1")

# With pre-configured boto3 client
import boto3
session = boto3.Session(profile_name="my-profile")
client = session.client("bedrock-runtime", region_name="us-east-1")
embedder = BedrockEmbeddings(client=client)
```

**Environment Variables** (handled by boto3 automatically):
- `AWS_ACCESS_KEY_ID` - Access key
- `AWS_SECRET_ACCESS_KEY` - Secret key
- `AWS_SESSION_TOKEN` - Session token (optional, for temporary credentials)
- `AWS_REGION` or `AWS_DEFAULT_REGION` - Default region
- `AWS_PROFILE` - Named profile from ~/.aws/credentials

### Dependency Management

**New Optional Dependency**:
- Package: `boto3`
- Version: `>=1.35.0,<2.0.0`
- Installation: `pip install "neo4j-graphrag[bedrock]"`

**Entry in pyproject.toml**:
```toml
bedrock = ["boto3>=1.35.0,<2.0.0"]
```

### File Structure

New files:
- `src/neo4j_graphrag/embeddings/bedrock.py` - Implementation
- `tests/unit/embeddings/test_bedrock_embeddings.py` - Unit tests

Modified files:
- `src/neo4j_graphrag/embeddings/__init__.py` - Add export
- `pyproject.toml` - Add optional dependency

### Error Handling

Map Bedrock-specific errors to the project's exception hierarchy:

| Bedrock Error | neo4j-graphrag Exception |
|---------------|--------------------------|
| `ThrottlingException` | Trigger rate limit retry |
| `ValidationException` | `EmbeddingsGenerationError` |
| `ModelNotReadyException` | `EmbeddingsGenerationError` |
| `AccessDeniedException` | `EmbeddingsGenerationError` |
| `ServiceQuotaExceededException` | Trigger rate limit retry |

### Rate Limiting

Use the existing `RetryRateLimitHandler` for exponential backoff. Detect throttling via boto3 exception types.

---

## Design Decisions (Keep It Simple)

### Synchronous Only

The implementation will be **synchronous only**, matching the base `Embedder` class which only defines `embed_query()`.

### No Caching

Do not implement caching. Keep the implementation simple.

### No Batch Embeddings

The project does not currently support batch embeddings. Do not add batch support.

### No Multimodal Support

Focus on text embeddings only with Titan Text Embeddings V2.

---

## Implementation Plan

### Phase 1: Core Implementation
- [x] Create `src/neo4j_graphrag/embeddings/bedrock.py`
- [x] Implement `BedrockEmbeddings` class
- [x] Support `model_id`, `region_name`, `inference_profile_id` parameters
- [x] Implement `embed_query()` method with Titan V2 request format
- [x] Add error handling and rate limit detection

### Phase 2: Unit Tests
- [x] Create `tests/unit/embeddings/test_bedrock_embedder.py`
- [x] Test missing dependency error
- [x] Test successful embedding generation (mocked)
- [x] Test error handling (validation, access denied)
- [x] Test rate limit retry behavior
- [x] Test inference profile usage

### Phase 3: Integration
- [x] Update `src/neo4j_graphrag/embeddings/__init__.py` with export
- [x] Update `pyproject.toml` with bedrock optional dependency
- [x] Verify import works correctly

---

## Testing Strategy

### Unit Tests

- Mock boto3 client responses
- Test successful embedding generation
- Test error handling (validation errors, access denied)
- Test rate limit retry behavior (throttling exceptions)
- Test inference profile parameter

### Integration Tests (E2E)

- Require AWS credentials (skip if not available)
- Test against actual Bedrock API
- Verify embedding dimensions match expected output

---

## Documentation

### Example Usage

```python
from neo4j_graphrag.embeddings import BedrockEmbeddings
from neo4j_graphrag.retrievers import VectorRetriever

# Create embedder (uses AWS credentials from environment)
embedder = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v2:0",
    region_name="us-east-1"
)

# Use with a retriever
retriever = VectorRetriever(
    driver=neo4j_driver,
    index_name="my_index",
    embedder=embedder
)

results = retriever.search(query_text="What is GraphRAG?")
```

---

## References

- [Amazon Bedrock Documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/)
- [Amazon Titan Embeddings Models](https://docs.aws.amazon.com/bedrock/latest/userguide/titan-embedding-models.html)
- [Boto3 Bedrock Runtime Documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock-runtime/client/invoke_model.html)
- [Bedrock Inference Profiles](https://docs.aws.amazon.com/bedrock/latest/userguide/inference-profiles.html)
