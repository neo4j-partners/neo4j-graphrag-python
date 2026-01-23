# Adding Custom Embedding and LLM Providers

To support models running on your own infrastructure this library provides the ability to create plugins. It does this by supporting embedding providers and LLM providers as plugins. To create the plugin your code will just need to implement the method signatures.  This it makes it easy to extend this library to add 

---

## How the Plugin System Works

The library validates providers at runtime using duck typing. It checks whether your object has the required methods, not whether it inherits from a specific class. If your embedder has an `embed_query` method that takes a string and returns a list of floats, it passes validation. Same principle applies to LLMs with their `invoke` method.

```python
# The library checks for this at runtime
def has_required_method(obj, method_name):
    return callable(getattr(obj, method_name, None))
```

---

## The Embeddings Contract

An embedder converts text into vectors. That's it. The `embed_query` method takes a string and returns a list of floats representing the semantic meaning of that text.

### What Your Embedder Must Implement

```python
def embed_query(self, text: str) -> list[float]:
    # Your implementation here
    pass
```

The input is always a single string. The output is always a list of floating-point numbers. Vector dimensions vary by model: OpenAI's `text-embedding-3-small` returns 1536 dimensions, `all-MiniLM-L6-v2` returns 384, and Cohere's models return 1024. The library accepts any dimension.

### Inheriting from the Base Class

While duck typing works, inheriting from `Embedder` gives you the rate limiting infrastructure for free.

```python
from neo4j_graphrag.embeddings.base import Embedder
from neo4j_graphrag.exceptions import EmbeddingsGenerationError
from neo4j_graphrag.utils.rate_limit import rate_limit_handler

class MyEmbedder(Embedder):
    @rate_limit_handler
    def embed_query(self, text: str) -> list[float]:
        try:
            # Call your model here
            return [0.1, 0.2, 0.3]  # Your actual vector
        except SomeAPIError as e:
            raise EmbeddingsGenerationError(f"Embedding failed: {e}")
```

The `@rate_limit_handler` decorator retries automatically with exponential backoff when your API returns 429 (rate limited) responses. Without it, a burst of embedding requests can crash your pipeline.

### Error Handling

When something breaks, raise `EmbeddingsGenerationError`. This lets the rest of the application handle embedding failures consistently, whether the problem is a network timeout, an invalid API key, or a malformed response.

```python
from neo4j_graphrag.exceptions import EmbeddingsGenerationError

# In your embed_query method:
if response.status_code == 401:
    raise EmbeddingsGenerationError("Invalid API key for embedding service")
if not response.json().get("embedding"):
    raise EmbeddingsGenerationError("API returned empty embedding")
```

---

## The LLM Contract

LLMs are more complex than embedders because they support two input styles and need both synchronous and asynchronous methods.

### Two Input Formats

**V1 (Legacy)** passes the user's input as a string, with optional message history and system instructions as separate parameters:

```python
def invoke(
    self,
    input: str,
    message_history: Optional[List[LLMMessage]] = None,
    system_instruction: Optional[str] = None,
) -> LLMResponse:
```

**V2 (Current)** passes a list of messages, matching the format that OpenAI, Anthropic, and most modern APIs use:

```python
def invoke(self, input: List[LLMMessage]) -> LLMResponse:
```

Each message is a dictionary with `role` ("system", "user", or "assistant") and `content` (the text):

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is GraphRAG?"},
]
```

New providers should support both formats for compatibility. The pattern is straightforward: check whether `input` is a string or a list, then dispatch to the appropriate internal method.

### Required Methods

Your LLM class needs four methods at minimum:

| Method | Purpose |
|--------|---------|
| `invoke` | Synchronous text generation |
| `ainvoke` | Asynchronous text generation |
| `invoke_with_tools` | Sync generation with function calling (optional) |
| `ainvoke_with_tools` | Async generation with function calling (optional) |

The tool-calling methods are optional. If your model doesn't support function calling, either skip them or raise `NotImplementedError`.

### Response Format

Both `invoke` and `ainvoke` must return an `LLMResponse` object with a `content` field containing the generated text:

```python
from neo4j_graphrag.llm.types import LLMResponse

def invoke(self, input, **kwargs) -> LLMResponse:
    text = self._call_my_model(input)
    return LLMResponse(content=text)
```

### Handling Both Interface Versions

Here's the pattern used by built-in providers to support V1 and V2:

```python
from typing import List, Optional, Union
from neo4j_graphrag.llm.base import LLMInterfaceV2
from neo4j_graphrag.llm.types import LLMResponse
from neo4j_graphrag.types import LLMMessage

class MyLLM(LLMInterfaceV2):
    def invoke(
        self,
        input: Union[str, List[LLMMessage]],
        message_history: Optional[List[LLMMessage]] = None,
        system_instruction: Optional[str] = None,
        **kwargs,
    ) -> LLMResponse:
        if isinstance(input, str):
            # V1 style: build message list from parts
            messages = self._build_messages(input, message_history, system_instruction)
        else:
            # V2 style: input is already a message list
            messages = input

        return self._call_api(messages, **kwargs)
```

---

## Building a Provider Step by Step

### 1. Understand Your Model's API

Before writing code, document these details about your model:

- **Authentication**: API key, OAuth token, or no auth for local models
- **Endpoint URL**: Where requests go
- **Request format**: JSON structure expected by the API
- **Response format**: Where the embedding or generated text lives in the response
- **Rate limits**: Requests per minute, tokens per minute, or both
- **Async support**: Does the SDK provide async clients?
- **Vector dimensions**: For embeddings, how many floats per vector
- **Tool calling**: For LLMs, does it support function calling?

### 2. Handle Missing Dependencies

If your provider needs an external package, import it inside your constructor and give a helpful error message when it's missing:

```python
class MyProvider:
    def __init__(self, model_name: str):
        try:
            import some_sdk
        except ImportError:
            raise ImportError(
                "Could not import some_sdk. "
                "Install it with: pip install some-sdk"
            )
        self.client = some_sdk.Client()
```

This pattern keeps the SDK optional. Users who don't need your provider won't get import errors just because they haven't installed its dependencies.

### 3. Set Up Clients in the Constructor

Initialize both sync and async clients if your SDK provides them. Store configuration that applies to every request:

```python
def __init__(
    self,
    model_name: str,
    api_key: str,
    base_url: str = "https://api.example.com",
    timeout: float = 30.0,
):
    self.model_name = model_name
    self.client = SomeSDK(api_key=api_key, base_url=base_url, timeout=timeout)
    self.async_client = AsyncSomeSDK(api_key=api_key, base_url=base_url, timeout=timeout)
```

### 4. Implement the Required Methods

For embedders, implement `embed_query`. For LLMs, implement `invoke` and `ainvoke` at minimum. Apply the rate limit decorators to any method that makes API calls:

```python
from neo4j_graphrag.utils.rate_limit import (
    rate_limit_handler,
    async_rate_limit_handler,
)

class MyLLM(LLMInterfaceV2):
    @rate_limit_handler
    def invoke(self, input: List[LLMMessage], **kwargs) -> LLMResponse:
        response = self.client.chat(messages=input, model=self.model_name)
        return LLMResponse(content=response.text)

    @async_rate_limit_handler
    async def ainvoke(self, input: List[LLMMessage], **kwargs) -> LLMResponse:
        response = await self.async_client.chat(messages=input, model=self.model_name)
        return LLMResponse(content=response.text)
```

### 5. Convert Errors

Catch your SDK's exceptions and convert them to the library's exception types:

```python
from neo4j_graphrag.exceptions import LLMGenerationError

try:
    response = self.client.chat(messages=input)
except SomeSDKAuthError as e:
    raise LLMGenerationError(f"Authentication failed: {e}")
except SomeSDKTimeoutError as e:
    raise LLMGenerationError(f"Request timed out after {self.timeout}s: {e}")
except SomeSDKError as e:
    raise LLMGenerationError(f"API call failed: {e}")
```

This gives callers a consistent exception type to catch, regardless of which provider they're using.

---

## Shortcuts for Common Scenarios

### Your Server Speaks OpenAI Protocol

vLLM, LocalAI, LM Studio, Ollama (with OpenAI compatibility mode), and many other inference servers expose OpenAI-compatible endpoints. Skip writing a custom provider entirely:

```python
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.embeddings import OpenAIEmbeddings

# Point the OpenAI provider at your server
llm = OpenAILLM(
    model_name="meta-llama/Llama-3-8b",
    base_url="http://localhost:8000/v1",
    api_key="not-needed",  # Many local servers ignore this
)

embedder = OpenAIEmbeddings(
    model="bge-large-en-v1.5",
    base_url="http://localhost:8001/v1",
    api_key="not-needed",
)
```

This works because the library's OpenAI provider accepts a `base_url` parameter. If your server's `/v1/chat/completions` and `/v1/embeddings` endpoints match OpenAI's schema, everything just works.

---

## Code Examples for Custom Providers

### Minimal Embeddings Provider

This example calls a custom HTTP API. Replace the endpoint and response parsing with your actual API's format:

```python
import requests
from typing import Any, Optional

from neo4j_graphrag.embeddings.base import Embedder
from neo4j_graphrag.exceptions import EmbeddingsGenerationError
from neo4j_graphrag.utils.rate_limit import RateLimitHandler, rate_limit_handler


class CustomEmbeddings(Embedder):
    """Embeddings provider for a self-hosted embedding service."""

    def __init__(
        self,
        api_url: str,
        model_name: str = "default",
        rate_limit_handler: Optional[RateLimitHandler] = None,
    ) -> None:
        super().__init__(rate_limit_handler)
        self.api_url = api_url
        self.model_name = model_name

    @rate_limit_handler
    def embed_query(self, text: str, **kwargs: Any) -> list[float]:
        """Generate embeddings by calling the custom API."""
        try:
            response = requests.post(
                f"{self.api_url}/embed",
                json={"text": text, "model": self.model_name},
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()

            # Adapt this to your API's response format
            return data["embedding"]
        except requests.RequestException as e:
            raise EmbeddingsGenerationError(f"Embedding request failed: {e}")


# Usage
embedder = CustomEmbeddings(
    api_url="http://localhost:8080",
    model_name="bge-large-en-v1.5"
)
vector = embedder.embed_query("What is knowledge graph construction?")
print(f"Vector has {len(vector)} dimensions")
```

### Minimal LLM Provider

This example uses `httpx` for both sync and async HTTP calls:

```python
import httpx
from typing import Any, List, Optional

from neo4j_graphrag.llm.base import LLMInterfaceV2
from neo4j_graphrag.llm.types import LLMResponse
from neo4j_graphrag.types import LLMMessage
from neo4j_graphrag.exceptions import LLMGenerationError
from neo4j_graphrag.utils.rate_limit import (
    RateLimitHandler,
    rate_limit_handler as rate_limit_handler_decorator,
    async_rate_limit_handler as async_rate_limit_handler_decorator,
)


class CustomLLM(LLMInterfaceV2):
    """LLM provider for a self-hosted language model."""

    def __init__(
        self,
        api_url: str,
        model_name: str,
        model_params: Optional[dict[str, Any]] = None,
        rate_limit_handler: Optional[RateLimitHandler] = None,
        timeout: float = 60.0,
    ):
        super().__init__(
            model_name=model_name,
            model_params=model_params,
            rate_limit_handler=rate_limit_handler,
        )
        self.api_url = api_url
        self.timeout = timeout
        self.client = httpx.Client(timeout=timeout)
        self.async_client = httpx.AsyncClient(timeout=timeout)

    @rate_limit_handler_decorator
    def invoke(self, input: List[LLMMessage], **kwargs: Any) -> LLMResponse:
        """Synchronously call the LLM API."""
        try:
            response = self.client.post(
                f"{self.api_url}/v1/chat/completions",
                json={
                    "model": self.model_name,
                    "messages": input,
                    **self.model_params,
                    **kwargs,
                },
            )
            response.raise_for_status()
            data = response.json()

            # Adapt this to your API's response format
            content = data["choices"][0]["message"]["content"]
            return LLMResponse(content=content)
        except httpx.HTTPError as e:
            raise LLMGenerationError(f"LLM request failed: {e}")

    @async_rate_limit_handler_decorator
    async def ainvoke(self, input: List[LLMMessage], **kwargs: Any) -> LLMResponse:
        """Asynchronously call the LLM API."""
        try:
            response = await self.async_client.post(
                f"{self.api_url}/v1/chat/completions",
                json={
                    "model": self.model_name,
                    "messages": input,
                    **self.model_params,
                    **kwargs,
                },
            )
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            return LLMResponse(content=content)
        except httpx.HTTPError as e:
            raise LLMGenerationError(f"LLM request failed: {e}")


# Usage
llm = CustomLLM(
    api_url="http://localhost:8000",
    model_name="llama-3-8b",
    model_params={"temperature": 0.7, "max_tokens": 512},
)

response = llm.invoke([
    {"role": "system", "content": "You answer questions about graphs."},
    {"role": "user", "content": "What is a knowledge graph?"},
])
print(response.content)
```

### In-Process Model with llama-cpp-python

For models running directly in your Python process without a server:

```python
from typing import Any, List, Optional

from neo4j_graphrag.llm.base import LLMInterfaceV2
from neo4j_graphrag.llm.types import LLMResponse
from neo4j_graphrag.types import LLMMessage
from neo4j_graphrag.exceptions import LLMGenerationError


class LlamaCppLLM(LLMInterfaceV2):
    """LLM using llama-cpp-python for local inference without a server."""

    def __init__(
        self,
        model_path: str,
        model_params: Optional[dict[str, Any]] = None,
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,
    ):
        super().__init__(model_name=model_path, model_params=model_params)
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "llama-cpp-python is required. "
                "Install with: pip install llama-cpp-python"
            )
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )

    def invoke(self, input: List[LLMMessage], **kwargs: Any) -> LLMResponse:
        try:
            params = {**self.model_params, **kwargs}

            # llama-cpp-python supports chat format directly
            response = self.llm.create_chat_completion(
                messages=input,
                **params,
            )
            content = response["choices"][0]["message"]["content"]
            return LLMResponse(content=content)
        except Exception as e:
            raise LLMGenerationError(f"Generation failed: {e}")

    async def ainvoke(self, input: List[LLMMessage], **kwargs: Any) -> LLMResponse:
        # llama-cpp-python doesn't have native async
        return self.invoke(input, **kwargs)


# Usage
llm = LlamaCppLLM(
    model_path="/models/llama-3-8b-instruct.Q4_K_M.gguf",
    model_params={"max_tokens": 512, "temperature": 0.7},
    n_gpu_layers=35,  # Offload 35 layers to GPU
)

response = llm.invoke([
    {"role": "user", "content": "Explain vector similarity search."},
])
print(response.content)
```

### Wiring Custom Providers into GraphRAG

Once your provider works, plug it into the library's retrieval and generation components:

```python
from neo4j import GraphDatabase
from neo4j_graphrag.retrievers import VectorRetriever
from neo4j_graphrag.generation import GraphRAG

# Your custom providers
embedder = CustomEmbeddings(api_url="http://localhost:8080")
llm = CustomLLM(api_url="http://localhost:8000", model_name="llama-3-8b")

driver = GraphDatabase.driver(
    "bolt://localhost:7687",
    auth=("neo4j", "password")
)

# The retriever uses your embedder to convert queries into vectors
retriever = VectorRetriever(
    driver=driver,
    index_name="document_embeddings",
    embedder=embedder,
)

# GraphRAG uses your LLM to generate answers from retrieved context
rag = GraphRAG(retriever=retriever, llm=llm)

response = rag.search(query_text="How do knowledge graphs improve RAG?")
print(response.answer)
```

---

## Protocol Reference

### Embeddings

| Requirement | Details |
|-------------|---------|
| Method | `embed_query(text: str) -> list[float]` |
| Error type | `EmbeddingsGenerationError` |
| Rate limiting | Apply `@rate_limit_handler` decorator |
| Base class | `neo4j_graphrag.embeddings.base.Embedder` (optional but recommended) |

### LLMs

| Requirement | Details |
|-------------|---------|
| Sync method | `invoke(input) -> LLMResponse` |
| Async method | `ainvoke(input) -> LLMResponse` |
| V1 input | `str` + optional `message_history` + optional `system_instruction` |
| V2 input | `List[LLMMessage]` where each message has `role` and `content` |
| Response | `LLMResponse(content=str)` |
| Error type | `LLMGenerationError` |
| Rate limiting | Apply `@rate_limit_handler` / `@async_rate_limit_handler` decorators |
| Tool calling | Optional `invoke_with_tools` / `ainvoke_with_tools` methods |
| Base class | `neo4j_graphrag.llm.base.LLMInterfaceV2` (recommended) |

---

## Testing Your Provider

### Unit Tests

Mock your SDK client to test the integration logic without making real API calls:

```python
from unittest.mock import Mock, patch

def test_embed_query_returns_vector():
    with patch("requests.post") as mock_post:
        mock_post.return_value.json.return_value = {
            "embedding": [0.1, 0.2, 0.3]
        }
        mock_post.return_value.raise_for_status = Mock()

        embedder = CustomEmbeddings(api_url="http://test")
        result = embedder.embed_query("test text")

        assert result == [0.1, 0.2, 0.3]
        mock_post.assert_called_once()

def test_embed_query_handles_api_error():
    with patch("requests.post") as mock_post:
        mock_post.side_effect = requests.RequestException("Connection failed")

        embedder = CustomEmbeddings(api_url="http://test")

        with pytest.raises(EmbeddingsGenerationError):
            embedder.embed_query("test text")
```

### Integration Tests

If you can run your model locally, write tests that make real requests:

```python
@pytest.mark.integration
def test_real_embedding_generation():
    embedder = CustomEmbeddings(api_url="http://localhost:8080")
    vector = embedder.embed_query("This is a test sentence.")

    assert isinstance(vector, list)
    assert len(vector) > 0
    assert all(isinstance(v, float) for v in vector)
```

### What to Test

- Basic functionality: does it return the expected types?
- Error handling: does it raise the right exceptions for API failures?
- Rate limiting: does the decorator get applied correctly?
- Edge cases: empty strings, very long inputs, special characters
