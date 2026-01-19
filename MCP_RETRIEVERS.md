# Proposal: Exposing GraphRAG Retrievers via MCP Server

## Executive Summary

This proposal explores how the neo4j-graphrag-python retrievers (Vector, Hybrid, Text2Cypher) could be made available to enterprise applications through the official Neo4j MCP server. The goal is to enable a workflow where knowledge graph data is created with embeddings using this library, then accessed across the enterprise via MCP-compatible clients (Claude Desktop, VS Code, Cursor, etc.).

---

## Background

### What We Have Today

**neo4j-graphrag-python** provides several retriever types:

1. **VectorRetriever** - Pure vector similarity search against Neo4j vector indexes
2. **HybridRetriever** - Combined vector and full-text search with ranking
3. **Text2CypherRetriever** - Natural language to Cypher query translation using an LLM
4. **VectorCypherRetriever / HybridCypherRetriever** - Extensions that add custom Cypher traversal after initial retrieval

**The Official Neo4j MCP Server** (`/Users/ryanknight/projects/mcp`) currently provides:

- `get-schema` - Introspect graph structure (labels, relationships, properties)
- `read-cypher` - Execute read-only Cypher queries
- `write-cypher` - Execute write Cypher queries
- `list-gds-procedures` - List available Graph Data Science procedures

### The Enterprise Use Case

After building a knowledge graph with embeddings using SimpleKGPipeline or the Pipeline class, organizations want to:

1. Make this knowledge accessible to LLM-powered tools across the enterprise
2. Provide a standardized interface that works with Claude Desktop, VS Code Copilot, and other MCP clients
3. Enable natural language queries against the graph without requiring Cypher expertise
4. Maintain security and access control through the MCP protocol

---

## Proposed Approach

### Option A: Retriever-Specific MCP Tools (Extension to Neo4j MCP Server)

Add new tools to the official Neo4j MCP server that correspond to each retriever type:

**Proposed Tools:**

| Tool Name | Purpose | Required Parameters |
|-----------|---------|---------------------|
| `vector-search` | Semantic similarity search | query_text, index_name, top_k |
| `hybrid-search` | Combined vector + fulltext search | query_text, vector_index, fulltext_index, top_k |
| `text2cypher` | Natural language to graph query | query_text, (optional) examples |

**How It Would Work:**

1. User creates knowledge graph with embeddings using neo4j-graphrag-python
2. User configures Neo4j MCP server with index names and embedding settings
3. MCP clients call retriever tools using natural language
4. MCP server executes appropriate vector/hybrid queries and returns formatted results

### Option B: Cypher-Based Approach (No Server Extensions Needed)

Leverage the existing `read-cypher` tool by having MCP clients generate the appropriate vector search Cypher directly:

```
User asks: "Find documents about climate change"

MCP client generates:
CALL db.index.vector.queryNodes('document_embeddings', 5, $embedding)
YIELD node, score
RETURN node.text, score
```

**Challenge:** This requires the MCP client (the LLM) to:
- Know the correct index names
- Have access to an embedding service to convert text to vectors
- Understand Neo4j vector search syntax

### Option C: Standalone Python MCP Server (Separate from Neo4j MCP)

Create a new Python-based MCP server that wraps neo4j-graphrag-python retrievers directly:

- Runs alongside or instead of the Go-based Neo4j MCP server
- Has direct access to Python embedder implementations
- Can use the full retriever API including custom formatters

---

## Required Extensions to Neo4j MCP Server

If pursuing Option A (recommended for demo purposes), the official Neo4j MCP server would need:

### 1. Embedding Service Integration

The server currently has no concept of embeddings. Vector search requires converting text queries to vectors. Options:

- **External embedding API** - Call OpenAI, Cohere, or other embedding services
- **Configuration-based** - Require embedding endpoint URL and API key in server config
- **Passthrough** - Require clients to provide pre-computed embedding vectors

### 2. New Tool Handlers

Following the existing pattern in `internal/tools/cypher/`, add:

- `internal/tools/retrieval/vector_search_handler.go`
- `internal/tools/retrieval/hybrid_search_handler.go`
- `internal/tools/retrieval/text2cypher_handler.go`

### 3. Index Discovery

Add a tool or resource to discover available vector and fulltext indexes:

- `list-vector-indexes` - Return available vector indexes with their configurations
- `list-fulltext-indexes` - Return available fulltext indexes

### 4. LLM Integration (for Text2Cypher)

The Text2Cypher retriever requires an LLM to translate natural language to Cypher. This would need:

- LLM provider configuration (OpenAI, Anthropic, etc.)
- Schema injection for context
- Few-shot example support

---

## Questioning the Validity of This Approach

### Is This a Common Pattern?

Based on research, MCP + RAG integration is an **emerging but not yet standardized pattern** in 2025:

**Evidence of adoption:**
- Teradata launched an MCP Server with built-in RAG capabilities for enterprise AI agents
- Multiple frameworks (RAG-MCP, Agentic RAG) are exploring this integration
- Neo4j explicitly positions MCP as part of their GenAI ecosystem

**However, the pattern has critics:**
- Some argue this adds unnecessary abstraction when direct API integration would suffice
- The "wrapping APIs in MCP" anti-pattern has been observed where teams implement MCP servers just to say they use the protocol

### Concerns With This Approach

**1. Redundancy with Existing Capabilities**

The current MCP server's `read-cypher` tool can already execute vector search queries. Adding retriever-specific tools may be solving a problem that doesn't exist if the LLM client is capable of generating the right Cypher.

**2. Embedding Service Complexity**

Vector search requires embeddings. The current MCP server is stateless and language-agnostic (Go). Adding embedding support means either:
- Depending on external embedding APIs (added latency, cost, API key management)
- Shipping embedding models with the server (significant complexity)
- Requiring clients to provide vectors (defeats the purpose of natural language access)

**3. Text2Cypher: LLM Calling LLM**

If an MCP client (powered by an LLM) calls a Text2Cypher tool (which uses another LLM to generate Cypher), you have:
- LLM #1 (client) interprets user intent
- LLM #2 (server) generates Cypher from that interpretation
- Potential for compounded hallucinations and errors
- Added latency and cost

**4. Security Implications**

Exposing graph search capabilities via MCP means:
- Any MCP client can query the knowledge graph
- Prompt injection could potentially extract sensitive data
- Access control becomes more complex than direct API integration

**5. When Direct Integration is Better**

For applications with:
- Predictable query patterns
- High performance requirements
- Sensitive data
- Custom business logic

Direct integration with neo4j-graphrag-python may be more appropriate than MCP abstraction.

### When This Approach Makes Sense

Despite concerns, MCP-based retrieval is valuable when:

1. **Diverse client ecosystem** - Many different LLM-powered tools need graph access
2. **Rapid prototyping** - Testing retrieval strategies across different AI assistants
3. **Standardization priority** - Organization wants unified tool protocol
4. **Read-only knowledge bases** - Public or semi-public information with low security risk
5. **Developer productivity** - Engineers want to query graphs from their IDE

---

## Recommended Demo Implementation

For a simple demonstration, the following minimal approach is suggested:

### Phase 1: Vector Search Tool (Minimal Viable Demo)

1. Add a `vector-search` tool to the Neo4j MCP server
2. Require the embedding vector as a parameter (client computes embedding)
3. Execute `db.index.vector.queryNodes()` and return results
4. Configure with index name via environment variable

This avoids embedding complexity while demonstrating the pattern.

### Phase 2: Text-Based Vector Search

1. Add embedding service configuration to MCP server
2. Accept text queries, compute embeddings server-side
3. Return semantically relevant results

### Phase 3: Hybrid and Text2Cypher

1. Add hybrid search combining vector and fulltext
2. Add text2cypher with LLM integration (optional, highest complexity)

---

## Alternative Consideration: Python MCP Server

Given that neo4j-graphrag-python already has:
- Full retriever implementations
- Embedder integrations (OpenAI, Cohere, sentence-transformers)
- LLM integrations for Text2Cypher

A Python-based MCP server wrapping this library directly may be more practical than extending the Go server. This would:

- Avoid reimplementing retriever logic in Go
- Leverage existing embedder infrastructure
- Maintain feature parity with the Python library
- Allow use of the same configuration patterns

The tradeoff is running two MCP servers (Go for core Neo4j, Python for GraphRAG) or replacing the Go server entirely.

---

## Conclusion

Exposing GraphRAG retrievers via MCP is technically feasible and aligns with emerging enterprise patterns. However, the value proposition should be carefully evaluated:

- For **diverse, multi-client environments** with standardization goals, MCP provides clear benefits
- For **single-application integrations** with performance requirements, direct Python API usage is likely superior
- The **Text2Cypher via MCP** pattern (LLM calling LLM) should be approached with caution

A phased demo starting with simple vector search would validate the approach before committing to full retriever implementation in the MCP server.

---

## Sources

Research informing this proposal:

- [Neo4j MCP Developer Guide](https://neo4j.com/developer/genai-ecosystem/model-context-protocol-mcp/)
- [MCP for RAG and Agentic AI](https://medium.com/@tam.tamanna18/model-context-protocol-mcp-for-retrieval-augmented-generation-rag-and-agentic-ai-6f9b4616d36e)
- [Integrating Agentic RAG with MCP Servers](https://becomingahacker.org/integrating-agentic-rag-with-mcp-servers-technical-implementation-guide-1aba8fd4e442)
- [RAG Servers vs MCP Servers](https://www.pgedge.com/blog/rag-servers-vs-mcp-servers-choosing-the-right-approach-for-ai-powered-database-access)
- [MCP vs APIs: When to Use Which](https://www.tinybird.co/blog/mcp-vs-apis-when-to-use-which-for-ai-agent-development)
- [Building Vertical AI Agents - MCP Anti-patterns](https://www.decodingai.com/p/building-vertical-ai-agents-case-study-1)
- [Teradata MCP Server for Enterprise AI](https://www.teradata.com/press-releases/2025/mcp-server-agentic-ai-at-scale)
