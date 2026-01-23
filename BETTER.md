# Why Use This Library?

You might be thinking: "I can just write some Cypher queries and call an embedding API. Why do I need a library for this?"

That's a fair question. Let's address it honestly.

## What You Get Beyond Plain Vector Search

**Retrieval Strategies:**
- **HybridRetriever** — Combines vector similarity + fulltext keyword search with two ranking algorithms:
  - *Linear weighted fusion*: Tunable alpha parameter blends normalized scores (`alpha * vector_score + (1-alpha) * fulltext_score`)
  - *Naive max ranking*: Takes the maximum normalized score per node across both indexes
- **Text2CypherRetriever** — LLM generates Cypher from natural language with automatic schema injection, few-shot examples, and syntax error handling
- **VectorCypherRetriever** — Vector search followed by custom Cypher traversal to gather related graph context
- **HybridCypherRetriever** — Hybrid search followed by custom Cypher traversal
- **ToolsRetriever** — LLM-driven tool selector that intelligently routes queries to the best retriever(s) and combines results
- **External vector store retrievers** — Weaviate, Pinecone, and Qdrant integrations that map results back to Neo4j nodes

**Knowledge Graph Construction Pipeline:**
- **Entity & relationship extraction** — LLM-powered extraction from unstructured text with configurable prompts and concurrent processing
- **Entity resolution / deduplication** — Three strategies to merge duplicate nodes:
  - *Exact match*: Same label + property value
  - *Fuzzy match*: RapidFuzz string similarity for typos and variations
  - *Semantic match*: spaCy embeddings with cosine similarity threshold
- **MERGE-based node writing** — Uses `apoc.merge.node()` to deduplicate entities across chunks during ingestion
- **Schema enforcement** — Validates extracted entities against defined schema, prunes invalid nodes/relationships
- **Lexical graph creation** — Automatically creates Document and Chunk nodes with structural relationships

**Text Processing:**
- **Chunking with overlap** — Fixed-size splitter with word-boundary awareness and configurable overlap
- **LangChain and LlamaIndex splitter wrappers** — Use your preferred chunking library

**LLM & Embedder Providers (unified interface):**
- OpenAI, Anthropic, Vertex AI, Cohere, Mistral AI, Ollama, AWS Bedrock, SentenceTransformers
- All implement the same interface — swap providers without code changes

**Production Features:**
- **Rate limiting with exponential backoff** — Tenacity-based retry logic for all API calls
- **Async-first design** — All operations have async variants for throughput
- **Metadata pre-filtering** — Filter nodes before vector search to reduce search space
- **Neo4j version compatibility checking** — Adapts queries for Neo4j 5.18.1+ features
- **Message history support** — Multi-turn conversations with context preservation

## The Illusion of Simplicity

Yes, a basic vector search is about five lines of code. Embedding text is a single API call. Writing a Cypher query to find similar nodes feels straightforward.

But here's what actually happens when you build a production GraphRAG system from scratch:

### Building the Knowledge Graph

Extracting entities and relationships from text sounds simple until you try it. You quickly discover:

**Entity extraction is messy.** Your LLM will call the same person "John Smith," "John," "Mr. Smith," and "he" across different paragraphs. Without entity resolution, you get a graph full of duplicate nodes representing the same thing. This is an important but often overlooked step in graph construction. Entity deduplication is essentially a cleaning step where you match multiple nodes that represent a single entity and merge them together for better graph structural integrity.

This library provides multiple resolver implementations: exact match, fuzzy matching for faster ingestion, and semantic matching using spaCy. You choose the tradeoff between speed and precision that fits your use case.

**Schema consistency is hard to maintain.** When you're extracting from thousands of documents, your entity types and relationship types will drift. What starts as a clean graph becomes a tangle of inconsistent labels. The library enforces schema constraints through a schema builder component that grounds LLM-extracted entities to your defined types.

**Chunking strategy matters more than you'd expect.** Recent research shows that chunking is arguably the most important factor for RAG performance. How you split documents affects your system's ability to find relevant information and give accurate answers. By refining document splitting, accuracy can jump from 65% to 92%.

The optimal chunk size typically ranges between 200 and 800 tokens, with 10-20% overlap recommended. But it depends on your documents and queries. Fixed-size chunking disrupts sentence structures. Semantic chunking groups sentences by embedding similarity. Recursive chunking applies rules hierarchically. The library provides these strategies so you can experiment without reimplementing each one.

**Document handling is a rabbit hole.** PDFs alone have dozens of edge cases: scanned images, multi-column layouts, tables, headers, footers. The experimental pipeline components handle these so you don't have to.

### Why Knowledge Graphs Beat Pure Vector Search

Vector databases are essentially black boxes. Data is transformed into numerical vectors, so if there's an error in the response, it can be difficult to trace back to the source. The opaque nature of vector representations makes building explainable AI systems challenging.

Vector databases also struggle with relationships. They might tell you that two documents are similar, but can't explain why or how they're connected. They have no structure for the most part, meaning you lose the hierarchical or relational structures that are essential for complex reasoning.

Knowledge graphs flip this around:

**Explainability and traceability.** If a knowledge graph RAG application makes an error, you can trace it back to the specific node or relationship where the incorrect information originated and correct it. This transparency is invaluable when debugging production issues.

**Multi-hop reasoning.** A knowledge graph enables reasoning by modeling entities and their relationships. Your LLM can trace how the answer was retrieved, not just get a list of similar chunks.

**More complete context retrieval.** With vector databases, the backing store only provides results semantically "close" to the user question. The generated context often lacks vital information the LLM needs. With a graph, once you find relevant nodes, you can traverse all connections to generate much richer context.

**Depth and breadth.** Graph structures give leverage for creating more complete answers. This kind of specific, deterministic retrieval is difficult to build with vector RAG alone.

### The Retriever Value Proposition

"I can just run a vector search" is technically true. But retrievers do more than search.

**Hybrid search outperforms pure vector search.** There's a growing consensus that relying solely on vector search may not always yield satisfactory results. Vector similarity search is good at handling typos, but not as good at precise matching on keywords, abbreviations, and names, which get lost in embeddings. Keyword search performs better there.

One practitioner reported building a hybrid search system that beats standard RAG by 35%. That's the difference between users finding what they need and giving up in frustration. More importantly, it's the difference between a RAG system that works in demos and one that works in production.

The HybridRetriever handles the scoring and fusion of results from vector and fulltext search. You don't need to figure out reciprocal rank fusion or how to weight the alpha parameter between sparse and dense scores.

**Text2Cypher is deceptively difficult.** Converting natural language to correct Cypher requires understanding your specific schema, handling edge cases gracefully, and generating efficient queries.

The challenges are real: LLMs struggle with complex multi-hop queries, leading to incomplete or incorrect outputs. The model needs to understand both user inputs and underlying graph schemas. Cypher includes flexible patterns, directional relationships, and variable-length path queries that add complexity. Models trained on one schema often struggle with new schemas.

The library handles this with comprehensive prompting, schema injection, and example queries you can customize for your data model. It also implements guardrails like syntax checks and retry logic when queries fail.

**Graph traversal after retrieval matters.** Finding a relevant node is often just the start. You frequently need to traverse relationships to gather context. What sets Neo4j apart is combining initial vector-based retrieval with powerful traversal capabilities. After finding the entry point using vector search, the database leverages explicit semantics from relationships to gather additional context and uncover meaningful connections.

**ToolsRetriever for intelligent routing.** The library's ToolsRetriever lets you use multiple retrievers as tools in a single query, with an LLM intelligently selecting the most appropriate retriever for the task. You don't have to hardcode which retrieval strategy to use.

### Production Concerns You'll Eventually Hit

When you roll your own, these problems tend to surface around month three:

**Rate limiting and retries.** LLM APIs fail. They rate limit you. They time out. Network glitches, temporary API unavailability, and occasional malformed outputs can disrupt your application's flow.

The recommended approach is exponential backoff with jitter: wait for a short random delay after an error, retry, and if it fails again, wait longer. Add randomness to prevent thundering herd problems. Honor Retry-After headers when APIs provide them. Set retry limits to prevent infinite loops.

The library uses tenacity for configurable retry logic with all these patterns built in. You'll build this yourself eventually, or you'll have production incidents.

**Provider lock-in.** You start with OpenAI, then need to support Anthropic for some users, then want to try a local model. The library's plugin architecture means swapping providers is configuration, not a rewrite. All providers implement the same LLMInterface and EmbedderInterface.

**Fallback strategies.** When your main LLM is throttled, you need to switch to a backup provider. The library's consistent interface makes this straightforward.

**Async operations for throughput.** Processing thousands of documents sequentially takes forever. The library provides async variants of operations so you can parallelize properly.

**Memory management.** Large documents and batch operations will blow up your memory if you're not careful. The pipeline components handle streaming and batching to keep memory usage reasonable.

**Neo4j version compatibility.** Different Neo4j versions have different capabilities and syntax quirks. The library checks versions and adapts accordingly.

## What This Library Actually Gives You

It's not about making easy things possible. It's about making hard things manageable.

**A tested abstraction over complexity.** The patterns for building knowledge graphs, implementing retrieval strategies, and orchestrating RAG pipelines have been refined across many production deployments.

**First-party support.** As an official Neo4j package, it offers long-term commitment and maintenance. New features and high-performing patterns ship quickly.

**Consistency across providers.** Whether you're using OpenAI, Vertex AI, Anthropic, Cohere, Mistral, or local models via Ollama, the interface is the same. Switch providers without rewriting your application.

**A pipeline architecture that scales.** The experimental pipeline system treats graph construction as a DAG of components: loaders, splitters, embedders, extractors, writers, and resolvers. This makes it possible to customize, extend, and parallelize complex workflows.

**Retrieval strategies that work together.** Vector search, hybrid search, and Text2Cypher can be combined and compared. The abstractions make it straightforward to try different approaches and find what works for your data.

## The Honest Tradeoff

Using a library means accepting its opinions about how things should work. If your use case is genuinely simple and will stay simple, you might not need this.

Knowledge graphs require structured data curation, which is resource-intensive to build and maintain. If you're only dealing with simple semantic similarity queries over unstructured text, a pure vector database might be sufficient.

But most GraphRAG projects that start simple don't stay simple. They grow to handle more document types, more retrieval strategies, more edge cases, more scale. When your RAG needs to model complex relationships between entities, when interconnections between data points are as important as the data points themselves, when you need explainability and multi-hop reasoning, then you need a knowledge graph.

This library is the accumulated knowledge of building these systems repeatedly. You can rebuild that knowledge yourself, or you can build on top of it.

The five-line vector search works great for a demo. For production, you'll want the other ten thousand lines this library provides.

---

## Further Reading

- [Neo4j GraphRAG Python Documentation](https://neo4j.com/docs/neo4j-graphrag-python/current/)
- [GraphRAG Python Package Announcement](https://neo4j.com/blog/news/graphrag-python-package/)
- [Knowledge Graph vs. Vector RAG: Benchmarking and Optimization](https://neo4j.com/blog/developer/knowledge-graph-vs-vector-rag/)
- [Entity Resolved Knowledge Graphs Tutorial](https://neo4j.com/blog/developer/entity-resolved-knowledge-graphs/)
- [Text2Cypher: Bridging Natural Language and Graph Databases](https://arxiv.org/html/2412.10064v1)
- [Constructing Knowledge Graphs with Neo4j GraphRAG](https://medium.com/neo4j/constructing-knowledge-graphs-with-neo4j-graphrag-for-python-2b3f1a42534d)
- [Hybrid Search for Better RAG Retrieval](https://machinelearningplus.com/gen-ai/hybrid-search-vector-keyword-techniques-for-better-rag/)
- [RAG Chunking Strategies Guide](https://www.firecrawl.dev/blog/best-chunking-strategies-rag-2025)
- [Why RAG Apps Fail in Production](https://www.traceloop.com/blog/why-your-rag-app-fails-in-production-even-when-code-hasnt-changed)
- [HybridRAG: Combining Vector and Graph](https://memgraph.com/blog/why-hybridrag)
