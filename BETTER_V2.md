# Neo4j GraphRAG Python: What You Get

The Neo4j GraphRAG Library provides production support for GraphRAG systems to create knowledge graphs with full document details including embeddings and relevant entities. It also includes advanced retrievers that provide flexible interfaces for searching your data with vector, hybrid, and natural language queries.

## Knowledge Graph Construction

Building knowledge graphs from unstructured text involves more than entity extraction. The library provides components for each step of the process.

**Entity Resolution** merges duplicate entities into single nodes. When your LLM extracts "John Smith," "John," "Mr. Smith," and "he" from different paragraphs, the resolver consolidates them. Three strategies let you choose your tradeoff between speed and precision: exact match for simple cases, fuzzy matching via RapidFuzz for typos and variations, and semantic matching via spaCy embeddings with cosine similarity thresholds.

**Schema Enforcement** keeps your entity and relationship types consistent across thousands of documents. The schema builder grounds LLM-extracted entities to your defined types, pruning nodes and relationships that don't match. This maintains a clean, well-structured graph as your data grows.

**MERGE-based Node Writing** uses `apoc.merge.node()` to deduplicate entities across chunks during ingestion. Nodes are matched and merged rather than duplicated.

**Lexical Graph Creation** automatically builds Document and Chunk nodes with structural relationships, maintaining the connection between your knowledge graph and source documents.

## Retrieval Strategies

Vector similarity search is one retrieval method. The library provides six others, each solving different problems.

**HybridRetriever** combines vector similarity with fulltext keyword search. Vector search excels at handling typos and semantic similarity, while keyword search provides precise matching on specific terms, abbreviations, and names. The retriever fuses results with two ranking algorithms: linear weighted fusion with a tunable alpha parameter, or naive max ranking that takes the maximum normalized score per node across both indexes.

**Text2CypherRetriever** converts natural language to Cypher queries. The retriever improves LLM accuracy on complex multi-hop queries through comprehensive prompting, automatic schema injection, few-shot examples you can customize for your data model, and retry logic for syntax validation.

**VectorCypherRetriever** and **HybridCypherRetriever** run vector or hybrid search, then execute custom Cypher traversals to gather related graph context. Finding a relevant node is often just the starting point. You frequently need to traverse relationships to build complete context.

**ToolsRetriever** uses an LLM to select the most appropriate retriever for each query. Instead of hardcoding which retrieval strategy to use, you define multiple retrievers as tools and let the model route queries intelligently.

**External Vector Store Retrievers** integrate Weaviate, Pinecone, and Qdrant with automatic mapping of results back to Neo4j nodes.

## LLM and Embedder Providers

All providers implement the same interface. Swap between them without rewriting your application.

OpenAI, Anthropic, Vertex AI, Cohere, Mistral AI, Ollama, AWS Bedrock, and SentenceTransformers all work. When your main LLM is throttled, switching to a backup provider is configuration, not a code change.

## Production Infrastructure

The library provides advanced options to configure and optimize your pipeline for production workloads.

**Rate Limiting and Retries** use tenacity for exponential backoff with jitter. The library implements standard resilience patterns: short random delays between retries, progressively longer waits, randomness to prevent thundering herd problems, and Retry-After header support.

**Async Operations** provide async variants of all operations for throughput when processing thousands of documents.

**Metadata Pre-filtering** reduces search space by filtering nodes before vector search rather than after.

**Neo4j Version Compatibility** automatically checks versions and adapts queries to use the optimal syntax and features available in your Neo4j instance.

**Message History Support** preserves context across multi-turn conversations.

## Text Processing

How you split documents affects your system's ability to find relevant information. Research shows chunking is arguably the most important factor for RAG performance. Optimal chunk sizes typically range between 200 and 800 tokens with 10-20% overlap, but it depends on your documents and queries.

The library provides fixed-size chunking with word-boundary awareness and configurable overlap. LangChain and LlamaIndex splitter wrappers let you use your preferred chunking library.

## Pipeline Architecture

The experimental pipeline system treats graph construction as a DAG of components: loaders, splitters, embedders, extractors, writers, and resolvers. This architecture makes it possible to customize individual steps, extend with new components, and parallelize complex workflows.

`SimpleKGPipeline` provides a simplified interface for common cases. The full `Pipeline` class offers complete control when you need it.

## When to Use This Library

Knowledge graphs excel when interconnections between data points matter as much as the data points themselves, when you need explainability, and when multi-hop reasoning is required. Most GraphRAG projects grow to handle more document types, retrieval strategies, and scale over time. This library provides the foundation to support that growth with tested patterns and production-ready components.

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
