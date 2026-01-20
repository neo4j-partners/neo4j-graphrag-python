# CREATE vs MERGE: Node Deduplication in KGWriter

## Summary

This document explains the changes made to the `Neo4jWriter` component to fix entity duplication issues that occur during knowledge graph construction from documents.

**Key Change:** The default behavior of `Neo4jWriter` now uses `MERGE` instead of `CREATE` for node creation, preventing duplicate nodes when the same entity is extracted from multiple chunks of a document.

---

## The Problem

### Symptom

When processing documents with `SimpleKGPipeline`, users encountered errors like:

```
IndexEntryConflictException{propertyValues=( String("Apple Inc.") ), addedEntityId=-1, existingEntityId=52}
Node(52) already exists with label `Company` and property `name` = 'Apple Inc.'
```

### Root Cause

The issue stems from how documents are processed:

1. **Chunking**: Documents are split into chunks (e.g., 2000 characters each)
2. **Independent Extraction**: The LLM extracts entities from each chunk independently
3. **Duplicate Extraction**: If "Apple Inc." appears in chunks 1, 3, and 7, the LLM extracts it three times
4. **Unique Internal IDs**: Each extraction gets a unique `__tmp_internal_id`
5. **CREATE Fails**: The original query used `CREATE`, attempting to create three separate nodes

The original Cypher query:

```cypher
UNWIND $rows AS row
CREATE (n:__KGBuilder__ {__tmp_internal_id: row.id})
SET n += row.properties
WITH n, row CALL apoc.create.addLabels(n, row.labels) YIELD node
...
```

This creates a new node for every row, even if entities have identical properties. When `apoc.create.addLabels` adds the `Company` label, a uniqueness constraint on `(:Company {name})` causes the second creation to fail.

### Why Entity Resolution Didn't Help

The pipeline includes an entity resolution step, but it runs **after** the writer:

```
Extractor → Pruner → Writer → Resolver
                       ↑
                    FAILS HERE
```

The constraint violation occurs at write time, before resolution can merge duplicates.

---

## The Solution

### New Behavior

`Neo4jWriter` now uses `apoc.merge.node` by default, which merges nodes based on their **primary entity label** and an identifying property (default: `name`):

```cypher
UNWIND $rows AS row
CALL apoc.merge.node(
    [row.labels[0]],                       -- Primary label only (e.g., 'Company')
    {`name`: row.properties.`name`},       -- Identity: merge on this
    row.properties,                         -- ON CREATE
    row.properties                          -- ON MATCH
) YIELD node AS n
WITH n, row
CALL apoc.create.addLabels(n, row.labels + ['__KGBuilder__']) YIELD node
SET n.__tmp_internal_id = row.id
...
```

**Key design decision:** The merge uses only the primary entity label (e.g., `Company`), not auxiliary labels like `__Entity__` or `__KGBuilder__`. This allows merging with pre-existing nodes created outside the pipeline (e.g., Company nodes loaded from CSV metadata). Additional labels are added after the merge.

This ensures:
- First extraction of "Apple Inc." → Creates/merges with Company node
- Second extraction of "Apple Inc." → Updates the existing node (no duplicate)
- Pre-existing nodes → LLM extractions merge with them seamlessly

### New Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_merge` | bool | `True` | Use MERGE instead of CREATE |
| `merge_property` | str | `"name"` | Property to use as the merge key |

### Enhanced Metadata Output

The `KGWriterModel` returned by `run()` now includes additional metrics:

| Metadata Field | Description |
|----------------|-------------|
| `node_count` | Total nodes in input graph |
| `nodes_created` | Nodes actually written to database |
| `nodes_skipped_missing_merge_property` | Nodes skipped due to missing merge key |
| `relationship_count` | Relationships written to database |

### Usage Examples

```python
from neo4j import GraphDatabase
from neo4j_graphrag.experimental.components.kg_writer import Neo4jWriter

driver = GraphDatabase.driver("neo4j://localhost:7687", auth=("neo4j", "password"))

# Default: Use MERGE with 'name' as the merge key (recommended)
writer = Neo4jWriter(driver=driver)

# Custom merge property (e.g., for nodes identified by 'id' instead of 'name')
writer = Neo4jWriter(driver=driver, merge_property="id")

# Disable MERGE for faster writes (only if duplicates are acceptable)
writer = Neo4jWriter(driver=driver, use_merge=False)
```

---

## Files Changed

### `src/neo4j_graphrag/neo4j_queries.py`

Added new function `upsert_node_query_merge()`:

```python
def upsert_node_query_merge(
    support_variable_scope_clause: bool,
    merge_property: str = "name",
) -> str:
    """Build a Cypher query to upsert nodes using MERGE for deduplication."""
    ...
```

Updated docstring for `upsert_node_query()` to warn about duplication issues.

### `src/neo4j_graphrag/experimental/components/kg_writer.py`

1. Added import for `upsert_node_query_merge`
2. Added `use_merge` and `merge_property` parameters to `Neo4jWriter.__init__`
3. Added `_validate_merge_property()` method to filter nodes missing the merge key
4. Modified `_upsert_nodes()` to:
   - Call validation before processing
   - Return count of skipped nodes
   - Handle empty batches gracefully
5. Modified `run()` to:
   - Track total skipped nodes across batches
   - Return enhanced metadata including `nodes_created` and `nodes_skipped_missing_merge_property`
6. Updated class docstring with new parameters and examples

### `tests/unit/experimental/components/test_kg_writer.py`

Added comprehensive tests for merge functionality:
- Query generation tests for both Neo4j version syntaxes
- Tests for custom `merge_property` values
- Validation tests for missing merge properties
- Tests for warning logging and metadata reporting

---

## Comparison: CREATE vs MERGE

| Aspect | CREATE (old default) | MERGE (new default) |
|--------|---------------------|---------------------|
| **Duplicate Handling** | Creates duplicates | Merges duplicates |
| **Uniqueness Constraints** | Fails if constraint exists | Works with constraints |
| **Performance** | Faster (no existence check) | Slightly slower |
| **Entity Resolution** | Required to clean up | Still useful for cross-document |
| **Missing Merge Key** | No validation | Skips node with warning |
| **Use Case** | Clean databases, no constraints | Production, constraints exist |

---

## Migration Guide

### For Most Users

**No action required.** The new default (`use_merge=True`) is more robust and works with existing code.

### To Preserve Old Behavior

If you need the old CREATE behavior (e.g., for performance in controlled environments):

```python
# Old behavior (not recommended for production)
writer = Neo4jWriter(driver=driver, use_merge=False)
```

### With SimpleKGPipeline

`SimpleKGPipeline` uses `Neo4jWriter` internally. The new default applies automatically:

```python
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline

# MERGE is now used by default - no changes needed
pipeline = SimpleKGPipeline(
    llm=llm,
    driver=driver,
    embedder=embedder,
)
```

---

## Technical Details

### Why `apoc.merge.node` Instead of Cypher MERGE

Cypher's native `MERGE` doesn't support dynamic labels:

```cypher
-- This doesn't work:
MERGE (n:$label {name: $name})  -- Labels can't be parameterized

-- apoc.merge.node supports dynamic labels:
CALL apoc.merge.node(['Company'], {name: 'Apple Inc.'}, ...) YIELD node
```

### Why Merge on Primary Label Only

The `row.labels` array typically contains `[EntityType, '__Entity__']` (e.g., `['Company', '__Entity__']`). If we merged on ALL labels, we would fail to match pre-existing nodes that don't have `__Entity__` or `__KGBuilder__` labels.

```cypher
-- WRONG: Would not find pre-existing Company node without __Entity__ label
CALL apoc.merge.node(['Company', '__Entity__'], {name: 'Apple Inc.'}, ...)

-- CORRECT: Finds any Company node with matching name
CALL apoc.merge.node(['Company'], {name: 'Apple Inc.'}, ...)
-- Then add auxiliary labels:
CALL apoc.create.addLabels(n, ['Company', '__Entity__', '__KGBuilder__'])
```

This allows the pipeline to seamlessly integrate with nodes created outside the pipeline (e.g., from CSV imports, other data sources).

### Lexical Graph Nodes Use CREATE

Lexical graph nodes (Chunk, Document) are always created with `CREATE`, not `MERGE`, because:
- Each chunk/document is unique and doesn't need deduplication
- Chunks don't have a `name` property (they have `text` and `index`)
- Merging on `text` would be inefficient and semantically wrong

The validation logic separates nodes into three categories:
1. **Entity nodes with merge_property** → Use MERGE
2. **Lexical graph nodes** → Use CREATE (always unique)
3. **Entity nodes missing merge_property** → Skip with warning

### Merge Key Selection

The `merge_property` parameter defaults to `"name"` because:

1. It's the standard identifying property in knowledge graphs
2. Most entity types (Company, Person, Product, etc.) use `name`
3. The schema typically defines `name` as the first property

For entities identified by other properties, pass a custom `merge_property`:

```python
# For nodes with 'id' as the identifier
writer = Neo4jWriter(driver=driver, merge_property="id")
```

### Validation for Missing Merge Properties

When `use_merge=True`, nodes that don't have the `merge_property` in their properties dictionary are automatically **skipped** to prevent unexpected merge behavior. This validation:

1. **Logs a warning** for each skipped node with details:
   ```
   WARNING - Node (id=123, label=Company) is missing merge_property 'name' and will be skipped. Available properties: ['title', 'description']
   ```

2. **Reports metrics** in the returned `KGWriterModel.metadata`:
   ```python
   result = await writer.run(graph=graph)
   print(result.metadata)
   # {
   #     "node_count": 100,           # Total nodes in input
   #     "nodes_created": 95,          # Nodes actually written
   #     "nodes_skipped_missing_merge_property": 5,  # Nodes skipped
   #     "relationship_count": 50
   # }
   ```

3. **Gracefully degrades** - The pipeline continues with valid nodes rather than failing entirely.

**Why validation instead of failing?** This matches the pattern used throughout the experimental components (e.g., `graph_pruning.py`, `entity_relation_extractor.py`) where partial success is preferred over complete failure during document processing.

**To fix skipped nodes:** Ensure your entity extraction prompts always extract the identifying property (default: `name`). If your entities use a different identifier, configure `merge_property` accordingly:

```python
# If your entities use 'id' instead of 'name'
writer = Neo4jWriter(driver=driver, merge_property="id")
```

---

## Relationship to Entity Resolution

Entity resolution (`perform_entity_resolution=True` in `SimpleKGPipeline`) still serves a purpose:

| Scenario | MERGE Handles | Entity Resolution Handles |
|----------|---------------|---------------------------|
| Same entity in same document | Yes | Yes |
| Same entity across documents | Partially* | Yes |
| Similar but not exact names | No | Yes |
| Typos and variations | No | Yes |

*MERGE handles cross-document duplicates if the names are exactly identical.

**Recommendation:** Keep entity resolution enabled for production use. MERGE prevents write-time failures; entity resolution improves graph quality.

---

## Requirements

- **APOC Plugin**: The new query uses `apoc.merge.node`, which requires the APOC plugin
- **Neo4j Version**: Works with Neo4j 4.4+ (tested with 5.x)

---

## Related Issues

This change addresses:
- Duplicate entity creation during document processing
- Uniqueness constraint violations with extracted entities
- Entity resolution running too late to prevent duplicates

For a detailed analysis of the original problem, see `CONFLICT_V2.md` in the workshop project.
