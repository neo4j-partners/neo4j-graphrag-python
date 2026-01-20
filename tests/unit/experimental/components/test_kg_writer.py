#  Copyright (c) "Neo4j"
#  Neo4j Sweden AB [https://neo4j.com]
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#      https://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
from __future__ import annotations

import logging
from typing import Any
from unittest import mock
from unittest.mock import MagicMock, Mock

import pytest
from neo4j_graphrag.experimental.components.kg_writer import Neo4jWriter, batched
from neo4j_graphrag.experimental.components.types import (
    LexicalGraphConfig,
    Neo4jGraph,
    Neo4jNode,
    Neo4jRelationship,
)
from neo4j_graphrag.neo4j_queries import (
    upsert_node_query,
    upsert_node_query_merge,
    upsert_relationship_query,
)


def test_batched() -> None:
    assert list(batched([1, 2, 3, 4], batch_size=2)) == [
        [1, 2],
        [3, 4],
    ]
    assert list(batched([1, 2, 3], batch_size=2)) == [
        [1, 2],
        [3],
    ]
    assert list(batched([1, 2, 3], batch_size=4)) == [
        [1, 2, 3],
    ]


@mock.patch(
    "neo4j_graphrag.experimental.components.kg_writer.get_version",
    return_value=((5, 22, 0), False, False),
)
@mock.patch(
    "neo4j_graphrag.experimental.components.kg_writer.Neo4jWriter._db_setup",
    return_value=None,
)
def test_upsert_nodes_with_create(_: Mock, driver: MagicMock) -> None:
    """Test node upsert using CREATE (use_merge=False)."""
    driver.execute_query.return_value = (
        [{"element_id": "#1"}],
        None,
        None,
    )
    neo4j_writer = Neo4jWriter(driver=driver, use_merge=False)
    node = Neo4jNode(id="1", label="Label", properties={"key": "value"})
    neo4j_writer._upsert_nodes(nodes=[node], lexical_graph_config=LexicalGraphConfig())
    driver.execute_query.assert_called_once_with(
        upsert_node_query(False),
        parameters_={
            "rows": [
                {
                    "label": "Label",
                    "labels": ["Label", "__Entity__"],
                    "id": "1",
                    "properties": {"key": "value"},
                    "embedding_properties": None,
                }
            ]
        },
        database_=None,
    )


@mock.patch(
    "neo4j_graphrag.experimental.components.kg_writer.get_version",
    return_value=((5, 22, 0), False, False),
)
@mock.patch(
    "neo4j_graphrag.experimental.components.kg_writer.Neo4jWriter._db_setup",
    return_value=None,
)
def test_upsert_nodes_with_embedding_and_create(
    _: Mock,
    driver: MagicMock,
) -> None:
    """Test node upsert with embeddings using CREATE (use_merge=False)."""
    driver.execute_query.return_value = (
        [{"element_id": "#1"}],
        None,
        None,
    )
    neo4j_writer = Neo4jWriter(driver=driver, use_merge=False)
    node = Neo4jNode(
        id="1",
        label="Label",
        properties={"key": "value"},
        embedding_properties={"embeddingProp": [1.0, 2.0, 3.0]},
    )
    neo4j_writer._upsert_nodes(nodes=[node], lexical_graph_config=LexicalGraphConfig())
    driver.execute_query.assert_any_call(
        upsert_node_query(False),
        parameters_={
            "rows": [
                {
                    "label": "Label",
                    "labels": ["Label", "__Entity__"],
                    "id": "1",
                    "properties": {"key": "value"},
                    "embedding_properties": {"embeddingProp": [1.0, 2.0, 3.0]},
                }
            ]
        },
        database_=None,
    )


@mock.patch(
    "neo4j_graphrag.experimental.components.kg_writer.get_version",
    return_value=((5, 22, 0), False, False),
)
@mock.patch(
    "neo4j_graphrag.experimental.components.kg_writer.Neo4jWriter._db_setup",
    return_value=None,
)
def test_upsert_relationship(_: Mock, driver: MagicMock) -> None:
    neo4j_writer = Neo4jWriter(driver=driver)
    rel = Neo4jRelationship(
        start_node_id="1",
        end_node_id="2",
        type="RELATIONSHIP",
        properties={"key": "value"},
    )
    neo4j_writer._upsert_relationships(
        rels=[rel],
    )
    parameters = {
        "rows": [
            {
                "type": "RELATIONSHIP",
                "start_node_id": "1",
                "end_node_id": "2",
                "properties": {"key": "value"},
                "embedding_properties": None,
            }
        ]
    }
    driver.execute_query.assert_called_once_with(
        upsert_relationship_query(False),
        parameters_=parameters,
        database_=None,
    )


@mock.patch(
    "neo4j_graphrag.experimental.components.kg_writer.get_version",
    return_value=((5, 22, 0), False, False),
)
@mock.patch(
    "neo4j_graphrag.experimental.components.kg_writer.Neo4jWriter._db_setup",
    return_value=None,
)
def test_upsert_relationship_with_embedding(_: Mock, driver: MagicMock) -> None:
    neo4j_writer = Neo4jWriter(driver=driver)
    rel = Neo4jRelationship(
        start_node_id="1",
        end_node_id="2",
        type="RELATIONSHIP",
        properties={"key": "value"},
        embedding_properties={"embeddingProp": [1.0, 2.0, 3.0]},
    )
    driver.execute_query.return_value.records = [{"elementId(r)": "rel_elem_id"}]
    neo4j_writer._upsert_relationships(
        rels=[rel],
    )
    parameters = {
        "rows": [
            {
                "type": "RELATIONSHIP",
                "start_node_id": "1",
                "end_node_id": "2",
                "properties": {"key": "value"},
                "embedding_properties": {"embeddingProp": [1.0, 2.0, 3.0]},
            }
        ]
    }
    driver.execute_query.assert_any_call(
        upsert_relationship_query(False),
        parameters_=parameters,
        database_=None,
    )


@mock.patch(
    "neo4j_graphrag.experimental.components.kg_writer.get_version",
    return_value=((5, 22, 0), False, False),
)
@pytest.mark.asyncio
@mock.patch(
    "neo4j_graphrag.experimental.components.kg_writer.Neo4jWriter._db_setup",
    return_value=None,
)
async def test_run_with_create(_: Mock, driver: MagicMock) -> None:
    """Test full pipeline run using CREATE (use_merge=False)."""
    driver.execute_query.return_value = (
        [
            {"element_id": "#1"},
            {"element_id": "#2"},
        ],
        None,
        None,
    )
    neo4j_writer = Neo4jWriter(driver=driver, use_merge=False)
    node = Neo4jNode(id="1", label="Label")
    rel = Neo4jRelationship(start_node_id="1", end_node_id="2", type="RELATIONSHIP")
    graph = Neo4jGraph(nodes=[node], relationships=[rel])
    await neo4j_writer.run(graph=graph)
    driver.execute_query.assert_any_call(
        upsert_node_query(False),
        parameters_={
            "rows": [
                {
                    "label": "Label",
                    "labels": ["Label", "__Entity__"],
                    "id": "1",
                    "properties": {},
                    "embedding_properties": None,
                }
            ]
        },
        database_=None,
    )
    parameters_: dict[str, Any] = {
        "rows": [
            {
                "type": "RELATIONSHIP",
                "start_node_id": "1",
                "end_node_id": "2",
                "properties": {},
                "embedding_properties": None,
            }
        ]
    }
    driver.execute_query.assert_any_call(
        upsert_relationship_query(False),
        parameters_=parameters_,
        database_=None,
    )


@pytest.mark.asyncio
@mock.patch(
    "neo4j_graphrag.experimental.components.kg_writer.Neo4jWriter._db_setup",
    return_value=None,
)
async def test_run_is_version_below_5_23_with_create(_: Mock) -> None:
    """Test CREATE behavior on Neo4j < 5.23."""
    driver = MagicMock()
    driver.execute_query = Mock(
        side_effect=(
            # get_version
            ([{"versions": ["5.22.0"], "edition": "enterpise"}], None, None),
            # upsert nodes
            ([{"_internal_id": "1", "element_id": "#1"}], None, None),
            # upsert relationships
            (None, None, None),
        )
    )

    neo4j_writer = Neo4jWriter(driver=driver, use_merge=False)

    node = Neo4jNode(id="1", label="Label")
    rel = Neo4jRelationship(start_node_id="1", end_node_id="2", type="RELATIONSHIP")
    graph = Neo4jGraph(nodes=[node], relationships=[rel])
    await neo4j_writer.run(graph=graph)

    driver.execute_query.assert_any_call(
        upsert_node_query(False),
        parameters_={
            "rows": [
                {
                    "label": "Label",
                    "labels": ["Label", "__Entity__"],
                    "id": "1",
                    "properties": {},
                    "embedding_properties": None,
                }
            ]
        },
        database_=None,
    )
    parameters_: dict[str, Any] = {
        "rows": [
            {
                "type": "RELATIONSHIP",
                "start_node_id": "1",
                "end_node_id": "2",
                "properties": {},
                "embedding_properties": None,
            }
        ]
    }
    driver.execute_query.assert_any_call(
        upsert_relationship_query(False),
        parameters_=parameters_,
        database_=None,
    )


@pytest.mark.asyncio
@mock.patch(
    "neo4j_graphrag.experimental.components.kg_writer.Neo4jWriter._db_setup",
    return_value=None,
)
async def test_run_is_version_5_23_or_above_with_create(_: Mock) -> None:
    """Test CREATE behavior on Neo4j >= 5.23."""
    driver = MagicMock()
    driver.execute_query = Mock(
        side_effect=(
            # get_version
            ([{"versions": ["5.23.0"], "edition": "enterpise"}], None, None),
            # upsert nodes
            ([{"element_id": "#1"}], None, None),
            # upsert relationships
            (None, None, None),
        )
    )

    neo4j_writer = Neo4jWriter(driver=driver, use_merge=False)
    neo4j_writer.is_version_5_23_or_above = True

    node = Neo4jNode(id="1", label="Label")
    rel = Neo4jRelationship(start_node_id="1", end_node_id="2", type="RELATIONSHIP")
    graph = Neo4jGraph(nodes=[node], relationships=[rel])
    await neo4j_writer.run(graph=graph)

    driver.execute_query.assert_any_call(
        upsert_node_query(True),
        parameters_={
            "rows": [
                {
                    "label": "Label",
                    "labels": ["Label", "__Entity__"],
                    "id": "1",
                    "properties": {},
                    "embedding_properties": None,
                }
            ]
        },
        database_=None,
    )
    parameters_: dict[str, Any] = {
        "rows": [
            {
                "type": "RELATIONSHIP",
                "start_node_id": "1",
                "end_node_id": "2",
                "properties": {},
                "embedding_properties": None,
            }
        ]
    }
    driver.execute_query.assert_any_call(
        upsert_relationship_query(True),
        parameters_=parameters_,
        database_=None,
    )


@pytest.mark.parametrize(
    "description, version, is_5_23_or_above",
    [
        ("SemVer, < 5.23", "5.22.0", False),
        ("SemVer, > 5.23", "5.24.0", True),
        ("SemVer, < 5.23, Aura", "5.22-aura", False),
        ("SemVer, > 5.23, Aura", "5.24-aura", True),
        ("CalVer", "2025.01.0", True),
        ("CalVer, Aura", "2025.01-aura", True),
    ],
)
@mock.patch(
    "neo4j_graphrag.experimental.components.kg_writer.Neo4jWriter._db_setup",
    return_value=None,
)
def test_get_version(
    _: Mock,
    driver: MagicMock,
    description: str,
    version: str,
    is_5_23_or_above: bool,
) -> None:
    execute_query_mock = MagicMock(
        return_value=(
            [
                {"versions": [version], "edition": "enterprise"},
            ],
            None,
            None,
        )
    )
    driver.execute_query = execute_query_mock
    neo4j_writer = Neo4jWriter(driver=driver)
    assert (
        neo4j_writer.is_version_5_23_or_above is is_5_23_or_above
    ), f"Failed is_version_5_23_or_above test case: {description}"


# =============================================================================
# Tests for MERGE functionality (use_merge=True, the default)
# =============================================================================


def test_upsert_node_query_merge_below_5_23() -> None:
    """Test that upsert_node_query_merge generates correct Cypher for Neo4j < 5.23."""
    query = upsert_node_query_merge(support_variable_scope_clause=False)
    # Should use CALL { WITH ... } syntax for older versions
    assert "CALL { WITH n,row" in query
    # Should use apoc.merge.node
    assert "apoc.merge.node" in query
    # Should use dot notation with backticks for property access
    assert "row.properties.`name`" in query
    # Should return elementId (not deprecated id())
    assert "elementId(n)" in query


def test_upsert_node_query_merge_5_23_or_above() -> None:
    """Test that upsert_node_query_merge generates correct Cypher for Neo4j >= 5.23."""
    query = upsert_node_query_merge(support_variable_scope_clause=True)
    # Should use CALL (vars) { } syntax for newer versions
    assert "CALL (n,row) {" in query
    # Should use apoc.merge.node
    assert "apoc.merge.node" in query
    # Should use dot notation with backticks for property access
    assert "row.properties.`name`" in query


def test_upsert_node_query_merge_custom_property() -> None:
    """Test that upsert_node_query_merge handles custom merge_property."""
    query = upsert_node_query_merge(
        support_variable_scope_clause=True,
        merge_property="id",
    )
    # Should use custom property in identity map
    assert "{`id`: row.properties.`id`}" in query
    # Should NOT use default 'name'
    assert "row.properties.`name`" not in query


def test_upsert_node_query_merge_special_chars_in_property() -> None:
    """Test that merge_property with special characters is properly escaped."""
    query = upsert_node_query_merge(
        support_variable_scope_clause=True,
        merge_property="my property",
    )
    # Backticks should handle property names with spaces
    assert "{`my property`: row.properties.`my property`}" in query


@mock.patch(
    "neo4j_graphrag.experimental.components.kg_writer.get_version",
    return_value=((5, 22, 0), False, False),
)
@mock.patch(
    "neo4j_graphrag.experimental.components.kg_writer.Neo4jWriter._db_setup",
    return_value=None,
)
def test_upsert_nodes_with_merge(_: Mock, driver: MagicMock) -> None:
    """Test node upsert using MERGE (use_merge=True, the default)."""
    driver.execute_query.return_value = (
        [{"element_id": "#1"}],
        None,
        None,
    )
    neo4j_writer = Neo4jWriter(driver=driver)  # use_merge=True is default
    node = Neo4jNode(id="1", label="Label", properties={"name": "TestEntity"})
    neo4j_writer._upsert_nodes(nodes=[node], lexical_graph_config=LexicalGraphConfig())
    driver.execute_query.assert_called_once_with(
        upsert_node_query_merge(False),
        parameters_={
            "rows": [
                {
                    "label": "Label",
                    "labels": ["Label", "__Entity__"],
                    "id": "1",
                    "properties": {"name": "TestEntity"},
                    "embedding_properties": None,
                }
            ]
        },
        database_=None,
    )


@mock.patch(
    "neo4j_graphrag.experimental.components.kg_writer.get_version",
    return_value=((5, 23, 0), False, False),
)
@mock.patch(
    "neo4j_graphrag.experimental.components.kg_writer.Neo4jWriter._db_setup",
    return_value=None,
)
def test_upsert_nodes_with_merge_5_23(_: Mock, driver: MagicMock) -> None:
    """Test MERGE uses correct CALL syntax for Neo4j >= 5.23."""
    driver.execute_query.return_value = (
        [{"element_id": "#1"}],
        None,
        None,
    )
    neo4j_writer = Neo4jWriter(driver=driver)  # use_merge=True is default
    node = Neo4jNode(id="1", label="Label", properties={"name": "TestEntity"})
    neo4j_writer._upsert_nodes(nodes=[node], lexical_graph_config=LexicalGraphConfig())
    driver.execute_query.assert_called_once_with(
        upsert_node_query_merge(True),  # True for 5.23+ syntax
        parameters_={
            "rows": [
                {
                    "label": "Label",
                    "labels": ["Label", "__Entity__"],
                    "id": "1",
                    "properties": {"name": "TestEntity"},
                    "embedding_properties": None,
                }
            ]
        },
        database_=None,
    )


@mock.patch(
    "neo4j_graphrag.experimental.components.kg_writer.get_version",
    return_value=((5, 22, 0), False, False),
)
@mock.patch(
    "neo4j_graphrag.experimental.components.kg_writer.Neo4jWriter._db_setup",
    return_value=None,
)
def test_upsert_nodes_with_custom_merge_property(_: Mock, driver: MagicMock) -> None:
    """Test node upsert using custom merge_property."""
    driver.execute_query.return_value = (
        [{"element_id": "#1"}],
        None,
        None,
    )
    neo4j_writer = Neo4jWriter(driver=driver, merge_property="id")
    node = Neo4jNode(id="1", label="Label", properties={"id": "entity-123"})
    neo4j_writer._upsert_nodes(nodes=[node], lexical_graph_config=LexicalGraphConfig())
    driver.execute_query.assert_called_once_with(
        upsert_node_query_merge(False, merge_property="id"),
        parameters_={
            "rows": [
                {
                    "label": "Label",
                    "labels": ["Label", "__Entity__"],
                    "id": "1",
                    "properties": {"id": "entity-123"},
                    "embedding_properties": None,
                }
            ]
        },
        database_=None,
    )


@pytest.mark.asyncio
@mock.patch(
    "neo4j_graphrag.experimental.components.kg_writer.Neo4jWriter._db_setup",
    return_value=None,
)
async def test_run_with_merge_default(_: Mock) -> None:
    """Test full pipeline run using MERGE (the default behavior)."""
    driver = MagicMock()
    driver.execute_query = Mock(
        side_effect=(
            # get_version
            ([{"versions": ["5.22.0"], "edition": "enterprise"}], None, None),
            # upsert nodes
            ([{"element_id": "#1"}], None, None),
            # upsert relationships
            (None, None, None),
        )
    )

    neo4j_writer = Neo4jWriter(driver=driver)  # use_merge=True is default

    node = Neo4jNode(id="1", label="Company", properties={"name": "Acme Corp"})
    rel = Neo4jRelationship(start_node_id="1", end_node_id="2", type="WORKS_FOR")
    graph = Neo4jGraph(nodes=[node], relationships=[rel])
    await neo4j_writer.run(graph=graph)

    # Should use MERGE query
    driver.execute_query.assert_any_call(
        upsert_node_query_merge(False),
        parameters_={
            "rows": [
                {
                    "label": "Company",
                    "labels": ["Company", "__Entity__"],
                    "id": "1",
                    "properties": {"name": "Acme Corp"},
                    "embedding_properties": None,
                }
            ]
        },
        database_=None,
    )


# =============================================================================
# Tests for merge_property validation
# =============================================================================


@mock.patch(
    "neo4j_graphrag.experimental.components.kg_writer.get_version",
    return_value=((5, 22, 0), False, False),
)
@mock.patch(
    "neo4j_graphrag.experimental.components.kg_writer.Neo4jWriter._db_setup",
    return_value=None,
)
def test_validate_merge_property_skips_nodes_missing_property(
    _: Mock, driver: MagicMock
) -> None:
    """Test that entity nodes without merge_property are skipped."""
    driver.execute_query.return_value = ([{"element_id": "#1"}], None, None)
    neo4j_writer = Neo4jWriter(driver=driver)  # use_merge=True, merge_property="name"

    # Entity node missing 'name' property - should be skipped
    node_missing = Neo4jNode(id="1", label="Company", properties={"other": "value"})
    # Entity node with 'name' property - should go to merge_nodes
    node_valid = Neo4jNode(id="2", label="Company", properties={"name": "ValidEntity"})

    merge_nodes, create_nodes, skipped = neo4j_writer._validate_merge_property(
        [node_missing, node_valid], LexicalGraphConfig()
    )

    assert skipped == 1
    assert len(merge_nodes) == 1
    assert merge_nodes[0].id == "2"
    assert len(create_nodes) == 0


@mock.patch(
    "neo4j_graphrag.experimental.components.kg_writer.get_version",
    return_value=((5, 22, 0), False, False),
)
@mock.patch(
    "neo4j_graphrag.experimental.components.kg_writer.Neo4jWriter._db_setup",
    return_value=None,
)
def test_validate_merge_property_separates_lexical_nodes(
    _: Mock, driver: MagicMock
) -> None:
    """Test that lexical graph nodes (Chunk, Document) go to create_nodes."""
    driver.execute_query.return_value = ([{"element_id": "#1"}], None, None)
    neo4j_writer = Neo4jWriter(driver=driver)

    # Entity node with name - should go to merge_nodes
    entity_node = Neo4jNode(id="1", label="Company", properties={"name": "Acme"})
    # Chunk node - should go to create_nodes (lexical graph)
    chunk_node = Neo4jNode(id="2", label="Chunk", properties={"text": "some text"})
    # Document node - should go to create_nodes (lexical graph)
    doc_node = Neo4jNode(id="3", label="Document", properties={"path": "/doc.pdf"})

    merge_nodes, create_nodes, skipped = neo4j_writer._validate_merge_property(
        [entity_node, chunk_node, doc_node], LexicalGraphConfig()
    )

    assert skipped == 0
    assert len(merge_nodes) == 1
    assert merge_nodes[0].id == "1"
    assert len(create_nodes) == 2
    assert {n.id for n in create_nodes} == {"2", "3"}


@mock.patch(
    "neo4j_graphrag.experimental.components.kg_writer.get_version",
    return_value=((5, 22, 0), False, False),
)
@mock.patch(
    "neo4j_graphrag.experimental.components.kg_writer.Neo4jWriter._db_setup",
    return_value=None,
)
def test_validate_merge_property_logs_warning(
    _: Mock, driver: MagicMock, caplog: pytest.LogCaptureFixture
) -> None:
    """Test that warning is logged for skipped entity nodes."""
    driver.execute_query.return_value = ([{"element_id": "#1"}], None, None)
    neo4j_writer = Neo4jWriter(driver=driver)

    node_missing = Neo4jNode(
        id="test-id", label="TestLabel", properties={"other": "value"}
    )

    with caplog.at_level(logging.WARNING):
        neo4j_writer._validate_merge_property([node_missing], LexicalGraphConfig())

    assert "Node (id=test-id, label=TestLabel)" in caplog.text
    assert "merge_property 'name'" in caplog.text
    assert "will be skipped" in caplog.text


@mock.patch(
    "neo4j_graphrag.experimental.components.kg_writer.get_version",
    return_value=((5, 22, 0), False, False),
)
@mock.patch(
    "neo4j_graphrag.experimental.components.kg_writer.Neo4jWriter._db_setup",
    return_value=None,
)
def test_validate_merge_property_skipped_when_use_merge_false(
    _: Mock, driver: MagicMock
) -> None:
    """Test that all nodes go to merge_nodes when use_merge=False."""
    driver.execute_query.return_value = ([{"element_id": "#1"}], None, None)
    neo4j_writer = Neo4jWriter(driver=driver, use_merge=False)

    # Node missing 'name' property - should NOT be skipped when use_merge=False
    node_missing = Neo4jNode(id="1", label="Label", properties={"other": "value"})

    merge_nodes, create_nodes, skipped = neo4j_writer._validate_merge_property(
        [node_missing], LexicalGraphConfig()
    )

    assert skipped == 0
    assert len(merge_nodes) == 1
    assert len(create_nodes) == 0


@pytest.mark.asyncio
@mock.patch(
    "neo4j_graphrag.experimental.components.kg_writer.Neo4jWriter._db_setup",
    return_value=None,
)
async def test_run_returns_skipped_count_in_metadata(_: Mock) -> None:
    """Test that skipped count is returned in metadata."""
    driver = MagicMock()
    driver.execute_query = Mock(
        side_effect=(
            # get_version
            ([{"versions": ["5.22.0"], "edition": "enterprise"}], None, None),
            # upsert nodes (only valid node is included)
            ([{"element_id": "#1"}], None, None),
            # upsert relationships
            (None, None, None),
        )
    )

    neo4j_writer = Neo4jWriter(driver=driver)

    # One node with 'name', one without
    node_valid = Neo4jNode(id="1", label="Company", properties={"name": "Acme"})
    node_invalid = Neo4jNode(id="2", label="Company", properties={"title": "Other"})
    graph = Neo4jGraph(nodes=[node_valid, node_invalid], relationships=[])

    result = await neo4j_writer.run(graph=graph)

    assert result.status == "SUCCESS"
    assert result.metadata is not None
    assert result.metadata["node_count"] == 2
    assert result.metadata["nodes_created"] == 1
    assert result.metadata["nodes_skipped_missing_merge_property"] == 1


@pytest.mark.asyncio
@mock.patch(
    "neo4j_graphrag.experimental.components.kg_writer.Neo4jWriter._db_setup",
    return_value=None,
)
async def test_run_skips_empty_batch_after_validation(_: Mock) -> None:
    """Test that empty batches after validation don't cause errors."""
    driver = MagicMock()
    driver.execute_query = Mock(
        side_effect=(
            # get_version
            ([{"versions": ["5.22.0"], "edition": "enterprise"}], None, None),
            # No upsert calls should happen - all nodes are invalid
        )
    )

    neo4j_writer = Neo4jWriter(driver=driver)

    # All nodes missing 'name' property
    node1 = Neo4jNode(id="1", label="Company", properties={"title": "A"})
    node2 = Neo4jNode(id="2", label="Company", properties={"title": "B"})
    graph = Neo4jGraph(nodes=[node1, node2], relationships=[])

    result = await neo4j_writer.run(graph=graph)

    assert result.status == "SUCCESS"
    assert result.metadata is not None
    assert result.metadata["nodes_created"] == 0
    assert result.metadata["nodes_skipped_missing_merge_property"] == 2
