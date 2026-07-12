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

import json
from typing import Any, Generator
from unittest.mock import MagicMock, patch

import pytest

from neo4j_graphrag.exceptions import LLMGenerationError
from neo4j_graphrag.experimental.components.types import Neo4jGraph
from neo4j_graphrag.llm import BedrockLLM
from neo4j_graphrag.types import LLMMessage


@pytest.fixture
def mock_boto3() -> Generator[MagicMock, None, None]:
    with patch("neo4j_graphrag.llm.bedrock_llm.boto3") as mock_boto:
        mock_client = MagicMock()
        mock_boto.client.return_value = mock_client
        yield mock_boto


def _make_converse_response(text: str = "generated text") -> dict[str, Any]:
    return {
        "output": {
            "message": {
                "role": "assistant",
                "content": [{"text": text}],
            }
        }
    }


def test_bedrock_llm_missing_dependency() -> None:
    with patch("neo4j_graphrag.llm.bedrock_llm.boto3", None):
        with pytest.raises(ImportError) as exc:
            BedrockLLM(model_name="us.anthropic.claude-sonnet-4-5-20250929-v1:0")
        assert "Could not import boto3 python client" in str(exc.value)


def test_bedrock_llm_default_model_from_env(mock_boto3: MagicMock) -> None:
    with patch.dict("os.environ", {"BEDROCK_LLM_MODEL": "custom-llm-model"}):
        import importlib
        import sys

        original_boto3 = sys.modules.get("boto3")
        sys.modules["boto3"] = mock_boto3

        try:
            import neo4j_graphrag.llm.bedrock_llm as bedrock_llm_mod

            importlib.reload(bedrock_llm_mod)

            assert bedrock_llm_mod.DEFAULT_BEDROCK_LLM_MODEL == "custom-llm-model"

            llm = bedrock_llm_mod.BedrockLLM()
            assert llm.model_name == "custom-llm-model"
        finally:
            if original_boto3 is not None:
                sys.modules["boto3"] = original_boto3
            importlib.reload(bedrock_llm_mod)


def test_bedrock_invoke_happy_path(mock_boto3: MagicMock) -> None:
    mock_client = mock_boto3.client.return_value
    mock_client.converse.return_value = _make_converse_response("hello world")

    llm = BedrockLLM("us.anthropic.claude-sonnet-4-5-20250929-v1:0")
    response = llm.invoke("hello")

    assert response.content == "hello world"
    mock_client.converse.assert_called_once()


def test_bedrock_invoke_with_message_history(mock_boto3: MagicMock) -> None:
    mock_client = mock_boto3.client.return_value
    mock_client.converse.return_value = _make_converse_response("response")

    llm = BedrockLLM("us.anthropic.claude-sonnet-4-5-20250929-v1:0")
    history: list[LLMMessage] = [
        {"role": "user", "content": "previous question"},
        {"role": "assistant", "content": "previous answer"},
    ]
    response = llm.invoke("follow up", message_history=history)

    assert response.content == "response"
    call_kwargs = mock_client.converse.call_args[1]
    # 2 history messages + 1 new user message
    assert len(call_kwargs["messages"]) == 3


def test_bedrock_invoke_with_system_instruction(mock_boto3: MagicMock) -> None:
    mock_client = mock_boto3.client.return_value
    mock_client.converse.return_value = _make_converse_response("response")

    llm = BedrockLLM("us.anthropic.claude-sonnet-4-5-20250929-v1:0")
    response = llm.invoke("hello", system_instruction="You are a bot")

    assert response.content == "response"
    call_kwargs = mock_client.converse.call_args[1]
    assert call_kwargs["system"] == [{"text": "You are a bot"}]


@pytest.mark.asyncio
async def test_bedrock_ainvoke_happy_path(mock_boto3: MagicMock) -> None:
    mock_client = mock_boto3.client.return_value
    mock_client.converse.return_value = _make_converse_response("async response")

    llm = BedrockLLM("us.anthropic.claude-sonnet-4-5-20250929-v1:0")
    response = await llm.ainvoke("hello")

    assert response.content == "async response"
    mock_client.converse.assert_called_once()


def test_bedrock_invoke_v2_happy_path(mock_boto3: MagicMock) -> None:
    mock_client = mock_boto3.client.return_value
    mock_client.converse.return_value = _make_converse_response("v2 response")

    messages: list[LLMMessage] = [
        {"role": "system", "content": "You are a bot"},
        {"role": "user", "content": "hello"},
    ]

    llm = BedrockLLM("us.anthropic.claude-sonnet-4-5-20250929-v1:0")
    response = llm.invoke(messages)

    assert response.content == "v2 response"
    call_kwargs = mock_client.converse.call_args[1]
    assert call_kwargs["system"] == [{"text": "You are a bot"}]
    # only user message, system is extracted
    assert len(call_kwargs["messages"]) == 1


@pytest.mark.asyncio
async def test_bedrock_ainvoke_v2_happy_path(mock_boto3: MagicMock) -> None:
    mock_client = mock_boto3.client.return_value
    mock_client.converse.return_value = _make_converse_response("async v2")

    messages: list[LLMMessage] = [{"role": "user", "content": "hello"}]

    llm = BedrockLLM("us.anthropic.claude-sonnet-4-5-20250929-v1:0")
    response = await llm.ainvoke(messages)

    assert response.content == "async v2"


def test_bedrock_invoke_error(mock_boto3: MagicMock) -> None:
    mock_client = mock_boto3.client.return_value
    mock_client.converse.side_effect = Exception("API error")

    llm = BedrockLLM("us.anthropic.claude-sonnet-4-5-20250929-v1:0")
    with pytest.raises(LLMGenerationError):
        llm.invoke("hello")


def test_bedrock_invoke_empty_response(mock_boto3: MagicMock) -> None:
    mock_client = mock_boto3.client.return_value
    mock_client.converse.return_value = {"output": {"message": {"content": []}}}

    llm = BedrockLLM("us.anthropic.claude-sonnet-4-5-20250929-v1:0")
    with pytest.raises(LLMGenerationError, match="LLM returned empty response"):
        llm.invoke("hello")


def test_bedrock_supports_structured_output() -> None:
    assert BedrockLLM.supports_structured_output is True


def test_bedrock_invoke_v2_with_dict_response_format(mock_boto3: MagicMock) -> None:
    mock_client = mock_boto3.client.return_value
    mock_client.converse.return_value = _make_converse_response('{"answer": 42}')

    messages: list[LLMMessage] = [{"role": "user", "content": "hello"}]
    output_config = {"textFormat": {"type": "json_schema", "structure": {}}}

    llm = BedrockLLM("us.anthropic.claude-sonnet-4-5-20250929-v1:0")
    response = llm.invoke(messages, response_format=output_config)

    # a raw dict is passed straight through as the outputConfig
    call_kwargs = mock_client.converse.call_args[1]
    assert call_kwargs["outputConfig"] == output_config
    # non-pydantic response_format leaves the content untouched
    assert response.content == '{"answer": 42}'


def test_bedrock_invoke_v2_with_pydantic_response_format(mock_boto3: MagicMock) -> None:
    # the model is constrained to emit open maps as key/value-pair arrays
    constrained = json.dumps(
        {
            "nodes": [
                {
                    "id": "1",
                    "label": "Person",
                    "properties": [{"key": "name", "value": "Paul"}],
                    "embedding_properties": [],
                }
            ],
            "relationships": [],
        }
    )
    mock_client = mock_boto3.client.return_value
    mock_client.converse.return_value = _make_converse_response(constrained)

    messages: list[LLMMessage] = [{"role": "user", "content": "extract"}]
    llm = BedrockLLM("us.anthropic.claude-sonnet-4-5-20250929-v1:0")
    response = llm.invoke(messages, response_format=Neo4jGraph)

    # outputConfig carries a json_schema with the schema as a JSON string
    call_kwargs = mock_client.converse.call_args[1]
    json_schema = call_kwargs["outputConfig"]["textFormat"]["structure"]["jsonSchema"]
    assert json_schema["name"] == "Neo4jGraph"
    schema = json.loads(json_schema["schema"])
    # open maps are closed and every object forbids extra properties
    assert schema["additionalProperties"] is False

    # the key/value arrays are restored to maps, so the content validates
    graph = Neo4jGraph.model_validate_json(response.content)
    assert graph.nodes[0].properties == {"name": "Paul"}
    assert graph.nodes[0].embedding_properties == {}


@pytest.mark.asyncio
async def test_bedrock_ainvoke_v2_with_pydantic_response_format(
    mock_boto3: MagicMock,
) -> None:
    constrained = json.dumps(
        {
            "nodes": [
                {
                    "id": "1",
                    "label": "Person",
                    "properties": [],
                    "embedding_properties": [],
                }
            ],
            "relationships": [],
        }
    )
    mock_client = mock_boto3.client.return_value
    mock_client.converse.return_value = _make_converse_response(constrained)

    messages: list[LLMMessage] = [{"role": "user", "content": "extract"}]
    llm = BedrockLLM("us.anthropic.claude-sonnet-4-5-20250929-v1:0")
    response = await llm.ainvoke(messages, response_format=Neo4jGraph)

    call_kwargs = mock_client.converse.call_args[1]
    assert "outputConfig" in call_kwargs
    graph = Neo4jGraph.model_validate_json(response.content)
    assert graph.nodes[0].properties == {}


def test_bedrock_invoke_with_model_params(mock_boto3: MagicMock) -> None:
    mock_client = mock_boto3.client.return_value
    mock_client.converse.return_value = _make_converse_response("response")

    llm = BedrockLLM(
        "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        model_params={"temperature": 0.5, "maxTokens": 512},
    )
    llm.invoke("hello")

    call_kwargs = mock_client.converse.call_args[1]
    assert call_kwargs["inferenceConfig"] == {"temperature": 0.5, "maxTokens": 512}
