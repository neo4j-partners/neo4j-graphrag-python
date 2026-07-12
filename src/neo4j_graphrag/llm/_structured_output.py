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

# ---------------------------------------------------------------------------
# Shared helpers for providers whose structured-output APIs use constrained
# decoding over a closed JSON Schema subset (currently Anthropic's
# ``output_config`` and Bedrock's ``outputConfig.textFormat``).
#
# Both APIs reject open-ended maps (Pydantic ``dict[str, X]`` -> a schema under
# ``additionalProperties``) with a 400, and both require every object to set
# ``additionalProperties: false``. Naively forcing ``additionalProperties: false``
# would instead make those maps un-fillable and silently drop every property
# value.
#
# To fix the 400 *without* dropping properties, and without touching the shared
# components, :func:`to_constrained_json_schema` transforms open maps into closed
# key/value-pair arrays on the way out, and :func:`restore_open_maps` converts
# them back to maps on the way in, so the returned content stays byte-compatible
# with the caller's Pydantic model (e.g. ``Neo4jGraph``).
# ---------------------------------------------------------------------------
from __future__ import annotations

import json
from typing import Any, Optional, Type, Union, cast

from pydantic import BaseModel


def _is_open_map(schema: dict[str, Any]) -> bool:
    """True if *schema* is an open-ended map (``dict[str, X]``) rather than a
    fixed-property object."""
    return (
        schema.get("type") == "object"
        and isinstance(schema.get("additionalProperties"), dict)
        and not schema.get("properties")
    )


def to_constrained_json_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Rewrite a JSON schema into the constrained-decoding subset.

    Open maps become closed ``[{"key": ..., "value": ...}]`` arrays, and every
    fixed-property object gets ``additionalProperties: false`` plus a full
    ``required`` list.
    """
    schema = dict(schema)
    if _is_open_map(schema):
        value_schema = to_constrained_json_schema(schema["additionalProperties"])
        return {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"key": {"type": "string"}, "value": value_schema},
                "required": ["key", "value"],
                "additionalProperties": False,
            },
        }
    if schema.get("type") == "object" and "properties" in schema:
        schema["properties"] = {
            key: to_constrained_json_schema(prop)
            for key, prop in schema["properties"].items()
        }
        schema["additionalProperties"] = False
        schema["required"] = list(schema["properties"].keys())
    if "items" in schema:
        schema["items"] = to_constrained_json_schema(schema["items"])
    for combinator in ("anyOf", "oneOf", "allOf"):
        if combinator in schema:
            schema[combinator] = [
                to_constrained_json_schema(variant) for variant in schema[combinator]
            ]
    if "$defs" in schema:
        schema["$defs"] = {
            name: to_constrained_json_schema(def_schema)
            for name, def_schema in schema["$defs"].items()
        }
    return schema


def _resolve_ref(schema: dict[str, Any], defs: dict[str, Any]) -> dict[str, Any]:
    """Resolve a local ``$ref`` against *defs*, if present."""
    ref = schema.get("$ref")
    if isinstance(ref, str):
        return cast("dict[str, Any]", defs.get(ref.split("/")[-1], {}))
    return schema


def restore_open_maps(value: Any, schema: dict[str, Any], defs: dict[str, Any]) -> Any:
    """Convert key/value-pair arrays produced for constrained decoding back into maps.

    Walks *value* alongside the caller's *original* (untransformed) JSON schema,
    so empty maps (``[]`` -> ``{}``) and genuine empty arrays are disambiguated
    correctly.
    """
    schema = _resolve_ref(schema, defs)
    if _is_open_map(schema) and isinstance(value, list):
        value_schema = schema["additionalProperties"]
        return {
            item["key"]: restore_open_maps(item["value"], value_schema, defs)
            for item in value
        }
    if schema.get("type") == "object" and isinstance(value, dict):
        properties = schema.get("properties", {})
        return {
            key: (
                restore_open_maps(val, properties[key], defs)
                if key in properties
                else val
            )
            for key, val in value.items()
        }
    if schema.get("type") == "array" and isinstance(value, list):
        item_schema = schema.get("items", {})
        return [restore_open_maps(item, item_schema, defs) for item in value]
    return value


def restore_structured_output_text(
    text: str,
    response_format: Optional[Union[Type[BaseModel], dict[str, Any]]],
) -> str:
    """Reverse :func:`to_constrained_json_schema` on a response's text content.

    Converts the key/value-pair arrays the model was constrained to emit back
    into the open maps expected by the caller's Pydantic model, so ``text``
    round-trips to a document that validates against *response_format*. Any
    non-Pydantic *response_format* (``None`` or a raw schema dict) and any text
    that is not valid JSON are returned unchanged.
    """
    if not (
        isinstance(response_format, type) and issubclass(response_format, BaseModel)
    ):
        return text
    original_schema = response_format.model_json_schema()
    defs = original_schema.get("$defs", {})
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return text
    restored = restore_open_maps(data, original_schema, defs)
    return json.dumps(restored)
