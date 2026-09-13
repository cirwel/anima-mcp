"""Tool contract tests — verify tool definitions, handlers, and schema consistency.

These catch tool definition drift before it reaches production:
- Every TOOLS entry has a HANDLERS entry and vice versa
- Schema validity (required fields exist in properties)
- No duplicate tool names
- Handler functions can be imported
- Governance bridge uses documented tool names
"""


import pytest

from anima_mcp.tool_registry import (
    TOOLS,
    HANDLERS,
    _create_tool_wrapper,
    _json_type_to_python,
)


# ============================================================
# TOOLS ↔ HANDLERS parity
# ============================================================

def test_every_tool_has_a_handler():
    """Every tool in TOOLS must have a corresponding HANDLERS entry."""
    tool_names = {t.name for t in TOOLS}
    handler_names = set(HANDLERS.keys())
    missing = tool_names - handler_names
    assert not missing, f"Tools without handlers: {missing}"


def test_every_handler_has_a_tool():
    """Every handler in HANDLERS must have a corresponding TOOLS entry."""
    tool_names = {t.name for t in TOOLS}
    handler_names = set(HANDLERS.keys())
    extra = handler_names - tool_names
    assert not extra, f"Handlers without tool definitions: {extra}"


# ============================================================
# No duplicates
# ============================================================

def test_no_duplicate_tool_names():
    """No two tools should share the same name."""
    names = [t.name for t in TOOLS]
    dupes = [n for n in names if names.count(n) > 1]
    assert not dupes, f"Duplicate tool names: {set(dupes)}"


# ============================================================
# Schema validity
# ============================================================

def test_all_tools_have_valid_input_schema():
    """Every tool must have a valid inputSchema with 'type': 'object'."""
    for tool in TOOLS:
        schema = tool.inputSchema
        assert isinstance(schema, dict), f"{tool.name}: inputSchema is not a dict"
        assert schema.get("type") == "object", f"{tool.name}: inputSchema type must be 'object'"


def test_required_fields_exist_in_properties():
    """Every field listed in 'required' must exist in 'properties'."""
    for tool in TOOLS:
        schema = tool.inputSchema
        required = schema.get("required", [])
        properties = schema.get("properties", {})
        for field in required:
            assert field in properties, (
                f"{tool.name}: required field '{field}' not in properties"
            )


def test_property_types_are_known():
    """Every property type must be a recognized JSON Schema type."""
    known_types = {"string", "integer", "number", "boolean", "array", "object", "null"}
    for tool in TOOLS:
        schema = tool.inputSchema
        for prop_name, prop_def in schema.get("properties", {}).items():
            prop_type = prop_def.get("type")
            if prop_type is None:
                continue  # No type specified — allowed for enum-only
            if isinstance(prop_type, list):
                for t in prop_type:
                    assert t in known_types, (
                        f"{tool.name}.{prop_name}: unknown type '{t}'"
                    )
            else:
                assert prop_type in known_types, (
                    f"{tool.name}.{prop_name}: unknown type '{prop_type}'"
                )


def test_enum_properties_have_values():
    """If a property uses 'enum', it must have at least one value."""
    for tool in TOOLS:
        schema = tool.inputSchema
        for prop_name, prop_def in schema.get("properties", {}).items():
            if "enum" in prop_def:
                assert len(prop_def["enum"]) > 0, (
                    f"{tool.name}.{prop_name}: enum is empty"
                )


def test_manage_display_schema_exposes_auto_rotate_control():
    """The advertised MCP contract includes the restart-safe rotation setter."""
    tool = next(tool for tool in TOOLS if tool.name == "manage_display")
    properties = tool.inputSchema["properties"]

    assert "set_auto_rotate" in properties["action"]["enum"]
    assert properties["enabled"]["type"] == "boolean"
    assert {
        "if": {
            "properties": {"action": {"const": "set_auto_rotate"}},
            "required": ["action"],
        },
        "then": {"required": ["enabled"]},
    } in tool.inputSchema["allOf"]


def test_boolean_schema_generates_boolean_annotation():
    """FastMCP must not advertise strings for JSON Schema boolean values."""
    assert _json_type_to_python("boolean") is bool


def test_manage_display_wrapper_keeps_enabled_boolean_only():
    """The generated FastMCP signature allows bool or omission, never strings."""
    import inspect
    from types import NoneType
    from typing import get_args

    tool = next(tool for tool in TOOLS if tool.name == "manage_display")
    wrapper = _create_tool_wrapper(HANDLERS["manage_display"], tool.name, tool)
    annotation = inspect.signature(wrapper).parameters["enabled"].annotation

    assert set(get_args(annotation)) == {bool, NoneType}


def test_fastmcp_manage_display_schema_preserves_contract(monkeypatch):
    """The production FastMCP listing retains enum and conditional constraints."""
    import asyncio
    from anima_mcp import tool_registry

    monkeypatch.setattr(tool_registry, "_fastmcp", None)
    server = tool_registry.get_fastmcp()
    tools = asyncio.run(server.list_tools())
    schema = next(tool.inputSchema for tool in tools if tool.name == "manage_display")

    assert "set_auto_rotate" in schema["properties"]["action"]["enum"]
    assert schema["properties"]["enabled"]["type"] == "boolean"
    assert schema["allOf"][0]["then"] == {"required": ["enabled"]}


# ============================================================
# Handler importability
# ============================================================

def test_all_handlers_are_callable():
    """Every handler function must be callable."""
    for name, handler in HANDLERS.items():
        assert callable(handler), f"Handler for '{name}' is not callable"


def test_all_handlers_are_async():
    """Every handler function should be an async function."""
    import inspect
    for name, handler in HANDLERS.items():
        assert inspect.iscoroutinefunction(handler), (
            f"Handler for '{name}' is not async"
        )


# ============================================================
# Governance bridge tool names
# ============================================================

def _governance_tool_names(module_file: str) -> set:
    """String literals used as a ``"name"`` dict value in an anima_mcp module."""
    import ast
    from pathlib import Path

    path = Path(__file__).parent.parent / "src" / "anima_mcp" / module_file
    found = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if (isinstance(key, ast.Constant) and key.value == "name"
                        and isinstance(value, ast.Constant) and isinstance(value.value, str)):
                    found.add(value.value)
    return found


# Names UNITARES's /mcp/ endpoint advertises in tools/list. /mcp/ does NOT
# resolve the alias table (unitares src/mcp_handlers/tool_stability.py) that
# REST /v1/tools/call and stdio do — an unadvertised name there comes back as
# {"isError": true, "text": "Unknown tool: ..."}, and a caller that only checks
# for a "result" key reads that as success. Measured 2026-09-13: both
# update_agent_metadata and store_knowledge_graph had been refused this way.
# Before adding a name here, confirm it with tools/list against /mcp/.
GOVERNANCE_CALLS = {
    "unitares_bridge.py": {
        "sync_state",  # advertised alias of process_agent_update (unitares c737b24c)
        "identity",
        "agent",  # action="update"; was update_agent_metadata
        "record_result",  # advertised alias of outcome_event (unitares c737b24c)
    },
    "unitares_knowledge.py": {
        "knowledge",  # action="store"; was store_knowledge_graph
    },
}

PRE_CONSOLIDATION_NAMES = {"update_agent_metadata", "store_knowledge_graph"}


@pytest.mark.parametrize("module_file", sorted(GOVERNANCE_CALLS))
def test_governance_bridge_tool_names_are_valid(module_file):
    """Every UNITARES tool name a module calls is one /mcp/ advertises."""
    found = _governance_tool_names(module_file)

    assert found == GOVERNANCE_CALLS[module_file], (
        f"{module_file} calls {sorted(found)}; expected {sorted(GOVERNANCE_CALLS[module_file])}. "
        "A new name must be confirmed against /mcp/ tools/list first."
    )
    assert not found & PRE_CONSOLIDATION_NAMES


# ============================================================
# Tool count sanity
# ============================================================

def test_tool_count_minimum():
    """Sanity check: at least 20 tools should be registered."""
    assert len(TOOLS) >= 20, f"Only {len(TOOLS)} tools registered (expected >= 20)"
    assert len(HANDLERS) >= 20, f"Only {len(HANDLERS)} handlers registered (expected >= 20)"
