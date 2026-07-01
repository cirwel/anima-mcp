"""Tool contract tests — verify tool definitions, handlers, and schema consistency.

These catch tool definition drift before it reaches production:
- Every TOOLS entry has a HANDLERS entry and vice versa
- Schema validity (required fields exist in properties)
- No duplicate tool names
- Handler functions can be imported
- Governance bridge uses documented tool names
"""



from anima_mcp.tool_registry import TOOLS, HANDLERS


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

def test_governance_bridge_tool_names_are_valid():
    """Tool names used by unitares_bridge.py must be documented constants."""
    # These are the tool names called on the UNITARES governance MCP server.
    # If any of these change in governance-mcp, the bridge will silently fail.
    expected_tools = {
        "sync_state",  # advertised alias of process_agent_update (unitares c737b24c)
        "identity",
        "update_agent_metadata",
        "record_result",  # advertised alias of outcome_event (unitares c737b24c)
    }

    # Verify they appear as string literals in unitares_bridge.py
    import ast
    from pathlib import Path

    bridge_path = Path(__file__).parent.parent / "src" / "anima_mcp" / "unitares_bridge.py"
    source = bridge_path.read_text()
    tree = ast.parse(source)

    # Collect all string literals that appear as values for "name" keys in dicts
    found_tool_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if (isinstance(key, ast.Constant) and key.value == "name"
                        and isinstance(value, ast.Constant) and isinstance(value.value, str)):
                    found_tool_names.add(value.value)

    missing = expected_tools - found_tool_names
    assert not missing, (
        f"Expected governance tool names not found in unitares_bridge.py: {missing}"
    )


# ============================================================
# Tool count sanity
# ============================================================

def test_tool_count_minimum():
    """Sanity check: at least 20 tools should be registered."""
    assert len(TOOLS) >= 20, f"Only {len(TOOLS)} tools registered (expected >= 20)"
    assert len(HANDLERS) >= 20, f"Only {len(HANDLERS)} handlers registered (expected >= 20)"
