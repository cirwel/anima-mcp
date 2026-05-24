"""Tests for OAuth wiring in tool_registry.py."""
import os
from unittest.mock import patch


class TestOAuthWiring:
    def test_fastmcp_created_without_oauth_when_no_env(self):
        """When ANIMA_OAUTH_ISSUER_URL is not set, FastMCP has no auth."""
        import anima_mcp.tool_registry as tr
        tr._fastmcp = None
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ANIMA_OAUTH_ISSUER_URL", None)
            mcp = tr.get_fastmcp()
            if mcp:
                assert mcp.settings.auth is None

    def test_fastmcp_created_with_oauth_when_env_set(self):
        """When ANIMA_OAUTH_ISSUER_URL is set, FastMCP has auth configured."""
        import anima_mcp.tool_registry as tr
        tr._fastmcp = None
        env = {
            "ANIMA_OAUTH_ISSUER_URL": "https://lumen.example.com",
            "ANIMA_OAUTH_SECRET": "test-secret",
        }
        with patch.dict(os.environ, env):
            mcp = tr.get_fastmcp()
            if mcp:
                assert mcp.settings.auth is not None
                assert str(mcp.settings.auth.issuer_url) == "https://lumen.example.com/"

    def test_resource_server_url_defaults_to_mcp_path(self):
        """resource_server_url must include /mcp/ so RFC 9728 metadata's
        `resource` field matches the URL claude.ai has stored. A bare-host
        resource (the prior default) caused claude.ai to mark the connector
        errored due to resource-mismatch validation."""
        import anima_mcp.tool_registry as tr
        tr._fastmcp = None
        env = {
            "ANIMA_OAUTH_ISSUER_URL": "https://lumen.example.com",
            "ANIMA_OAUTH_SECRET": "test-secret",
        }
        with patch.dict(os.environ, env):
            env.pop("ANIMA_OAUTH_RESOURCE_URL", None)
            os.environ.pop("ANIMA_OAUTH_RESOURCE_URL", None)
            mcp = tr.get_fastmcp()
            if mcp:
                assert str(mcp.settings.auth.resource_server_url) == "https://lumen.example.com/mcp/"

    def test_resource_server_url_env_override(self):
        """ANIMA_OAUTH_RESOURCE_URL overrides the default."""
        import anima_mcp.tool_registry as tr
        tr._fastmcp = None
        env = {
            "ANIMA_OAUTH_ISSUER_URL": "https://lumen.example.com",
            "ANIMA_OAUTH_RESOURCE_URL": "https://lumen.example.com/custom/",
            "ANIMA_OAUTH_SECRET": "test-secret",
        }
        with patch.dict(os.environ, env):
            mcp = tr.get_fastmcp()
            if mcp:
                assert str(mcp.settings.auth.resource_server_url) == "https://lumen.example.com/custom/"
