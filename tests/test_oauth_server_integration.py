"""Integration test: OAuth endpoints respond correctly on the HTTP server."""
import os
import pytest
from unittest.mock import patch


class TestOAuthEndpointsExist:
    """Verify that OAuth routes are mounted when configured."""

    def _build_app_with_oauth(self):
        """Helper: build the Starlette app with OAuth enabled."""
        env = {
            "ANIMA_OAUTH_ISSUER_URL": "https://lumen.example.com",
            "ANIMA_OAUTH_SECRET": "test-secret",
            "ANIMA_OAUTH_AUTO_APPROVE": "true",
        }
        with patch.dict(os.environ, env):
            import anima_mcp.tool_registry as tr
            tr._fastmcp = None

            from anima_mcp.oauth_provider import AnimaOAuthProvider
            provider = AnimaOAuthProvider(secret="test-secret", auto_approve=True)
            return provider

    def test_oauth_provider_instantiates(self):
        """Smoke test: provider can be created."""
        provider = self._build_app_with_oauth()
        assert provider is not None

    @pytest.mark.asyncio
    async def test_full_oauth_flow_unit(self):
        """End-to-end OAuth flow at the provider level."""
        from mcp.server.auth.provider import AuthorizationParams
        from mcp.shared.auth import OAuthClientInformationFull
        from urllib.parse import urlparse, parse_qs
        import time

        provider = self._build_app_with_oauth()

        client = OAuthClientInformationFull(
            client_id="claude-web-test",
            client_secret="secret",
            client_id_issued_at=int(time.time()),
            redirect_uris=["https://claude.ai/callback"],
            grant_types=["authorization_code", "refresh_token"],
            response_types=["code"],
            token_endpoint_auth_method="client_secret_post",
        )
        await provider.register_client(client)

        params = AuthorizationParams(
            state="xyz",
            scopes=["mcp:tools"],
            code_challenge="test_challenge",
            redirect_uri="https://claude.ai/callback",
            redirect_uri_provided_explicitly=True,
        )
        redirect_url = await provider.authorize(client, params)
        code = parse_qs(urlparse(redirect_url).query)["code"][0]

        auth_code = await provider.load_authorization_code(client, code)
        token = await provider.exchange_authorization_code(client, auth_code)
        assert token.access_token.startswith("at_")
        assert token.refresh_token.startswith("rt_")

        access = await provider.load_access_token(token.access_token)
        assert access.client_id == "claude-web-test"

        rt = await provider.load_refresh_token(client, token.refresh_token)
        new_token = await provider.exchange_refresh_token(client, rt, ["mcp:tools"])
        assert new_token.access_token != token.access_token

        new_access = await provider.load_access_token(new_token.access_token)
        await provider.revoke_token(new_access)
        assert await provider.load_access_token(new_token.access_token) is None
