"""Tests for AnimaOAuthProvider — in-memory OAuth 2.1 provider."""
import time
import pytest
from anima_mcp.oauth_provider import AnimaOAuthProvider
from mcp.shared.auth import OAuthClientInformationFull


@pytest.fixture
def provider():
    return AnimaOAuthProvider(secret="test-secret", auto_approve=True)


@pytest.fixture
def sample_client_info():
    return OAuthClientInformationFull(
        client_id="test-client-123",
        client_secret="test-secret-456",
        client_id_issued_at=int(time.time()),
        redirect_uris=["https://example.com/callback"],
        client_name="Test Client",
        grant_types=["authorization_code", "refresh_token"],
        response_types=["code"],
        token_endpoint_auth_method="client_secret_post",
    )


class TestClientRegistration:
    async def test_register_and_get_client(self, provider, sample_client_info):
        await provider.register_client(sample_client_info)
        result = await provider.get_client("test-client-123")
        assert result is not None
        assert result.client_id == "test-client-123"
        assert result.client_name == "Test Client"

    async def test_get_unknown_client_returns_none(self, provider):
        result = await provider.get_client("nonexistent")
        assert result is None


class TestAuthorize:
    async def test_authorize_auto_approve_returns_redirect_with_code(self, provider, sample_client_info):
        from mcp.server.auth.provider import AuthorizationParams
        await provider.register_client(sample_client_info)
        params = AuthorizationParams(
            state="test-state",
            scopes=["mcp:tools"],
            code_challenge="challenge123",
            redirect_uri="https://example.com/callback",
            redirect_uri_provided_explicitly=True,
        )
        redirect_url = await provider.authorize(sample_client_info, params)
        assert "https://example.com/callback" in redirect_url
        assert "code=" in redirect_url
        assert "state=test-state" in redirect_url

    async def test_authorize_no_auto_approve_returns_consent_page(self, sample_client_info):
        provider = AnimaOAuthProvider(secret="test-secret", auto_approve=False)
        from mcp.server.auth.provider import AuthorizationParams
        await provider.register_client(sample_client_info)
        params = AuthorizationParams(
            state="test-state",
            scopes=["mcp:tools"],
            code_challenge="challenge123",
            redirect_uri="https://example.com/callback",
            redirect_uri_provided_explicitly=True,
        )
        redirect_url = await provider.authorize(sample_client_info, params)
        assert "code=" in redirect_url


class TestTokenExchange:
    async def test_exchange_authorization_code(self, provider, sample_client_info):
        from mcp.server.auth.provider import AuthorizationParams
        await provider.register_client(sample_client_info)
        params = AuthorizationParams(
            state="s", scopes=["mcp:tools"], code_challenge="ch",
            redirect_uri="https://example.com/callback",
            redirect_uri_provided_explicitly=True,
        )
        redirect_url = await provider.authorize(sample_client_info, params)
        from urllib.parse import urlparse, parse_qs
        parsed = urlparse(redirect_url)
        code = parse_qs(parsed.query)["code"][0]

        auth_code = await provider.load_authorization_code(sample_client_info, code)
        assert auth_code is not None

        token = await provider.exchange_authorization_code(sample_client_info, auth_code)
        assert token.access_token
        assert token.refresh_token
        assert token.token_type == "Bearer"
        assert token.expires_in == 3600

        used = await provider.load_authorization_code(sample_client_info, code)
        assert used is None

    async def test_exchange_refresh_token(self, provider, sample_client_info):
        from mcp.server.auth.provider import AuthorizationParams
        await provider.register_client(sample_client_info)
        params = AuthorizationParams(
            state="s", scopes=["mcp:tools"], code_challenge="ch",
            redirect_uri="https://example.com/callback",
            redirect_uri_provided_explicitly=True,
        )
        redirect_url = await provider.authorize(sample_client_info, params)
        from urllib.parse import urlparse, parse_qs
        code = parse_qs(urlparse(redirect_url).query)["code"][0]
        auth_code = await provider.load_authorization_code(sample_client_info, code)
        token = await provider.exchange_authorization_code(sample_client_info, auth_code)

        rt = await provider.load_refresh_token(sample_client_info, token.refresh_token)
        assert rt is not None
        new_token = await provider.exchange_refresh_token(sample_client_info, rt, ["mcp:tools"])
        assert new_token.access_token != token.access_token
        assert new_token.refresh_token

    async def test_refresh_token_has_expires_at_attr(self, provider, sample_client_info):
        """MCP SDK's token handler reads `refresh_token.expires_at` directly.
        Missing the attribute → AttributeError → 500 on /token → claude.ai
        connector drops after ~1h (access_token_ttl). Regression test for the
        2026-05-24 incident."""
        import time as _t
        from mcp.server.auth.provider import AuthorizationParams
        await provider.register_client(sample_client_info)
        params = AuthorizationParams(
            state="s", scopes=["mcp:tools"], code_challenge="ch",
            redirect_uri="https://example.com/callback",
            redirect_uri_provided_explicitly=True,
        )
        redirect_url = await provider.authorize(sample_client_info, params)
        from urllib.parse import urlparse, parse_qs
        code = parse_qs(urlparse(redirect_url).query)["code"][0]
        auth_code = await provider.load_authorization_code(sample_client_info, code)
        token = await provider.exchange_authorization_code(sample_client_info, auth_code)

        rt = await provider.load_refresh_token(sample_client_info, token.refresh_token)
        assert rt is not None
        assert hasattr(rt, "expires_at"), "SDK token handler reads .expires_at"
        assert rt.expires_at is not None
        assert rt.expires_at > _t.time()
        # Mirror the exact SDK check at mcp/server/auth/handlers/token.py:209
        assert not (rt.expires_at and rt.expires_at < _t.time())

        # Refresh-issued refresh tokens must also have expires_at
        new_token = await provider.exchange_refresh_token(sample_client_info, rt, ["mcp:tools"])
        rt2 = await provider.load_refresh_token(sample_client_info, new_token.refresh_token)
        assert rt2 is not None
        assert rt2.expires_at is not None
        assert rt2.expires_at > _t.time()


class TestTokenVerification:
    async def test_load_access_token(self, provider, sample_client_info):
        from mcp.server.auth.provider import AuthorizationParams
        await provider.register_client(sample_client_info)
        params = AuthorizationParams(
            state="s", scopes=["mcp:tools"], code_challenge="ch",
            redirect_uri="https://example.com/callback",
            redirect_uri_provided_explicitly=True,
        )
        redirect_url = await provider.authorize(sample_client_info, params)
        from urllib.parse import urlparse, parse_qs
        code = parse_qs(urlparse(redirect_url).query)["code"][0]
        auth_code = await provider.load_authorization_code(sample_client_info, code)
        token = await provider.exchange_authorization_code(sample_client_info, auth_code)

        access = await provider.load_access_token(token.access_token)
        assert access is not None
        assert access.client_id == "test-client-123"
        assert "mcp:tools" in access.scopes

    async def test_load_invalid_token_returns_none(self, provider):
        result = await provider.load_access_token("bogus-token")
        assert result is None


class TestRevocation:
    async def test_revoke_access_token(self, provider, sample_client_info):
        from mcp.server.auth.provider import AuthorizationParams
        await provider.register_client(sample_client_info)
        params = AuthorizationParams(
            state="s", scopes=["mcp:tools"], code_challenge="ch",
            redirect_uri="https://example.com/callback",
            redirect_uri_provided_explicitly=True,
        )
        redirect_url = await provider.authorize(sample_client_info, params)
        from urllib.parse import urlparse, parse_qs
        code = parse_qs(urlparse(redirect_url).query)["code"][0]
        auth_code = await provider.load_authorization_code(sample_client_info, code)
        token = await provider.exchange_authorization_code(sample_client_info, auth_code)

        access = await provider.load_access_token(token.access_token)
        assert access is not None
        await provider.revoke_token(access)
        assert await provider.load_access_token(token.access_token) is None
