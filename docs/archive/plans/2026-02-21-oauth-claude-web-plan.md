# OAuth 2.1 for Claude.ai MCP Connector — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add self-contained OAuth 2.1 to anima-mcp so Claude.ai can connect as a custom MCP integration via `lumen-anima.ngrok.io`.

**Architecture:** Implement the MCP SDK's `OAuthAuthorizationServerProvider` protocol with in-memory storage. Wire it into the existing FastMCP instance and manually-built Starlette app. OAuth protects only `/mcp`; dashboard/REST endpoints remain open.

**Tech Stack:** MCP SDK 1.26.0 (`mcp.server.auth`), Starlette, Python 3.11+, pytest + httpx for testing.

**Design doc:** `docs/plans/2026-02-21-oauth-claude-web-design.md`

---

### Task 1: OAuth Provider — Tests

**Files:**
- Create: `tests/test_oauth_provider.py`

**Step 1: Write failing tests for the OAuth provider**

```python
"""Tests for AnimaOAuthProvider — in-memory OAuth 2.1 provider."""
import time
import pytest
from anima_mcp.oauth_provider import AnimaOAuthProvider, AuthCodeEntry
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
        # When auto_approve=False, should still redirect (consent is sync for simplicity)
        # In a real impl, this would show a page — but for single-user we auto-approve
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
        # Extract code from redirect URL
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

        # Code should be consumed (not reusable)
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

        # Now refresh
        rt = await provider.load_refresh_token(sample_client_info, token.refresh_token)
        assert rt is not None
        new_token = await provider.exchange_refresh_token(sample_client_info, rt, ["mcp:tools"])
        assert new_token.access_token != token.access_token
        assert new_token.refresh_token


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
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/cirwel/projects/anima-mcp && python -m pytest tests/test_oauth_provider.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'anima_mcp.oauth_provider'`

**Step 3: Commit test file**

```bash
cd /Users/cirwel/projects/anima-mcp
git add tests/test_oauth_provider.py
git commit -m "test: add OAuth 2.1 provider tests (red)"
```

---

### Task 2: OAuth Provider — Implementation

**Files:**
- Create: `src/anima_mcp/oauth_provider.py`

**Step 1: Implement the OAuth provider**

```python
"""
In-memory OAuth 2.1 Authorization Server Provider for anima-mcp.

Implements the MCP SDK's OAuthAuthorizationServerProvider protocol.
Tokens stored in-memory — reset on server restart (Claude.ai re-authenticates).
"""
from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import time
from dataclasses import dataclass, field
from urllib.parse import urlencode

from mcp.server.auth.provider import (
    AccessToken,
    AuthorizationParams,
    OAuthAuthorizationServerProvider,
    OAuthToken,
)
from mcp.shared.auth import OAuthClientInformationFull


@dataclass
class AuthCodeEntry:
    """Stored authorization code with metadata."""
    code: str
    client_id: str
    redirect_uri: str
    code_challenge: str
    scopes: list[str]
    created_at: float = field(default_factory=time.time)
    resource: str | None = None

    def is_expired(self, ttl: int = 300) -> bool:
        return time.time() - self.created_at > ttl


@dataclass
class RefreshTokenEntry:
    """Stored refresh token with metadata."""
    token: str
    client_id: str
    scopes: list[str]
    created_at: float = field(default_factory=time.time)

    def is_expired(self, ttl: int = 604800) -> bool:  # 7 days
        return time.time() - self.created_at > ttl


class AnimaOAuthProvider:
    """
    In-memory OAuth 2.1 Authorization Server for anima-mcp.

    Implements OAuthAuthorizationServerProvider protocol from the MCP SDK.
    Single-user, personal server — optimized for simplicity.
    """

    def __init__(
        self,
        secret: str | None = None,
        auto_approve: bool = True,
        access_token_ttl: int = 3600,
        refresh_token_ttl: int = 604800,
        auth_code_ttl: int = 300,
    ):
        self._secret = secret or secrets.token_hex(32)
        self._auto_approve = auto_approve
        self._access_token_ttl = access_token_ttl
        self._refresh_token_ttl = refresh_token_ttl
        self._auth_code_ttl = auth_code_ttl

        # In-memory stores
        self._clients: dict[str, OAuthClientInformationFull] = {}
        self._auth_codes: dict[str, AuthCodeEntry] = {}
        self._access_tokens: dict[str, AccessToken] = {}
        self._refresh_tokens: dict[str, RefreshTokenEntry] = {}

    def _generate_token(self, prefix: str = "at") -> str:
        """Generate a cryptographically random token."""
        raw = secrets.token_hex(32)
        return f"{prefix}_{raw}"

    # --- Client Registration ---

    async def get_client(self, client_id: str) -> OAuthClientInformationFull | None:
        return self._clients.get(client_id)

    async def register_client(self, client_info: OAuthClientInformationFull) -> None:
        if not client_info.client_id:
            client_info.client_id = f"anima_{secrets.token_hex(16)}"
        if not client_info.client_secret:
            client_info.client_secret = secrets.token_hex(32)
        if not client_info.client_id_issued_at:
            client_info.client_id_issued_at = int(time.time())
        self._clients[client_info.client_id] = client_info

    # --- Authorization ---

    async def authorize(
        self, client: OAuthClientInformationFull, params: AuthorizationParams
    ) -> str:
        # Generate authorization code (>= 160 bits entropy per spec)
        code = secrets.token_hex(24)  # 192 bits

        entry = AuthCodeEntry(
            code=code,
            client_id=client.client_id,
            redirect_uri=str(params.redirect_uri),
            code_challenge=params.code_challenge,
            scopes=params.scopes or [],
            resource=params.resource,
        )
        self._auth_codes[code] = entry

        # Build redirect URL with code and state
        query = {"code": code}
        if params.state:
            query["state"] = params.state
        redirect = str(params.redirect_uri)
        separator = "&" if "?" in redirect else "?"
        return f"{redirect}{separator}{urlencode(query)}"

    # --- Authorization Code ---

    async def load_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: str
    ) -> AuthCodeEntry | None:
        entry = self._auth_codes.get(authorization_code)
        if entry is None:
            return None
        if entry.client_id != client.client_id:
            return None
        if entry.is_expired(self._auth_code_ttl):
            del self._auth_codes[authorization_code]
            return None
        return entry

    async def exchange_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: AuthCodeEntry
    ) -> OAuthToken:
        # Remove code (single-use)
        self._auth_codes.pop(authorization_code.code, None)

        # Generate tokens
        access_token_str = self._generate_token("at")
        refresh_token_str = self._generate_token("rt")
        expires_at = int(time.time()) + self._access_token_ttl

        # Store access token
        self._access_tokens[access_token_str] = AccessToken(
            token=access_token_str,
            client_id=client.client_id,
            scopes=authorization_code.scopes or ["mcp:tools"],
            expires_at=expires_at,
            resource=authorization_code.resource,
        )

        # Store refresh token
        self._refresh_tokens[refresh_token_str] = RefreshTokenEntry(
            token=refresh_token_str,
            client_id=client.client_id,
            scopes=authorization_code.scopes or ["mcp:tools"],
        )

        return OAuthToken(
            access_token=access_token_str,
            token_type="Bearer",
            expires_in=self._access_token_ttl,
            scope=" ".join(authorization_code.scopes) if authorization_code.scopes else "mcp:tools",
            refresh_token=refresh_token_str,
        )

    # --- Refresh Token ---

    async def load_refresh_token(
        self, client: OAuthClientInformationFull, refresh_token: str
    ) -> RefreshTokenEntry | None:
        entry = self._refresh_tokens.get(refresh_token)
        if entry is None:
            return None
        if entry.client_id != client.client_id:
            return None
        if entry.is_expired(self._refresh_token_ttl):
            del self._refresh_tokens[refresh_token]
            return None
        return entry

    async def exchange_refresh_token(
        self,
        client: OAuthClientInformationFull,
        refresh_token: RefreshTokenEntry,
        scopes: list[str],
    ) -> OAuthToken:
        # Revoke old refresh token (rotation)
        self._refresh_tokens.pop(refresh_token.token, None)

        # Generate new tokens
        access_token_str = self._generate_token("at")
        new_refresh_str = self._generate_token("rt")
        expires_at = int(time.time()) + self._access_token_ttl
        effective_scopes = scopes or refresh_token.scopes or ["mcp:tools"]

        self._access_tokens[access_token_str] = AccessToken(
            token=access_token_str,
            client_id=client.client_id,
            scopes=effective_scopes,
            expires_at=expires_at,
        )

        self._refresh_tokens[new_refresh_str] = RefreshTokenEntry(
            token=new_refresh_str,
            client_id=client.client_id,
            scopes=effective_scopes,
        )

        return OAuthToken(
            access_token=access_token_str,
            token_type="Bearer",
            expires_in=self._access_token_ttl,
            scope=" ".join(effective_scopes),
            refresh_token=new_refresh_str,
        )

    # --- Access Token ---

    async def load_access_token(self, token: str) -> AccessToken | None:
        entry = self._access_tokens.get(token)
        if entry is None:
            return None
        if entry.expires_at and entry.expires_at < int(time.time()):
            del self._access_tokens[token]
            return None
        return entry

    # --- Revocation ---

    async def revoke_token(self, token: AccessToken | RefreshTokenEntry) -> None:
        if isinstance(token, AccessToken):
            self._access_tokens.pop(token.token, None)
            # Also revoke associated refresh tokens for this client
            to_remove = [
                k for k, v in self._refresh_tokens.items()
                if v.client_id == token.client_id
            ]
            for k in to_remove:
                del self._refresh_tokens[k]
        elif isinstance(token, RefreshTokenEntry):
            self._refresh_tokens.pop(token.token, None)
```

**Step 2: Run tests to verify they pass**

Run: `cd /Users/cirwel/projects/anima-mcp && python -m pytest tests/test_oauth_provider.py -v`
Expected: All tests PASS

**Step 3: Commit**

```bash
cd /Users/cirwel/projects/anima-mcp
git add src/anima_mcp/oauth_provider.py
git commit -m "feat: add in-memory OAuth 2.1 provider for Claude.ai"
```

---

### Task 3: Wire OAuth into FastMCP — Tests

**Files:**
- Create: `tests/test_oauth_wiring.py`

**Step 1: Write integration test for OAuth wiring**

This tests that `get_fastmcp()` creates a FastMCP instance with OAuth enabled when env vars are set.

```python
"""Tests for OAuth wiring in tool_registry.py."""
import os
import pytest
from unittest.mock import patch


class TestOAuthWiring:
    def test_fastmcp_created_without_oauth_when_no_env(self):
        """When ANIMA_OAUTH_ISSUER_URL is not set, FastMCP has no auth."""
        # Clear any cached instance
        import anima_mcp.tool_registry as tr
        tr._fastmcp = None
        with patch.dict(os.environ, {}, clear=False):
            # Remove the key if it exists
            os.environ.pop("ANIMA_OAUTH_ISSUER_URL", None)
            mcp = tr.get_fastmcp()
            if mcp:
                assert mcp.settings.auth is None

    def test_fastmcp_created_with_oauth_when_env_set(self):
        """When ANIMA_OAUTH_ISSUER_URL is set, FastMCP has auth configured."""
        import anima_mcp.tool_registry as tr
        tr._fastmcp = None
        env = {
            "ANIMA_OAUTH_ISSUER_URL": "https://lumen-anima.ngrok.io",
            "ANIMA_OAUTH_SECRET": "test-secret",
        }
        with patch.dict(os.environ, env):
            mcp = tr.get_fastmcp()
            if mcp:
                assert mcp.settings.auth is not None
                assert str(mcp.settings.auth.issuer_url) == "https://lumen-anima.ngrok.io/"
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/cirwel/projects/anima-mcp && python -m pytest tests/test_oauth_wiring.py -v`
Expected: FAIL — `test_fastmcp_created_with_oauth_when_env_set` fails because `get_fastmcp()` doesn't configure auth yet

**Step 3: Commit**

```bash
cd /Users/cirwel/projects/anima-mcp
git add tests/test_oauth_wiring.py
git commit -m "test: add OAuth wiring integration tests (red)"
```

---

### Task 4: Wire OAuth into FastMCP — Implementation

**Files:**
- Modify: `src/anima_mcp/tool_registry.py:584-607` (the `get_fastmcp()` function)

**Step 1: Modify `get_fastmcp()` to conditionally enable OAuth**

In `src/anima_mcp/tool_registry.py`, find the `get_fastmcp()` function (line ~584). Replace the `FastMCP(...)` construction with OAuth-aware version:

```python
def get_fastmcp() -> "FastMCP":
    """Get or create the FastMCP server instance."""
    global _fastmcp
    if _fastmcp is None and HAS_FASTMCP:
        # --- OAuth 2.1 configuration (optional, enabled by env var) ---
        oauth_issuer_url = os.environ.get("ANIMA_OAUTH_ISSUER_URL")
        oauth_provider = None
        auth_settings = None

        if oauth_issuer_url:
            from mcp.server.auth.settings import AuthSettings
            from .oauth_provider import AnimaOAuthProvider

            oauth_secret = os.environ.get("ANIMA_OAUTH_SECRET")
            auto_approve = os.environ.get("ANIMA_OAUTH_AUTO_APPROVE", "true").lower() in ("true", "1", "yes")
            oauth_provider = AnimaOAuthProvider(secret=oauth_secret, auto_approve=auto_approve)
            auth_settings = AuthSettings(
                issuer_url=oauth_issuer_url,
                resource_server_url=oauth_issuer_url,
            )
            print(f"[FastMCP] OAuth 2.1 enabled (issuer: {oauth_issuer_url})", file=sys.stderr, flush=True)

        _fastmcp = FastMCP(
            name="anima-mcp",
            host="0.0.0.0",
            auth_server_provider=oauth_provider,
            auth=auth_settings,
            transport_security=TransportSecuritySettings(
                enable_dns_rebinding_protection=True,
                allowed_hosts=[
                    "127.0.0.1:*", "localhost:*", "[::1]:*",
                    "192.168.1.165:*", "192.168.1.151:*",
                    "<PI_TAILSCALE_IP>:*",
                    "lumen-anima.ngrok.io",
                    "0.0.0.0:*",
                ],
                allowed_origins=[
                    "http://127.0.0.1:*", "http://localhost:*", "http://[::1]:*",
                    "http://192.168.1.165:*", "http://192.168.1.151:*",
                    "https://lumen-anima.ngrok.io",
                    "null",
                ],
            ),
        )

        # ... rest of tool registration stays the same ...
```

Note: `import os` should already be available at module level. Verify and add if needed.

**Step 2: Run tests to verify they pass**

Run: `cd /Users/cirwel/projects/anima-mcp && python -m pytest tests/test_oauth_wiring.py tests/test_oauth_provider.py -v`
Expected: All PASS

**Step 3: Commit**

```bash
cd /Users/cirwel/projects/anima-mcp
git add src/anima_mcp/tool_registry.py
git commit -m "feat: wire OAuth provider into FastMCP when ANIMA_OAUTH_ISSUER_URL is set"
```

---

### Task 5: Wire OAuth into Starlette app in server.py — Tests

**Files:**
- Create: `tests/test_oauth_server_integration.py`

**Step 1: Write integration test that exercises the full HTTP OAuth flow**

```python
"""Integration test: OAuth endpoints respond correctly on the HTTP server."""
import os
import pytest
from unittest.mock import patch


class TestOAuthEndpointsExist:
    """Verify that OAuth routes are mounted when configured."""

    def _build_app_with_oauth(self):
        """Helper: build the Starlette app with OAuth enabled."""
        env = {
            "ANIMA_OAUTH_ISSUER_URL": "https://lumen-anima.ngrok.io",
            "ANIMA_OAUTH_SECRET": "test-secret",
            "ANIMA_OAUTH_AUTO_APPROVE": "true",
        }
        with patch.dict(os.environ, env):
            # Reset cached FastMCP to pick up env
            import anima_mcp.tool_registry as tr
            tr._fastmcp = None

            from anima_mcp.oauth_provider import AnimaOAuthProvider
            provider = AnimaOAuthProvider(secret="test-secret", auto_approve=True)
            return provider

    def test_oauth_provider_instantiates(self):
        """Smoke test: provider can be created."""
        provider = self._build_app_with_oauth()
        assert provider is not None

    async def test_full_oauth_flow_unit(self):
        """End-to-end OAuth flow at the provider level."""
        from mcp.server.auth.provider import AuthorizationParams
        from mcp.shared.auth import OAuthClientInformationFull
        from urllib.parse import urlparse, parse_qs
        import time

        provider = self._build_app_with_oauth()

        # 1. Register client
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

        # 2. Authorize
        params = AuthorizationParams(
            state="xyz",
            scopes=["mcp:tools"],
            code_challenge="test_challenge",
            redirect_uri="https://claude.ai/callback",
            redirect_uri_provided_explicitly=True,
        )
        redirect_url = await provider.authorize(client, params)
        code = parse_qs(urlparse(redirect_url).query)["code"][0]

        # 3. Exchange code
        auth_code = await provider.load_authorization_code(client, code)
        token = await provider.exchange_authorization_code(client, auth_code)
        assert token.access_token.startswith("at_")
        assert token.refresh_token.startswith("rt_")

        # 4. Verify token
        access = await provider.load_access_token(token.access_token)
        assert access.client_id == "claude-web-test"

        # 5. Refresh
        rt = await provider.load_refresh_token(client, token.refresh_token)
        new_token = await provider.exchange_refresh_token(client, rt, ["mcp:tools"])
        assert new_token.access_token != token.access_token

        # 6. Revoke
        new_access = await provider.load_access_token(new_token.access_token)
        await provider.revoke_token(new_access)
        assert await provider.load_access_token(new_token.access_token) is None
```

**Step 2: Run tests**

Run: `cd /Users/cirwel/projects/anima-mcp && python -m pytest tests/test_oauth_server_integration.py -v`
Expected: PASS (these test the provider, not server wiring yet)

**Step 3: Commit**

```bash
cd /Users/cirwel/projects/anima-mcp
git add tests/test_oauth_server_integration.py
git commit -m "test: add OAuth full-flow integration test"
```

---

### Task 6: Wire OAuth routes into server.py Starlette app

**Files:**
- Modify: `src/anima_mcp/server.py:2680-2695` (session manager setup)
- Modify: `src/anima_mcp/server.py:3249-3269` (Starlette app construction)

This is the most delicate task. The anima server builds its own Starlette app manually rather than using `FastMCP.streamable_http_app()`. We need to conditionally add OAuth middleware and routes.

**Step 1: Add OAuth route and middleware wiring**

Near the top of `run_http_server()` (around line 2680), after the `StreamableHTTPSessionManager` is created, add the OAuth setup:

```python
        # --- OAuth 2.1 setup (conditional) ---
        _oauth_issuer_url = os.environ.get("ANIMA_OAUTH_ISSUER_URL")
        _oauth_auth_routes = []
        _oauth_middleware = []
        _oauth_token_verifier = None

        if _oauth_issuer_url and hasattr(mcp, '_auth_server_provider') and mcp._auth_server_provider:
            try:
                from mcp.server.auth.routes import create_auth_routes, create_protected_resource_routes
                from mcp.server.auth.provider import ProviderTokenVerifier
                from mcp.server.auth.middleware.bearer_auth import BearerAuthBackend, RequireAuthMiddleware
                from mcp.server.auth.middleware.auth_context import AuthContextMiddleware
                from starlette.middleware import Middleware
                from starlette.middleware.authentication import AuthenticationMiddleware

                _oauth_token_verifier = ProviderTokenVerifier(mcp._auth_server_provider)

                # Auth routes: /.well-known/oauth-authorization-server, /authorize, /token, /register, /revoke
                _oauth_auth_routes = create_auth_routes(
                    provider=mcp._auth_server_provider,
                    issuer_url=mcp.settings.auth.issuer_url,
                )

                # Protected resource metadata: /.well-known/oauth-protected-resource
                _oauth_auth_routes.extend(
                    create_protected_resource_routes(
                        resource_url=mcp.settings.auth.resource_server_url,
                        authorization_servers=[mcp.settings.auth.issuer_url],
                        scopes_supported=mcp.settings.auth.required_scopes,
                    )
                )

                # Middleware stack for bearer token validation
                _oauth_middleware = [
                    Middleware(AuthenticationMiddleware, backend=BearerAuthBackend(_oauth_token_verifier)),
                    Middleware(AuthContextMiddleware),
                ]

                print(f"[Server] OAuth 2.1 routes enabled ({len(_oauth_auth_routes)} routes)", file=sys.stderr, flush=True)
            except Exception as e:
                print(f"[Server] OAuth setup failed, continuing without auth: {e}", file=sys.stderr, flush=True)
                _oauth_auth_routes = []
                _oauth_middleware = []
                _oauth_token_verifier = None
```

Then modify the Starlette app construction (~line 3249). Replace:

```python
        app = Starlette(routes=[
            Mount("/mcp", app=streamable_mcp_asgi),
            ...
        ])
```

With:

```python
        # === Build Starlette app with all routes ===
        # Wrap /mcp with OAuth if configured
        if _oauth_token_verifier:
            from mcp.server.auth.middleware.bearer_auth import RequireAuthMiddleware
            from mcp.server.auth.routes import build_resource_metadata_url
            resource_metadata_url = build_resource_metadata_url(mcp.settings.auth.resource_server_url)
            mcp_endpoint = RequireAuthMiddleware(
                streamable_mcp_asgi,
                required_scopes=mcp.settings.auth.required_scopes or [],
                resource_metadata_url=resource_metadata_url,
            )
        else:
            mcp_endpoint = streamable_mcp_asgi

        all_routes = [
            *_oauth_auth_routes,  # OAuth routes first (/.well-known/*, /authorize, /token, etc.)
            Mount("/mcp", app=mcp_endpoint),
            Route("/health", health_check, methods=["GET"]),
            Route("/health/detailed", rest_health_detailed, methods=["GET"]),
            Route("/v1/tools/call", rest_tool_call, methods=["POST"]),
            Route("/dashboard", dashboard, methods=["GET"]),
            Route("/state", rest_state, methods=["GET"]),
            Route("/qa", rest_qa, methods=["GET"]),
            Route("/answer", rest_answer, methods=["POST"]),
            Route("/message", rest_message, methods=["POST"]),
            Route("/messages", rest_messages, methods=["GET"]),
            Route("/learning", rest_learning, methods=["GET"]),
            Route("/voice", rest_voice, methods=["GET"]),
            Route("/gallery", rest_gallery, methods=["GET"]),
            Route("/gallery/{filename}", rest_gallery_image, methods=["GET"]),
            Route("/gallery-page", rest_gallery_page, methods=["GET"]),
            Route("/layers", rest_layers, methods=["GET"]),
            Route("/architecture", rest_architecture_page, methods=["GET"]),
        ]
        app = Starlette(routes=all_routes, middleware=_oauth_middleware)
        print("[Server] Starlette app created with all routes", file=sys.stderr, flush=True)
```

**Step 2: Run all tests**

Run: `cd /Users/cirwel/projects/anima-mcp && python -m pytest tests/test_oauth_provider.py tests/test_oauth_wiring.py tests/test_oauth_server_integration.py -v`
Expected: All PASS

**Step 3: Commit**

```bash
cd /Users/cirwel/projects/anima-mcp
git add src/anima_mcp/server.py
git commit -m "feat: wire OAuth routes and middleware into HTTP server"
```

---

### Task 7: Manual smoke test with local server

**Step 1: Start the server with OAuth enabled**

```bash
cd /Users/cirwel/projects/anima-mcp
ANIMA_OAUTH_ISSUER_URL=http://localhost:8766 ANIMA_OAUTH_SECRET=test python -m anima_mcp.server --http --port 8766
```

**Step 2: Verify OAuth metadata endpoints**

```bash
# Protected resource metadata
curl -s http://localhost:8766/.well-known/oauth-protected-resource | python3 -m json.tool

# Authorization server metadata
curl -s http://localhost:8766/.well-known/oauth-authorization-server | python3 -m json.tool
```

Expected: Both return valid JSON with correct URLs.

**Step 3: Verify dashboard still works (no auth)**

```bash
curl -s -o /dev/null -w "%{http_code}" http://localhost:8766/dashboard
curl -s -o /dev/null -w "%{http_code}" http://localhost:8766/health
curl -s -o /dev/null -w "%{http_code}" http://localhost:8766/state
```

Expected: All return `200`

**Step 4: Verify /mcp requires auth**

```bash
curl -s -o /dev/null -w "%{http_code}" -X POST http://localhost:8766/mcp/
```

Expected: `401` (unauthorized — OAuth Bearer token required)

**Step 5: Stop the server and commit results**

```bash
cd /Users/cirwel/projects/anima-mcp
git add -A
git commit -m "chore: verify OAuth smoke test passes"
```

---

### Task 8: Deploy to Pi and test via ngrok

**Step 1: Add env vars to Pi's systemd service**

SSH to Pi and add to the anima systemd service file (or `.env`):

```bash
ANIMA_OAUTH_ISSUER_URL=https://lumen-anima.ngrok.io
ANIMA_OAUTH_SECRET=<generate-a-real-secret>
ANIMA_OAUTH_AUTO_APPROVE=true
```

**Step 2: Deploy using the deploy-to-pi skill**

Use the `deploy-to-pi` skill to push the code and restart the service.

**Step 3: Verify via ngrok**

```bash
curl -s https://lumen-anima.ngrok.io/.well-known/oauth-protected-resource | python3 -m json.tool
curl -s https://lumen-anima.ngrok.io/.well-known/oauth-authorization-server | python3 -m json.tool
curl -s https://lumen-anima.ngrok.io/dashboard  # should still work
```

**Step 4: Connect Claude.ai**

1. Go to Claude.ai Settings → Integrations → Add Custom Integration
2. Enter URL: `https://lumen-anima.ngrok.io`
3. Claude.ai should discover OAuth metadata and initiate the flow
4. Auto-approve grants access
5. Anima MCP tools should appear in Claude.ai

**Step 5: Commit any deployment config changes**

```bash
cd /Users/cirwel/projects/anima-mcp
git add -A
git commit -m "deploy: enable OAuth 2.1 for Claude.ai integration"
```

---

### Task 9: Run full test suite — verify no regressions

**Step 1: Run all tests**

```bash
cd /Users/cirwel/projects/anima-mcp && python -m pytest tests/ -v --timeout=30
```

Expected: All existing tests pass, plus the 3 new test files.

**Step 2: Final commit if needed**

```bash
cd /Users/cirwel/projects/anima-mcp
git add -A
git commit -m "test: verify full test suite passes with OAuth integration"
```
