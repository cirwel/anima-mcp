"""Tests for AnimaOAuthProvider SQLite persistence.

OAuth state (registered clients, access tokens, refresh tokens) must survive
server restart so claude.ai does not re-authenticate on every Pi reboot /
deploy / WiFi recovery cycle.
"""
import time
from pathlib import Path

import pytest

from anima_mcp.oauth_provider import AnimaOAuthProvider
from mcp.server.auth.provider import AuthorizationParams
from mcp.shared.auth import OAuthClientInformationFull


def _client(client_id: str = "cid-1") -> OAuthClientInformationFull:
    return OAuthClientInformationFull(
        client_id=client_id,
        client_secret="csecret",
        client_id_issued_at=int(time.time()),
        redirect_uris=["https://example.com/cb"],
        client_name="Test",
        grant_types=["authorization_code", "refresh_token"],
        response_types=["code"],
        token_endpoint_auth_method="client_secret_post",
    )


async def _complete_auth(provider: AnimaOAuthProvider, client: OAuthClientInformationFull):
    from urllib.parse import urlparse, parse_qs
    params = AuthorizationParams(
        state="s", scopes=["mcp:tools"], code_challenge="ch",
        redirect_uri="https://example.com/cb",
        redirect_uri_provided_explicitly=True,
    )
    redirect = await provider.authorize(client, params)
    code = parse_qs(urlparse(redirect).query)["code"][0]
    auth_code = await provider.load_authorization_code(client, code)
    return await provider.exchange_authorization_code(client, auth_code)


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "oauth.db"


class TestClientPersistence:
    async def test_registered_client_survives_restart(self, db_path):
        p1 = AnimaOAuthProvider(secret="s", db_path=db_path)
        client = _client()
        await p1.register_client(client)

        p2 = AnimaOAuthProvider(secret="s", db_path=db_path)
        loaded = await p2.get_client("cid-1")
        assert loaded is not None
        assert loaded.client_id == "cid-1"
        assert loaded.client_name == "Test"
        assert "https://example.com/cb" in [str(u) for u in loaded.redirect_uris]

    async def test_unknown_client_after_restart_returns_none(self, db_path):
        AnimaOAuthProvider(secret="s", db_path=db_path)
        p2 = AnimaOAuthProvider(secret="s", db_path=db_path)
        assert await p2.get_client("never-registered") is None


class TestTokenPersistence:
    async def test_access_token_valid_after_restart(self, db_path):
        p1 = AnimaOAuthProvider(secret="s", db_path=db_path)
        client = _client()
        await p1.register_client(client)
        token = await _complete_auth(p1, client)

        p2 = AnimaOAuthProvider(secret="s", db_path=db_path)
        loaded = await p2.load_access_token(token.access_token)
        assert loaded is not None
        assert loaded.client_id == "cid-1"
        assert "mcp:tools" in loaded.scopes

    async def test_refresh_token_valid_after_restart(self, db_path):
        p1 = AnimaOAuthProvider(secret="s", db_path=db_path)
        client = _client()
        await p1.register_client(client)
        token = await _complete_auth(p1, client)

        p2 = AnimaOAuthProvider(secret="s", db_path=db_path)
        rt = await p2.load_refresh_token(client, token.refresh_token)
        assert rt is not None
        new_token = await p2.exchange_refresh_token(client, rt, ["mcp:tools"])
        assert new_token.access_token
        assert new_token.access_token != token.access_token

    async def test_expired_access_token_not_returned_after_restart(self, db_path):
        p1 = AnimaOAuthProvider(secret="s", db_path=db_path, access_token_ttl=1)
        client = _client()
        await p1.register_client(client)
        token = await _complete_auth(p1, client)

        time.sleep(2.1)
        p2 = AnimaOAuthProvider(secret="s", db_path=db_path)
        assert await p2.load_access_token(token.access_token) is None


class TestRevocationPersistence:
    async def test_revoked_access_token_stays_revoked_after_restart(self, db_path):
        p1 = AnimaOAuthProvider(secret="s", db_path=db_path)
        client = _client()
        await p1.register_client(client)
        token = await _complete_auth(p1, client)
        access = await p1.load_access_token(token.access_token)
        await p1.revoke_token(access)

        p2 = AnimaOAuthProvider(secret="s", db_path=db_path)
        assert await p2.load_access_token(token.access_token) is None


class TestInMemoryBackcompat:
    async def test_no_db_path_behaves_as_before(self):
        p = AnimaOAuthProvider(secret="s")
        client = _client()
        await p.register_client(client)
        assert await p.get_client("cid-1") is not None
