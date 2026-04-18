"""
OAuth 2.1 Authorization Server Provider for anima-mcp.

Implements the MCP SDK's OAuthAuthorizationServerProvider protocol. When a
`db_path` is provided, registered clients, access tokens, and refresh tokens
persist to SQLite so claude.ai does not re-authenticate after every server
restart. Without `db_path`, state is in-memory only (original behavior).

Auth codes are short-lived (5 min default) and stay in-memory — they do not
need to survive restart.
"""
from __future__ import annotations

import json
import secrets
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlencode

from mcp.server.auth.provider import (
    AccessToken,
    AuthorizationParams,
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
    redirect_uri_provided_explicitly: bool = True
    created_at: float = field(default_factory=time.time)
    expires_at: float = 0.0
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

    def is_expired(self, ttl: int = 604800) -> bool:
        return time.time() - self.created_at > ttl


_SCHEMA = """
CREATE TABLE IF NOT EXISTS oauth_clients (
    client_id TEXT PRIMARY KEY,
    data TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS oauth_access_tokens (
    token TEXT PRIMARY KEY,
    client_id TEXT NOT NULL,
    scopes TEXT NOT NULL,
    expires_at INTEGER NOT NULL,
    resource TEXT
);
CREATE TABLE IF NOT EXISTS oauth_refresh_tokens (
    token TEXT PRIMARY KEY,
    client_id TEXT NOT NULL,
    scopes TEXT NOT NULL,
    created_at REAL NOT NULL
);
"""


class AnimaOAuthProvider:
    """
    OAuth 2.1 Authorization Server for anima-mcp.

    Single-user, personal server. State optionally persisted to SQLite so
    registered clients and tokens survive process restart.
    """

    def __init__(
        self,
        secret: str | None = None,
        auto_approve: bool = True,
        access_token_ttl: int = 3600,
        refresh_token_ttl: int = 604800,
        auth_code_ttl: int = 300,
        db_path: str | Path | None = None,
    ):
        self._secret = secret or secrets.token_hex(32)
        self._auto_approve = auto_approve
        self._access_token_ttl = access_token_ttl
        self._refresh_token_ttl = refresh_token_ttl
        self._auth_code_ttl = auth_code_ttl

        self._db_path: Path | None = Path(db_path) if db_path else None
        self._clients: dict[str, OAuthClientInformationFull] = {}
        self._auth_codes: dict[str, AuthCodeEntry] = {}
        self._access_tokens: dict[str, AccessToken] = {}
        self._refresh_tokens: dict[str, RefreshTokenEntry] = {}

        if self._db_path is not None:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._init_db()
            self._load_from_db()

    # --- SQLite helpers ---

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _load_from_db(self) -> None:
        now = int(time.time())
        with self._connect() as conn:
            for row in conn.execute("SELECT client_id, data FROM oauth_clients"):
                try:
                    self._clients[row["client_id"]] = OAuthClientInformationFull.model_validate_json(row["data"])
                except Exception:
                    continue

            for row in conn.execute(
                "SELECT token, client_id, scopes, expires_at, resource FROM oauth_access_tokens"
            ):
                if row["expires_at"] and row["expires_at"] < now:
                    continue
                self._access_tokens[row["token"]] = AccessToken(
                    token=row["token"],
                    client_id=row["client_id"],
                    scopes=json.loads(row["scopes"]),
                    expires_at=row["expires_at"],
                    resource=row["resource"],
                )

            refresh_cutoff = time.time() - self._refresh_token_ttl
            for row in conn.execute(
                "SELECT token, client_id, scopes, created_at FROM oauth_refresh_tokens"
            ):
                if row["created_at"] < refresh_cutoff:
                    continue
                self._refresh_tokens[row["token"]] = RefreshTokenEntry(
                    token=row["token"],
                    client_id=row["client_id"],
                    scopes=json.loads(row["scopes"]),
                    created_at=row["created_at"],
                )

    def _persist_client(self, client: OAuthClientInformationFull) -> None:
        if self._db_path is None:
            return
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO oauth_clients (client_id, data) VALUES (?, ?)",
                (client.client_id, client.model_dump_json()),
            )

    def _persist_access_token(self, token: AccessToken) -> None:
        if self._db_path is None:
            return
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO oauth_access_tokens "
                "(token, client_id, scopes, expires_at, resource) VALUES (?, ?, ?, ?, ?)",
                (
                    token.token,
                    token.client_id,
                    json.dumps(list(token.scopes)),
                    int(token.expires_at or 0),
                    token.resource,
                ),
            )

    def _persist_refresh_token(self, entry: RefreshTokenEntry) -> None:
        if self._db_path is None:
            return
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO oauth_refresh_tokens "
                "(token, client_id, scopes, created_at) VALUES (?, ?, ?, ?)",
                (
                    entry.token,
                    entry.client_id,
                    json.dumps(list(entry.scopes)),
                    entry.created_at,
                ),
            )

    def _delete_access_token(self, token: str) -> None:
        if self._db_path is None:
            return
        with self._connect() as conn:
            conn.execute("DELETE FROM oauth_access_tokens WHERE token = ?", (token,))

    def _delete_refresh_token(self, token: str) -> None:
        if self._db_path is None:
            return
        with self._connect() as conn:
            conn.execute("DELETE FROM oauth_refresh_tokens WHERE token = ?", (token,))

    def _delete_refresh_tokens_for_client(self, client_id: str) -> None:
        if self._db_path is None:
            return
        with self._connect() as conn:
            conn.execute("DELETE FROM oauth_refresh_tokens WHERE client_id = ?", (client_id,))

    # --- Token generation ---

    def _generate_token(self, prefix: str = "at") -> str:
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
        self._persist_client(client_info)

    # --- Authorization ---

    async def authorize(
        self, client: OAuthClientInformationFull, params: AuthorizationParams
    ) -> str:
        code = secrets.token_hex(24)

        entry = AuthCodeEntry(
            code=code,
            client_id=client.client_id,
            redirect_uri=str(params.redirect_uri),
            code_challenge=params.code_challenge,
            scopes=params.scopes or [],
            redirect_uri_provided_explicitly=params.redirect_uri_provided_explicitly,
            expires_at=time.time() + self._auth_code_ttl,
            resource=params.resource,
        )
        self._auth_codes[code] = entry

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
        self._auth_codes.pop(authorization_code.code, None)

        access_token_str = self._generate_token("at")
        refresh_token_str = self._generate_token("rt")
        expires_at = int(time.time()) + self._access_token_ttl
        scopes = authorization_code.scopes or ["mcp:tools"]

        access = AccessToken(
            token=access_token_str,
            client_id=client.client_id,
            scopes=scopes,
            expires_at=expires_at,
            resource=authorization_code.resource,
        )
        refresh = RefreshTokenEntry(
            token=refresh_token_str,
            client_id=client.client_id,
            scopes=scopes,
        )

        self._access_tokens[access_token_str] = access
        self._refresh_tokens[refresh_token_str] = refresh
        self._persist_access_token(access)
        self._persist_refresh_token(refresh)

        return OAuthToken(
            access_token=access_token_str,
            token_type="Bearer",
            expires_in=self._access_token_ttl,
            scope=" ".join(scopes),
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
            self._delete_refresh_token(refresh_token)
            return None
        return entry

    async def exchange_refresh_token(
        self,
        client: OAuthClientInformationFull,
        refresh_token: RefreshTokenEntry,
        scopes: list[str],
    ) -> OAuthToken:
        self._refresh_tokens.pop(refresh_token.token, None)
        self._delete_refresh_token(refresh_token.token)

        access_token_str = self._generate_token("at")
        new_refresh_str = self._generate_token("rt")
        expires_at = int(time.time()) + self._access_token_ttl
        effective_scopes = scopes or refresh_token.scopes or ["mcp:tools"]

        access = AccessToken(
            token=access_token_str,
            client_id=client.client_id,
            scopes=effective_scopes,
            expires_at=expires_at,
        )
        refresh = RefreshTokenEntry(
            token=new_refresh_str,
            client_id=client.client_id,
            scopes=effective_scopes,
        )

        self._access_tokens[access_token_str] = access
        self._refresh_tokens[new_refresh_str] = refresh
        self._persist_access_token(access)
        self._persist_refresh_token(refresh)

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
            self._delete_access_token(token)
            return None
        return entry

    # --- Revocation ---

    async def revoke_token(self, token: AccessToken | RefreshTokenEntry) -> None:
        if isinstance(token, AccessToken):
            self._access_tokens.pop(token.token, None)
            self._delete_access_token(token.token)
            to_remove = [
                k for k, v in self._refresh_tokens.items()
                if v.client_id == token.client_id
            ]
            for k in to_remove:
                del self._refresh_tokens[k]
            self._delete_refresh_tokens_for_client(token.client_id)
        elif isinstance(token, RefreshTokenEntry):
            self._refresh_tokens.pop(token.token, None)
            self._delete_refresh_token(token.token)
