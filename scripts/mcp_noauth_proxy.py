#!/usr/bin/env python3
"""Local no-auth reverse proxy for anima-mcp.

The upstream Lumen/Anima HTTP server advertises OAuth discovery metadata for
public/tunnel clients, while local tools such as Hermes should use the MCP route
without OAuth. This proxy gives local clients a stable endpoint that suppresses
OAuth discovery probes and forwards real traffic to the upstream server.
"""
from __future__ import annotations

import argparse
import http.client
import ipaddress
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Iterable
from urllib.parse import urlsplit

HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}

OAUTH_DISCOVERY_MARKERS = (
    "/.well-known/oauth-authorization-server",
    "/.well-known/oauth-protected-resource",
    "/.well-known/openid-configuration",
)

LOCALHOST_NAMES = {"localhost"}
STREAM_CHUNK_BYTES = 64 * 1024
DEFAULT_MAX_REQUEST_BODY_BYTES = 16 * 1024 * 1024


class ProxyRequestError(Exception):
    """Client request error that can be rendered before proxying upstream."""

    def __init__(self, status_code: int, message: str) -> None:
        """Store an HTTP status code and safe response message."""
        super().__init__(message)
        self.status_code = status_code
        self.message = message


class NoAuthProxyHandler(BaseHTTPRequestHandler):
    """HTTP handler that suppresses OAuth discovery and proxies everything else."""

    upstream_host: str = "127.0.0.1"
    upstream_port: int = 8766
    max_request_body_bytes: int = DEFAULT_MAX_REQUEST_BODY_BYTES
    _response_started: bool = False

    def log_message(self, fmt: str, *args: object) -> None:
        """Write compact access logs with a stable prefix."""
        print(
            f"[noauth-proxy] {self.address_string()} {fmt % args}",
            file=sys.stdout,
            flush=True,
        )

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        """Proxy GET requests unless they are OAuth discovery probes."""
        self._handle_request()

    def do_HEAD(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        """Proxy HEAD requests unless they are OAuth discovery probes."""
        self._handle_request(body_allowed=False)

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        """Proxy POST requests, including Streamable HTTP MCP JSON-RPC calls."""
        self._handle_request()

    def do_PUT(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        """Proxy PUT requests."""
        self._handle_request()

    def do_PATCH(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        """Proxy PATCH requests."""
        self._handle_request()

    def do_DELETE(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        """Proxy DELETE requests, including MCP session termination."""
        self._handle_request()

    def do_OPTIONS(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        """Proxy OPTIONS requests for clients that probe capabilities/CORS."""
        self._handle_request()

    def _handle_request(self, body_allowed: bool = True) -> None:
        """Return 404 for OAuth discovery; otherwise forward to upstream."""
        self._response_started = False
        parsed = urlsplit(self.path)
        if self._is_oauth_discovery_probe(parsed.path):
            self._send_text_response(404, "Not Found\n")
            return

        try:
            self._proxy_to_upstream(body_allowed=body_allowed)
        except ProxyRequestError as exc:
            if not self._response_started:
                self._send_text_response(exc.status_code, f"{exc.message}\n")
        except Exception as exc:  # pragma: no cover - defensive network boundary
            if not self._response_started:
                self._send_text_response(502, "upstream proxy error\n")
            print(f"[noauth-proxy] upstream error: {exc}", file=sys.stderr, flush=True)

    @staticmethod
    def _is_oauth_discovery_probe(path: str) -> bool:
        """Return True when a path is an OAuth/OIDC discovery request."""
        return any(
            path == marker or path.startswith(f"{marker}/")
            for marker in OAUTH_DISCOVERY_MARKERS
        )

    def _send_text_response(self, status_code: int, text: str) -> None:
        """Return a small close-delimited text response."""
        body = text.encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self._response_started = True
        if self.command != "HEAD":
            self.wfile.write(body)

    def _request_body(self) -> bytes:
        """Read and bound the incoming request body, if any."""
        transfer_encoding = self.headers.get("Transfer-Encoding")
        if transfer_encoding and transfer_encoding.strip().lower() != "identity":
            raise ProxyRequestError(501, "unsupported request Transfer-Encoding")

        length_values = self.headers.get_all("Content-Length", [])
        if len(length_values) > 1:
            raise ProxyRequestError(400, "duplicate Content-Length")
        if not length_values:
            return b""

        length = length_values[0]
        try:
            body_bytes = int(length)
        except ValueError as exc:
            raise ProxyRequestError(400, "invalid Content-Length") from exc
        if body_bytes < 0:
            raise ProxyRequestError(400, "invalid Content-Length")
        if body_bytes > self.max_request_body_bytes:
            raise ProxyRequestError(413, "request body too large")
        return self.rfile.read(body_bytes)

    @staticmethod
    def _connection_header_tokens(headers: Iterable[tuple[str, str]]) -> set[str]:
        """Extract header names nominated by Connection headers."""
        tokens: set[str] = set()
        for key, value in headers:
            if key.lower() != "connection":
                continue
            tokens.update(token.strip().lower() for token in value.split(",") if token.strip())
        return tokens

    def _forward_headers(self) -> dict[str, str]:
        """Build upstream headers, stripping hop-by-hop fields."""
        header_items = list(self.headers.items())
        connection_tokens = self._connection_header_tokens(header_items)
        headers: dict[str, str] = {}
        for key, value in header_items:
            lower = key.lower()
            if lower in HOP_BY_HOP_HEADERS or lower in connection_tokens:
                continue
            if lower == "host":
                headers["Host"] = f"{self.upstream_host}:{self.upstream_port}"
                continue
            if lower == "accept":
                headers[key] = self._broaden_accept_header(value)
                continue
            headers[key] = value
        headers.setdefault("Host", f"{self.upstream_host}:{self.upstream_port}")
        headers["Connection"] = "close"
        return headers

    @staticmethod
    def _broaden_accept_header(value: str) -> str:
        """Ensure JSON Streamable HTTP responses are acceptable to strict clients."""
        lowered = value.lower()
        if "text/event-stream" in lowered and "application/json" not in lowered:
            return f"{value}, application/json"
        return value

    def _response_headers(
        self,
        upstream_headers: Iterable[tuple[str, str]],
        body_allowed: bool,
    ) -> list[tuple[str, str]]:
        """Filter upstream response headers for a close-delimited local reply."""
        header_items = list(upstream_headers)
        connection_tokens = self._connection_header_tokens(header_items)
        filtered: list[tuple[str, str]] = []
        for key, value in header_items:
            lower = key.lower()
            if lower in HOP_BY_HOP_HEADERS or lower in connection_tokens:
                continue
            if lower == "content-length" and not body_allowed:
                continue
            filtered.append((key, value))
        return filtered

    def _proxy_to_upstream(self, body_allowed: bool) -> None:
        """Forward the request to the configured upstream and stream its response."""
        body = self._request_body()
        parsed = urlsplit(self.path)
        upstream_path = parsed.path or "/"
        if parsed.query:
            upstream_path = f"{upstream_path}?{parsed.query}"

        connection = http.client.HTTPConnection(
            self.upstream_host,
            self.upstream_port,
            timeout=30,
        )
        try:
            connection.request(
                method=self.command,
                url=upstream_path,
                body=body if body else None,
                headers=self._forward_headers(),
            )
            response = connection.getresponse()

            self.send_response(response.status, response.reason)
            for key, value in self._response_headers(response.getheaders(), body_allowed):
                self.send_header(key, value)
            self.send_header("Connection", "close")
            self.end_headers()
            self._response_started = True

            if body_allowed and self.command != "HEAD":
                self._stream_response_body(response)
        finally:
            connection.close()

    def _stream_response_body(self, response: http.client.HTTPResponse) -> None:
        """Stream the upstream response body to the downstream client."""
        read_chunk = getattr(response, "read1", response.read)
        while True:
            chunk = read_chunk(STREAM_CHUNK_BYTES)
            if not chunk:
                break
            self.wfile.write(chunk)
            self.wfile.flush()


def is_loopback_host(host: str) -> bool:
    """Return True if the listen host is explicitly loopback/local-only."""
    normalized = host.strip().lower()
    if normalized in LOCALHOST_NAMES:
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(description="Local no-auth proxy for anima-mcp")
    parser.add_argument("--listen-host", default="127.0.0.1")
    parser.add_argument("--listen-port", type=int, default=8769)
    parser.add_argument("--upstream-host", default="127.0.0.1")
    parser.add_argument("--upstream-port", type=int, default=8766)
    parser.add_argument(
        "--allow-non-loopback-upstream",
        action="store_true",
        help="allow proxying to a non-loopback upstream host",
    )
    parser.add_argument(
        "--max-request-body-bytes",
        type=int,
        default=DEFAULT_MAX_REQUEST_BODY_BYTES,
    )
    return parser


def main() -> None:
    """Run the proxy server until interrupted."""
    args = build_parser().parse_args()
    if not is_loopback_host(args.listen_host):
        raise SystemExit(
            "Refusing non-loopback listen host for no-auth proxy: "
            f"{args.listen_host!r}. Use 127.0.0.1 or localhost."
        )
    if not args.allow_non_loopback_upstream and not is_loopback_host(args.upstream_host):
        raise SystemExit(
            "Refusing non-loopback upstream host for no-auth proxy: "
            f"{args.upstream_host!r}. Use --allow-non-loopback-upstream to override."
        )
    if args.max_request_body_bytes < 0:
        raise SystemExit("--max-request-body-bytes must be non-negative")

    NoAuthProxyHandler.upstream_host = args.upstream_host
    NoAuthProxyHandler.upstream_port = args.upstream_port
    NoAuthProxyHandler.max_request_body_bytes = args.max_request_body_bytes

    server = ThreadingHTTPServer((args.listen_host, args.listen_port), NoAuthProxyHandler)
    print(
        "[noauth-proxy] forwarding "
        f"http://{args.listen_host}:{args.listen_port} -> "
        f"http://{args.upstream_host}:{args.upstream_port}",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
