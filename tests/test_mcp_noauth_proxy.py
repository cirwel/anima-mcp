"""Tests for the local no-auth anima-mcp reverse proxy script."""
from __future__ import annotations

import http.client
import importlib.util
import json
import socket
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import ModuleType


def load_proxy_module() -> ModuleType:
    """Load the script module without requiring it to be package-installed."""
    script = Path(__file__).resolve().parents[1] / "scripts" / "mcp_noauth_proxy.py"
    spec = importlib.util.spec_from_file_location("mcp_noauth_proxy", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ProbeUpstream(BaseHTTPRequestHandler):
    """Tiny upstream server used by proxy behavior tests."""

    seen: dict[str, object] = {}

    def log_message(self, *_args: object) -> None:
        """Suppress test server access logs."""

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        """Serve either a streaming SSE-like body or a simple text body."""
        if self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            first = b"data: first\n\n"
            self.wfile.write(f"{len(first):x}\r\n".encode() + first + b"\r\n")
            self.wfile.flush()
            time.sleep(0.75)
            second = b"data: second\n\n"
            self.wfile.write(
                f"{len(second):x}\r\n".encode() + second + b"\r\n0\r\n\r\n"
            )
            self.wfile.flush()
            return

        body = b"upstream-ok"
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_DELETE(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        """Record DELETE proxying for MCP session termination."""
        self.seen["delete_path"] = self.path
        self.send_response(204)
        self.end_headers()

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        """Record forwarded request headers/body."""
        length = int(self.headers.get("Content-Length", "0"))
        self.seen["accept"] = self.headers.get("Accept")
        self.seen["body"] = self.rfile.read(length).decode()
        self.seen["hop_value"] = self.headers.get("X-Hop-Test")
        body = json.dumps({"jsonrpc": "2.0", "id": 1, "result": {"ok": True}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def start_proxy_pair() -> tuple[ModuleType, ThreadingHTTPServer, ThreadingHTTPServer]:
    """Start an upstream test server and a proxy server on ephemeral ports."""
    module = load_proxy_module()
    ProbeUpstream.seen = {}
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), ProbeUpstream)
    module.NoAuthProxyHandler.upstream_host = "127.0.0.1"
    module.NoAuthProxyHandler.upstream_port = upstream.server_address[1]
    proxy = ThreadingHTTPServer(("127.0.0.1", 0), module.NoAuthProxyHandler)
    for server in (upstream, proxy):
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
    return module, upstream, proxy


def stop_servers(*servers: ThreadingHTTPServer) -> None:
    """Stop test HTTP servers."""
    for server in servers:
        server.shutdown()
        server.server_close()


def test_loopback_validation_and_oauth_discovery_matching() -> None:
    """The no-auth proxy must stay local and only suppress discovery routes."""
    module = load_proxy_module()

    assert module.is_loopback_host("127.0.0.1")
    assert module.is_loopback_host("localhost")
    assert module.is_loopback_host("::1")
    assert not module.is_loopback_host("0.0.0.0")
    assert not module.is_loopback_host("192.168.1.10")

    is_probe = module.NoAuthProxyHandler._is_oauth_discovery_probe
    assert is_probe("/.well-known/oauth-protected-resource")
    assert is_probe("/.well-known/oauth-protected-resource/mcp")
    assert not is_probe("/x/.well-known/oauth-protected-resource")


def test_proxy_core_http_behaviors() -> None:
    """Proxy MCP-relevant methods/headers and suppress OAuth discovery."""
    _module, upstream, proxy = start_proxy_pair()
    base = f"http://127.0.0.1:{proxy.server_address[1]}"
    try:
        with urllib.request.urlopen(f"{base}/health", timeout=5) as response:
            assert response.status == 200
            assert response.read() == b"upstream-ok"

        try:
            urllib.request.urlopen(
                f"{base}/.well-known/oauth-protected-resource",
                timeout=5,
            )
        except urllib.error.HTTPError as exc:
            assert exc.code == 404
        else:  # pragma: no cover - assertion branch
            raise AssertionError("OAuth discovery probe should return 404")

        request = urllib.request.Request(f"{base}/mcp/session?sid=1", method="DELETE")
        with urllib.request.urlopen(request, timeout=5) as response:
            assert response.status == 204
        assert ProbeUpstream.seen["delete_path"] == "/mcp/session?sid=1"

        connection = http.client.HTTPConnection("127.0.0.1", proxy.server_address[1], timeout=5)
        connection.request(
            "POST",
            "/mcp/",
            body=b'{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}',
            headers={
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
                "Connection": "X-Hop-Test",
                "X-Hop-Test": "remove-me",
            },
        )
        response = connection.getresponse()
        payload = json.loads(response.read())
        connection.close()

        assert response.status == 200
        assert payload["result"]["ok"] is True
        assert "application/json" in str(ProbeUpstream.seen["accept"])
        assert "initialize" in str(ProbeUpstream.seen["body"])
        assert ProbeUpstream.seen["hop_value"] is None
    finally:
        stop_servers(proxy, upstream)


def test_streamed_upstream_chunks_are_forwarded_promptly() -> None:
    """A streaming upstream chunk should not be buffered until stream close."""
    _module, upstream, proxy = start_proxy_pair()
    try:
        raw = socket.create_connection(("127.0.0.1", proxy.server_address[1]), timeout=5)
        raw.settimeout(0.35)
        raw.sendall(
            b"GET /stream HTTP/1.1\r\n"
            b"Host: 127.0.0.1\r\n"
            b"Connection: close\r\n\r\n"
        )
        received = b""
        deadline = time.monotonic() + 0.35
        while time.monotonic() < deadline and b"data: first\n\n" not in received:
            try:
                received += raw.recv(4096)
            except socket.timeout:
                pass
        raw.close()
        assert b"data: first\n\n" in received
    finally:
        stop_servers(proxy, upstream)


def test_chunked_request_bodies_are_rejected() -> None:
    """Do not silently drop unsupported chunked request bodies."""
    _module, upstream, proxy = start_proxy_pair()
    try:
        connection = http.client.HTTPConnection("127.0.0.1", proxy.server_address[1], timeout=5)
        connection.putrequest("POST", "/mcp/")
        connection.putheader("Transfer-Encoding", "chunked")
        connection.endheaders()
        response = connection.getresponse()
        body = response.read().decode()
        connection.close()

        assert response.status == 501
        assert "unsupported request Transfer-Encoding" in body
    finally:
        stop_servers(proxy, upstream)
