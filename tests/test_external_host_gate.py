"""The public-hostname check must fold case the same way on every gate.

``_EXTERNAL_HOSTS`` is stored lowercased. The MCP mount used to compare the raw
Host header against it, so a case variant of the public hostname skipped the
OAuth gate while the REST gate (which lowercased) still caught it.
"""

import inspect

import pytest

from anima_mcp import rest_api, server


@pytest.fixture
def external_hosts(monkeypatch):
    monkeypatch.setattr(rest_api, "_EXTERNAL_HOSTS", {"lumen.cirwel.org"})


@pytest.mark.parametrize(
    "raw_host",
    [
        "lumen.cirwel.org",
        "LUMEN.cirwel.org",
        "Lumen.Cirwel.Org",
        "LUMEN.CIRWEL.ORG:443",
        " lumen.cirwel.org ",
    ],
)
def test_public_hostname_is_external_in_any_case(external_hosts, raw_host):
    assert rest_api._host_is_external(raw_host)


@pytest.mark.parametrize("raw_host", ["", "localhost:8766", "192.168.1.165", "lumen.local"])
def test_other_hosts_are_not_external(external_hosts, raw_host):
    assert not rest_api._host_is_external(raw_host)


def test_request_host_uses_the_shared_normalizer(external_hosts):
    class _Req:
        headers = {"host": "LUMEN.cirwel.org:443"}

    assert rest_api._request_host(_Req()) == "lumen.cirwel.org"
    assert rest_api._is_external_request(_Req())


def test_mcp_mount_routes_through_the_shared_check():
    """The MCP endpoint is a closure, so pin its wiring at the source level:
    it must call the shared check, never compare a raw Host to the set."""
    src = inspect.getsource(server)
    assert "_host_is_external(host)" in src
    assert "host in _EXTERNAL_HOSTS" not in src
