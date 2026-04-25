#!/usr/bin/env python3
"""Lightweight TCP proxy — forward local port to remote host.

Usage: tcp_proxy.py <local_port> <remote_host> <remote_port>

Used to expose Pi's anima-mcp (Tailscale) on localhost so a tunnel (cloudflared, etc.) can forward it.
"""

import asyncio
import sys


async def pipe(reader, writer):
    try:
        while True:
            data = await reader.read(65536)
            if not data:
                break
            writer.write(data)
            await writer.drain()
    except (ConnectionResetError, BrokenPipeError, OSError):
        pass
    finally:
        writer.close()


async def handle_client(local_reader, local_writer, remote_host, remote_port):
    try:
        remote_reader, remote_writer = await asyncio.wait_for(
            asyncio.open_connection(remote_host, remote_port),
            timeout=10.0,
        )
    except Exception as e:
        print(f"[proxy] Failed to connect to {remote_host}:{remote_port}: {e}", flush=True)
        local_writer.close()
        return

    await asyncio.gather(
        pipe(local_reader, remote_writer),
        pipe(remote_reader, local_writer),
    )


async def main(local_port, remote_host, remote_port):
    async def on_connect(reader, writer):
        asyncio.create_task(handle_client(reader, writer, remote_host, remote_port))

    server = await asyncio.start_server(on_connect, "127.0.0.1", local_port)
    print(f"[proxy] Forwarding 127.0.0.1:{local_port} → {remote_host}:{remote_port}", flush=True)
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <local_port> <remote_host> <remote_port>")
        sys.exit(1)
    asyncio.run(main(int(sys.argv[1]), sys.argv[2], int(sys.argv[3])))
