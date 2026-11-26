"""Wait for a TCP port or an HTTP URL to become ready. Standard library only, works on Windows.

    python scripts/wait_for.py tcp 127.0.0.1:55432 --timeout 120
    python scripts/wait_for.py http http://127.0.0.1:58000/health --timeout 60

Exit code 0 once the target answers, 1 on timeout (with what was last seen), 2 on a bad argument.
Useful when something needs a service that `docker compose --wait` does not gate on, and for the
in-process mock API during a manual experiment.
"""

from __future__ import annotations

import argparse
import socket
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from functools import partial

DEFAULT_TIMEOUT = 60.0
DEFAULT_INTERVAL = 0.5

Probe = Callable[[], "tuple[bool, str]"]


def parse_host_port(target: str) -> tuple[str, int]:
    """`127.0.0.1:55432` -> ("127.0.0.1", 55432). Rejects anything without a port."""
    host, sep, port = target.rpartition(":")
    if not sep or not host or not port.isdigit():
        raise ValueError(f"expected HOST:PORT, got {target!r}")
    return host, int(port)


def tcp_ready(host: str, port: int, timeout: float = 2.0) -> tuple[bool, str]:
    """A TCP handshake is the cheapest honest readiness probe: the listener accepted us."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True, "connected"
    except OSError as exc:
        return False, str(exc)


def http_ready(url: str, expect: int, timeout: float = 5.0) -> tuple[bool, str]:
    """GET the URL; ready when the status code matches the expected one."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            status = int(response.status)
    except urllib.error.HTTPError as exc:
        status = int(exc.code)
    except OSError as exc:
        return False, str(exc)
    return status == expect, f"HTTP {status}"


def wait_until(probe: Probe, timeout: float, interval: float) -> tuple[bool, str]:
    """Poll `probe` until it reports ready or the timeout expires. Returns (ready, last detail)."""
    deadline = time.monotonic() + timeout
    detail = "no attempt made"
    while True:
        ready, detail = probe()
        if ready or time.monotonic() >= deadline:
            return ready, detail
        time.sleep(interval)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python scripts/wait_for.py",
        description="Wait for a TCP port or HTTP URL, with a timeout.",
    )
    parser.add_argument("kind", choices=("tcp", "http"), help="what kind of target to wait for")
    parser.add_argument("target", help="HOST:PORT for tcp, a full URL for http")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="seconds to wait")
    parser.add_argument(
        "--interval", type=float, default=DEFAULT_INTERVAL, help="seconds between tries"
    )
    parser.add_argument("--expect", type=int, default=200, help="expected HTTP status (default 200)")
    args = parser.parse_args(argv)

    probe: Probe
    if args.kind == "tcp":
        try:
            host, port = parse_host_port(args.target)
        except ValueError as exc:
            print(exc, file=sys.stderr)
            return 2
        probe = partial(tcp_ready, host, port)
    else:
        probe = partial(http_ready, args.target, args.expect)

    ready, detail = wait_until(probe, args.timeout, args.interval)
    if ready:
        print(f"ready: {args.target} ({detail})")
        return 0
    print(f"timeout after {args.timeout:g}s waiting for {args.target}: {detail}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
