"""Wait for a TCP port or an HTTP URL to become ready. Standard library only, works on Windows.

    python scripts/wait_for.py tcp 127.0.0.1:55432 --timeout 120
    python scripts/wait_for.py http http://127.0.0.1:58000/health --timeout 60

Exit code 0 once the target answers, 1 on timeout (with what was last seen), 2 on a bad argument.
The `http` form accepts a loopback URL only (127.0.0.1, localhost, ::1): this waits for local
services, and `urllib` would otherwise also honour a `file:` URL from the command line.
Useful when something needs a service that `docker compose --wait` does not gate on, and for the
in-process mock API during a manual experiment.
"""

from __future__ import annotations

import argparse
import socket
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from functools import partial

DEFAULT_TIMEOUT = 60.0
DEFAULT_INTERVAL = 0.5

LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
"""This script waits for services on *this* machine, so any other host is a mistake."""

Probe = Callable[[], "tuple[bool, str]"]


def parse_host_port(target: str) -> tuple[str, int]:
    """`127.0.0.1:55432` -> ("127.0.0.1", 55432). Rejects anything without a port."""
    host, sep, port = target.rpartition(":")
    if not sep or not host:
        raise ValueError(f"expected HOST:PORT, got {target!r}")
    try:
        number = int(port)
    except ValueError:
        raise ValueError(f"expected HOST:PORT with a numeric port, got {target!r}") from None
    if not 0 < number <= 65535:
        raise ValueError(f"expected a port from 1 to 65535, got {number} in {target!r}")
    return host, number


def check_local_http_url(url: str) -> str:
    """Return `url` when it is an http(s) URL on a loopback host; raise `ValueError` otherwise.

    `urllib` also speaks `file:`, so a URL taken from the command line is narrowed to a scheme and
    a host *before* the request is made, never trusted afterwards.
    """
    parts = urllib.parse.urlsplit(url)
    if parts.scheme not in ("http", "https"):
        raise ValueError(f"expected an http(s) URL, got {url!r}")
    host = parts.hostname or ""
    if host not in LOOPBACK_HOSTS:
        raise ValueError(
            f"refusing host {host!r}: this script waits for loopback services only "
            "(127.0.0.1, localhost, ::1)"
        )
    return url


def tcp_ready(host: str, port: int, timeout: float = 2.0) -> tuple[bool, str]:
    """A TCP handshake is the cheapest honest readiness probe: the listener accepted us."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True, "connected"
    except OSError as exc:
        return False, str(exc)


def http_ready(url: str, expect: int, timeout: float = 5.0) -> tuple[bool, str]:
    """GET the URL; ready when the status code matches the expected one."""
    check_local_http_url(url)  # the only URL that reaches urllib: http(s) on loopback
    try:
        # nosemgrep: python.lang.security.audit.dynamic-urllib-use-detected.dynamic-urllib-use-detected
        # The line above already refused every scheme but http(s) and every host but loopback, so
        # the `file:` escape this rule warns about cannot arrive here. Semgrep matches on shape,
        # not dataflow, and no built-in alternative probes a URL for readiness as plainly.
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
    parser.add_argument(
        "--expect", type=int, default=200, help="expected HTTP status (default 200)"
    )
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
        try:
            check_local_http_url(args.target)
        except ValueError as exc:
            print(exc, file=sys.stderr)
            return 2
        probe = partial(http_ready, args.target, args.expect)

    ready, detail = wait_until(probe, args.timeout, args.interval)
    if ready:
        print(f"ready: {args.target} ({detail})")
        return 0
    print(f"timeout after {args.timeout:g}s waiting for {args.target}: {detail}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
