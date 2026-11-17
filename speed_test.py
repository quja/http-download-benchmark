#!/usr/bin/env python3
"""Measure ten sequential complete HTTP downloads using only the stdlib."""

from __future__ import annotations

import argparse
import http.client
import math
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from email.message import Message

REQUEST_COUNT = 10
CHUNK_SIZE = 64 * 1024
DEFAULT_TIMEOUT = 30.0
MAX_TIMEOUT = 86_400.0
REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})


class BenchmarkError(Exception):
    """A failed measurement; no aggregate result may be reported."""


@dataclass(frozen=True)
class RequestResult:
    url: str
    status: int
    byte_count: int
    duration: float
    redirect_url: str | None = None


@dataclass(frozen=True)
class Statistics:
    total_bytes: int
    total_seconds: float
    average_seconds: float
    min_seconds: float
    max_seconds: float
    megabytes_per_second: float
    megabits_per_second: float


class NoAutomaticRedirects(urllib.request.HTTPRedirectHandler):
    """Expose redirects as responses so the CLI can reject incomplete downloads."""

    def http_error_302(
        self,
        req: urllib.request.Request,
        fp: http.client.HTTPResponse,
        code: int,
        msg: str,
        headers: Message,
    ) -> http.client.HTTPResponse:
        return fp

    http_error_301 = http_error_302
    http_error_303 = http_error_302
    http_error_307 = http_error_302
    http_error_308 = http_error_302


def validate_url(url: str) -> str:
    """Accept an explicit ASCII HTTP(S) URL, without credentials or controls."""
    if not url or any(ord(char) <= 32 or ord(char) >= 127 for char in url):
        raise ValueError(
            "URL must be ASCII with no whitespace or control characters; "
            "use percent-encoding and an IDNA hostname if needed."
        )
    if "\\" in url:
        raise ValueError("URL must not contain backslashes.")
    try:
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme.lower() not in {"http", "https"}:
            raise ValueError("URL scheme must be http or https.")
        if not parsed.hostname:
            raise ValueError("URL must include a hostname.")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("Credentials in the URL are not supported.")
        if parsed.port is not None and parsed.port == 0:
            raise ValueError("URL port must be between 1 and 65535.")
        if parsed.netloc.endswith(":"):
            raise ValueError("URL port must not be empty.")
    except ValueError as exc:
        raise ValueError(f"Invalid URL: {exc}") from exc
    # Fragments are local identifiers and must never be sent to the server.
    return urllib.parse.urlunsplit(
        parsed._replace(scheme=parsed.scheme.lower(), fragment="")
    )


def positive_timeout(value: str) -> float:
    try:
        timeout = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("timeout must be a number") from exc
    if not math.isfinite(timeout) or not 0 < timeout <= MAX_TIMEOUT:
        raise argparse.ArgumentTypeError(
            "timeout must be finite and in (0, 86400] seconds"
        )
    return timeout


def expected_body_length(headers: Message) -> int | None:
    """Reject ambiguous framing and validate Content-Length when present."""
    lengths = headers.get_all("Content-Length", [])
    transfers = headers.get_all("Transfer-Encoding", [])
    if transfers:
        if lengths or len(transfers) != 1 or transfers[0].strip().lower() != "chunked":
            raise BenchmarkError("Unsupported or ambiguous HTTP response framing.")
        return None
    if not lengths:
        return None
    if len(lengths) != 1 or not lengths[0].isascii() or not lengths[0].isdigit():
        raise BenchmarkError("Invalid or repeated Content-Length header.")
    try:
        return int(lengths[0])
    except ValueError as exc:
        raise BenchmarkError("Invalid Content-Length header.") from exc


def download_once(
    url: str, timeout: float, opener: urllib.request.OpenerDirector
) -> RequestResult:
    """Read a single response in bounded blocks, including a redirect body."""
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "http-download-benchmark/1.0",
            "Accept-Encoding": "identity",
            "Cache-Control": "no-cache, no-store, max-age=0",
            "Pragma": "no-cache",
        },
    )
    started = time.perf_counter()

    def check_budget() -> None:
        if time.perf_counter() - started >= timeout:
            raise BenchmarkError(f"Request exceeded the {timeout:g} s time budget.")

    try:
        with opener.open(request, timeout=timeout) as response:
            check_budget()
            status = response.status
            if not (200 <= status < 300 or status in REDIRECT_STATUSES):
                raise BenchmarkError(f"Unexpected HTTP status {status}.")
            if status == 206 or response.headers.get("Content-Range") is not None:
                raise BenchmarkError(
                    "Partial HTTP responses are not valid measurements."
                )
            expected = expected_body_length(response.headers)
            byte_count = 0
            while True:
                check_budget()
                # read1 performs at most one underlying read. Unlike read(n), it
                # lets us check the time budget even with a slowly trickling body.
                block = response.read1(CHUNK_SIZE)
                check_budget()
                if not block:
                    break
                byte_count += len(block)
            if expected is not None and byte_count != expected:
                raise BenchmarkError(
                    f"Incomplete response: expected {expected} bytes, received {byte_count}."
                )
            location = (
                response.headers.get("Location")
                if status in REDIRECT_STATUSES
                else None
            )
        # Include opening, headers, the complete body and closing the response.
        duration = time.perf_counter() - started
    except urllib.error.HTTPError as exc:
        exc.close()
        raise BenchmarkError(
            f"HTTP {exc.code}: the server rejected the request."
        ) from exc
    except urllib.error.URLError as exc:
        reason = exc.reason
        if isinstance(reason, TimeoutError):
            raise BenchmarkError(f"Network timeout after {timeout:g} s.") from exc
        raise BenchmarkError(f"Network error: {reason}") from exc
    except TimeoutError as exc:
        raise BenchmarkError(f"Network timeout after {timeout:g} s.") from exc
    except (OSError, http.client.HTTPException) as exc:
        raise BenchmarkError(f"Connection or HTTP protocol error: {exc}") from exc

    redirect_url = None
    if status in REDIRECT_STATUSES:
        if not location:
            raise BenchmarkError(f"HTTP {status} redirect has no Location header.")
        try:
            # Validate before urljoin, which otherwise strips some controls.
            if any(ord(char) <= 32 or ord(char) >= 127 for char in location):
                raise ValueError("Redirect Location contains unsafe characters.")
            redirect_url = validate_url(urllib.parse.urljoin(url, location))
            if urllib.parse.urlsplit(
                url
            ).scheme == "https" and not redirect_url.startswith("https:"):
                raise ValueError("HTTPS to HTTP redirects are not allowed.")
        except ValueError as exc:
            raise BenchmarkError(f"Invalid redirect: {exc}") from exc
    return RequestResult(url, status, byte_count, duration, redirect_url)


def run_benchmark(
    url: str,
    timeout: float = DEFAULT_TIMEOUT,
    on_result: Callable[[int, RequestResult], None] | None = None,
) -> list[RequestResult]:
    """Run exactly ten complete GETs on success; stop on first failure."""
    url = validate_url(url)
    if not math.isfinite(timeout) or not 0 < timeout <= MAX_TIMEOUT:
        raise ValueError("Timeout must be finite and in (0, 86400] seconds.")
    opener = urllib.request.build_opener(NoAutomaticRedirects())
    results: list[RequestResult] = []
    for index in range(1, REQUEST_COUNT + 1):
        try:
            result = download_once(url, timeout, opener)
            # A redirect is not the requested resource's complete body.
            # Reject it instead of mixing the redirect body into speed metrics.
            if result.redirect_url is not None:
                raise BenchmarkError(
                    "HTTP redirect received; provide the final direct resource URL "
                    "to measure ten complete downloads."
                )
        except BenchmarkError as exc:
            raise BenchmarkError(
                f"Request {index}/{REQUEST_COUNT} failed: {exc}"
            ) from exc
        results.append(result)
        if on_result is not None:
            on_result(index, result)
    return results


def calculate_statistics(results: Sequence[RequestResult]) -> Statistics:
    if len(results) != REQUEST_COUNT or results[-1].redirect_url is not None:
        raise ValueError(
            "Statistics require ten complete requests ending without a redirect."
        )
    if any(
        not math.isfinite(item.duration) or item.duration <= 0 or item.byte_count < 0
        for item in results
    ):
        raise ValueError(
            "Measurements must have positive finite durations and nonnegative byte counts."
        )
    total_bytes = sum(item.byte_count for item in results)
    total_seconds = math.fsum(item.duration for item in results)
    if not math.isfinite(total_seconds):
        raise ValueError("Total duration is not finite.")
    bytes_per_second = total_bytes / total_seconds
    return Statistics(
        total_bytes=total_bytes,
        total_seconds=total_seconds,
        average_seconds=total_seconds / REQUEST_COUNT,
        min_seconds=min(item.duration for item in results),
        max_seconds=max(item.duration for item in results),
        megabytes_per_second=bytes_per_second / 1_000_000,
        megabits_per_second=bytes_per_second * 8 / 1_000_000,
    )


def print_progress(index: int, result: RequestResult) -> None:
    print(
        f"[{index:2}/{REQUEST_COUNT}] HTTP {result.status} | {result.duration:.6f} s"
        f" | {result.byte_count:,} bytes",
        flush=True,
    )
    if result.redirect_url is not None:
        print(f"        Redirect -> {result.redirect_url}", flush=True)


def print_summary(stats: Statistics) -> None:
    rows = [
        ("Requests", str(REQUEST_COUNT)),
        ("Total bytes", f"{stats.total_bytes:,} bytes"),
        ("Total data", f"{stats.total_bytes / 1_000_000:.6f} MB"),
        ("Total request time", f"{stats.total_seconds:.6f} s"),
        ("Average request time", f"{stats.average_seconds:.6f} s"),
        ("Min / max time", f"{stats.min_seconds:.6f} / {stats.max_seconds:.6f} s"),
        ("Download speed", f"{stats.megabytes_per_second:.6f} MB/s"),
        ("Download speed", f"{stats.megabits_per_second:.6f} Mbps"),
    ]
    value_width = max(len(value) for _, value in rows)
    border = "+----------------------+" + "-" * (value_width + 2) + "+"
    print("\nResults")
    print(border)
    for label, value in rows:
        print(f"| {label:<20} | {value:>{value_width}} |")
    print(border)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Measure ten sequential complete HTTP downloads (direct URLs only)."
    )
    parser.add_argument(
        "url", type=validate_url, help="HTTP(S) resource URL, without credentials"
    )
    parser.add_argument(
        "--timeout",
        type=positive_timeout,
        default=DEFAULT_TIMEOUT,
        metavar="SECONDS",
        help="socket timeout and checked request budget (default: 30)",
    )
    args = parser.parse_args(argv)
    print("HTTP Download Benchmark")
    print(f"URL: {args.url}")
    print(
        f"Requests: {REQUEST_COUNT} | Timeout: {args.timeout:g} s | Block: {CHUNK_SIZE:,} bytes\n",
        flush=True,
    )
    try:
        results = run_benchmark(args.url, args.timeout, print_progress)
        stats = calculate_statistics(results)
    except (BenchmarkError, ValueError) as exc:
        print(
            f"Error: {exc}\nMeasurement incomplete; no summary produced.",
            file=sys.stderr,
        )
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted; no summary produced.", file=sys.stderr)
        return 130
    print_summary(stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
