"""Offline tests: real HTTP exchanges, CLI subprocesses and pure calculations."""

from __future__ import annotations

import http.client
import io
import math
import socket
import ssl
import subprocess
import sys
import threading
import time
import tracemalloc
import unittest
import urllib.error
import urllib.request
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from email.message import Message
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import speed_test as app

PAYLOAD = bytes(range(256)) * 513 + b"last byte matters"
LARGE_SIZE = 5 * 1024 * 1024 + 17
SCRIPT = Path(app.__file__).resolve()


class ServerState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.requests: list[tuple[str, dict[str, str]]] = []
        self.events: list[str] = []
        self.active = 0
        self.max_active = 0


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format, *args) -> None:
        pass

    def do_GET(self) -> None:
        state = self.server.state
        with state.lock:
            state.requests.append((self.path, dict(self.headers)))
            state.active += 1
            state.max_active = max(state.max_active, state.active)
            state.events.append("start")
        self.completed = False
        self.close_connection = True
        try:
            self.serve_response()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            # Expected when a client rejects an error or times out.
            pass
        finally:
            self.mark_complete()

    def mark_complete(self) -> None:
        if not self.completed:
            with self.server.state.lock:
                self.server.state.active -= 1
                self.server.state.events.append("complete")
            self.completed = True

    def headers_for(self, status: int, length: int | None, **headers: str) -> None:
        self.send_response(status)
        if length is not None:
            self.send_header("Content-Length", str(length))
        self.send_header("Connection", "close")
        for name, value in headers.items():
            self.send_header(name.replace("_", "-"), value)
        self.end_headers()

    def body(self, data: bytes, delay: float = 0.0) -> None:
        if data:
            self.wfile.write(data[:-1])
            self.wfile.flush()
            if delay:
                time.sleep(delay)
        # Release the active counter BEFORE the last byte becomes available.
        # This avoids a race after a correct client's complete body read, while
        # exposing a client that starts the next GET after headers/partial data.
        self.mark_complete()
        if data:
            self.wfile.write(data[-1:])

    def serve_response(self) -> None:
        path = self.path
        if path in {"/404", "/500"}:
            self.headers_for(int(path[1:]), 5)
            self.body(b"error")
        elif path == "/truncated":
            self.headers_for(200, 100)
            self.body(b"short")
        elif path == "/broken-chunked":
            self.headers_for(200, None, Transfer_Encoding="chunked")
            self.wfile.write(b"a\r\nabc")
        elif path == "/chunked":
            self.headers_for(200, None, Transfer_Encoding="chunked")
            for chunk in (PAYLOAD[:1000], PAYLOAD[1000:]):
                self.wfile.write(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n")
            self.mark_complete()
            self.wfile.write(b"0\r\n\r\n")
        elif path == "/slow":
            self.headers_for(200, 5)
            time.sleep(0.15)
            self.body(b"hello")
        elif path == "/trickle":
            self.headers_for(200, 30)
            for _ in range(30):
                self.wfile.write(b"x")
                self.wfile.flush()
                time.sleep(0.01)
        elif path == "/late-failure" and len(self.server.state.requests) == 4:
            self.headers_for(500, 0)
        elif path == "/empty":
            self.headers_for(204, 0)
        elif path == "/large":
            self.headers_for(200, LARGE_SIZE)
            chunk = b"x" * app.CHUNK_SIZE
            remaining = LARGE_SIZE - 1
            while remaining:
                block = chunk[: min(remaining, len(chunk))]
                self.wfile.write(block)
                remaining -= len(block)
            self.mark_complete()
            self.wfile.write(b"z")
        elif path in {"/redirect", "/loop", "/unsafe-redirect", "/missing-location"}:
            location = {
                "/redirect": "/data",
                "/loop": "/loop",
                "/unsafe-redirect": "file:///etc/passwd",
            }.get(path)
            self.headers_for(302, 4, **({"Location": location} if location else {}))
            self.body(b"move")
        elif path.startswith("/chain/"):
            step = int(path.rsplit("/", 1)[1])
            self.headers_for(307, 0, Location=f"/chain/{step + 1}")
        elif path == "/partial":
            self.headers_for(206, 4, Content_Range="bytes 0-3/100")
            self.body(b"part")
        elif path == "/bad-length":
            self.headers_for(200, None, Content_Length="invalid")
            self.body(b"data")
        elif path == "/ambiguous":
            self.headers_for(200, 0, Transfer_Encoding="chunked")
            self.body(b"0\r\n\r\n")
        elif path == "/no-length":
            self.headers_for(200, None)
            self.body(PAYLOAD)
        else:
            self.headers_for(200, len(PAYLOAD))
            self.body(PAYLOAD, delay=0.003)


@contextmanager
def local_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = False
    server.state = ServerState()
    worker = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.01}
    )
    worker.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", server.state
    finally:
        server.shutdown()
        worker.join(timeout=3)
        server.server_close()


class HTTPTests(unittest.TestCase):
    def setUp(self) -> None:
        # Ignore ambient proxy variables; the production utility still honours
        # them. Never modify the real process/user proxy configuration.
        self.proxy_patch = patch.dict("os.environ", {"no_proxy": "*", "NO_PROXY": "*"})
        self.proxy_patch.start()
        self.addCleanup(self.proxy_patch.stop)

    def test_exactly_ten_sequential_complete_downloads(self) -> None:
        progress = []
        with local_server() as (base, state):
            results = app.run_benchmark(
                base + "/data", on_result=lambda i, r: progress.append(i)
            )
        self.assertEqual(len(state.requests), 10)
        self.assertEqual(progress, list(range(1, 11)))
        self.assertEqual(state.max_active, 1)
        self.assertEqual(state.events, ["start", "complete"] * 10)
        self.assertEqual([r.byte_count for r in results], [len(PAYLOAD)] * 10)
        self.assertEqual(
            app.calculate_statistics(results).total_bytes, len(PAYLOAD) * 10
        )
        self.assertTrue(all(r.duration > 0 for r in results))

    def test_cache_and_compression_headers(self) -> None:
        with local_server() as (base, state):
            app.run_benchmark(base + "/data")
        for _, headers in state.requests:
            self.assertIn("no-cache", headers["Cache-Control"])
            self.assertIn("no-store", headers["Cache-Control"])
            self.assertEqual(headers["Pragma"], "no-cache")
            self.assertEqual(headers["Accept-Encoding"], "identity")

    def test_slow_tail_is_included_in_request_time(self) -> None:
        with local_server() as (base, _):
            results = app.run_benchmark(base + "/data")
        self.assertTrue(all(r.duration >= 0.003 for r in results))

    def test_large_responses_use_bounded_memory(self) -> None:
        with local_server() as (base, state):
            tracemalloc.start()
            try:
                results = app.run_benchmark(base + "/large")
                _, peak = tracemalloc.get_traced_memory()
            finally:
                tracemalloc.stop()
        self.assertEqual(len(state.requests), 10)
        self.assertEqual(sum(r.byte_count for r in results), LARGE_SIZE * 10)
        self.assertLess(peak, 3 * 1024 * 1024, f"Peak traced memory: {peak} bytes")

    def test_chunked_complete_body(self) -> None:
        with local_server() as (base, _):
            results = app.run_benchmark(base + "/chunked")
        self.assertEqual([r.byte_count for r in results], [len(PAYLOAD)] * 10)

    def test_close_delimited_body(self) -> None:
        with local_server() as (base, _):
            results = app.run_benchmark(base + "/no-length")
        self.assertEqual(sum(r.byte_count for r in results), len(PAYLOAD) * 10)

    def test_empty_body_is_a_valid_zero_speed(self) -> None:
        with local_server() as (base, state):
            results = app.run_benchmark(base + "/empty")
        stats = app.calculate_statistics(results)
        self.assertEqual(len(state.requests), 10)
        self.assertEqual(stats.total_bytes, 0)
        self.assertEqual(stats.megabytes_per_second, 0)
        self.assertEqual(stats.megabits_per_second, 0)

    def test_http_errors_stop_immediately(self) -> None:
        for status in (404, 500):
            with self.subTest(status=status), local_server() as (base, state):
                with self.assertRaisesRegex(
                    app.BenchmarkError, f"Request 1/10 failed: HTTP {status}"
                ):
                    app.run_benchmark(base + f"/{status}")
                self.assertEqual(len(state.requests), 1)

    def test_late_error_stops_without_retry(self) -> None:
        with local_server() as (base, state):
            with self.assertRaisesRegex(app.BenchmarkError, "Request 4/10 failed"):
                app.run_benchmark(base + "/late-failure")
            self.assertEqual(len(state.requests), 4)

    def test_truncated_content_length(self) -> None:
        with local_server() as (base, state):
            with self.assertRaisesRegex(
                app.BenchmarkError, "expected 100 bytes, received 5"
            ):
                app.run_benchmark(base + "/truncated")
            self.assertEqual(len(state.requests), 1)

    def test_broken_chunked_body(self) -> None:
        with local_server() as (base, state):
            with self.assertRaisesRegex(app.BenchmarkError, "protocol error"):
                app.run_benchmark(base + "/broken-chunked")
            self.assertEqual(len(state.requests), 1)

    def test_socket_timeout(self) -> None:
        with local_server() as (base, state):
            with self.assertRaisesRegex(app.BenchmarkError, "timeout|time budget"):
                app.run_benchmark(base + "/slow", timeout=0.05)
            self.assertEqual(len(state.requests), 1)

    def test_trickling_body_cannot_evade_checked_budget(self) -> None:
        with local_server() as (base, state):
            started = time.perf_counter()
            with self.assertRaisesRegex(app.BenchmarkError, "time budget"):
                app.run_benchmark(base + "/trickle", timeout=0.08)
            self.assertLess(time.perf_counter() - started, 1)
            self.assertEqual(len(state.requests), 1)

    def test_redirect_requires_direct_url_without_false_summary(self) -> None:
        with local_server() as (base, state):
            with self.assertRaisesRegex(
                app.BenchmarkError, "Request 1/10 failed: HTTP redirect"
            ):
                app.run_benchmark(base + "/redirect")
        self.assertEqual([path for path, _ in state.requests], ["/redirect"])

    def test_redirect_loop_and_unsafe_target(self) -> None:
        for path, message in (
            ("/loop", "HTTP redirect"),
            ("/unsafe-redirect", "Invalid redirect"),
            ("/missing-location", "no Location"),
        ):
            with self.subTest(path=path), local_server() as (base, state):
                with self.assertRaisesRegex(app.BenchmarkError, message):
                    app.run_benchmark(base + path)
                self.assertEqual(len(state.requests), 1)

    def test_redirect_chain_is_rejected_on_first_step(self) -> None:
        with local_server() as (base, state):
            with self.assertRaisesRegex(app.BenchmarkError, "HTTP redirect"):
                app.run_benchmark(base + "/chain/0")
            self.assertEqual(len(state.requests), 1)

    def test_cli_redirect_has_no_speed_summary(self) -> None:
        with local_server() as (base, state):
            completed = self.cli(base + "/redirect")
        self.assertEqual(completed.returncode, 1)
        self.assertEqual(len(state.requests), 1)
        self.assertIn("final direct resource URL", completed.stderr)
        self.assertNotIn("Results\n", completed.stdout)

    def test_partial_or_ambiguous_response_is_not_success(self) -> None:
        for path in ("/partial", "/bad-length", "/ambiguous"):
            with self.subTest(path=path), local_server() as (base, state):
                with self.assertRaises(app.BenchmarkError):
                    app.run_benchmark(base + path)
                self.assertEqual(len(state.requests), 1)

    def cli(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(SCRIPT), *args],
            capture_output=True,
            text=True,
            encoding="ascii",
            timeout=10,
        )

    def test_cli_success_and_ascii_output(self) -> None:
        with local_server() as (base, state):
            completed = self.cli(base + "/data")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(len(state.requests), 10)
        self.assertEqual(completed.stderr, "")
        self.assertIn("[ 1/10]", completed.stdout)
        self.assertIn("[10/10]", completed.stdout)
        self.assertIn("Results\n", completed.stdout)
        self.assertIn("MB/s", completed.stdout)
        self.assertIn("Mbps", completed.stdout)
        self.assertIn(f"{len(PAYLOAD) * 10:,} bytes", completed.stdout)

    def test_cli_failure_has_no_summary(self) -> None:
        for path in ("/500", "/truncated", "/late-failure", "/partial"):
            with self.subTest(path=path), local_server() as (base, _):
                completed = self.cli(base + path)
            self.assertEqual(completed.returncode, 1)
            self.assertIn("Error:", completed.stderr)
            self.assertNotIn("Traceback", completed.stderr)
            self.assertNotIn("Results\n", completed.stdout)
            self.assertNotIn("MB/s", completed.stdout)

    def test_cli_timeout_exit_code(self) -> None:
        with local_server() as (base, _):
            completed = self.cli(base + "/slow", "--timeout", "0.05")
        self.assertEqual(completed.returncode, 1)
        self.assertNotIn("Results\n", completed.stdout)


class CalculationTests(unittest.TestCase):
    def test_formulas_with_unequal_sizes_and_durations(self) -> None:
        results = [
            app.RequestResult("http://example.test", 200, i * 1_000_000, float(i))
            for i in range(1, 11)
        ]
        stats = app.calculate_statistics(results)
        self.assertEqual(stats.total_bytes, 55_000_000)
        self.assertEqual(stats.total_seconds, 55)
        self.assertEqual(stats.average_seconds, 5.5)
        self.assertEqual(stats.min_seconds, 1)
        self.assertEqual(stats.max_seconds, 10)
        self.assertEqual(stats.megabytes_per_second, 1)
        self.assertEqual(stats.megabits_per_second, 8)

    def test_speed_is_ratio_of_totals_not_mean_of_speeds(self) -> None:
        results = [app.RequestResult("http://example.test", 200, 1_000_000, 1.0)] * 9
        results.append(app.RequestResult("http://example.test", 200, 1_000_000, 10.0))
        stats = app.calculate_statistics(results)
        self.assertAlmostEqual(stats.megabytes_per_second, 10 / 19)
        self.assertAlmostEqual(stats.megabits_per_second, 80 / 19)
        self.assertEqual(stats.average_seconds, 1.9)

    def test_rejects_incomplete_or_invalid_measurements(self) -> None:
        good = app.RequestResult("http://example.test", 200, 1, 1.0)
        for count in (0, 9, 11):
            with self.subTest(count=count), self.assertRaises(ValueError):
                app.calculate_statistics([good] * count)
        for duration in (0, -1, math.inf, math.nan):
            with self.subTest(duration=duration), self.assertRaises(ValueError):
                app.calculate_statistics(
                    [app.RequestResult(good.url, 200, 1, duration)] * 10
                )
        with self.assertRaises(ValueError):
            app.calculate_statistics([app.RequestResult(good.url, 200, -1, 1)] * 10)
        with self.assertRaises(ValueError):
            app.calculate_statistics(
                [app.RequestResult(good.url, 302, 0, 1, good.url)] * 10
            )


class ValidationTests(unittest.TestCase):
    def test_invalid_urls(self) -> None:
        for value in (
            "",
            "example.com",
            "ftp://example.com/a",
            "file:///tmp/a",
            "http://",
            "http:///path",
            "http://user:password@example.com",
            "http://host:0",
            "http://host:65536",
            "http://host:bad",
            "http://host:",
            "http://[broken",
            " http://host",
            "http://host/a b",
            "http://host/\r\nx",
            "http://host/\x7f",
            "http://host/файл",
            "http://host\\path",
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                app.validate_url(value)

    def test_valid_urls_and_fragment_removal(self) -> None:
        self.assertEqual(
            app.validate_url("https://example.com/a%20b?q=1#fragment"),
            "https://example.com/a%20b?q=1",
        )
        self.assertEqual(app.validate_url("http://[::1]:8080/"), "http://[::1]:8080/")

    def test_invalid_cli_arguments_exit_two_without_network(self) -> None:
        for args in (
            [],
            ["ftp://host/a"],
            ["http://host", "--timeout", "0"],
            ["http://host", "--timeout", "nan"],
            ["http://host", "--timeout", "inf"],
            ["http://host", "--timeout", "no"],
            ["http://host", "--timeout", "1e300"],
            ["http://host", "--unknown"],
        ):
            with self.subTest(args=args):
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as raised:
                        app.main(args)
                self.assertEqual(raised.exception.code, 2)

    def test_help_exits_zero(self) -> None:
        with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as raised:
            app.main(["--help"])
        self.assertEqual(raised.exception.code, 0)

    def test_invalid_timeout_rejected_before_opening(self) -> None:
        for value in (0, -1, math.inf, math.nan, 1e300):
            with (
                self.subTest(value=value),
                patch("urllib.request.build_opener") as opened,
            ):
                with self.assertRaises(ValueError):
                    app.run_benchmark("http://example.test", timeout=value)
                opened.assert_not_called()

    def test_keyboard_interrupt_exit_130_without_summary(self) -> None:
        output, errors = io.StringIO(), io.StringIO()
        with patch.object(app, "run_benchmark", side_effect=KeyboardInterrupt):
            with redirect_stdout(output), redirect_stderr(errors):
                code = app.main(["http://example.test"])
        self.assertEqual(code, 130)
        self.assertIn("Interrupted", errors.getvalue())
        self.assertNotIn("Results\n", output.getvalue())

    def test_unreachable_local_port_is_readable(self) -> None:
        # Keep a port bound but not listening, eliminating a port-reuse race.
        with socket.socket() as reserved:
            reserved.bind(("127.0.0.1", 0))
            port = reserved.getsockname()[1]
            with patch.dict("os.environ", {"no_proxy": "*", "NO_PROXY": "*"}):
                # Windows may time out here instead of reporting ECONNREFUSED.
                with self.assertRaisesRegex(
                    app.BenchmarkError, "Network (error|timeout)"
                ):
                    app.run_benchmark(f"http://127.0.0.1:{port}/", timeout=0.5)

    def test_expected_length_rejects_duplicate_headers(self) -> None:
        headers = Message()
        headers.add_header("Content-Length", "5")
        headers.add_header("Content-Length", "5")
        with self.assertRaises(app.BenchmarkError):
            app.expected_body_length(headers)

    def test_https_downgrade_is_rejected(self) -> None:
        class Response:
            status = 302
            headers = Message()
            headers.add_header("Content-Length", "0")
            headers.add_header("Location", "http://example.test/")

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def read1(self, size):
                return b""

        with patch.object(
            urllib.request.OpenerDirector, "open", return_value=Response()
        ):
            with self.assertRaisesRegex(app.BenchmarkError, "HTTPS to HTTP"):
                app.run_benchmark("https://example.test/")

    def test_connection_timeout_is_readable(self) -> None:
        with patch.object(
            urllib.request.OpenerDirector,
            "open",
            side_effect=urllib.error.URLError(TimeoutError("timed out")),
        ) as opened:
            with self.assertRaisesRegex(app.BenchmarkError, "Network timeout"):
                app.run_benchmark("http://example.test/", timeout=1)
        self.assertEqual(opened.call_count, 1)

    def test_tls_certificate_failure_is_not_ignored(self) -> None:
        with patch.object(
            urllib.request.OpenerDirector,
            "open",
            side_effect=urllib.error.URLError(
                ssl.SSLCertVerificationError("certificate verify failed")
            ),
        ) as opened:
            with self.assertRaisesRegex(
                app.BenchmarkError, "certificate verify failed"
            ):
                app.run_benchmark("https://example.test/")
        self.assertEqual(opened.call_count, 1)

    def test_body_read_errors_close_response(self) -> None:
        for error in (
            ConnectionResetError("reset"),
            http.client.IncompleteRead(b"x", 10),
        ):
            response = MagicMock()
            response.__enter__.return_value = response
            response.status = 200
            response.headers = Message()
            response.read1.side_effect = error
            with (
                self.subTest(error=error),
                patch.object(
                    urllib.request.OpenerDirector,
                    "open",
                    return_value=response,
                ) as opened,
            ):
                with self.assertRaisesRegex(app.BenchmarkError, "protocol error"):
                    app.run_benchmark("http://example.test/")
            response.__exit__.assert_called_once()
            self.assertEqual(opened.call_count, 1)

    def test_stream_reads_bounded_blocks_and_closes_response(self) -> None:
        response = MagicMock()
        response.__enter__.return_value = response
        response.status = 200
        response.headers = Message()
        response.headers.add_header("Content-Length", str(app.CHUNK_SIZE + 4))
        response.read1.side_effect = [b"x" * app.CHUNK_SIZE, b"tail", b""]
        opener = MagicMock(spec=urllib.request.OpenerDirector)
        opener.open.return_value = response
        result = app.download_once("http://example.test/", 1, opener)
        self.assertEqual(result.byte_count, app.CHUNK_SIZE + 4)
        self.assertEqual(response.read1.call_args_list, [call(app.CHUNK_SIZE)] * 3)
        response.__exit__.assert_called_once()


if __name__ == "__main__":
    unittest.main()
