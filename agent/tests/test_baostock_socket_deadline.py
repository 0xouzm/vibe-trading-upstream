"""Regression tests for #1492: baostock socket IO must stay bounded.

baostock 0.9.3's send_msg reads with no timeout and spins at 100% CPU when
the peer closes the connection. The loader swaps in a bounded send_msg and a
connect with a deadline for the duration of every fetch.
"""

from __future__ import annotations

import socket
import threading
import time

import pytest

from backtest.loaders.baostock_loader import (
    _DEFAULT_READ_TIMEOUT,
    _READ_TIMEOUT_ENV,
    _baostock_socket_guard,
    _bounded_send_msg,
    _connect_with_timeout,
    _read_timeout,
)

bs = pytest.importorskip("baostock")
import baostock.common.contants as cons  # noqa: E402
import baostock.common.context as context  # noqa: E402

_TERMINATOR = b"<![CDATA[]]>\n"


def _serve_once(handler):
    """Run handler(client_sock) on one accepted connection; returns the port."""
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]

    def run():
        conn, _ = srv.accept()
        with conn:
            handler(conn)
        srv.close()

    threading.Thread(target=run, daemon=True).start()
    return port


def _connect(port: int) -> socket.socket:
    return socket.create_connection(("127.0.0.1", port), timeout=2)


@pytest.fixture(autouse=True)
def _clean_default_socket():
    yield
    sock = getattr(context, "default_socket", None)
    if sock is not None:
        try:
            sock.close()
        except OSError:
            pass
        setattr(context, "default_socket", None)


def test_silent_server_times_out() -> None:
    port = _serve_once(lambda conn: time.sleep(5))
    setattr(context, "default_socket", _connect(port))
    started = time.monotonic()
    assert _bounded_send_msg("login\1anonymous", timeout=0.3) is None
    assert time.monotonic() - started < 3


def test_closed_connection_returns_none_instead_of_spinning() -> None:
    # recv() returns b"" forever here; stock baostock spins on this at 100% CPU.
    port = _serve_once(lambda conn: None)
    setattr(context, "default_socket", _connect(port))
    started = time.monotonic()
    assert _bounded_send_msg("login\1anonymous", timeout=5) is None
    assert time.monotonic() - started < 3


def test_healthy_reply_roundtrip_unchanged() -> None:
    body = "hello"
    header = (
        cons.BAOSTOCK_CLIENT_VERSION
        + cons.MESSAGE_SPLIT
        + "00"  # any type outside COMPRESSED_MESSAGE_TYPE_TUPLE
        + cons.MESSAGE_SPLIT
        + str(len(body)).zfill(10)
    )
    assert len(header) == cons.MESSAGE_HEADER_LENGTH
    reply = (header + body).encode() + _TERMINATOR

    def handler(conn: socket.socket) -> None:
        conn.recv(4096)
        conn.sendall(reply)

    port = _serve_once(handler)
    setattr(context, "default_socket", _connect(port))
    assert _bounded_send_msg("login\1anonymous", timeout=2) == reply.decode()


def test_malformed_reply_returns_none() -> None:
    def handler(conn: socket.socket) -> None:
        conn.recv(4096)
        conn.sendall(b"garbage" + _TERMINATOR)

    port = _serve_once(handler)
    setattr(context, "default_socket", _connect(port))
    assert _bounded_send_msg("login\1anonymous", timeout=2) is None


def test_connect_failure_degrades_to_none(monkeypatch) -> None:
    # Closed loopback port: refused fast. Stock 0.9.3 raises NameError here via
    # the unbound socket in SocketUtil.connect.
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    closed_port = probe.getsockname()[1]
    probe.close()
    monkeypatch.setattr(cons, "BAOSTOCK_SERVER_IP", "127.0.0.1")
    monkeypatch.setattr(cons, "BAOSTOCK_SERVER_PORT", closed_port)
    _connect_with_timeout(None, timeout=0.5)
    assert getattr(context, "default_socket", None) is None
    assert _bounded_send_msg("login\1anonymous", timeout=0.5) is None


def test_guard_patches_and_restores() -> None:
    socketutil = bs.util.socketutil
    original_send = socketutil.send_msg
    original_connect = socketutil.SocketUtil.connect
    with _baostock_socket_guard(0.5):
        assert socketutil.send_msg is not original_send
        assert socketutil.SocketUtil.connect is not original_connect
    assert socketutil.send_msg is original_send
    assert socketutil.SocketUtil.connect is original_connect


def test_read_timeout_env(monkeypatch) -> None:
    assert _read_timeout() == _DEFAULT_READ_TIMEOUT
    monkeypatch.setenv(_READ_TIMEOUT_ENV, "0.7")
    assert _read_timeout() == 0.7
    monkeypatch.setenv(_READ_TIMEOUT_ENV, "not-a-number")
    assert _read_timeout() == _DEFAULT_READ_TIMEOUT
    monkeypatch.setenv(_READ_TIMEOUT_ENV, "-3")
    assert _read_timeout() == _DEFAULT_READ_TIMEOUT
