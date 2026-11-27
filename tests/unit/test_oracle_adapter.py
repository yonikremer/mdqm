"""Unit tests for the Oracle adapter's pure logic (task A07).

No Docker, no network: everything here is a table of cases over functions that only reshape values
(types, ``HIGH_VALUE`` text, plan rows, config text). The database claims -- the catalog, the
interval partitions with their real bounds, a cancelled statement, Oracle's own refusal of a write
-- live in ``tests/integration/test_oracle_adapter.py`` because only Oracle itself can answer them.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

# pyarrow ships no stubs; mypy already overrides it in pyproject, pyright needs the same note.
import pyarrow as pa  # pyright: ignore[reportMissingTypeStubs]
import pytest

pytest.importorskip("oracledb", reason="the `oracle` extra is not installed")

import oracledb  # noqa: E402 - after importorskip

from mdqm.adapters.base import OPTIONAL_METHODS  # noqa: E402 - after importorskip
from mdqm.adapters.oracle import (  # noqa: E402 - after importorskip
    OracleAdapter,
    OracleCancelled,
    OracleConfigError,
    OracleTimeout,
    _bounds,
    _connection_parts,
    _list_values,
    _partitioning,
    _raw_queries,
    _stream_error,
    arrow_type,
    base_type,
    interval_granularity,
    normalized_type,
    parse_high_value,
    plan_summary,
)
from mdqm.config.settings import SourceConfig  # noqa: E402 - after importorskip
from mdqm.models import (  # noqa: E402 - after importorskip
    NormalizedType,
    PartitionGranularity,
    PartitionStrategy,
    PruningProof,
)

# --- HIGH_VALUE parsing (the bound a partition really has) -----------------------------------

HIGH_VALUE_CASES: list[tuple[str, datetime | None]] = [
    ("TIMESTAMP' 2025-07-01 00:00:00'", datetime(2025, 7, 1, tzinfo=UTC)),
    ("TIMESTAMP' 2000-01-01 00:00:00'", datetime(2000, 1, 1, tzinfo=UTC)),
    ("DATE ' 2026-01-01 00:00:00'", datetime(2026, 1, 1, tzinfo=UTC)),
    ("MAXVALUE", None),
    ("maxvalue", None),
    ("DEFAULT", None),
    ("", None),
    ("not a bound", None),
    ("TIMESTAMP' 2025-13-01 00:00:00'", None),
]


@pytest.mark.parametrize(("text", "expected"), HIGH_VALUE_CASES)
def test_parse_high_value(text: str, expected: datetime | None) -> None:
    """Oracle stores the bound as DDL text with a space after the quote; a bad one is ``None``."""
    assert parse_high_value(text) == expected


def test_parse_high_value_never_invents_a_bound() -> None:
    """A ``None`` column and an unparsable one both stay ``None``: a wrong bound prunes wrongly."""
    assert parse_high_value(None) is None
    assert parse_high_value(12345) is None


# --- interval -> granularity ----------------------------------------------------------------

INTERVAL_CASES: list[tuple[str | None, PartitionGranularity | None]] = [
    ("NUMTOYMINTERVAL(1, 'MONTH')", PartitionGranularity.MONTH),
    ("NUMTOYMINTERVAL(3, 'MONTH')", PartitionGranularity.QUARTER),
    ("NUMTOYMINTERVAL(1, 'YEAR')", PartitionGranularity.YEAR),
    ("NUMTODSINTERVAL(1, 'DAY')", PartitionGranularity.DAY),
    ("NUMTODSINTERVAL(7, 'DAY')", PartitionGranularity.WEEK),
    ("NUMTODSINTERVAL(6, 'HOUR')", None),
    ("NUMTOYMINTERVAL(2, 'MONTH')", None),
    (None, None),
    ("", None),
    ("something else entirely", None),
]


@pytest.mark.parametrize(("interval", "expected"), INTERVAL_CASES)
def test_interval_granularity(interval: str | None, expected: PartitionGranularity | None) -> None:
    """Only a whole-month or whole-day interval has a granularity; the rest report ``None``."""
    assert interval_granularity(interval) is expected


# --- type mapping ---------------------------------------------------------------------------


def test_base_type_strips_the_size_suffix() -> None:
    assert base_type("TIMESTAMP(6)") == "TIMESTAMP"
    assert base_type("VARCHAR2") == "VARCHAR2"
    assert base_type("NUMBER") == "NUMBER"
    assert base_type("INTERVAL DAY(2) TO SECOND(6)") == "INTERVAL DAY TO SECOND"


TYPE_CASES: list[tuple[str, int | None, int | None, NormalizedType]] = [
    ("NUMBER", 12, 0, NormalizedType.INTEGER),
    ("NUMBER", 19, 0, NormalizedType.INTEGER),
    ("NUMBER", 12, 2, NormalizedType.DECIMAL),
    ("NUMBER", None, None, NormalizedType.DECIMAL),
    ("VARCHAR2", None, None, NormalizedType.STRING),
    ("CHAR", None, None, NormalizedType.STRING),
    ("CLOB", None, None, NormalizedType.STRING),
    ("TIMESTAMP(6)", None, 6, NormalizedType.TIMESTAMP),
    ("TIMESTAMP(6) WITH TIME ZONE", None, 6, NormalizedType.TIMESTAMP),
    ("DATE", None, None, NormalizedType.DATE),
    ("BINARY_DOUBLE", None, None, NormalizedType.FLOAT),
    ("RAW", 16, None, NormalizedType.BINARY),
    ("JSON", None, None, NormalizedType.JSON),
    ("SDO_GEOMETRY", None, None, NormalizedType.OTHER),
]


@pytest.mark.parametrize(("data_type", "precision", "scale", "expected"), TYPE_CASES)
def test_normalized_type(
    data_type: str, precision: int | None, scale: int | None, expected: NormalizedType
) -> None:
    """A whole ``NUMBER`` is an integer, a fractional one is money: the conversion needs it."""
    assert normalized_type(data_type, precision, scale) is expected


ARROW_CASES: list[tuple[object, int | None, int | None, pa.DataType]] = [
    (oracledb.DB_TYPE_NUMBER, 12, 0, pa.int64()),
    (oracledb.DB_TYPE_NUMBER, 18, 0, pa.int64()),
    # 19 digits can exceed 2**63 - 1, so it stays an exact decimal rather than an int.
    (oracledb.DB_TYPE_NUMBER, 19, 0, pa.decimal128(38, 0)),
    (oracledb.DB_TYPE_NUMBER, 38, 0, pa.decimal128(38, 0)),
    (oracledb.DB_TYPE_NUMBER, 12, 2, pa.decimal128(38, 2)),
    (oracledb.DB_TYPE_NUMBER, None, None, pa.float64()),
    (oracledb.DB_TYPE_VARCHAR, None, None, pa.string()),
    (oracledb.DB_TYPE_CLOB, None, None, pa.string()),
    (oracledb.DB_TYPE_DATE, None, None, pa.timestamp("us")),
    (oracledb.DB_TYPE_TIMESTAMP, None, 6, pa.timestamp("us")),
    (oracledb.DB_TYPE_TIMESTAMP_TZ, None, 6, pa.timestamp("us", tz="UTC")),
    (oracledb.DB_TYPE_BINARY_DOUBLE, None, None, pa.float64()),
    (oracledb.DB_TYPE_BLOB, None, None, pa.binary()),
]


@pytest.mark.parametrize(("type_code", "precision", "scale", "expected"), ARROW_CASES)
def test_arrow_type(
    type_code: object, precision: int | None, scale: int | None, expected: pa.DataType
) -> None:
    """Money keeps its declared scale; an unconstrained ``NUMBER`` is a float, not a rounded int."""
    assert arrow_type(type_code, precision, scale) == expected


# --- bounds ---------------------------------------------------------------------------------

PARTITIONS: list[tuple[str, object, object]] = [
    ("P_INIT", "TIMESTAMP' 2000-01-01 00:00:00'", 0),
    ("SYS_P1", "TIMESTAMP' 2025-06-01 00:00:00'", 613),
    ("SYS_P2", "TIMESTAMP' 2025-07-01 00:00:00'", 606),
    ("SYS_P3", "MAXVALUE", None),
]


def test_bounds_chain_lower_to_the_previous_upper() -> None:
    """A range partition starts where the previous one ended; the first is unbounded below."""
    bounds = _bounds(PARTITIONS)
    assert [bound.name for bound in bounds] == ["P_INIT", "SYS_P1", "SYS_P2", "SYS_P3"]
    assert bounds[0].lower is None
    assert bounds[0].upper == datetime(2000, 1, 1, tzinfo=UTC)
    assert bounds[1].lower == bounds[0].upper
    assert bounds[2].lower == bounds[1].upper
    # MAXVALUE is unbounded above, not a date to be guessed, and the chain keeps the last bound.
    assert bounds[3].upper is None
    assert bounds[3].lower == bounds[2].upper
    assert [bound.row_estimate for bound in bounds] == [0, 613, 606, None]


def test_bounds_carry_list_values() -> None:
    """A LIST partition's ``HIGH_VALUE`` is the value list, so it is reported as ``values``."""
    bounds = _bounds([("P_A", "'A', 'C'", 10), ("P_REST", "DEFAULT", None)], list_values=True)
    assert bounds[0].values == ("A", "C")
    assert bounds[0].upper is None
    assert bounds[1].values == ("DEFAULT",)


def test_list_values_splits_quoted_values() -> None:
    assert _list_values("'active', 'churned'") == ("active", "churned")
    assert _list_values(None) == ()
    assert _list_values("MAXVALUE") == ("MAXVALUE",)


def test_partitioning_reports_strategy_key_and_granularity() -> None:
    """The shape the planner reads: RANGE on one column, monthly, with the catalog's bounds."""
    entry = _partitioning(
        strategy="RANGE",
        subpartitioning_type="NONE",
        interval="NUMTOYMINTERVAL(1, 'MONTH')",
        key=("START_TIME",),
        partitions=PARTITIONS,
        table="MDQM_BILLING.USAGE_EVENTS",
    )
    assert entry.strategy is PartitionStrategy.RANGE
    assert entry.key == ("START_TIME",)
    assert entry.granularity is PartitionGranularity.MONTH
    assert len(entry.bounds) == len(PARTITIONS)


def test_every_oracle_partitioning_shape_has_an_answer() -> None:
    """RANGE/LIST/HASH map onto the model, SYSTEM is an expression, composite is NotSupported."""
    for strategy, expected in (
        ("RANGE", PartitionStrategy.RANGE),
        ("LIST", PartitionStrategy.LIST),
        ("HASH", PartitionStrategy.HASH),
        ("SYSTEM", PartitionStrategy.EXPRESSION),
    ):
        entry = _partitioning(
            strategy=strategy,
            subpartitioning_type="NONE",
            interval=None,
            key=() if strategy == "SYSTEM" else ("ID",),
            partitions=PARTITIONS,
            table="T",
        )
        assert entry.strategy is expected, strategy

    with pytest.raises(Exception) as composite:
        _partitioning(
            strategy="RANGE",
            subpartitioning_type="LIST",
            interval=None,
            key=("START_TIME",),
            partitions=PARTITIONS,
            table="T",
        )
    assert OPTIONAL_METHODS["extract_partitions"] in str(composite.value)

    with pytest.raises(Exception) as unknown:
        _partitioning(
            strategy="REFERENCE",
            subpartitioning_type="NONE",
            interval=None,
            key=("ID",),
            partitions=PARTITIONS,
            table="T",
        )
    assert OPTIONAL_METHODS["extract_partitions"] in str(unknown.value)


# --- the plan summary and the pruning proof -------------------------------------------------

PLAN_LITERAL: list[tuple[Any, ...]] = [
    (0, None, "SELECT STATEMENT", None, None, 1, 5, None, None),
    (1, 0, "SORT", "AGGREGATE", None, 1, None, None, None),
    (2, 1, "PARTITION RANGE", "ITERATOR", None, 1246, 5, "312", "1048575"),
    (3, 2, "INDEX", "RANGE SCAN", "USAGE_EVENTS_START_TIME_IDX", 1246, 5, "312", "1048575"),
]
PLAN_BIND: list[tuple[Any, ...]] = [
    (0, None, "SELECT STATEMENT", None, None, 1, 2, None, None),
    (2, 1, "PARTITION RANGE", "ITERATOR", None, 250, 2, "KEY", "1048575"),
    (3, 2, "INDEX", "RANGE SCAN", "USAGE_EVENTS_START_TIME_IDX", 250, 2, "KEY", "1048575"),
]
PLAN_PLAIN: list[tuple[Any, ...]] = [
    (0, None, "SELECT STATEMENT", None, None, 1, 3, None, None),
    (1, 0, "TABLE ACCESS", "FULL", "ACCOUNTS", 200, 3, None, None),
]


def test_literal_filter_plan_is_proven_pruning() -> None:
    """Numeric ``PSTART``/``PSTOP`` is the optimizer pruning at parse time: ``PROVEN``."""
    summary = plan_summary(PLAN_LITERAL, partitions_total=9)
    assert summary.pruning_proof is PruningProof.PROVEN
    assert summary.partitions_total == 9
    assert summary.partitions_scanned is None, "Oracle's partition numbers are not a count"
    assert summary.uses_index is True
    assert summary.indexes_used == ("USAGE_EVENTS_START_TIME_IDX",)
    assert summary.est_rows == 1.0
    assert summary.est_cost == 5.0


def test_bind_filter_plan_is_unproven_pruning() -> None:
    """``PSTART = KEY``: the value is known at run time only, so ``UNPROVEN``, never ``PROVEN``."""
    summary = plan_summary(PLAN_BIND, partitions_total=9)
    assert summary.pruning_proof is PruningProof.UNPROVEN


def test_a_plan_without_partition_operations_proves_nothing() -> None:
    """A table with no partitions has nothing to prove: ``UNKNOWN``, not ``PROVEN``."""
    summary = plan_summary(PLAN_PLAIN, partitions_total=None)
    assert summary.pruning_proof is PruningProof.UNKNOWN
    assert summary.partitions_total is None
    assert summary.uses_index is False or summary.uses_index is None
    assert summary.indexes_used == ()


def test_an_empty_plan_is_still_a_summary() -> None:
    """``PLAN_TABLE`` empty (a probe that found nothing) must not crash the parse."""
    summary = plan_summary([], partitions_total=None)
    assert summary.pruning_proof is PruningProof.UNKNOWN
    assert summary.est_rows is None and summary.est_cost is None
    assert summary.raw is None


def test_plan_summary_keeps_the_verbatim_plan() -> None:
    """``raw`` is the engine's own text, so a human can check the proof instead of trusting it."""
    summary = plan_summary(PLAN_LITERAL, partitions_total=9, raw="| 2 | PARTITION RANGE | 312 |")
    assert summary.raw is not None and "PARTITION RANGE" in summary.raw


# --- the query store ------------------------------------------------------------------------


def test_raw_queries_turns_cumulative_counters_into_mean_time() -> None:
    """``V$SQL`` counts are cumulative: one row, N calls, a mean derived from the total."""
    rows = [("select 1 from dual", 4, 3925472, 1, datetime(2025, 11, 27, 11, 51, 36))]
    mined = _raw_queries(rows)
    assert len(mined) == 1
    query = mined[0]
    assert query.dialect == "oracle"
    assert query.calls == 4
    assert query.total_time_ms == pytest.approx(3925.472)
    assert query.mean_time_ms == pytest.approx(3925.472 / 4)
    assert query.last_seen == datetime(2025, 11, 27, 11, 51, 36, tzinfo=UTC)
    assert query.user is None, "V$SQL names the parsing schema, not the executor: claim neither"


def test_raw_queries_skips_a_row_without_text() -> None:
    """A ``NULL`` SQL text is not a query; reporting it would be an empty statement."""
    assert _raw_queries([(None, 1, 1000, 0, None)]) == []


# --- error translation ----------------------------------------------------------------------


def _db_error(code: int, message: str) -> oracledb.DatabaseError:
    error = oracledb.DatabaseError.__new__(oracledb.DatabaseError)
    error.args = (_FakeError(code, message),)
    return error


class _FakeError:
    """The shape ``oracledb`` puts in ``DatabaseError.args[0]``: ``.code`` and ``.message``."""

    def __init__(self, code: int, message: str) -> None:
        self.code = code
        self.message = message

    def __str__(self) -> str:
        return self.message


def test_a_cancelled_statement_is_a_cancellation() -> None:
    """``ORA-01013`` after ``cancel()`` is our own cancellation, not a deadline."""
    translated = _stream_error(
        _db_error(1013, "ORA-01013: User requested cancel of current operation."),
        cancelled=True,
        timed_out=False,
        timeout_s=5.0,
    )
    assert isinstance(translated, OracleCancelled)


def test_ora_01013_without_a_cancel_is_a_deadline() -> None:
    """The same error without our ``cancel`` means something else stopped the statement."""
    translated = _stream_error(
        _db_error(1013, "ORA-01013: User requested cancel"),
        cancelled=False,
        timed_out=True,
        timeout_s=5.0,
    )
    assert isinstance(translated, OracleTimeout)
    assert "5.0s" in str(translated)


def test_a_call_timeout_is_a_deadline() -> None:
    """``DPY-4024`` is ``Connection.call_timeout`` firing: the server was told to abort."""
    translated = _stream_error(
        _db_error(0, "DPY-4024: call timeout of 1000 ms exceeded"),
        cancelled=False,
        timed_out=False,
        timeout_s=1.0,
    )
    assert isinstance(translated, OracleTimeout)


def test_any_other_error_is_left_alone() -> None:
    """A refusal is a refusal: the driver's error keeps its code and its message."""
    original = _db_error(41900, 'ORA-41900: missing INSERT privilege on "T"."C"')
    translated = _stream_error(original, cancelled=False, timed_out=False, timeout_s=1.0)
    assert translated is original


def test_a_lost_connection_is_a_deadline_only_when_the_deadline_passed() -> None:
    """``DPY-4011`` after the deadline is the backstop ``call_timeout`` losing the session.

    It is still a deadline for the caller (``OracleTimeout``); without the deadline having passed it
    is a real connection problem and must keep the driver's error.
    """
    lost = _db_error(0, "DPY-4011: the database or network closed the connection")
    assert isinstance(
        _stream_error(lost, cancelled=False, timed_out=True, timeout_s=1.0), OracleTimeout
    )
    assert _stream_error(lost, cancelled=False, timed_out=False, timeout_s=1.0) is lost


# --- configuration ---------------------------------------------------------------------------


class _StaticSecrets:
    def get(self, name: str) -> str:
        assert name
        return "throwaway-dev-password"


def _source(**options: Any) -> SourceConfig:
    return SourceConfig(
        name="billing_oracle", adapter="oracle", secret="billing_ro", options=options
    )


def test_connection_parts_builds_an_easy_connect_dsn() -> None:
    dsn, user, password = _connection_parts(
        _source(host="db.internal", port=1521, service_name="MDQM", user="MDQM_RO"),
        _StaticSecrets(),
    )
    assert dsn == "db.internal:1521/MDQM"
    assert user == "MDQM_RO"
    assert password == "throwaway-dev-password"


def test_connection_parts_accepts_a_whole_dsn() -> None:
    dsn, _, _ = _connection_parts(
        _source(dsn="db.internal:1521/MDQM", user="MDQM_RO"), _StaticSecrets()
    )
    assert dsn == "db.internal:1521/MDQM"


CONFIG_ERRORS: list[tuple[dict[str, Any], str, str]] = [
    ({"host": "h", "service_name": "s"}, "options.user", "no user"),
    ({"user": "u"}, "options.host", "no host"),
    ({"host": "h", "user": "u"}, "options.service_name", "no service"),
    ({"host": "h", "service_name": "s", "user": "u", "port": "abc"}, "options.port", "bad port"),
]


@pytest.mark.parametrize(("options", "expected", "label"), CONFIG_ERRORS)
def test_configuration_errors_name_the_offending_key(
    options: dict[str, Any], expected: str, label: str
) -> None:
    with pytest.raises(OracleConfigError) as excinfo:
        _connection_parts(_source(**options), _StaticSecrets())
    assert expected in str(excinfo.value), label


def test_a_password_must_come_from_a_secret_reference() -> None:
    """A source with no ``secret:`` cannot log in, and the error says how to fix it."""
    config = SourceConfig(
        name="billing_oracle",
        adapter="oracle",
        options={"host": "h", "service_name": "s", "user": "u"},
    )
    with pytest.raises(OracleConfigError) as excinfo:
        _connection_parts(config, None)
    assert "secret" in str(excinfo.value)


def test_a_secret_reference_needs_a_provider() -> None:
    with pytest.raises(OracleConfigError) as excinfo:
        _connection_parts(_source(host="h", service_name="s", user="u"), None)
    assert "secret provider" in str(excinfo.value)


def test_the_adapter_never_shows_the_password() -> None:
    """``repr`` names the source and the user: the password must never reach a log line."""
    adapter = OracleAdapter(
        _source(host="h", service_name="s", user="MDQM_RO"),
        _StaticSecrets(),
    )
    try:
        assert "throwaway-dev-password" not in repr(adapter)
        assert adapter.kind == "oracle"
        assert adapter.dialect() == "oracle"
        assert adapter.source == "billing_oracle"
    finally:
        adapter.close()


def test_the_wrong_adapter_kind_is_a_config_error() -> None:
    config = SourceConfig(
        name="x", adapter="postgresql", options={"host": "h", "service_name": "s", "user": "u"}
    )
    with pytest.raises(OracleConfigError):
        OracleAdapter(config, _StaticSecrets())
