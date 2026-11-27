"""Unit tests for the SQL Server adapter's pure logic (task A08). No driver, no network, no Docker.

Everything here is a table of cases over functions that take a value and return a value: the plan
parser, the partition-bound reconstruction, the type mapping and the connection-string builder. The
live behaviour (catalog, streaming, cancel, the plan XML compared with an actual plan) is proved in
``tests/integration/test_mssql_adapter.py``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pyarrow as pa
import pytest

from mdqm.adapters.base import NotSupported
from mdqm.adapters.sqlserver import (
    SqlServerAdapter,
    SqlServerCancelled,
    SqlServerConfigError,
    SqlServerTimeout,
    _parse_boundary,
    _stream_error,
    arrow_type,
    granularity,
    normalized_type,
    parse_partition_bounds,
    plan_summary,
)
from mdqm.config.settings import SourceConfig
from mdqm.models import NormalizedType, PartitionGranularity, PruningProof

NOW = datetime(2026, 1, 1, tzinfo=UTC)
"""A fixed ``exec_ts``: the adapter never substitutes the clock, so a constant is honest."""

SHOWPLAN_DOCUMENT = """<ShowPlanXML xmlns="http://schemas.microsoft.com/sqlserver/2004/07/showplan"
  Version="1.564" Build="16.0.4135.4"><BatchSequence><Batch><Statements>
  <StmtSimple StatementText="SELECT ..." StatementType="SELECT" StatementSubTreeCost="0.44"
    StatementEstRows="12.5" StatementOptmLevel="FULL">
  <QueryPlan><RelOp NodeId="0" PhysicalOp="Hash Match" LogicalOp="Inner Join" Partitioned="0">
  <RelOp NodeId="1" PhysicalOp="Clustered Index Scan" LogicalOp="Clustered Index Scan"
    Partitioned="1"><IndexScan><Object Database="[erp]" Schema="[dbo]" Table="[usage_events]"
      Index="[usage_events_pk]" IndexKind="Clustered"/><SeekPredicates><SeekPredicateNew><SeekKeys>
      <StartRange ScanType="GE"><RangeColumns>
        <ColumnReference Column="PtnId1000"/></RangeColumns>
        <RangeExpressions><ScalarOperator ScalarString="(9)"/></RangeExpressions></StartRange>
      <EndRange ScanType="LE"><RangeColumns>
        <ColumnReference Column="PtnId1000"/></RangeColumns>
        <RangeExpressions><ScalarOperator ScalarString="(11)-(1)"/></RangeExpressions></EndRange>
    </SeekKeys></SeekPredicateNew></SeekPredicates></IndexScan></RelOp>
  </RelOp></QueryPlan></StmtSimple></Statements></Batch></BatchSequence></ShowPlanXML>"""

BOUND_DOCUMENT = """<ShowPlanXML xmlns="http://schemas.microsoft.com/sqlserver/2004/07/showplan">
  <BatchSequence><Batch><Statements><StmtSimple StatementEstRows="1" StatementSubTreeCost="0.5">
  <QueryPlan><RelOp NodeId="0" PhysicalOp="Clustered Index Scan" Partitioned="1"><IndexScan>
  <Object Database="[erp]" Schema="[dbo]" Table="[usage_events]" Index="[usage_events_pk]"/>
  <SeekPredicates><SeekPredicateNew><SeekKeys>
  <StartRange ScanType="GE"><RangeColumns><ColumnReference Column="PtnId1000"/></RangeColumns>
  <RangeExpressions><ScalarOperator
    ScalarString="RangePartitionNew([@1],(1),'2024-11-01')"/></RangeExpressions></StartRange>
  </SeekKeys></SeekPredicateNew></SeekPredicates></IndexScan></RelOp></QueryPlan></StmtSimple>
  </Statements></Batch></BatchSequence></ShowPlanXML>"""

UNPARTITIONED_DOCUMENT = """<ShowPlanXML xmlns="http://schemas.microsoft.com/sqlserver/2004/07/showplan">
  <BatchSequence><Batch><Statements><StmtSimple StatementEstRows="3" StatementSubTreeCost="0.01">
  <QueryPlan><RelOp NodeId="0" PhysicalOp="Table Scan" Partitioned="0"><TableScan>
  <Object Database="[erp]" Schema="[dbo]" Table="[customers]" IndexKind="Heap"/>
  </TableScan></RelOp></QueryPlan></StmtSimple></Statements></Batch></BatchSequence></ShowPlanXML>"""


# --- the plan parser --------------------------------------------------------------------------


def test_a_literal_partition_range_is_proven_with_a_counted_scan() -> None:
    """``(9)`` to ``(11)-(1)`` is partitions 9 and 10: two scanned, and the plan says so."""
    summary = plan_summary(SHOWPLAN_DOCUMENT, partitions_total=18)
    assert summary.pruning_proof is PruningProof.PROVEN
    assert summary.partitions_scanned == 2
    assert summary.partitions_total == 18
    assert summary.uses_index is True
    assert summary.indexes_used == ("[usage_events_pk]",)
    assert summary.est_rows == pytest.approx(12.5)
    assert summary.est_cost == pytest.approx(0.44)
    assert summary.raw is not None and "ShowPlanXML" in summary.raw


def test_a_parameterised_partition_range_is_unproven() -> None:
    """``RangePartitionNew(...)`` is a runtime expression: the ends are unknown before execution."""
    summary = plan_summary(BOUND_DOCUMENT, partitions_total=18)
    assert summary.pruning_proof is PruningProof.UNPROVEN
    assert summary.partitions_scanned is None
    assert summary.partitions_total == 18
    assert summary.indexes_used == ("[usage_events_pk]",)


def test_a_plan_without_a_partitioned_table_has_nothing_to_prove() -> None:
    """``UNKNOWN`` is for "there is nothing to prove", never for "I could not tell"."""
    summary = plan_summary(UNPARTITIONED_DOCUMENT, partitions_total=None)
    assert summary.pruning_proof is PruningProof.UNKNOWN
    assert summary.partitions_scanned is None
    assert summary.partitions_total is None
    assert summary.uses_index is None
    assert summary.indexes_used == ()


def test_the_summary_keeps_the_plan_verbatim_but_bounded() -> None:
    """``raw`` is the document itself, cut at the limit so a wide plan cannot flood the caller."""
    summary = plan_summary(UNPARTITIONED_DOCUMENT)
    assert summary.raw == UNPARTITIONED_DOCUMENT
    huge = UNPARTITIONED_DOCUMENT.replace("</ShowPlanXML>", " " * 40_000 + "</ShowPlanXML>")
    bounded = plan_summary(huge)
    assert bounded.raw is not None
    assert len(bounded.raw) < len(huge)
    assert bounded.raw.endswith("<!-- truncated -->")


# --- partition bounds -------------------------------------------------------------------------


def test_n_tuples_define_n_plus_one_half_open_partitions() -> None:
    """RANGE RIGHT: the tuple is the lower bound of the partition to its right."""
    tuples = (
        datetime(2024, 11, 1, tzinfo=UTC),
        datetime(2024, 12, 1, tzinfo=UTC),
        datetime(2025, 1, 1, tzinfo=UTC),
    )
    bounds = parse_partition_bounds(
        scheme="ps_usage",
        fanout=4,
        boundaries=tuples,
        filegroups={1: "PRIMARY", 2: "PRIMARY", 3: "PRIMARY", 4: "PRIMARY"},
        table="dbo.usage_events",
        row_counts={("dbo", "usage_events", 1): 0, ("dbo", "usage_events", 4): 7},
    )
    assert [bound.lower for bound in bounds] == [None, *tuples]
    assert [bound.upper for bound in bounds] == [*tuples, None]
    assert bounds[0].row_estimate == 0
    assert bounds[3].row_estimate == 7
    assert bounds[0].name == "ps_usage.1 (PRIMARY)"


def test_a_fanout_that_does_not_match_the_tuples_is_refused() -> None:
    """A planner pruning on a wrong count is worse than no metadata: refuse, do not patch."""
    with pytest.raises(NotSupported, match="fanout"):
        parse_partition_bounds(
            scheme="ps_usage",
            fanout=5,
            boundaries=(datetime(2024, 11, 1, tzinfo=UTC),),
            filegroups={},
            table="dbo.usage_events",
        )


@pytest.mark.parametrize(
    ("boundaries", "expected"),
    [
        (
            (datetime(2024, 11, 1), datetime(2024, 12, 1), datetime(2025, 1, 1)),
            PartitionGranularity.MONTH,
        ),
        (
            (datetime(2024, 11, 1), datetime(2024, 11, 2), datetime(2024, 11, 3)),
            PartitionGranularity.DAY,
        ),
        (
            (datetime(2024, 11, 1), datetime(2024, 11, 8), datetime(2024, 11, 15)),
            PartitionGranularity.WEEK,
        ),
        ((datetime(2024, 11, 1), datetime(2024, 12, 15)), None),
        ((datetime(2024, 11, 1),), None),
        ((), None),
    ],
)
def test_granularity_is_reported_only_when_every_step_is_the_same(
    boundaries: tuple[datetime, ...], expected: PartitionGranularity | None
) -> None:
    """A monthly function steps by one calendar month, which is 28..31 days long."""
    assert granularity(boundaries) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("2024-11-01T00:00:00", datetime(2024, 11, 1, tzinfo=UTC)),
        ("2024-11-01T00:00:00.1234560", datetime(2024, 11, 1, 0, 0, 0, 123456, tzinfo=UTC)),
        ("2024-11-01", datetime(2024, 11, 1, tzinfo=UTC)),
        ("2024-11-01T00:00:00Z", datetime(2024, 11, 1, tzinfo=UTC)),
        ("Nov  1 2024 12:00AM", None),
        ("", None),
        ("5", None),
    ],
)
def test_boundary_parsing_never_guesses(text: str, expected: datetime | None) -> None:
    """Only the ISO shapes ``CONVERT(..., 126)`` produces are accepted; else ``None``."""
    assert _parse_boundary(text) == expected


# --- type mapping -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("type_name", "expected"),
    [
        ("nvarchar", NormalizedType.STRING),
        ("datetime2", NormalizedType.TIMESTAMP),
        ("date", NormalizedType.DATE),
        ("time", NormalizedType.TIME),
        ("bigint", NormalizedType.INTEGER),
        ("bit", NormalizedType.BOOLEAN),
        ("money", NormalizedType.DECIMAL),
        ("varbinary", NormalizedType.BINARY),
        ("uniqueidentifier", NormalizedType.UUID),
        ("hierarchyid", NormalizedType.OTHER),
        ("something_new", NormalizedType.OTHER),
    ],
)
def test_normalized_type(type_name: str, expected: NormalizedType) -> None:
    """SQL Server's vocabulary maps onto the portable one, and the unknown lands in OTHER."""
    assert normalized_type(type_name) is expected


@pytest.mark.parametrize(
    ("py_type", "precision", "scale", "expected"),
    [
        (int, 10, 0, pa.int64()),
        (str, None, None, pa.string()),
        (bytes, None, None, pa.binary()),
        (datetime, None, None, pa.timestamp("us", tz="UTC")),
        (Decimal, 10, 2, pa.decimal128(10, 2)),
        (Decimal, None, None, pa.string()),  # no precision: text, never a made-up decimal
        (object, None, None, pa.string()),
    ],
)
def test_arrow_type(
    py_type: object, precision: int | None, scale: int | None, expected: pa.DataType
) -> None:
    """``mssql-python`` describes a column with the Python class it produces, so we map that."""
    assert arrow_type(py_type, precision, scale) == expected


# --- errors -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("message", "cancelled", "expected"),
    [
        (
            "Driver Error: Operation canceled; DDBC Error: [Microsoft]Operation canceled",
            True,
            SqlServerCancelled,
        ),
        (
            "Driver Error: Operation canceled; DDBC Error: [Microsoft]Operation canceled",
            False,
            SqlServerTimeout,
        ),
        (
            "Driver Error: Timeout expired; DDBC Error: [Microsoft]Query timeout expired",
            False,
            SqlServerTimeout,
        ),
        (
            "Driver Error: Syntax error; DDBC Error: [Microsoft][SQL Server]The INSERT permission "
            "was denied",
            False,
            Exception,
        ),
    ],
)
def test_stream_error_translation(message: str, cancelled: bool, expected: type[Exception]) -> None:
    """A cancellation and a deadline are answers about this execution, not about the source."""
    translated = _stream_error(
        Exception(message), cancelled=cancelled, timed_out=False, timeout_s=1.0
    )
    assert type(translated) is expected
    assert not isinstance(translated, NotSupported)


def test_a_watchdog_that_fired_is_a_deadline_even_for_a_plain_driver_error() -> None:
    """When our own watchdog fired the deadline is the reason; saying so beats a raw error."""
    translated = _stream_error(
        Exception("connection reset by peer"), cancelled=False, timed_out=True, timeout_s=2.0
    )
    assert isinstance(translated, SqlServerTimeout)


# --- configuration ----------------------------------------------------------------------------


def _config(**options: object) -> SourceConfig:
    return SourceConfig(
        name="erp_mssql",
        adapter="sqlserver",
        secret="erp_ro",
        options=options,
    )


def test_config_error_names_the_missing_key_never_a_value() -> None:
    """A configuration problem is an operator's fix, so the message names the key."""
    with pytest.raises(SqlServerConfigError, match="options.user"):
        SqlServerAdapter(_config(database="erp"), _StaticSecrets("unused"))
    with pytest.raises(SqlServerConfigError, match="options.database"):
        SqlServerAdapter(_config(host="127.0.0.1", user="mdqm_ro"), _StaticSecrets("unused"))
    with pytest.raises(SqlServerConfigError, match="no secret provider"):
        SqlServerAdapter(_config(host="h", database="d", user="u"), None)
    with pytest.raises(SqlServerConfigError, match="positive integer"):
        SqlServerAdapter(
            _config(host="h", database="d", user="u", port=0), _StaticSecrets("unused")
        )


def test_the_adapter_refuses_another_adapters_config() -> None:
    """A mismatch between ``adapter:`` and the class is a wiring bug, not a runtime surprise."""
    with pytest.raises(SqlServerConfigError, match="names adapter"):
        SqlServerAdapter(
            SourceConfig(name="x", adapter="oracle", secret="s", options={}),
            _StaticSecrets("unused"),
        )


def test_repr_never_carries_the_password() -> None:
    """The connection string holds the password; ``repr`` must stay safe to log."""
    adapter = SqlServerAdapter(
        _config(host="mssql.internal", port=1433, database="erp", user="mdqm_ro"),
        _StaticSecrets("hunter2"),
    )
    assert "hunter2" not in repr(adapter)
    assert "mssql.internal" in repr(adapter)
    assert adapter.kind == "sqlserver"
    assert adapter.dialect() == "tsql"


def test_a_password_never_reaches_the_config_error() -> None:
    """The secret is resolved at connect time, so a config error cannot quote it."""
    try:
        SqlServerAdapter(_config(host="h", user="u"), _StaticSecrets("hunter2"))
    except SqlServerConfigError as exc:
        assert "hunter2" not in str(exc)
    else:  # pragma: no cover - the call above must fail
        raise AssertionError("a config without options.database must be refused")


class _StaticSecrets:
    """A secret provider returning one throwaway dev password for any reference."""

    def __init__(self, password: str) -> None:
        self._password = password

    def get(self, name: str) -> str:
        """Resolve any reference to the throwaway dev password (the name is what config carries)."""
        assert name, "a secret reference must be a name, never an empty string"
        return self._password
