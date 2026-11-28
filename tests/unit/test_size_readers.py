"""The three non-PostgreSQL size readers: what each engine's row shape becomes.

None of these runs a database call, on purpose. A reader is two halves -- the statement and the
parse -- and only the parse can be proven here, from captured row shapes (an Oracle ``ALL_TABLES``
row, a SQL Server ``sys.dm_db_partition_stats`` row, an Elasticsearch ``_cat/indices`` row). The
statement half is pinned by parsing it with sqlglot and running it through the read-only guard, so a
typo that makes it un-parsable or a ``DELETE`` that sneaks in fails here.

The live query is **unverified against a real Oracle, SQL Server or Elasticsearch**: no engine was
running when this was written. What is verified is that a captured row means what the adapter says
it means, including the cases that matter most -- bytes the account may not read, ``Decimal`` rows
from Oracle, ``_cat``'s ``"-"`` for "no figure", and the partition name matching the bound the probe
reports.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import pytest
import sqlglot

from mdqm.adapters.elastic import EsTable, parse_index_sizes, table_sizes
from mdqm.adapters.oracle import _SIZES_SQL as ORACLE_SIZES_SQL
from mdqm.adapters.sizes import SIZE_COLUMNS, parse_sizes
from mdqm.adapters.sqlserver import (
    _SIZES_SQL as MSSQL_SIZES_SQL,
)
from mdqm.adapters.sqlserver import (
    parse_partition_bounds,
    parse_size_rows,
    partition_name,
)
from mdqm.sql.guard import guard

NOW = datetime(2026, 11, 28, tzinfo=UTC)


# --- Oracle: the shared seven-column shape ------------------------------------------------


def test_oracle_rows_report_bytes_it_may_not_read_as_unknown_not_zero() -> None:
    """A read-only account is usually denied ``ALL_SEGMENTS``: NULL there means "not reported"."""
    (table,) = parse_sizes(
        [
            ("BILLING", "CUSTOMERS", "CUSTOMERS", 0, 0, Decimal(2048), None),
        ],
        source="billing_oracle",
        captured_at=NOW,
    )

    assert (table.table, table.rows, table.bytes) == ("CUSTOMERS", 2048, None)
    assert table.partitions == ()


def test_oracle_reports_a_partitioned_parent_without_rows_of_its_own() -> None:
    """``ALL_TABLES`` keeps global statistics for the parent; the partitions are the measurement."""
    (table,) = parse_sizes(
        [
            ("BILLING", "USAGE_EVENTS", "USAGE_EVENTS", 0, 1, None, Decimal(0)),
            ("BILLING", "USAGE_EVENTS", "P_2026_01", 1, 1, Decimal(40_000), Decimal(14_000_000)),
            ("BILLING", "USAGE_EVENTS", "P_2026_02", 1, 1, Decimal(50_006), None),
        ],
        source="billing_oracle",
        captured_at=NOW,
    )

    assert (table.rows, table.bytes) == (None, None)
    assert [(item.name, item.rows, item.bytes) for item in table.partitions] == [
        ("P_2026_01", 40_000, 14_000_000),
        ("P_2026_02", 50_006, None),
    ]


def test_oracle_a_number_the_dictionary_did_not_report_is_not_zero() -> None:
    """``NUM_ROWS`` is NULL until the table is analyzed; that is "unknown", never an empty table."""
    (table,) = parse_sizes(
        [("BILLING", "FRESH", "FRESH", 0, 0, None, None)],
        source="billing_oracle",
        captured_at=NOW,
    )

    assert (table.rows, table.bytes) == (None, None)


def test_oracle_sizes_statement_parses_and_is_read_only() -> None:
    """The statement half: one Oracle statement, accepted by sqlglot and by the guard."""
    assert len(sqlglot.parse(ORACLE_SIZES_SQL, read="oracle")) == 1
    checked = guard(ORACLE_SIZES_SQL, "oracle")
    assert checked.ok, checked.error


# --- SQL Server: a partition has a number, not a name -------------------------------------


def test_a_table_without_a_partition_scheme_reports_its_own_rows_and_bytes() -> None:
    (table,) = parse_size_rows(
        [("dbo", "customers", 1, None, None, 2_000, 1_100_000)],
        source="erp_mssql",
        captured_at=NOW,
    )

    assert (table.table, table.rows, table.bytes, table.partitions) == (
        "customers",
        2_000,
        1_100_000,
        (),
    )


def test_a_partitioned_table_names_its_partitions_the_way_the_bounds_do() -> None:
    """One spelling for one partition: a size no bound can be matched to is a size nobody uses."""
    rows = [
        ("dbo", "usage_events", 1, "ps_month", "PRIMARY", 1_000, 8_192_000),
        ("dbo", "usage_events", 2, "ps_month", None, 20_000, 160_000_000),
    ]

    (table,) = parse_size_rows(rows, source="erp_mssql", captured_at=NOW)

    assert (table.rows, table.bytes) == (None, None), (
        "a partitioned table has no storage of its own"
    )
    assert [item.name for item in table.partitions] == [
        partition_name("ps_month", 1, "PRIMARY"),
        partition_name("ps_month", 2, None),
    ]


def test_the_partition_name_matches_the_bounds_the_probe_reports() -> None:
    """The cross-check that matters: the same scheme and filegroups produce the same names."""
    bounds = parse_partition_bounds(
        scheme="ps_month",
        fanout=3,
        boundaries=(datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 2, 1, tzinfo=UTC)),
        filegroups={1: "PRIMARY", 2: "PRIMARY", 3: "ARCHIVE"},
    )

    (table,) = parse_size_rows(
        [
            ("dbo", "usage_events", number, "ps_month", name, 0, 0)
            for number, name in ((1, "PRIMARY"), (2, "PRIMARY"), (3, "ARCHIVE"))
        ],
        source="erp_mssql",
        captured_at=NOW,
    )

    assert [item.name for item in table.partitions] == [bound.name for bound in bounds]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (0, 0),
        (8_192, 8_192),
        (None, None),  # the view returned no row for that index
        ("16384", 16_384),
        (-1, None),  # a negative page count is not a size
    ],
    ids=["zero", "int", "absent", "text", "negative"],
)
def test_a_sqlserver_size_is_read_leniently_but_never_invented(
    raw: object, expected: int | None
) -> None:
    (table,) = parse_size_rows(
        [("dbo", "t", 1, None, None, raw, raw)], source="erp_mssql", captured_at=NOW
    )

    assert (table.rows, table.bytes) == (expected, expected)


def test_sqlserver_sizes_statement_parses_and_is_read_only() -> None:
    assert len(sqlglot.parse(MSSQL_SIZES_SQL, read="tsql")) == 1
    checked = guard(MSSQL_SIZES_SQL, "tsql")
    assert checked.ok, checked.error


# --- Elasticsearch: _cat JSON -------------------------------------------------------------


def _cat_row(name: str, docs: object, size: object, status: str = "open") -> dict[str, Any]:
    return {"index": name, "docs.count": docs, "store.size": size, "status": status}


def test_cat_rows_become_index_sizes_and_a_missing_figure_stays_missing() -> None:
    """``_cat`` writes every number as text and ``"-"`` when the cluster has no figure."""
    sizes = parse_index_sizes(
        [
            _cat_row("usage_events_2026_01", "1000", "2048000"),
            _cat_row("usage_events_2026_02", "2000", "-"),
            _cat_row(".kibana", "5", "1024"),
            _cat_row("closed_index", "5", "1024", status="close"),
        ]
    )

    assert sizes == {
        "usage_events_2026_01": (1_000, 2_048_000),
        "usage_events_2026_02": (2_000, None),
    }


def test_an_alias_is_a_table_and_its_members_are_its_partitions() -> None:
    """The alias has no storage of its own; the monthly indices behind it do."""
    tables = (
        EsTable(
            name="usage_events",
            members=("usage_events_2026_01", "usage_events_2026_02"),
            fields=(),
            docs=3_000,
            aliased=True,
            time_field=None,
            granularity=None,
        ),
        EsTable(
            name="customers",
            members=("customers",),
            fields=(),
            docs=10,
            aliased=False,
            time_field=None,
            granularity=None,
        ),
    )
    sizes = {
        "usage_events_2026_01": (1_000, 2_048_000),
        "usage_events_2026_02": (2_000, None),
        "customers": (10, 16_384),
    }

    alias, lone = table_sizes(tables, sizes, source="es", captured_at=NOW)

    assert (alias.table, alias.rows, alias.bytes) == ("usage_events", None, None)
    assert [(item.name, item.rows, item.bytes) for item in alias.partitions] == [
        ("usage_events_2026_01", 1_000, 2_048_000),
        ("usage_events_2026_02", 2_000, None),
    ]
    assert (lone.table, lone.rows, lone.bytes, lone.partitions) == ("customers", 10, 16_384, ())


def test_an_alias_whose_member_was_not_reported_keeps_the_measured_ones() -> None:
    """A member the ``_cat`` answer omitted is absent from the partitions, not reported as zero."""
    tables = (
        EsTable(
            name="usage_events",
            members=("usage_events_2026_01", "usage_events_2026_02"),
            fields=(),
            docs=1_000,
            aliased=True,
            time_field=None,
            granularity=None,
        ),
    )

    (only,) = table_sizes(
        tables, {"usage_events_2026_01": (1_000, 2_048_000)}, source="es", captured_at=NOW
    )

    assert [item.name for item in only.partitions] == ["usage_events_2026_01"]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, None),
        ("-", None),
        ("", None),
        ("not a number", None),
        ("-4", None),
        (True, None),
        ("4096", 4_096),
    ],
    ids=["absent", "dash", "empty", "text", "negative", "boolean", "number"],
)
def test_a_cat_figure_that_is_not_a_number_is_unknown(raw: object, expected: int | None) -> None:
    sizes = parse_index_sizes([_cat_row("some_index", raw, raw)])

    assert sizes["some_index"] == (expected, expected)


def test_the_shared_row_shape_is_exactly_seven_columns() -> None:
    """The adapters write their queries against this list; a drift here breaks both engines."""
    assert SIZE_COLUMNS == (
        "schema_name",
        "table_name",
        "relation_name",
        "is_partition",
        "is_partitioned_parent",
        "rows",
        "bytes",
    )
    rows: Sequence[Sequence[Any]] = [("s", "t", "t", 0, 0, 1, 2)]
    assert parse_sizes(rows, source="x", captured_at=NOW)[0].rows == 1
