"""Unit tests for the Oracle loader's pure logic (task D05).

No Docker, no network: everything here is a table of cases over functions that only reshape
values. The database claims (partitions, privileges, PSTART/PSTOP) live in
``tests/integration/test_seed_oracle.py`` because they can only be asserted against Oracle itself.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

pytest.importorskip("oracledb", reason="the `oracle` extra is not installed")

from mdqm.testing.loaders.oracle import (
    GENERATED_PARTITION_PREFIX,
    INITIAL_PARTITION,
    LoaderError,
    OracleTarget,
    Partition,
    _batched,
    find_dotenv,
    generation_expected_months,
    month_start,
    parse_high_value,
    read_dotenv,
    target_from_environ,
    to_stored_timestamp,
)
from mdqm.testing.world import WorldConfig, build_world


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("TIMESTAMP' 2025-07-01 00:00:00'", datetime(2025, 7, 1)),
        ("timestamp' 2026-01-01 00:00:00'", datetime(2026, 1, 1)),
        ("  TIMESTAMP' 2000-01-01 00:00:00'  ", datetime(2000, 1, 1)),
        ("2025-07-01 00:00:00", datetime(2025, 7, 1)),
        ("MAXVALUE", None),
        ("", None),
        ("not a date", None),
    ],
)
def test_parse_high_value(text: str, expected: datetime | None) -> None:
    """The catalog hands ``HIGH_VALUE`` over as a LONG string; the loader needs the instant."""
    assert parse_high_value(text) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 1, 1)),
        (datetime(2025, 6, 30, 21, 0, tzinfo=UTC), datetime(2025, 6, 30, 21, 0)),
        (datetime(2025, 6, 30, 21, 0), None),
    ],
)
def test_to_stored_timestamp(value: datetime, expected: datetime | None) -> None:
    """Aware UTC becomes naive UTC (Oracle cannot interval-partition a WITH TIME ZONE column)."""
    if expected is None:
        with pytest.raises(LoaderError, match="naive timestamp"):
            to_stored_timestamp(value)
    else:
        assert to_stored_timestamp(value) == expected


def test_month_start_and_generated_flag() -> None:
    assert month_start(datetime(2025, 5, 17, 23, 59)) == datetime(2025, 5, 1)
    assert Partition("SYS_P661", "x", None).generated
    assert Partition(INITIAL_PARTITION, "x", None).generated is False
    assert not INITIAL_PARTITION.startswith(GENERATED_PARTITION_PREFIX)


def test_generation_expected_months_is_the_data_months() -> None:
    """The number Oracle must derive is the number of distinct event months, no more."""
    world = build_world(WorldConfig.small())
    months = generation_expected_months(world)
    assert list(months) == sorted(set(months)), "unique and ordered"
    assert all(month == month_start(month) for month in months)
    assert len(months) == len(
        {(event.start_time.year, event.start_time.month) for event in world.billing.usage_events}
    )


@pytest.mark.parametrize(
    ("count", "size", "batches"), [(0, 10, []), (5, 10, [5]), (25, 10, [10, 10, 5])]
)
def test_batched(count: int, size: int, batches: list[int]) -> None:
    """Batches are lists (``executemany`` rejects a tuple of tuples with DPY-2004)."""
    rows = [(index,) for index in range(count)]
    out = list(_batched(rows, size))
    assert [len(batch) for batch in out] == batches
    assert all(isinstance(batch, list) for batch in out)
    assert [row for batch in out for row in batch] == rows


def test_read_dotenv_ignores_comments_and_quotes(tmp_path: Path) -> None:
    path = tmp_path / "env"
    path.write_text('# comment\nA=1\nB="two"\n\nC=3\n', encoding="utf-8")
    assert read_dotenv(path) == {"A": "1", "B": "two", "C": "3"}
    assert read_dotenv(tmp_path / "missing") == {}


def test_find_dotenv_walks_up_from_a_worktree(tmp_path: Path) -> None:
    """A checkout under ``.worktrees/<task>`` reuses the parent's credentials file."""
    (tmp_path / ".env").write_text("A=1\n", encoding="utf-8")
    nested = tmp_path / ".worktrees" / "D05b"
    nested.mkdir(parents=True)
    assert find_dotenv(nested) == tmp_path / ".env"


def test_target_from_environ_names_what_is_missing(tmp_path: Path) -> None:
    """A missing password is a ``LoaderError`` naming the variable and the file it looked at."""
    empty = tmp_path / "none"
    with pytest.raises(LoaderError, match="MDQM_ORACLE_PASSWORD"):
        target_from_environ(environ={}, dotenv_path=empty)


def test_target_from_environ_reads_values_and_rejects_a_bad_port(tmp_path: Path) -> None:
    path = tmp_path / "env"
    path.write_text(
        "MDQM_ORACLE_PASSWORD=devpass\nMDQM_RO_PASSWORD=ro\nMDQM_ORACLE_PORT=51521\n",
        encoding="utf-8",
    )
    target = target_from_environ(environ={}, dotenv_path=path)
    assert target == OracleTarget(
        host="127.0.0.1",
        port=51521,
        admin_password="devpass",
        ro_password="ro",
    )
    assert target.dsn == "127.0.0.1:51521/MDQM"
    assert target.owner_password == "devpass"

    path.write_text("MDQM_ORACLE_PASSWORD=devpass\nMDQM_ORACLE_PORT=not-a-port\n", encoding="utf-8")
    with pytest.raises(LoaderError, match="MDQM_ORACLE_PORT must be an integer"):
        target_from_environ(environ={}, dotenv_path=path)
