"""M1: DuckDB must spill into the cache tree, not into the working directory.

DuckDB's default ``temp_directory`` is ``.tmp`` *relative to the process CWD*, which is ``/app`` in
the image and read-only there, and ``TMPDIR`` does not move it. Neither component used to set it
unless a caller passed ``spill_dir``, and no caller did, so a spill on a read-only root filesystem
failed the query.

Each test builds the component through its production constructor path (no ``spill_dir`` argument)
and asks DuckDB itself what it applied: that is provable in-process, with no container and no
cluster. ``MDQM_PATHS__CACHE_DIR`` is the operator-facing knob, so it is what moves the answer.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import duckdb
import pytest

from mdqm.cache.segments import SegmentCache
from mdqm.executor.federate import FederatedExecutor
from mdqm.governor.core import Governor
from mdqm.semantic.models import SemanticLayer

TEMP_DIRECTORY = "SELECT current_setting('temp_directory')"


@pytest.fixture
def cache_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """The cache root the settings resolve, so the expected spill dir is known exactly."""
    root = tmp_path / "cache"
    monkeypatch.setenv("MDQM_PATHS__CACHE_DIR", str(root))
    yield root / "spill"
    for leftover in tmp_path.rglob("*"):  # a data file locked by Windows cannot be unlinked
        if leftover.is_file():
            leftover.unlink(missing_ok=True)


def test_duckdb_default_is_the_working_directory() -> None:
    """Why this matters: the default is not a temp dir the platform gives us."""
    connection = duckdb.connect(":memory:")
    try:
        assert connection.execute(TEMP_DIRECTORY).fetchone() == (".tmp",)
    finally:
        connection.close()


def test_segment_cache_sets_the_spill_directory(cache_dir: Path) -> None:
    store = SegmentCache()
    try:
        applied = store._con.execute(TEMP_DIRECTORY).fetchone()
    finally:
        store.close()

    assert applied is not None
    assert Path(applied[0]) == cache_dir
    assert cache_dir.is_dir(), "the directory must exist before DuckDB spills into it"
    assert applied[0] != ".tmp"


def test_federated_executor_sets_the_spill_directory(cache_dir: Path) -> None:
    governor = Governor({})  # no source is contacted: the constructor only opens DuckDB
    runner = FederatedExecutor(governor, layer=SemanticLayer())
    try:
        applied = runner._con.execute(TEMP_DIRECTORY).fetchone()
    finally:
        runner.close()

    assert applied is not None
    assert Path(applied[0]) == cache_dir
    assert cache_dir.is_dir()


def test_an_explicit_spill_dir_still_wins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests and one-off jobs pass their own: the settings default must not override it."""
    monkeypatch.setenv("MDQM_PATHS__CACHE_DIR", str(tmp_path / "cache"))
    explicit = tmp_path / "elsewhere"

    store = SegmentCache(spill_dir=explicit)
    try:
        applied = store._con.execute(TEMP_DIRECTORY).fetchone()
    finally:
        store.close()

    assert applied is not None
    assert Path(applied[0]) == explicit
