"""Stage D5: profiling a table cheaply -- optimizer statistics first, a bounded sample second.

DESIGN.md section 10 puts profiling among the *heavy* stages, and AGENTS.md principle 6 says a
primary database with no replica pays for every row we read. So this stage has one rule above all:
**never scan a table to learn something the optimizer already knows.**

Two families of fact, and each fact records which one answered it (:class:`Provenance`):

* **distribution** -- null rate, distinct count, min/max, top-N values. PostgreSQL's ``pg_stats``
  (``null_frac``, ``n_distinct``, ``histogram_bounds``, ``most_common_vals``/``most_common_freqs``)
  already holds all four, and so do the equivalents on Oracle, SQL Server and MySQL. They are read
  through :class:`OptimizerStatistics`, a plain injectable port, so a vendor adapter wires its own
  query without core code learning a dialect. Statistics older than ``stats_max_age_days``, or with
  no recorded ``analyze`` time at all, count as **stale**: freshness we cannot prove is not fresh.
* **value shape** -- string format (``client_no`` is ten zero-padded digits) and numeric granularity
  (``credit_limit`` is a multiple of 5000). No statistics table can answer these; only values can.
  They come from one bounded fetch per table.

Order, therefore: statistics answer every distribution fact they can prove fresh; the bounded fetch
answers the shape facts, and replaces the distribution facts when the statistics are absent or
stale. ``shape=False`` asks for a statistics-only pass, where a table with fresh statistics costs
**no data query at all** -- that is the cheap mode, and the acceptance test that "sampling happens
only when statistics are absent or stale" asserts it.

The fetch is bounded three ways and cannot become a scan: the ``LIMIT``/``TABLESAMPLE`` is rendered
by sqlglot from an identifier validated against the catalog, :class:`~mdqm.models.limits.Limits`
caps the streamed rows and bytes and enforces the timeout, and a table whose estimate is over the
cap is reported as ``sample_complete: false`` so a later reader knows it saw a sample, not the
table.

Nothing here runs outside a :class:`~mdqm.config.quiet.QuietWindow` unless forced: the runner
refuses a heavy stage first (``PipelineContext.force`` passes it through), and
:meth:`ProfilingStage.run` repeats the check so a caller that invokes the stage directly cannot
skip it either.

Deterministic: the sample seed, the caps and the ``exec_ts`` are inputs, ``exec_ts`` decides
staleness (never the wall clock), and every list comes out sorted.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timedelta
from decimal import Decimal
from enum import StrEnum
from typing import Any, Final, Protocol

import pyarrow as pa
from pydantic import Field
from sqlglot import exp

from mdqm.adapters.base import NotSupported, SourceAdapter, supports_method
from mdqm.models.base import MDQMModel, UtcDatetime
from mdqm.models.catalog import Column, NormalizedType, Table
from mdqm.models.fragment import Fragment
from mdqm.models.limits import Limits
from mdqm.pipeline.core import (
    PipelineContext,
    StageReport,
    StageStatus,
    hash_inputs,
    next_quiet_start,
    quiet_allows,
)
from mdqm.sql.guard import guard

__all__ = [
    "DEFAULT_GRANULARITY_SHARE",
    "DEFAULT_SAMPLE_ROWS",
    "DEFAULT_STATS_MAX_AGE_DAYS",
    "DEFAULT_TOP_N",
    "GRANULARITY_LADDER",
    "STAGE_VERSION",
    "SUPPORTED_FACTS",
    "ColumnProfile",
    "ColumnStat",
    "FormatKind",
    "GranularityShare",
    "NumericGranularity",
    "OptimizerStatistics",
    "ProfilingStage",
    "Provenance",
    "QuietWindowClosed",
    "SamplePlan",
    "SourceProfile",
    "StaticStatistics",
    "StringFormat",
    "TableProfile",
    "TopValue",
    "detect_format",
    "detect_granularity",
    "parse_pg_stats",
    "sample_query",
]

logger = logging.getLogger(__name__)

#: Bumped when a change to the measurement itself changes the fact sheet, so a rerun refreshes it.
STAGE_VERSION: Final[int] = 1

#: Rows one bounded fetch may return. Enough for a granularity and a format check, far below a scan.
DEFAULT_SAMPLE_ROWS: Final[int] = 5_000

#: How many most-common values per column reach the fact sheet.
DEFAULT_TOP_N: Final[int] = 10

#: A granularity is reported when at least this share of the non-null values are multiples of it.
#: Not 1.0: one stray value must not collapse a real 5000 to 1, and the full candidate ladder
#: travels with it so D8 sees that 5000 fits 0.3% of a raw, unrounded column.
DEFAULT_GRANULARITY_SHARE: Final[float] = 0.99

#: Statistics with no ``analyze`` time, or older than this, are treated as stale and re-measured.
DEFAULT_STATS_MAX_AGE_DAYS: Final[int] = 7

#: The granularities worth reporting, largest first. ``1`` is the floor: integers.
GRANULARITY_LADDER: Final[tuple[int, ...]] = (5000, 1000, 500, 100, 50, 10, 5, 1)

#: The facts this stage can measure at all. Named so a report can say what it left out.
SUPPORTED_FACTS: Final[tuple[str, ...]] = (
    "bounds",
    "distinct",
    "format",
    "granularity",
    "null_rate",
    "top_values",
)

#: A catalog identifier we are willing to put in SQL. The catalog is the source of the name, but a
#: source that reports a wild name must not reach the query text: sqlglot quotes it *and* this gate.
_IDENTIFIER: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*$")

#: Dialects whose ``TABLESAMPLE SYSTEM`` takes a percentage of pages. Postgres needs a table big
#: enough that a small percentage still touches pages, or the sample comes back empty.
_PERCENT_SAMPLED: Final[frozenset[str]] = frozenset({"postgres", "postgresql"})

#: A table must hold this many times the sample size before a percentage sample is used on it.
_NATIVE_SAMPLE_RATIO: Final[int] = 50

#: ASCII digits: the shape ``client_no`` has, and the only shape a zero pad can be read from.
_DIGITS: Final[re.Pattern[str]] = re.compile(r"^[0-9]+$")


class Provenance(StrEnum):
    """Where one measured number came from -- the point of the whole stage.

    ``STATS`` is the source's own optimizer statistics, ``SAMPLED`` is a bounded fetch we made, and
    ``ABSENT`` means neither could answer (the source has no statistics and could not be sampled, or
    the table held no rows). A reader trusts each number only as far as this says.
    """

    STATS = "stats"
    SAMPLED = "sampled"
    ABSENT = "absent"


class FormatKind(StrEnum):
    """The coarse string shape :func:`detect_format` recognises. Not a full regex learner."""

    DIGITS = "digits"
    LETTERS = "letters"
    ALPHANUMERIC = "alphanumeric"
    OTHER = "other"


#: Tie break for :class:`FormatKind`: the most common class wins, and this order settles a draw.
_KIND_ORDER: Final[tuple[FormatKind, ...]] = (
    FormatKind.DIGITS,
    FormatKind.LETTERS,
    FormatKind.ALPHANUMERIC,
    FormatKind.OTHER,
)


class QuietWindowClosed(RuntimeError):
    """A heavy pipeline stage was asked to run outside its quiet windows without ``force``.

    The runner refuses before a stage is called; this exists so a direct caller cannot either, which
    is what keeps profiling off a primary database during the working day.
    """


# --- the facts --------------------------------------------------------------------------------


class TopValue(MDQMModel):
    """One most-common value and how often it was seen."""

    value: str
    count: int = Field(ge=0)


class GranularityShare(MDQMModel):
    """One candidate granularity and the share of non-null values that are exact multiples of it."""

    value: int = Field(gt=0)
    share: float = Field(ge=0.0, le=1.0)


class NumericGranularity(MDQMModel):
    """The largest granularity enough values fit, plus every candidate's share.

    D8 needs both: ``value``/``share`` answer "what is this number rounded to", and ``candidates``
    shows how close the runner-up was (a raw column where 5000 fits 0.3% is *not* rounded to 5000).
    """

    value: int = Field(gt=0)
    share: float = Field(ge=0.0, le=1.0)
    candidates: tuple[GranularityShare, ...] = ()


class StringFormat(MDQMModel):
    """The shape of a string column: its character class, width and zero padding share.

    ``digits`` is set only when every matching value has the same character count, which is what
    makes ``client_no`` "ten zero-padded digits" rather than "digits of varying width".
    """

    kind: FormatKind
    digits: int | None = Field(default=None, ge=1)
    zero_padded: bool = False
    length_min: int | None = Field(default=None, ge=0)
    length_max: int | None = Field(default=None, ge=0)
    share: float = Field(ge=0.0, le=1.0)


class ColumnProfile(MDQMModel):
    """What one column looks like, and which of the two sources answered each fact.

    ``distribution_provenance`` covers ``null_rate``, ``distinct``, ``minimum``, ``maximum`` and
    ``top_values`` (one family, one decision, so the numbers cannot disagree about their origin);
    ``value_shape_provenance`` covers ``granularity`` and ``format``.
    """

    name: str = Field(min_length=1)
    normalized_type: NormalizedType
    null_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    distinct: int | None = Field(default=None, ge=0)
    distinct_exact: bool = Field(
        default=False,
        description="True when the count came from the whole table (or exact statistics).",
    )
    minimum: str | None = None
    maximum: str | None = None
    top_values: tuple[TopValue, ...] = ()
    granularity: NumericGranularity | None = None
    format: StringFormat | None = None
    distribution_provenance: Provenance = Provenance.ABSENT
    value_shape_provenance: Provenance = Provenance.ABSENT
    stats_analyzed_at: UtcDatetime | None = Field(
        default=None, description="When the statistics used here were collected, if they were."
    )


class TableProfile(MDQMModel):
    """One table's fact sheet: the row estimate, what the fetch covered, and every column.

    ``sample_complete`` is the honest answer to "did this see the table": it is ``True`` only when
    the catalog's estimate was at or under the cap. An unknown estimate is read as *over* the cap,
    because assuming a table is small is how a bounded fetch turns into a scan.
    """

    table: str = Field(min_length=1)
    rows_estimate: int | None = Field(default=None, ge=0)
    rows_estimate_provenance: Provenance = Provenance.ABSENT
    rows_sampled: int = Field(default=0, ge=0)
    sample_rows_cap: int = Field(gt=0)
    sample_complete: bool = False
    sample_method: str = Field(default="none")
    sample_seed: int = 0
    columns: tuple[ColumnProfile, ...] = ()


class SourceProfile(MDQMModel):
    """The whole fact sheet for one source: what feeds D7/D8 and the A-stages."""

    source: str = Field(min_length=1)
    stage_version: int = STAGE_VERSION
    captured_at: UtcDatetime
    tables: tuple[TableProfile, ...] = ()
    notes: tuple[str, ...] = Field(
        default=(), description="What could not be measured, in words: never silently absent."
    )

    def column(self, table: str, column: str) -> ColumnProfile | None:
        """The profile of one column, or ``None`` when this sheet does not hold it."""
        for entry in self.tables:
            if entry.table != table:
                continue
            return next((item for item in entry.columns if item.name == column), None)
        return None


# --- optimizer statistics, and the pg_stats shape ---------------------------------------------


class ColumnStat(MDQMModel):
    """One column's row from the optimizer's statistics, in a source-neutral form."""

    table: str = Field(min_length=1)
    column: str = Field(min_length=1)
    null_fraction: float | None = Field(default=None, ge=0.0, le=1.0)
    distinct: int | None = Field(default=None, ge=0)
    minimum: str | None = None
    maximum: str | None = None
    top_values: tuple[TopValue, ...] = ()
    analyzed_at: UtcDatetime | None = None

    def answers_distribution(self) -> bool:
        """True when this row can answer all four distribution facts without reading a row."""
        return (
            self.null_fraction is not None
            and self.distinct is not None
            and (self.minimum is not None or self.maximum is not None)
        )


class OptimizerStatistics(Protocol):
    """The port a vendor adapter implements to hand over its own optimizer statistics.

    PostgreSQL reads ``pg_stats`` plus ``pg_class.reltuples``; Oracle ``DBMS_STATS`` views; SQL
    Server ``sys.dm_db_stats_*``; MySQL ``information_schema`` / ``mysql.innodb_index_stats``. Core
    code never learns which: it asks, or it takes the declared fallback and samples.
    """

    # pi-lens-ignore: no-ellipsis-body -- Protocol stub: pyright requires an ellipsis body
    def column_stats(self, table: str) -> Sequence[ColumnStat] | None:
        """Statistics rows for ``table``, or ``None`` when this source has none for it."""
        ...


class StaticStatistics:
    """A :class:`OptimizerStatistics` over an in-memory mapping -- what a probe result becomes.

    Also the seam a test uses to prove the statistics-first order without a PostgreSQL server, and
    the natural landing place for :func:`parse_pg_stats` output once an adapter has fetched it.
    """

    def __init__(self, stats: Mapping[str, Sequence[ColumnStat]]) -> None:
        self._stats = {table: tuple(rows) for table, rows in stats.items()}

    def column_stats(self, table: str) -> Sequence[ColumnStat] | None:
        """The rows declared for ``table``, or ``None`` when that table was not described."""
        return self._stats.get(table)


def _distinct_from_n_distinct(value: float, row_estimate: int) -> int:
    """Resolve ``pg_stats.n_distinct``: positive is a count, negative is a fraction of the rows.

    A negative value is how PostgreSQL says "too many distinct values to store a count": ``-0.5``
    means half the rows are distinct, so the count is an estimate by construction.
    """
    if value >= 0:
        return round(value)
    return round(abs(value) * max(0, row_estimate))


def _bounds_from(
    histogram: Sequence[Any] | None, commons: Sequence[Any] | None
) -> tuple[str | None, str | None]:
    """min/max from the histogram when there is one, else from the most-common values.

    A low-cardinality column has no histogram (the MCVs are the whole distribution); a
    high-cardinality one has no MCVs. Whichever exists carries the bounds.
    """
    for source in (histogram, commons):
        if source:
            return (_as_text(source[0]), _as_text(source[-1]))
    return (None, None)


def parse_pg_stats(
    rows: Iterable[Mapping[str, Any]],
    *,
    row_estimate: int,
    analyzed_at: datetime | None = None,
) -> tuple[ColumnStat, ...]:
    """Turn ``pg_stats`` rows for one table into :class:`ColumnStat` values, sorted by column.

    The input keeps ``pg_stats``' own column names (``attname``, ``null_frac``, ``n_distinct``,
    ``most_common_vals``, ``most_common_freqs``, ``histogram_bounds``), so an adapter hands over
    what it read and the mapping happens here, once, in a pure function that a fixture can test.
    ``row_estimate`` is the table's ``pg_class.reltuples``: the MCV frequencies are fractions of it.
    """
    parsed: list[ColumnStat] = []
    for row in rows:
        column = row.get("attname")
        if not isinstance(column, str) or not column:
            continue
        commons = row.get("most_common_vals")
        frequencies = row.get("most_common_freqs")
        counts: list[TopValue] = []
        if isinstance(commons, Sequence) and isinstance(frequencies, Sequence):
            for value, frequency in zip(commons, frequencies, strict=False):
                if isinstance(frequency, (int, float)):
                    count = round(frequency * row_estimate)
                    counts.append(TopValue(value=_as_text(value) or "", count=count))
        histogram = row.get("histogram_bounds")
        minimum, maximum = _bounds_from(
            histogram if isinstance(histogram, Sequence) else None,
            commons if isinstance(commons, Sequence) else None,
        )
        null_frac = row.get("null_frac")
        n_distinct = row.get("n_distinct")
        distinct = _as_float(n_distinct)
        parsed.append(
            ColumnStat(
                table=str(row.get("tablename") or ""),
                column=column,
                null_fraction=_as_float(null_frac),
                distinct=(
                    None if distinct is None else _distinct_from_n_distinct(distinct, row_estimate)
                ),
                minimum=minimum,
                maximum=maximum,
                top_values=tuple(counts[:DEFAULT_TOP_N]),
                analyzed_at=analyzed_at,
            )
        )
    return tuple(sorted(parsed, key=lambda stat: stat.column))


# --- value shaping (pure) ---------------------------------------------------------------------


def _as_text(value: object) -> str | None:
    """One cell as the text the fact sheet stores; ``None`` for NULL and for binary payloads."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value) if isinstance(value, float) else str(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return None  # a binary column has no readable text shape
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _as_int(value: object) -> int | None:
    """The exact integer a cell holds, or ``None`` when it holds something else.

    A float counts only when it is integral: ``12.0`` is 12, but ``12.5`` is not a multiple of any
    integer granularity and must lower every candidate's share instead of being rounded away.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        try:
            number = int(value)
        except (ValueError, OverflowError):  # NaN and +-Inf are not an integer a column holds
            return None
        return number if number == value else None
    if isinstance(value, str):
        text = value.strip()
        try:
            number = int(text)
        except ValueError:
            return None
        return number
    return None


def _as_float(value: object) -> float | None:
    """A statistic as a finite ``float``, or ``None`` when it is missing or not a number.

    Drivers differ on what they hand over for one statistic (a float, a ``Decimal``, text from a
    text protocol), so the conversion is wrapped: a value we cannot read means "no statistic",
    never a failure on the profile of a table we do not control.
    """
    if value is None or isinstance(value, bool):
        return None
    if not isinstance(value, (int, float, str, Decimal)):
        return None
    try:
        number = float(value)
    except (ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def detect_granularity(
    values: Iterable[object], *, threshold: float = DEFAULT_GRANULARITY_SHARE
) -> NumericGranularity | None:
    """The largest :data:`GRANULARITY_LADDER` value enough non-null values are multiples of.

    Returns ``None`` when the column holds no numbers, or when nothing on the ladder reaches
    ``threshold`` (a decimal column, where every integer candidate fits nothing).
    """
    present = [value for value in values if value is not None]
    if not present:
        return None
    candidates = tuple(
        GranularityShare(
            value=candidate,
            share=sum(1 for value in present if _is_multiple(value, candidate)) / len(present),
        )
        for candidate in GRANULARITY_LADDER
    )
    chosen = next((candidate for candidate in candidates if candidate.share >= threshold), None)
    if chosen is None:
        return None
    return NumericGranularity(value=chosen.value, share=chosen.share, candidates=candidates)


def _is_multiple(value: object, granularity: int) -> bool:
    """True when ``value`` is an exact integer multiple of ``granularity``."""
    number = _as_int(value)
    return number is not None and number % granularity == 0


def _kind_of(text: str) -> FormatKind:
    """Classify one string's characters into a :class:`FormatKind`.

    Letters are tested by Unicode category, not by ``[A-Za-z]``: a Hebrew name column is a letters
    column, and both languages matter here (AGENTS.md section 5).
    """
    if not text:
        return FormatKind.OTHER
    if _DIGITS.match(text):
        return FormatKind.DIGITS
    if all(character.isalpha() or character.isspace() for character in text):
        return FormatKind.LETTERS
    if text.isprintable():
        return FormatKind.ALPHANUMERIC
    return FormatKind.OTHER


def detect_format(values: Iterable[object]) -> StringFormat | None:
    """The dominant string shape, its width when every matching value agrees, and its share.

    A column of mixed shapes reports the most common one with ``share`` below 1, so a reader sees
    that the signature describes most values rather than all of them.
    """
    texts = [text for text in (_as_text(value) for value in values) if text is not None]
    if not texts:
        return None
    kinds = Counter(_kind_of(text) for text in texts)
    kind = min(_KIND_ORDER, key=lambda candidate: (-kinds[candidate], _KIND_ORDER.index(candidate)))
    matching = [text for text in texts if _kind_of(text) is kind]
    widths = sorted({len(text) for text in matching})
    digits = widths[0] if len(widths) == 1 else None
    lengths = [len(text) for text in texts]
    zero_padded = (
        kind is FormatKind.DIGITS
        and digits is not None
        and digits > 1
        and any(text.startswith("0") for text in matching)
    )
    return StringFormat(
        kind=kind,
        digits=digits,
        zero_padded=zero_padded,
        length_min=min(lengths),
        length_max=max(lengths),
        share=len(matching) / len(texts),
    )


def _bounds(values: Sequence[object]) -> tuple[str | None, str | None]:
    """min/max of the non-null values: numerically when every one is an integer, else by text."""
    present = [value for value in values if value is not None]
    if not present:
        return (None, None)
    numbers = [_as_int(value) for value in present]
    if all(number is not None for number in numbers):
        ordered = sorted(number for number in numbers if number is not None)
        return (str(ordered[0]), str(ordered[-1]))
    texts = [text for text in (_as_text(value) for value in present) if text is not None]
    if not texts:
        return (None, None)
    ordered_text = sorted(texts)
    return (ordered_text[0], ordered_text[-1])


def _top_values(values: Sequence[object], limit: int) -> tuple[TopValue, ...]:
    """The most common values, ordered by count then by value so two runs agree."""
    counts = Counter(text for text in (_as_text(value) for value in values) if text is not None)
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]
    return tuple(TopValue(value=value, count=count) for value, count in ranked)


# --- the bounded fetch ------------------------------------------------------------------------


class SamplePlan(MDQMModel):
    """The statement and the method name that goes into the fact sheet."""

    sql: str
    method: str


def sample_query(
    table: Table,
    *,
    dialect: str,
    rows: int,
    seed: int,
    native: bool,
    row_estimate: int | None,
) -> SamplePlan | None:
    """The bounded fetch for one table, or ``None`` when this dialect cannot be sampled.

    Rendered by sqlglot from the catalog's own identifier, so ``LIMIT`` becomes ``FETCH FIRST`` on
    Oracle and ``SELECT TOP`` on SQL Server without a dialect table here. With ``native`` (the
    probe reported the source's ``sampling`` capability) and a dialect that really implements it, a
    seeded ``TABLESAMPLE`` replaces the plain ``LIMIT``: same bound, and the same seed returns the
    same rows, which is what makes a repeat profile comparable to the last one.
    """
    if rows < 1:
        raise ValueError(f"rows must be at least 1, got {rows}")
    if not _IDENTIFIER.match(table.name) or (
        table.schema_name is not None and not _IDENTIFIER.match(table.schema_name)
    ):
        raise ValueError(
            f"refusing to profile {table.name!r}: the catalog's name is not a plain identifier"
        )
    source = exp.Table(
        this=exp.to_identifier(table.name, quoted=True),
        db=(
            exp.to_identifier(table.schema_name, quoted=True)
            if table.schema_name is not None
            else None
        ),
    )
    statement = exp.select("*").from_(source)
    method = "limit"
    if native and dialect == "duckdb":
        source.set(
            "sample",
            exp.TableSample(
                size=exp.Literal.number(rows),
                method=exp.var("RESERVOIR"),
                seed=exp.Literal.number(seed),
            ),
        )
        method = "tablesample"
    elif (
        native
        and dialect in _PERCENT_SAMPLED
        and row_estimate is not None
        and row_estimate >= rows * _NATIVE_SAMPLE_RATIO
    ):
        percent = round(100.0 * rows / row_estimate, 4)
        percent = min(100.0, max(0.01, percent))
        source.set(
            "sample",
            exp.TableSample(
                percent=exp.Literal.number(percent),
                method=exp.var("SYSTEM"),
                seed=exp.Literal.number(seed),
            ),
        )
        statement = statement.limit(rows)
        method = "tablesample"
    else:
        statement = statement.limit(rows)
    try:
        sql = statement.sql(dialect=dialect)
    except ValueError:
        # sqlglot raises ValueError for a name it has no dialect for (an API or ES source).
        logger.debug("no SQL dialect %r: the source cannot be sampled", dialect)
        return None
    return SamplePlan(sql=sql, method=method)


# --- the stage --------------------------------------------------------------------------------


class ProfilingStage:
    """Stage D5. Construct with the adapters to profile; ``run`` fills in the fact sheet.

    ``statistics`` is keyed by source and may omit a source entirely -- then that source's columns
    are all measured from the sample, which is exactly the fallback the probe declares when a source
    has no optimizer statistics exposed.
    """

    name: str = "D5"
    heavy: bool = True

    def __init__(
        self,
        adapters: Mapping[str, SourceAdapter],
        *,
        statistics: Mapping[str, OptimizerStatistics] | None = None,
        sample_rows: int = DEFAULT_SAMPLE_ROWS,
        top_n: int = DEFAULT_TOP_N,
        granularity_share: float = DEFAULT_GRANULARITY_SHARE,
        stats_max_age_days: int = DEFAULT_STATS_MAX_AGE_DAYS,
        shape: bool = True,
        seed: int = 0,
        timeout_s: float = 30.0,
        max_bytes: int = 8 * 1024 * 1024,
        tables: Sequence[str] | None = None,
    ) -> None:
        if sample_rows < 1:
            raise ValueError(f"sample_rows must be at least 1, got {sample_rows}")
        if top_n < 1:
            raise ValueError(f"top_n must be at least 1, got {top_n}")
        if not 0.0 < granularity_share <= 1.0:
            raise ValueError(f"granularity_share must be in (0, 1], got {granularity_share}")
        if stats_max_age_days < 0:
            raise ValueError(f"stats_max_age_days must be >= 0, got {stats_max_age_days}")
        self._adapters = dict(adapters)
        self._statistics = dict(statistics or {})
        self._sample_rows = sample_rows
        self._top_n = top_n
        self._granularity_share = granularity_share
        self._stats_max_age_days = stats_max_age_days
        self._shape = shape
        self._seed = seed
        self._timeout_s = timeout_s
        self._max_bytes = max_bytes
        self._tables = tuple(tables) if tables is not None else None
        self._capabilities: dict[str, Mapping[str, Any]] = {}

    @property
    def sources(self) -> tuple[str, ...]:
        """The sources this instance profiles, sorted (what the runner gates and filters on)."""
        return tuple(sorted(self._adapters))

    def inputs_hash(self, ctx: PipelineContext) -> str:
        """sha256 over the declared inputs: sources, caps, the seed and the day being profiled.

        The **date** of ``exec_ts`` is deliberate. A profile of a table that grows every day is not
        the same fact tomorrow, so a nightly pass must re-measure it; two passes on one day measure
        the same day and the second one skips. Nothing reads the data itself, so the hash stays a
        pure function of declared inputs, as :class:`~mdqm.pipeline.core.Stage` requires.
        """
        return hash_inputs(
            self.name,
            STAGE_VERSION,
            sorted(self._adapters),
            self._tables,
            ctx.source,
            ctx.exec_ts.date().isoformat(),
            self._sample_rows,
            self._top_n,
            self._granularity_share,
            self._stats_max_age_days,
            self._shape,
            self._seed,
        )

    def run(self, ctx: PipelineContext) -> StageReport:
        """Measure every selected source and buffer one ``generated/profile_<source>.yaml`` each."""
        if not ctx.force and not quiet_allows(ctx.quiet_windows, ctx.exec_ts):
            raise QuietWindowClosed(self._refusal(ctx))
        selected = (ctx.source,) if ctx.source is not None else self.sources
        missing = [source for source in selected if source not in self._adapters]
        if missing:
            raise ValueError(f"no adapter for source(s) {', '.join(missing)}")
        counts: dict[str, int] = {
            "tables": 0,
            "columns": 0,
            "sampled_tables": 0,
            "stats_columns": 0,
        }
        skipped: list[str] = []
        fixes: list[str] = []
        for source in selected:
            profile = self._profile_source(source, ctx, skipped=skipped, fixes=fixes)
            ctx.write_generated(_section(source), profile.model_dump(mode="json"))
            counts["tables"] += len(profile.tables)
            counts["columns"] += sum(len(table.columns) for table in profile.tables)
            counts["sampled_tables"] += sum(1 for table in profile.tables if table.rows_sampled > 0)
            counts["stats_columns"] += sum(
                1
                for table in profile.tables
                for column in table.columns
                if column.distribution_provenance is Provenance.STATS
            )
        return StageReport(
            stage=self.name,
            status=StageStatus.RAN,
            counts=counts,
            skipped=tuple(skipped),
            applied_fixes=tuple(fixes),
        )

    # -- internals -------------------------------------------------------------------------

    def _refusal(self, ctx: PipelineContext) -> str:
        """The message a direct caller sees: same rule and same next-window hint as the runner."""
        upcoming = next_quiet_start(ctx.quiet_windows, ctx.exec_ts)
        detail = (
            f"the next window opens at {upcoming[0].isoformat()}"
            if upcoming is not None
            else "no quiet window is configured so none ever opens"
        )
        return (
            f"stage {self.name} profiling is heavy and its quiet window is closed ({detail}); "
            "run it inside the window or pass --force to profile now"
        )

    def _profile_source(
        self,
        source: str,
        ctx: PipelineContext,
        *,
        skipped: list[str],
        fixes: list[str],
    ) -> SourceProfile:
        """Measure one source's tables, in name order, and collect what could not be measured."""
        adapter = self._adapters[source]
        catalog = adapter.extract_catalog()
        notes: list[str] = []
        wanted = None if self._tables is None else set(self._tables)
        tables: list[TableProfile] = []
        for table in sorted(catalog.tables, key=lambda item: item.name):
            if wanted is not None and table.name not in wanted:
                continue
            profiled = self._profile_table(
                source,
                adapter,
                table,
                exec_ts=ctx.exec_ts,
                skipped=skipped,
                fixes=fixes,
                notes=notes,
            )
            if profiled is not None:
                tables.append(profiled)
        return SourceProfile(
            source=source,
            captured_at=ctx.exec_ts,
            tables=tuple(tables),
            notes=tuple(sorted(notes)),
        )

    def _profile_table(
        self,
        source: str,
        adapter: SourceAdapter,
        table: Table,
        *,
        exec_ts: datetime,
        skipped: list[str],
        fixes: list[str],
        notes: list[str],
    ) -> TableProfile | None:
        """One table: statistics first, then the single bounded fetch those facts still need."""
        if table.kind.value in ("endpoint", "data_stream"):
            skipped.append(f"{source}.{table.name}: {table.kind.value} is not profiled by D5")
            return None
        fresh = self._fresh_stats(source, table, exec_ts)
        for stale_column in sorted(set(self._declared_columns(source, table)) - set(fresh)):
            notes.append(
                f"{source}.{table.name}.{stale_column}: no statistics newer than "
                f"{self._stats_max_age_days}d, measured from the sample"
            )
        rows: list[Mapping[str, Any]] = []
        plan: SamplePlan | None = None
        complete = False
        needs_sample = self._shape or len(fresh) < len(table.columns)
        if needs_sample:
            rows, plan, complete = self._fetch(
                source, adapter, table, exec_ts=exec_ts, skipped=skipped, fixes=fixes
            )
        columns = tuple(
            self._profile_column(
                column,
                stat=fresh.get(column.name),
                rows=rows,
                complete=complete,
                estimate=table.row_estimate,
            )
            for column in table.columns
        )
        return TableProfile(
            table=table.name,
            rows_estimate=table.row_estimate,
            rows_estimate_provenance=(
                Provenance.STATS if table.row_estimate is not None else Provenance.ABSENT
            ),
            rows_sampled=len(rows),
            sample_rows_cap=self._sample_rows,
            sample_complete=complete,
            sample_method=plan.method if plan is not None else "none",
            sample_seed=self._seed,
            columns=columns,
        )

    def _declared_columns(self, source: str, table: Table) -> tuple[str, ...]:
        """The columns the statistics provider claims to describe for this table."""
        provider = self._statistics.get(source)
        if provider is None:
            return ()
        declared = provider.column_stats(table.name) or ()
        return tuple(stat.column for stat in declared)

    def _fresh_stats(self, source: str, table: Table, exec_ts: datetime) -> dict[str, ColumnStat]:
        """The per-column statistics that are present *and* provably fresh, keyed by column."""
        provider = self._statistics.get(source)
        if provider is None:
            return {}
        known = {column.name for column in table.columns}
        fresh: dict[str, ColumnStat] = {}
        for stat in provider.column_stats(table.name) or ():
            if stat.column not in known or not stat.answers_distribution():
                continue
            if not self._is_fresh(stat, exec_ts):
                continue
            fresh[stat.column] = stat
        return fresh

    def _is_fresh(self, stat: ColumnStat, exec_ts: datetime) -> bool:
        """True when the statistics were collected recently enough to trust today.

        No recorded analyze time means freshness cannot be proven, so it is stale: the conservative
        reading, and the one that keeps a silently-shifting table from being trusted. The comparison
        is against the injected ``exec_ts``, never the wall clock.
        """
        if stat.analyzed_at is None:
            return False
        return stat.analyzed_at >= exec_ts - timedelta(days=self._stats_max_age_days)

    def _fetch(
        self,
        source: str,
        adapter: SourceAdapter,
        table: Table,
        *,
        exec_ts: datetime,
        skipped: list[str],
        fixes: list[str],
    ) -> tuple[list[Mapping[str, Any]], SamplePlan | None, bool]:
        """One bounded fetch. Returns the rows, the plan, and whether it covered the whole table."""
        dialect = adapter.dialect()
        plan = self._plan(adapter, table, dialect=dialect, source=source, skipped=skipped)
        if plan is None:
            return ([], None, False)
        fragment = Fragment(
            source=source,
            dialect=dialect,
            sql=plan.sql,
            tables=(table.name,),
            exec_ts=exec_ts,
        )
        limits = Limits(
            timeout_s=self._timeout_s,
            max_rows=self._sample_rows,
            max_bytes=self._max_bytes,
            max_partitions=1,
        )
        if table.row_estimate is None:
            fixes.append(
                f"{source}.{table.name}: no row estimate, so the {self._sample_rows}-row cap was "
                "read as a sample rather than a full read"
            )
        rows = self._read(adapter, fragment, limits, source=source, table=table, skipped=skipped)
        complete = table.row_estimate is not None and table.row_estimate <= self._sample_rows
        return (rows, plan, complete)

    def _plan(
        self,
        adapter: SourceAdapter,
        table: Table,
        *,
        dialect: str,
        source: str,
        skipped: list[str],
    ) -> SamplePlan | None:
        """The statement to run: the adapter's own when it has one, else the sqlglot bound."""
        if supports_method(adapter, "sample_sql"):
            # Optional method: absent on most adapters, so it is reached through a plain lookup.
            # It returns raw SQL text, so it is a trusted-plugin surface: the statement is guarded
            # before it runs (see :meth:`_read`), and a plugin that returns a write is reported.
            build = getattr(adapter, "sample_sql")  # noqa: B009 - not a declared protocol member
            sql = build(table.name, self._sample_rows, self._seed)
            if not isinstance(sql, str) or not sql:
                raise ValueError(f"sample_sql returned {type(sql).__name__}, expected SQL text")
            return SamplePlan(sql=sql, method="adapter")
        plan = sample_query(
            table,
            dialect=dialect,
            rows=self._sample_rows,
            seed=self._seed,
            native=self._declares_sampling(source, adapter),
            row_estimate=table.row_estimate,
        )
        if plan is None:
            skipped.append(
                f"{source}.{table.name}: dialect {dialect!r} cannot be sampled with SQL, so only "
                "statistics were read"
            )
        return plan

    def _read(
        self,
        adapter: SourceAdapter,
        fragment: Fragment,
        limits: Limits,
        *,
        source: str,
        table: Table,
        skipped: list[str],
    ) -> list[Mapping[str, Any]]:
        """Run the bounded fragment and materialise its rows; a failed fetch yields no rows.

        The statement is guarded here because this is the one place in the pipeline that hands a
        fragment straight to an adapter with no governor in between, and ``sample_sql()`` can
        return raw SQL text from a plugin. A refusal is reported in ``skipped``, never run.
        """
        if fragment.sql is not None:
            checked = guard(fragment.sql, fragment.dialect)
            if not checked.ok:
                code = checked.error.code if checked.error is not None else "WRITE_BLOCKED"
                skipped.append(
                    f"{source}.{table.name}: the sample statement was refused by the read-only "
                    f"guard ({code}), so only statistics were read"
                )
                return []
        # The adapter protocol method, not a SQL sink: the statement was guarded just above.
        stream = (  # pi-lens-ignore: python-sql-injection -- adapter protocol call
            adapter.execute(fragment, limits)
        )
        try:
            return [row for batch in stream for row in _rows_of(batch)]
        except (NotSupported, TimeoutError) as exc:
            # One table that cannot be fetched (or runs past `timeout_s`) must not lose the others:
            # the fact sheet records the miss in `skipped` and marks the columns absent.
            skipped.append(f"{source}.{table.name}: the bounded fetch stopped early ({exc})")
            return []
        finally:
            stream.close()

    def _declares_sampling(self, source: str, adapter: SourceAdapter) -> bool:
        """True when the probe positively reported the source's own ``sampling`` capability."""
        capabilities = self._capabilities.get(source)
        if capabilities is None:
            try:
                capabilities = adapter.probe().capabilities
            except NotSupported:
                capabilities = {}
            self._capabilities[source] = capabilities
        verdict = capabilities.get("sampling")
        return verdict is not None and getattr(verdict, "supported", None) is True

    def _profile_column(
        self,
        column: Column,
        *,
        stat: ColumnStat | None,
        rows: Sequence[Mapping[str, Any]],
        complete: bool,
        estimate: int | None,
    ) -> ColumnProfile:
        """One column: statistics where they answer, the sample for shape and for the rest."""
        values = [row.get(column.name) for row in rows]
        sampled = len(rows) > 0
        if stat is not None:
            distribution = Provenance.STATS
            null_rate: float | None = stat.null_fraction
            distinct: int | None = stat.distinct
            distinct_exact = False
            minimum, maximum = stat.minimum, stat.maximum
            top_values = stat.top_values
        elif sampled:
            distribution = Provenance.SAMPLED
            present = [value for value in values if value is not None]
            null_rate = 1.0 - len(present) / len(values)
            observed = {text for text in (_as_text(value) for value in present) if text is not None}
            distinct_exact = complete
            distinct = len(observed) if complete else _scaled(len(observed), len(values), estimate)
            minimum, maximum = _bounds(values)
            top_values = _top_values(values, self._top_n)
        else:
            distribution = Provenance.ABSENT
            null_rate, distinct, distinct_exact = (None, None, False)
            minimum = maximum = None
            top_values = ()
        shapes = self._shape and sampled
        return ColumnProfile(
            name=column.name,
            normalized_type=column.normalized_type,
            null_rate=null_rate,
            distinct=distinct,
            distinct_exact=distinct_exact,
            minimum=minimum,
            maximum=maximum,
            top_values=top_values,
            granularity=(
                detect_granularity(values, threshold=self._granularity_share) if shapes else None
            ),
            format=detect_format(values) if shapes else None,
            distribution_provenance=distribution,
            value_shape_provenance=Provenance.SAMPLED if shapes else Provenance.ABSENT,
            stats_analyzed_at=stat.analyzed_at if stat is not None else None,
        )


def _scaled(observed: int, seen: int, estimate: int | None) -> int:
    """Distinct count for a partial sample: scale to the table, never past its row count."""
    if seen <= 0 or estimate is None or estimate <= seen:
        return observed
    return min(estimate, max(observed, round(observed * estimate / seen)))


def _rows_of(batch: pa.RecordBatch) -> list[Mapping[str, Any]]:
    """One Arrow batch as row mappings, ready for the pure shaping functions above.

    Typed as the batch the adapter contract already promises (:class:`~mdqm.adapters.base.
    ArrowStream` yields ``pa.RecordBatch``), so no defensive ``isinstance`` stands between the
    protocol and the call.
    """
    return list(batch.to_pylist())


def _section(source: str) -> str:
    """The generated section name for a source: its name, made safe for a file stem."""
    slug = re.sub(r"[^A-Za-z0-9_.-]", "_", source).strip(".") or "source"
    return f"profile_{slug}"
