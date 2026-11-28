"""Short statistics lines for the Context Pack and ``find_data``.

A weak agent gets one line per table it already matched, not a dump of every column. Shares are
shown only when they are non-zero, except NULL, which is always shown when it was measured.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from mdqm.context.pack import ContextPack, OmissionReason, OmittedItem, PackItem, PackKind
from mdqm.results.summary import estimate_tokens
from mdqm.stats.models import CategoryColumnStats, ColumnQuality

__all__ = ["attach_briefs", "column_line", "table_brief"]

_TOP = 3


def _percent(share: float) -> str:
    return f"{round(share * 100)}%"


def column_line(quality: ColumnQuality, distribution: CategoryColumnStats | None = None) -> str:
    """One column, a few tokens: nulls, sentinels that actually occur, distinct, top values."""
    parts: list[str] = []
    if quality.null_share is not None:
        parts.append(f"null {_percent(quality.null_share)}")
    if quality.zero_share:
        parts.append(f"zero {_percent(quality.zero_share)}")
    if quality.empty_share:
        parts.append(f"empty {_percent(quality.empty_share)}")
    if quality.whitespace_share:
        parts.append(f"blank {_percent(quality.whitespace_share)}")
    if quality.distinct_count is not None:
        parts.append(f"distinct {quality.distinct_count}")
    if quality.minimum is not None and quality.maximum is not None:
        parts.append(f"range {quality.minimum}..{quality.maximum}")
    if distribution is not None and distribution.values:
        top = ", ".join(f"{item.value} {item.count}" for item in distribution.values[:_TOP])
        parts.append(f"top {top}")
    body = ", ".join(parts) if parts else "no stats"
    return f"{quality.column}: {body}"


def table_brief(
    table: str,
    qualities: Sequence[ColumnQuality],
    distributions: Mapping[str, CategoryColumnStats] | None = None,
    *,
    limit: int = 4,
) -> str:
    """Up to ``limit`` columns of one table, column name ascending, on one line."""
    chosen = sorted(qualities, key=lambda item: item.column)[:limit]
    lines = [
        column_line(item, None if distributions is None else distributions.get(item.column))
        for item in chosen
    ]
    return f"{table}: " + " | ".join(lines)


def _mentioned(pack: ContextPack, key: str) -> bool:
    """True when the pack already talks about this table (``source.table``)."""
    for item in pack.items:
        if item.ref == key or item.ref.startswith(key + ".") or key.startswith(item.ref + "."):
            return True
    return False


def _with_stats(base: str, lines: Sequence[str]) -> str:
    block = "stats:\n" + "\n".join(f"  {line}" for line in lines)
    return base + "\n" + block


def attach_briefs(pack: ContextPack, briefs: Mapping[str, str]) -> ContextPack:
    """Append stats lines for tables the pack already named, inside the same token budget.

    An empty pack stays empty: a question that matched nothing must not gain a statistics dump.
    Lines that do not fit are listed in ``omitted`` with reason ``budget``. Briefs whose table
    is not in the pack are ignored, not omitted: they were not relevant.
    """
    if pack.is_empty or not briefs:
        return pack
    chosen = [(key, briefs[key]) for key in sorted(briefs) if _mentioned(pack, key)]
    if not chosen:
        return pack
    kept: list[tuple[str, str]] = []
    omitted = list(pack.omitted)
    for index, (key, line) in enumerate(chosen):
        candidate = _with_stats(pack.text, [item for _, item in [*kept, (key, line)]])
        if estimate_tokens(candidate) > pack.budget_tokens:
            omitted.extend(
                OmittedItem(kind=PackKind.STATS, ref=rest, reason=OmissionReason.BUDGET)
                for rest, _ in chosen[index:]
            )
            break
        kept.append((key, line))
    if not kept:
        return pack.model_copy(update={"omitted": tuple(omitted)})
    text = _with_stats(pack.text, [line for _, line in kept])
    items = list(pack.items)
    items.extend(
        PackItem(kind=PackKind.STATS, ref=key, text=line, matched_terms=1) for key, line in kept
    )
    return pack.model_copy(
        update={
            "items": tuple(items),
            "omitted": tuple(omitted),
            "text": text,
            "tokens_estimated": estimate_tokens(text),
        }
    )
