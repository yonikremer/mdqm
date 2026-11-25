"""One executable unit for one source.

A fragment carries exactly one payload: SQL for a SQL source, Query DSL for Elasticsearch, or an
HTTP request for an API source. ``exec_ts`` is one value per execution and already substituted into
the payload; adapters never call ``now()`` themselves.
"""

from __future__ import annotations

from pydantic import Field, model_validator

from mdqm.models.base import JsonObject, MDQMModel, UtcDatetime


class Fragment(MDQMModel):
    """A single-source query, ready to hand to an adapter."""

    source: str = Field(min_length=1)
    dialect: str = Field(
        min_length=1, description="sqlglot dialect name, or 'dsl' for ES and APIs."
    )
    sql: str | None = None
    dsl: JsonObject | None = None
    request: JsonObject | None = None
    params: JsonObject = Field(
        default_factory=dict,
        description="Bound values. JSON scalars only, so the fragment round-trips through JSON.",
    )
    tables: tuple[str, ...] = ()
    exec_ts: UtcDatetime

    @model_validator(mode="after")
    def _check_single_payload(self) -> Fragment:
        present = [
            name
            for name, value in (("sql", self.sql), ("dsl", self.dsl), ("request", self.request))
            if value is not None
        ]
        if len(present) != 1:
            raise ValueError(
                f"a fragment carries exactly one of sql, dsl, request; got {present or 'none'}."
            )
        return self

    @property
    def kind(self) -> str:
        """Which payload is set: ``sql``, ``dsl`` or ``request``."""
        if self.sql is not None:
            return "sql"
        if self.dsl is not None:
            return "dsl"
        return "request"
