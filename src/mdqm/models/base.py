"""Shared base and reusable field types for the mdqm domain models.

The models are the wire format between lanes (adapters, compiler, planner, cache, server), so they
are frozen and reject unknown fields: a typo must fail loudly instead of being dropped silently.
Every model must survive ``model_validate_json(model.model_dump_json()) == model``, which is why the
JSON-carrying fields reject values that do not round-trip (datetimes, tuples, NaN, ...).
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Annotated, Any, TypeAlias

from pydantic import AfterValidator, AwareDatetime, BaseModel, ConfigDict


class MDQMModel(BaseModel):
    """Base for every shared domain model: frozen, no unknown fields."""

    model_config = ConfigDict(frozen=True, extra="forbid")


def _to_utc(value: datetime) -> datetime:
    """Normalise to UTC: timezones are converted at the edges, UTC internally."""
    return value.astimezone(UTC)


# Timezone-aware timestamp stored as UTC. A naive datetime is a bug, not a default.
UtcDatetime: TypeAlias = Annotated[AwareDatetime, AfterValidator(_to_utc)]


def _reject_non_json(value: Any) -> Any:
    """Reject values JSON cannot represent, so a dumped model validates back to an equal model."""
    stack: list[Any] = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            for key, nested in item.items():
                if not isinstance(key, str):
                    raise ValueError(
                        f"JSON object keys must be strings, got {type(key).__name__}."
                    )
                stack.append(nested)
        elif isinstance(item, list):
            stack.extend(item)
        elif isinstance(item, float) and not math.isfinite(item):
            raise ValueError("JSON numbers must be finite: NaN and Infinity do not round-trip.")
        elif item is None or isinstance(item, (bool, int, float, str)):
            continue
        else:
            raise ValueError(
                f"{type(item).__name__} is not JSON native "
                "(use str/int/float/bool/None/list/dict); the model would not round-trip."
            )
    return value


# A JSON object whose values survive a dump/validate round-trip unchanged.
JsonObject: TypeAlias = Annotated[dict[str, Any], AfterValidator(_reject_non_json)]
