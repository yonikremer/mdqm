"""The error type a broken semantic file raises.

Same discipline as :class:`mdqm.config.ConfigError`: the message names the file and the problem,
never the value, so it is safe to log and to show to an agent. A validation problem additionally
names the path inside the document and the line, because a human edits this file by hand.
"""

from __future__ import annotations

from collections.abc import Sequence

__all__ = ["SemanticError"]


class SemanticError(ValueError):
    """A semantic layer file that cannot be loaded.

    ``source`` is the file (or directory) the operator must fix. ``problems`` are individual
    strings such as ``"tables.billing.usage_events.mutability.class: line 48: ..."``.
    """

    def __init__(self, source: str, problems: Sequence[str]) -> None:
        self.source = source
        self.problems = tuple(problems)
        super().__init__(f"{source}: " + "; ".join(self.problems))
