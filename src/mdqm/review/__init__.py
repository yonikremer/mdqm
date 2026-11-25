"""The human gate: proposals queue here, and only an approval reaches ``curated/*.yaml``.

``docs/DESIGN.md`` section 10: deterministic stages write facts, a model only proposes, code
validates, and a human approves. This package is the last step: :mod:`mdqm.review.queue` stores
what awaits a decision, records every decision (including rejections, which are remembered as
negative memory), and writes an approved fact into the curated layer with its provenance.
"""

from mdqm.review.queue import (
    Proposal,
    ProposalKind,
    ProposalStatus,
    ReviewError,
    ReviewQueue,
    SubmitAction,
)

__all__ = [
    "Proposal",
    "ProposalKind",
    "ProposalStatus",
    "ReviewError",
    "ReviewQueue",
    "SubmitAction",
]
