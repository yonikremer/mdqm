"""The weekly digest (S13): at most ten yes/no/other questions, by email or as a static page.

The human in the loop has a day job. DESIGN section 11 says a model proposes and a human approves,
which only works if the asking is cheap: one short question per pending proposal, ten at most, the
same order every run, and an answer that is one click of the reply key. This module renders that
question list and delivers it. It **decides nothing**: applying an answer goes through
:class:`~mdqm.review.queue.ReviewQueue` (:func:`record_answer`), so the queue stays the only writer
of ``curated/*.yaml``.

Three properties are load-bearing:

* **Never more than ten questions, always the same ten.** Selection is ``limit=10`` over proposals
  sorted by ``(-confidence, id)``. Mining's usage score would be the better "impact" signal, but a
  proposal row carries no fingerprint to join the mined counts on, so confidence (already
  confidence-major, id-minor in the queue) is the only honest ranking we have. The sort is redone
  here rather than trusted from the queue, so a queue change cannot silently make the digest
  non-deterministic. Two runs over the same rows produce byte-identical output.
* **Nothing is sent without explicit configuration.** Email needs ``smtp.host`` (and an address to
  send to). Anything missing -- no host, no sender, no recipients -- degrades to the static page
  with the reason in :attr:`DigestDelivery.reason` and in the log. A failure *inside* the send (DNS,
  connection refused, a rejected recipient, TLS) degrades the same way: page written, no exception
  raised. A digest that crashes the nightly run is worse than one a human reads a day later.
* **No secret reaches the page, the message or the log.** The SMTP password comes from
  ``SmtpSettings.password`` (a :class:`~pydantic.SecretStr`, populated from the environment; YAML
  refuses a literal ``password``). It is unpacked only at login and scrubbed from any exception text
  that is logged or returned, so a relay that echoes credentials in an error cannot leak them.

The page is HTML because a human reads it in a browser; the email carries the same content as a
plain-text body plus an HTML alternative. Hebrew pages are ``lang="he" dir="rtl"`` and Hebrew
templates are used for the whole page, not mixed into an English shell (a right-to-left page with
left-to-right question marks and parentheses reads badly).
"""

from __future__ import annotations

import hashlib
import html
import json
import logging
import smtplib
import ssl
from collections.abc import Callable, Iterable, Sequence
from email.message import EmailMessage
from email.policy import SMTP as SMTP_POLICY
from enum import StrEnum
from pathlib import Path
from typing import Final, Protocol

from pydantic import Field

from mdqm.cache.freshness import Clock, SystemClock
from mdqm.config.settings import Settings
from mdqm.models.base import MDQMModel, UtcDatetime
from mdqm.review.queue import Proposal, ProposalKind, ProposalStatus, ReviewQueue

__all__ = [
    "MAX_QUESTIONS",
    "SMTP_TIMEOUT_S",
    "Answer",
    "AnswerResult",
    "DeliveryChannel",
    "Digest",
    "DigestDelivery",
    "DigestError",
    "DigestItem",
    "DigestLanguage",
    "SmtpClient",
    "build_digest",
    "deliver",
    "page_path",
    "record_answer",
    "render_html",
    "render_text",
    "write_page",
]

logger = logging.getLogger(__name__)

#: Hard cap on questions in one digest. The reader's attention is the scarce resource.
MAX_QUESTIONS: Final[int] = 10
#: Socket timeout for the SMTP conversation: an unreachable relay must not hang the nightly run.
SMTP_TIMEOUT_S: Final[float] = 30.0


class DigestError(RuntimeError):
    """A digest problem the caller must not ignore (a naive clock, an empty language)."""


class DigestLanguage(StrEnum):
    """The language the whole page is written in: templates are not mixed."""

    HEBREW = "he"
    ENGLISH = "en"


class Answer(StrEnum):
    """The three replies a question allows. ``OTHER`` is a reply, not a decision."""

    YES = "yes"
    NO = "no"
    OTHER = "other"


#: The word a human writes for each answer, in the page's own language.
_ANSWER_LABELS: Final[dict[DigestLanguage, dict[Answer, str]]] = {
    DigestLanguage.HEBREW: {Answer.YES: "כן", Answer.NO: "לא", Answer.OTHER: "אחר"},
    DigestLanguage.ENGLISH: {Answer.YES: "yes", Answer.NO: "no", Answer.OTHER: "other"},
}

#: How a proposal kind is named to a human; the enum value is wire format, not prose.
_KIND_LABELS: Final[dict[DigestLanguage, dict[ProposalKind, str]]] = {
    DigestLanguage.HEBREW: {
        ProposalKind.NAME: "שם",
        ProposalKind.SYNONYM: "מילה נרדפת",
        ProposalKind.VALUE_MAP: "מיפוי ערכים",
        ProposalKind.MEASURE: "מדד",
        ProposalKind.JOIN: "קישור בין מקורות",
        ProposalKind.MUTABILITY: "תכיפות שינוי",
        ProposalKind.OTHER: "שינוי אחר",
    },
    DigestLanguage.ENGLISH: {
        ProposalKind.NAME: "name",
        ProposalKind.SYNONYM: "synonym",
        ProposalKind.VALUE_MAP: "value mapping",
        ProposalKind.MEASURE: "measure",
        ProposalKind.JOIN: "join",
        ProposalKind.MUTABILITY: "mutability",
        ProposalKind.OTHER: "other change",
    },
}

_TITLES: Final[dict[DigestLanguage, str]] = {
    DigestLanguage.HEBREW: "תקציר שבועי לאישור הצעות",
    DigestLanguage.ENGLISH: "Weekly review digest",
}
_EVIDENCE_HEADINGS: Final[dict[DigestLanguage, str]] = {
    DigestLanguage.HEBREW: "הראיות",
    DigestLanguage.ENGLISH: "Evidence",
}
_ANSWER_HINT: Final[dict[DigestLanguage, str]] = {
    DigestLanguage.HEBREW: "השיבו: כן / לא / אחר",
    DigestLanguage.ENGLISH: "Reply: yes / no / other",
}
_NO_ITEMS: Final[dict[DigestLanguage, str]] = {
    DigestLanguage.HEBREW: "אין הצעות ממתינות. אין שאלות השבוע.",
    DigestLanguage.ENGLISH: "Nothing is pending. No questions this week.",
}
_OMITTED: Final[dict[DigestLanguage, str]] = {
    DigestLanguage.HEBREW: "עוד {count} הצעות ממתינות ממתינות לשבוע הבא.",
    DigestLanguage.ENGLISH: "{count} more pending proposals are held for next week.",
}


def _yes_no_other(language: DigestLanguage) -> str:
    """The three answers as one line, in the page's language."""
    labels = _ANSWER_LABELS[language]
    return " / ".join(labels[answer] for answer in Answer)


class DigestItem(MDQMModel):
    """One question: a pending proposal plus the sentence a human answers yes/no/other to."""

    number: int = Field(ge=1, le=MAX_QUESTIONS)
    #: The proposal id, so a reply can be traced back without guessing at the wording.
    proposal: str = Field(min_length=64, max_length=64)
    kind: ProposalKind
    #: ``<key>.<field>`` from the proposal: where the approved fact would land.
    target: str
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: tuple[str, ...] = ()
    #: The complete question sentence, already localised.
    prompt: str

    @property
    def short_id(self) -> str:
        """The first twelve hex characters: enough to quote in a reply, short enough to read."""
        return self.proposal[:12]


class Digest(MDQMModel):
    """One rendered digest: the questions, the counts behind them, and where it came from."""

    id: str = Field(min_length=64, max_length=64)
    language: DigestLanguage
    generated_at: UtcDatetime
    items: tuple[DigestItem, ...] = ()
    #: How many proposals were pending when the digest was built (items + omitted).
    pending_total: int = Field(default=0, ge=0)
    #: Pending proposals deliberately left for next week, so the count is never mysterious.
    omitted: int = Field(default=0, ge=0)

    @property
    def count(self) -> int:
        """How many questions this digest asks."""
        return len(self.items)


class DeliveryChannel(StrEnum):
    """What actually carried the digest. ``PAGE`` also means "email was not sent"."""

    EMAIL = "email"
    PAGE = "page"


class DigestDelivery(MDQMModel):
    """The outcome of one delivery attempt, safe to log: counts and a reason, never a secret."""

    channel: DeliveryChannel
    page_path: Path
    email_sent: bool = False
    recipients: int = Field(default=0, ge=0)
    #: Why email was skipped or failed. Empty only when email was sent.
    reason: str = ""


class AnswerResult(MDQMModel):
    """What recording one reply did, so a caller can report it instead of guessing."""

    proposal: str
    answer: Answer
    #: False for ``other``: an open reply is not a decision, so the proposal stays pending.
    recorded: bool
    status: ProposalStatus
    reason: str = ""


class SmtpClient(Protocol):
    """The slice of :class:`smtplib.SMTP` the digest uses, so a test can pass a double.

    Typed as a protocol rather than a concrete class because ``tests/unit`` blocks sockets: the
    only way to test the message we *would* send is to inject something that is not a socket.
    """

    def starttls(self, *, context: ssl.SSLContext | None = None) -> object: ...

    def login(self, user: str, password: str) -> object: ...

    def send_message(self, message: EmailMessage) -> object: ...

    def quit(self) -> object: ...


#: ``(host, port) -> client``. The default opens a real SMTP connection with a bounded timeout.
Connect = Callable[[str, int], SmtpClient]


def _default_connect(host: str, port: int) -> SmtpClient:
    """Open a real SMTP connection. Never called when SMTP is unconfigured."""
    return smtplib.SMTP(host, port, timeout=SMTP_TIMEOUT_S)


def _tls_context() -> ssl.SSLContext:
    """A verifying TLS context for STARTTLS: an unverified upgrade protects nobody.

    An internal relay whose certificate this host does not trust degrades to the static page (the
    failure is logged), which is the safe direction: the digest is not secret, the credentials are.
    """
    return ssl.create_default_context()


def _canonical(value: object) -> bytes:
    """Stable bytes for hashing: sorted keys, UTF-8 so Hebrew hashes the same on every OS."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()


def _redact(text: str, secret: str) -> str:
    """Remove a secret value from text bound for a log line or a reason."""
    if secret and secret in text:
        return text.replace(secret, "***")
    return text


def _now(clock: Clock) -> UtcDatetime:
    """One clock reading, normalised to UTC; a naive clock is a bug, not a default."""
    now = clock.now()
    if now.tzinfo is None:
        raise DigestError("the clock returned a naive datetime; the digest is dated in UTC")
    return now


def _localise(item: Proposal, language: DigestLanguage, number: int) -> DigestItem:
    """One proposal as one question, in the page's language."""
    kind = _KIND_LABELS[language][item.kind]
    target = item.target or item.key or item.section
    if language is DigestLanguage.HEBREW:
        prompt = f"לאשר {kind} עבור {target}?"
    else:
        prompt = f"Approve the {kind} for {target}?"
    return DigestItem(
        number=number,
        proposal=item.id,
        kind=item.kind,
        target=target,
        confidence=item.confidence,
        evidence=item.evidence,
        prompt=prompt,
    )


def _select(pending: Sequence[Proposal], limit: int) -> tuple[Proposal, ...]:
    """The questions to ask: pending proposals by ``(-confidence, id)``, at most ``limit``.

    Ties on confidence are broken by id, so the set does not depend on insertion order or on the
    queue's own ORDER BY. ``limit <= 0`` asks nothing, which is a legitimate (if odd) request.
    """
    if limit <= 0:
        return ()
    ordered = sorted(pending, key=lambda item: (-item.confidence, item.id))
    return tuple(ordered[:limit])


def build_digest(
    queue: ReviewQueue,
    *,
    language: DigestLanguage = DigestLanguage.HEBREW,
    limit: int = MAX_QUESTIONS,
    clock: Clock | None = None,
) -> Digest:
    """Collect at most ``limit`` pending proposals and render them as localised questions.

    The digest id is the content hash of (language, proposal ids), so the same week's queue gives
    the same id on every run and a re-render never looks like a new digest.
    """
    effective = min(limit, MAX_QUESTIONS)
    pending = queue.pending()
    pending_total = len(pending)
    chosen = _select(pending, effective)
    items = tuple(_localise(item, language, number) for number, item in enumerate(chosen, start=1))
    digest_id = hashlib.sha256(
        _canonical({"language": language.value, "proposals": [item.proposal for item in items]})
    ).hexdigest()
    return Digest(
        id=digest_id,
        language=language,
        generated_at=_now(clock if clock is not None else SystemClock()),
        items=items,
        pending_total=pending_total,
        omitted=max(pending_total - len(items), 0),
    )


def _confidence_label(item: DigestItem) -> str:
    """Confidence as a plain two-decimal number: no percent sign, no locale surprise."""
    return f"{item.confidence:.2f}"


def render_text(digest: Digest) -> str:
    """The plain-text body: what a text-only mail client and a ``.txt`` reader see."""
    language = digest.language
    title = _TITLES[language]
    lines = [
        title,
        f"{digest.generated_at.date().isoformat()} · {digest.count}/{MAX_QUESTIONS}",
        "",
    ]
    if not digest.items:
        lines.append(_NO_ITEMS[language])
    for item in digest.items:
        lines.append(f"{item.number}. {item.prompt}")
        lines.append(f"   {item.target} · {_confidence_label(item)} · {item.short_id}")
        if item.evidence:
            lines.append(f"   {_EVIDENCE_HEADINGS[language]}:")
            lines.extend(f"   - {line}" for line in item.evidence)
        lines.append(f"   {_ANSWER_HINT[language]}")
        lines.append("")
    if digest.omitted:
        lines.append(_OMITTED[language].format(count=digest.omitted))
    return "\n".join(lines).rstrip("\n") + "\n"


def _escaped(value: str) -> str:
    """HTML text with the three dangerous characters escaped; quotes stay readable."""
    return html.escape(value, quote=False)


def render_html(digest: Digest) -> str:
    """The page and the HTML mail alternative: one RTL document for Hebrew, LTR for English.

    Hebrew gets ``dir="rtl"`` on ``<html>`` and the whole template in Hebrew, because a mixed
    direction page renders punctuation on the wrong side. Evidence is escaped, so a value map with
    ``<script>`` in it cannot execute in the reader's browser.
    """
    language = digest.language
    direction = "rtl" if language is DigestLanguage.HEBREW else "ltr"
    title = _TITLES[language]
    parts = [
        "<!DOCTYPE html>",
        f'<html lang="{language.value}" dir="{direction}">',
        "<head>",
        '<meta charset="utf-8">',
        f"<title>{_escaped(title)}</title>",
        "<style>",
        "body{font-family:system-ui,sans-serif;max-width:48rem;margin:2rem auto;padding:0 1rem}",
        "li{margin:0 0 1.5rem 0}",
        ".prompt{font-weight:600;margin:0 0 .25rem 0}",
        ".meta,.ref{color:#555;font-size:.85rem;margin:.15rem 0}",
        ".answers{margin:.35rem 0 0 0}",
        "</style>",
        "</head>",
        "<body>",
        f"<h1>{_escaped(title)}</h1>",
        f'<p class="meta">{digest.generated_at.date().isoformat()} · '
        f"{digest.count}/{MAX_QUESTIONS}</p>",
    ]
    if not digest.items:
        parts.append(f"<p>{_escaped(_NO_ITEMS[language])}</p>")
    else:
        parts.append('<ol class="questions">')
        for item in digest.items:
            parts.append(f'<li id="q{item.number}">')
            parts.append(f'<p class="prompt">{_escaped(item.prompt)}</p>')
            parts.append(
                f'<p class="meta">{_escaped(item.target)} · {_confidence_label(item)} · '
                f"{_escaped(item.short_id)}</p>"
            )
            if item.evidence:
                parts.append(f"<p>{_escaped(_EVIDENCE_HEADINGS[language])}:</p>")
                parts.append("<ul>")
                parts.extend(f"<li>{_escaped(line)}</li>" for line in item.evidence)
                parts.append("</ul>")
            parts.append(f'<p class="answers">{_escaped(_yes_no_other(language))}</p>')
            parts.append("</li>")
        parts.append("</ol>")
    if digest.omitted:
        parts.append(
            f'<p class="meta">{_escaped(_OMITTED[language].format(count=digest.omitted))}</p>'
        )
    parts.extend(["</body>", "</html>"])
    return "\n".join(parts) + "\n"


def page_path(digest: Digest, directory: Path | str) -> Path:
    """The weekly file name: ``digest-<lang>-<year>-W<week>.html``, stable across re-renders."""
    year, week, _ = digest.generated_at.isocalendar()
    return Path(directory) / f"digest-{digest.language.value}-{year}-W{week:02d}.html"


def write_page(digest: Digest, directory: Path | str) -> Path:
    """Write the static page and return its path. This is the fallback, so it always happens."""
    path = page_path(digest, directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_html(digest), encoding="utf-8", newline="\n")
    return path


def build_message(digest: Digest, *, sender: str, recipients: Sequence[str]) -> EmailMessage:
    """The mail to send: one text body and one HTML alternative, both from the same digest."""
    title = _TITLES[digest.language]
    message = EmailMessage(policy=SMTP_POLICY)
    message["Subject"] = f"{title}: {digest.generated_at.date().isoformat()} ({digest.count})"
    message["From"] = sender
    message["To"] = ", ".join(recipients)
    message.set_content(render_text(digest), charset="utf-8")
    message.add_alternative(render_html(digest), subtype="html", charset="utf-8")
    return message


def _send(
    message: EmailMessage,
    settings: Settings,
    *,
    password: str,
    connect: Connect,
) -> str | None:
    """Try to send; return ``None`` on success or a redacted reason for the failure."""
    smtp = settings.smtp
    client: SmtpClient | None = None
    try:
        client = connect(smtp.host, smtp.port)
        if smtp.starttls:
            client.starttls(context=_tls_context())
        if smtp.username and password:
            client.login(smtp.username, password)
        client.send_message(message)
    except Exception as exc:  # noqa: BLE001 - any failure degrades to the page, never raises
        detail = f"{type(exc).__name__}: {exc}".rstrip(": ")
        return _redact(detail, password)
    finally:
        if client is not None:
            try:
                client.quit()
            except Exception:  # noqa: BLE001 - a broken QUIT must not undo a sent message
                logger.debug("digest: smtp quit failed", extra={"host": smtp.host})
    return None


def _addresses(candidates: Iterable[str], fallback: str) -> list[str]:
    """Usable addresses only: a header needs a name, not a blank line or a newline injection."""
    cleaned = [
        address.strip()
        for address in candidates
        if address and "\n" not in address and "\r" not in address
    ]
    if cleaned:
        return cleaned
    fallback = fallback.strip()
    return [fallback] if fallback and "\n" not in fallback and "\r" not in fallback else []


def deliver(
    digest: Digest,
    settings: Settings,
    *,
    recipients: Iterable[str] | None = None,
    page_dir: Path | str | None = None,
    connect: Connect | None = None,
) -> DigestDelivery:
    """Write the page, then email it when SMTP is configured and reachable.

    Email is attempted only with an explicit configuration: ``smtp.host`` set and at least one
    address (``recipients``, else ``smtp.sender``). Every other path -- and every failure inside the
    send -- returns a :class:`DigestDelivery` with ``channel=page``, the page already on disk, and
    the reason recorded; nothing is raised, so a nightly run cannot die on a mail server.
    """
    directory = Path(page_dir) if page_dir is not None else settings.paths.cache_dir / "digest"
    path = write_page(digest, directory)
    smtp = settings.smtp
    if not smtp.enabled:
        reason = "smtp.host is not configured: digest written to the static page only"
        logger.info("digest: page only: %s", reason, extra={"reason": reason, "page": str(path)})
        return DigestDelivery(channel=DeliveryChannel.PAGE, page_path=path, reason=reason)

    addresses = _addresses(recipients or (), smtp.sender)
    if not addresses:
        reason = "smtp.host is set but no sender or recipient address: page only"
        logger.warning("digest: page only: %s", reason, extra={"reason": reason, "page": str(path)})
        return DigestDelivery(channel=DeliveryChannel.PAGE, page_path=path, reason=reason)

    password = smtp.password.get_secret_value() if smtp.password is not None else ""
    sender = smtp.sender.strip() or addresses[0]
    message = build_message(digest, sender=sender, recipients=addresses)
    failure = _send(message, settings, password=password, connect=connect or _default_connect)
    if failure is not None:
        reason = f"smtp send failed ({failure}): page only"
        logger.warning(
            "digest: email failed: %s", reason, extra={"reason": reason, "page": str(path)}
        )
        return DigestDelivery(channel=DeliveryChannel.PAGE, page_path=path, reason=reason)
    return DigestDelivery(
        channel=DeliveryChannel.EMAIL,
        page_path=path,
        email_sent=True,
        recipients=len(addresses),
    )


def record_answer(queue: ReviewQueue, proposal: str, answer: Answer, *, by: str) -> AnswerResult:
    """Record one reply through the queue: ``yes`` approves, ``no`` rejects, ``other`` waits.

    ``other`` is deliberately not a decision: it means "neither", and guessing which of approve /
    reject / never ask the human meant would put a machine's guess in ``curated/``. The proposal
    stays pending and the caller is told so, which is the honest answer.
    """
    if answer is Answer.OTHER:
        current = queue.get(proposal)
        return AnswerResult(
            proposal=proposal,
            answer=answer,
            recorded=False,
            status=current.status,
            reason="'other' is not yes/no: the proposal stays pending for a human comment",
        )
    if answer is Answer.YES:
        decided = queue.approve(proposal, by=by)
    else:
        decided = queue.reject(proposal, by=by)
    return AnswerResult(proposal=proposal, answer=answer, recorded=True, status=decided.status)
