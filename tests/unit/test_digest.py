"""The weekly digest (S13): at most ten questions, emailed or served as a static page.

The acceptance criteria this file proves, one class each:

* :class:`TestSelection` -- never more than ten questions, and the same ten in the same order on
  every run whatever the insertion order was.
* :class:`TestRendering` -- golden HTML and text for ten sample proposals in Hebrew (with
  ``dir="rtl"``) and in English; evidence is escaped; every question carries yes/no/other.
* :class:`TestDeliveryWithoutSmtp` -- no SMTP configuration means page only, nothing is opened, and
  the reason is logged.
* :class:`TestDeliveryFallback` -- a relay that refuses the connection, fails TLS, fails the send or
  echoes the password in its error degrades to the page without raising and without leaking.
* :class:`TestDeliveryEmail` -- with an explicit configuration the message that *would* go out
  carries both bodies, the right headers, and no secret.
* :class:`TestRecordAnswer` -- a reply is applied through the review queue, so ``curated/`` is
  written by the queue alone and ``other`` decides nothing.
* :class:`TestDeterminism` -- two renders of one queue are byte-identical; a naive clock is a bug.

No real SMTP server exists in this environment and ``tests/unit`` blocks sockets, so the transport
is injected: the tests assert on the :class:`~email.message.EmailMessage` the digest would send.
"""

from __future__ import annotations

import logging
import ssl
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from email.message import EmailMessage
from pathlib import Path
from typing import Any

import pytest
from pydantic import SecretStr

from mdqm.config.settings import Settings, SmtpSettings
from mdqm.review.digest import (
    MAX_QUESTIONS,
    Answer,
    AnswerResult,
    DeliveryChannel,
    Digest,
    DigestError,
    DigestLanguage,
    build_digest,
    deliver,
    page_path,
    record_answer,
    render_html,
    render_text,
)
from mdqm.review.queue import (
    Proposal,
    ProposalKind,
    ProposalStatus,
    ReviewQueue,
    proposal_id,
)
from mdqm.store.db import Store

#: A Wednesday: ISO week 10 of 2026, so the weekly page name is predictable.
NOW = datetime(2026, 3, 4, 9, 30, tzinfo=UTC)
#: A throwaway password. It must never appear in a message, a page, a log line or a reason.
SECRET = "s3cr3t-relay-pass-דיגסט"

#: A CRLF pair, as an attacker would put it in a configured address.
CRLF = chr(13) + chr(10)

KINDS: tuple[ProposalKind, ...] = (
    ProposalKind.VALUE_MAP,
    ProposalKind.SYNONYM,
    ProposalKind.NAME,
    ProposalKind.MEASURE,
    ProposalKind.JOIN,
    ProposalKind.MUTABILITY,
)


class FakeClock:
    """A clock the test drives, so the weekly file name and every timestamp are deterministic."""

    def __init__(self, start: datetime = NOW) -> None:
        self._now = start

    def now(self) -> datetime:
        return self._now

    def advance(self, by: timedelta) -> None:
        self._now += by


class FakeSmtp:
    """A stand-in for :class:`smtplib.SMTP` that records the conversation instead of opening one."""

    def __init__(self, *, fail_on: str | None = None, detail: str = "") -> None:
        self.calls: list[str] = []
        self.logins: list[tuple[str, str]] = []
        self.sent: list[EmailMessage] = []
        self.contexts: list[object] = []
        self._fail_on = fail_on
        self._detail = detail

    def _record(self, name: str) -> None:
        self.calls.append(name)
        if self._fail_on == name:
            raise OSError(self._detail or f"{name} failed")

    def starttls(self, *, context: object = None) -> object:
        self.contexts.append(context)
        self._record("starttls")
        return None

    def login(self, user: str, password: str) -> object:
        self.logins.append((user, password))
        self._record("login")
        return None

    def send_message(self, message: EmailMessage) -> object:
        self._record("send_message")
        self.sent.append(message)
        return None

    def quit(self) -> object:
        self._record("quit")
        return None


class Connector:
    """The injected ``connect(host, port)``: hands back a fake client or raises."""

    def __init__(self, client: FakeSmtp | None = None, error: Exception | None = None) -> None:
        self.client = client
        self.error = error
        self.calls: list[tuple[str, int]] = []

    def __call__(self, host: str, port: int) -> FakeSmtp:
        self.calls.append((host, port))
        if self.error is not None:
            raise self.error
        assert self.client is not None, "Connector needs a client when error is None"
        return self.client


def make(
    number: int,
    *,
    confidence: float = 0.5,
    evidence: tuple[str, ...] = ("measured on 3 databases",),
    at: datetime = NOW,
) -> Proposal:
    """One pending proposal with a distinct key, built through the real id function."""
    kind = KINDS[number % len(KINDS)]
    return Proposal(
        id=proposal_id(kind, "value_maps", f"status_{number}", "", {"canonical": [f"v{number}"]}),
        kind=kind,
        section="value_maps",
        key=f"status_{number}",
        field="",
        patch={"canonical": [f"v{number}"]},
        evidence=evidence,
        confidence=confidence,
        proposed_by="mining",
        proposed_at=at,
    )


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """A semantic root with a curated directory, as the loader expects."""
    (tmp_path / "curated").mkdir()
    return tmp_path


@pytest.fixture
def store(tmp_path: Path) -> Iterator[Store]:
    opened = Store(tmp_path / "metadata.db")
    yield opened
    opened.close()


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def queue(store: Store, root: Path, clock: FakeClock) -> ReviewQueue:
    return ReviewQueue(store, root, clock=clock)


def seed(queue: ReviewQueue, count: int) -> tuple[Proposal, ...]:
    """Submit ``count`` proposals with descending confidence, returning them."""
    submitted: list[Proposal] = []
    for number in range(count):
        proposal = make(number, confidence=round(0.05 + number * 0.04, 2))
        queue.submit(proposal)
        submitted.append(proposal)
    return tuple(submitted)


def ten_samples(queue: ReviewQueue) -> None:
    """The ten proposals the golden files were rendered from: mixed kinds, Hebrew and English."""
    evidence = (
        ("3 values in crm.customers.status; billing.accounts.client_no joins at 99.8%",),
        ("ערכי הסטטוס תואמים ב-99.8% בין שלושת המקורות", "עמודת usage_events.status"),
        ("pg_stat_statements: 412 calls in 30 days",),
        ("אותו מדד מחושב בשתי דרכים: ROUND(x/5000)*5000",),
        ("join key needs LPAD(client_no, 10, '0')",),
        ("הטבלה מתעדכנת 3 פעמים ביום",),
        ("column renamed in erp on 2026-02-11",),
        ("ערכי הקטגוריה בעברית ובאנגלית",),
        ("measure appears in 7 saved questions",),
        ("partition key is start_time, events arrive late",),
    )
    for number in range(MAX_QUESTIONS):
        queue.submit(
            make(number, confidence=round(0.90 - number * 0.05, 2), evidence=evidence[number])
        )


def settings_with(
    *,
    host: str = "",
    sender: str = "",
    username: str = "",
    password: str | None = None,
    starttls: bool = True,
    port: int = 25,
) -> Settings:
    return Settings(
        smtp=SmtpSettings(
            host=host,
            port=port,
            sender=sender,
            username=username,
            password=None if password is None else SecretStr(password),
            starttls=starttls,
        )
    )


class TestSelection:
    """Acceptance: never more than 10 questions; ordering is deterministic."""

    def test_at_most_ten_questions_and_the_rest_is_counted(self, queue: ReviewQueue) -> None:
        seed(queue, 14)
        digest = build_digest(queue, clock=FakeClock())
        assert digest.count == MAX_QUESTIONS
        assert digest.pending_total == 14
        assert digest.omitted == 4

    def test_nothing_pending_asks_nothing(self, queue: ReviewQueue) -> None:
        digest = build_digest(queue, clock=FakeClock())
        assert digest.items == ()
        assert digest.pending_total == 0
        assert digest.omitted == 0

    def test_a_decided_proposal_is_not_asked_again(self, queue: ReviewQueue) -> None:
        submitted = seed(queue, 3)
        queue.reject(submitted[0].id, by="data-team")
        digest = build_digest(queue, clock=FakeClock())
        assert submitted[0].id not in [item.proposal for item in digest.items]
        assert digest.pending_total == 2

    def test_an_explicit_limit_cannot_exceed_the_cap(self, queue: ReviewQueue) -> None:
        seed(queue, 14)
        assert build_digest(queue, limit=99, clock=FakeClock()).count == MAX_QUESTIONS

    def test_order_is_confidence_then_id(self, queue: ReviewQueue) -> None:
        seed(queue, 12)
        digest = build_digest(queue, clock=FakeClock())
        keys = [(-item.confidence, item.proposal) for item in digest.items]
        assert keys == sorted(keys)
        assert [item.number for item in digest.items] == list(range(1, MAX_QUESTIONS + 1))

    def test_the_same_ten_whatever_the_insertion_order(self, store: Store, root: Path) -> None:
        """Determinism is about the rows, not about the order the pipeline happened to write."""
        forward = ReviewQueue(store, root, clock=FakeClock())
        seed(forward, 12)
        reversed_store = Store(root / "reversed.db")
        try:
            backward = ReviewQueue(reversed_store, root, clock=FakeClock())
            for proposal in reversed(seed(forward, 12)):
                backward.submit(proposal)
            first = build_digest(forward, clock=FakeClock())
            second = build_digest(backward, clock=FakeClock())
        finally:
            reversed_store.close()
        assert [item.proposal for item in first.items] == [item.proposal for item in second.items]


class TestRendering:
    """Acceptance: golden HTML/text for ten sample proposals in Hebrew and English."""

    def test_hebrew_html_is_golden(self, queue: ReviewQueue, golden: Any) -> None:
        ten_samples(queue)
        digest = build_digest(queue, language=DigestLanguage.HEBREW, clock=FakeClock())
        golden("digest_he.html", render_html(digest))

    def test_hebrew_text_is_golden(self, queue: ReviewQueue, golden: Any) -> None:
        ten_samples(queue)
        digest = build_digest(queue, language=DigestLanguage.HEBREW, clock=FakeClock())
        golden("digest_he.txt", render_text(digest))

    def test_english_html_is_golden(self, queue: ReviewQueue, golden: Any) -> None:
        ten_samples(queue)
        digest = build_digest(queue, language=DigestLanguage.ENGLISH, clock=FakeClock())
        golden("digest_en.html", render_html(digest))

    def test_english_text_is_golden(self, queue: ReviewQueue, golden: Any) -> None:
        ten_samples(queue)
        digest = build_digest(queue, language=DigestLanguage.ENGLISH, clock=FakeClock())
        golden("digest_en.txt", render_text(digest))

    def test_hebrew_page_is_right_to_left(self, queue: ReviewQueue) -> None:
        ten_samples(queue)
        page = render_html(build_digest(queue, clock=FakeClock()))
        assert 'dir="rtl"' in page
        assert 'lang="he"' in page
        assert "לאשר" in page
        assert "כן / לא / אחר" in page
        assert "yes / no / other" not in page

    def test_english_page_is_left_to_right(self, queue: ReviewQueue) -> None:
        ten_samples(queue)
        page = render_html(build_digest(queue, language=DigestLanguage.ENGLISH, clock=FakeClock()))
        assert 'dir="ltr"' in page
        assert 'lang="en"' in page
        assert "yes / no / other" in page
        assert "כן / לא / אחר" not in page

    def test_every_question_offers_three_answers(self, queue: ReviewQueue) -> None:
        seed(queue, 3)
        digest = build_digest(queue, clock=FakeClock())
        assert render_html(digest).count('class="answers">כן / לא / אחר') == digest.count
        english = build_digest(queue, language=DigestLanguage.ENGLISH, clock=FakeClock())
        assert render_html(english).count('class="answers">yes / no / other') == english.count

    def test_html_escapes_evidence(self, queue: ReviewQueue) -> None:
        """A value map with markup in it must not become markup in the reader's browser."""
        queue.submit(make(0, evidence=("<script>alert(1)</script>", "a & b")))
        page = render_html(build_digest(queue, clock=FakeClock()))
        assert "<script>alert(1)</script>" not in page
        assert "&lt;script&gt;alert(1)&lt;/script&gt;" in page
        assert "a &amp; b" in page

    def test_text_carries_evidence_and_the_short_id(self, queue: ReviewQueue) -> None:
        submitted = seed(queue, 2)
        body = render_text(build_digest(queue, clock=FakeClock()))
        assert "measured on 3 databases" in body
        assert submitted[0].id[:12] in body
        for number in (1, 2):
            assert f"{number}. " in body

    def test_nothing_pending_says_so_in_both_languages(self, queue: ReviewQueue) -> None:
        hebrew = render_text(build_digest(queue, clock=FakeClock()))
        english = render_text(
            build_digest(queue, language=DigestLanguage.ENGLISH, clock=FakeClock())
        )
        assert "אין הצעות ממתינות" in hebrew
        assert "Nothing is pending" in english
        assert "אין הצעות ממתינות" in render_html(build_digest(queue, clock=FakeClock()))


class TestDeliveryWithoutSmtp:
    """Acceptance: without an explicit configuration nothing is sent and the page is the digest."""

    def test_no_host_writes_the_page_and_opens_no_connection(
        self, queue: ReviewQueue, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        seed(queue, 3)
        digest = build_digest(queue, clock=FakeClock())
        connector = Connector()
        with caplog.at_level(logging.INFO, logger="mdqm.review.digest"):
            delivery = deliver(digest, settings_with(), page_dir=tmp_path, connect=connector)
        assert delivery.channel is DeliveryChannel.PAGE
        assert delivery.email_sent is False
        assert connector.calls == []
        assert delivery.page_path.is_file()
        assert delivery.page_path.read_text(encoding="utf-8") == render_html(digest)
        assert "smtp.host is not configured" in delivery.reason
        assert "digest: page only" in caplog.text

    def test_a_host_without_any_address_degrades_to_the_page(
        self, queue: ReviewQueue, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        digest = build_digest(queue, clock=FakeClock())
        connector = Connector(FakeSmtp())
        with caplog.at_level(logging.WARNING, logger="mdqm.review.digest"):
            delivery = deliver(
                digest, settings_with(host="relay.internal"), page_dir=tmp_path, connect=connector
            )
        assert delivery.channel is DeliveryChannel.PAGE
        assert connector.calls == []
        assert "no sender or recipient address" in delivery.reason
        assert delivery.page_path.is_file()

    def test_recipients_alone_do_not_replace_a_missing_host(
        self, queue: ReviewQueue, tmp_path: Path
    ) -> None:
        """An address list is not a configuration: without a relay nothing is sent."""
        digest = build_digest(queue, clock=FakeClock())
        connector = Connector(FakeSmtp())
        delivery = deliver(
            digest,
            settings_with(),
            recipients=["data-team@example.internal"],
            page_dir=tmp_path,
            connect=connector,
        )
        assert delivery.channel is DeliveryChannel.PAGE
        assert connector.calls == []


class TestDeliveryFallback:
    """Acceptance: an SMTP failure falls back to the page without raising."""

    def test_connection_refused_falls_back(self, queue: ReviewQueue, tmp_path: Path) -> None:
        digest = build_digest(queue, clock=FakeClock())
        connector = Connector(error=ConnectionRefusedError("connection refused"))
        delivery = deliver(
            digest,
            settings_with(host="relay.internal", sender="data-team@example.internal"),
            page_dir=tmp_path,
            connect=connector,
        )
        assert delivery.channel is DeliveryChannel.PAGE
        assert delivery.email_sent is False
        assert "ConnectionRefusedError" in delivery.reason
        assert "connection refused" in delivery.reason
        assert delivery.page_path.is_file()

    @pytest.mark.parametrize("stage", ["starttls", "login", "send_message"])
    def test_failure_at_any_stage_falls_back(
        self, queue: ReviewQueue, tmp_path: Path, stage: str
    ) -> None:
        digest = build_digest(queue, clock=FakeClock())
        client = FakeSmtp(fail_on=stage)
        delivery = deliver(
            digest,
            settings_with(
                host="relay.internal",
                sender="data-team@example.internal",
                username="mdqm",
                password=SECRET,
            ),
            page_dir=tmp_path,
            connect=Connector(client),
        )
        assert delivery.channel is DeliveryChannel.PAGE
        assert delivery.page_path.is_file()
        assert client.sent == []

    def test_a_failing_relay_is_logged(
        self, queue: ReviewQueue, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        digest = build_digest(queue, clock=FakeClock())
        with caplog.at_level(logging.WARNING, logger="mdqm.review.digest"):
            deliver(
                digest,
                settings_with(host="relay.internal", sender="data-team@example.internal"),
                page_dir=tmp_path,
                connect=Connector(error=TimeoutError("relay did not answer")),
            )
        assert "digest: email failed" in caplog.text
        assert "TimeoutError" in caplog.text

    def test_an_error_that_echoes_the_password_is_redacted(
        self, queue: ReviewQueue, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        digest = build_digest(queue, clock=FakeClock())
        with caplog.at_level(logging.WARNING, logger="mdqm.review.digest"):
            delivery = deliver(
                digest,
                settings_with(
                    host="relay.internal",
                    sender="data-team@example.internal",
                    username="mdqm",
                    password=SECRET,
                ),
                page_dir=tmp_path,
                connect=Connector(error=OSError(f"535 auth failed for {SECRET}")),
            )
        assert SECRET not in delivery.reason
        assert "***" in delivery.reason
        assert SECRET not in caplog.text
        assert SECRET not in delivery.page_path.read_text(encoding="utf-8")

    def test_a_broken_quit_does_not_undo_a_sent_digest(
        self, queue: ReviewQueue, tmp_path: Path
    ) -> None:
        digest = build_digest(queue, clock=FakeClock())
        client = FakeSmtp(fail_on="quit")
        delivery = deliver(
            digest,
            settings_with(host="relay.internal", sender="data-team@example.internal"),
            page_dir=tmp_path,
            connect=Connector(client),
        )
        assert delivery.channel is DeliveryChannel.EMAIL
        assert len(client.sent) == 1


class TestDeliveryEmail:
    """With an explicit configuration the digest is emailed, and the message holds no secret."""

    def test_the_message_is_sent_to_the_configured_addresses(
        self, queue: ReviewQueue, tmp_path: Path
    ) -> None:
        ten_samples(queue)
        digest = build_digest(queue, clock=FakeClock())
        client = FakeSmtp()
        delivery = deliver(
            digest,
            settings_with(
                host="relay.internal",
                port=2525,
                sender="mdqm@example.internal",
                username="mdqm",
                password=SECRET,
            ),
            recipients=["data-team@example.internal", "dba@example.internal"],
            page_dir=tmp_path,
            connect=Connector(client),
        )
        assert delivery.channel is DeliveryChannel.EMAIL
        assert delivery.email_sent is True
        assert delivery.recipients == 2
        assert delivery.reason == ""
        assert client.calls == ["starttls", "login", "send_message", "quit"]
        assert client.logins == [("mdqm", SECRET)]
        message = client.sent[0]
        assert message["From"] == "mdqm@example.internal"
        assert message["To"] == "data-team@example.internal, dba@example.internal"
        assert "2026-03-04" in str(message["Subject"])

    def test_the_message_carries_both_bodies(self, queue: ReviewQueue, tmp_path: Path) -> None:
        ten_samples(queue)
        digest = build_digest(queue, clock=FakeClock())
        client = FakeSmtp()
        deliver(
            digest,
            settings_with(host="relay.internal", sender="mdqm@example.internal"),
            recipients=["data-team@example.internal"],
            page_dir=tmp_path,
            connect=Connector(client),
        )
        message = client.sent[0]
        assert message.is_multipart()
        types = {part.get_content_type() for part in message.iter_parts()}
        assert types == {"text/plain", "text/html"}
        plain = message.get_body(preferencelist=("plain",))
        markup = message.get_body(preferencelist=("html",))
        assert plain is not None and markup is not None
        assert "לאשר" in plain.get_content()
        assert 'dir="rtl"' in markup.get_content()

    def test_the_sender_is_the_default_recipient(self, queue: ReviewQueue, tmp_path: Path) -> None:
        digest = build_digest(queue, clock=FakeClock())
        client = FakeSmtp()
        delivery = deliver(
            digest,
            settings_with(host="relay.internal", sender="data-team@example.internal"),
            page_dir=tmp_path,
            connect=Connector(client),
        )
        assert delivery.recipients == 1
        assert client.sent[0]["To"] == "data-team@example.internal"

    def test_starttls_off_and_no_credentials_skips_both_steps(
        self, queue: ReviewQueue, tmp_path: Path
    ) -> None:
        digest = build_digest(queue, clock=FakeClock())
        client = FakeSmtp()
        deliver(
            digest,
            settings_with(
                host="relay.internal", sender="mdqm@example.internal", starttls=False, port=25
            ),
            page_dir=tmp_path,
            connect=Connector(client),
        )
        assert client.calls == ["send_message", "quit"]
        assert client.logins == []

    def test_starttls_verifies_the_relay_certificate(
        self, queue: ReviewQueue, tmp_path: Path
    ) -> None:
        """An unverified TLS upgrade protects nobody: the context must check the certificate."""
        digest = build_digest(queue, clock=FakeClock())
        client = FakeSmtp()
        deliver(
            digest,
            settings_with(host="relay.internal", sender="mdqm@example.internal"),
            page_dir=tmp_path,
            connect=Connector(client),
        )
        context = client.contexts[0]
        assert isinstance(context, ssl.SSLContext)
        assert context.verify_mode == ssl.CERT_REQUIRED
        assert context.check_hostname is True

    def test_a_newline_in_an_address_is_not_a_header(
        self, queue: ReviewQueue, tmp_path: Path
    ) -> None:
        """A configured address must not be able to smuggle a second header into the message."""
        digest = build_digest(queue, clock=FakeClock())
        client = FakeSmtp()
        delivery = deliver(
            digest,
            settings_with(host="relay.internal", sender="mdqm@example.internal"),
            recipients=["data-team@example.internal" + CRLF + "Bcc: attacker@example.org"],
            page_dir=tmp_path,
            connect=Connector(client),
        )
        assert delivery.channel is DeliveryChannel.EMAIL
        assert client.sent[0]["Bcc"] is None
        assert client.sent[0]["To"] == "mdqm@example.internal"

    def test_a_hostile_sender_falls_back_to_the_page(
        self, queue: ReviewQueue, tmp_path: Path
    ) -> None:
        digest = build_digest(queue, clock=FakeClock())
        connector = Connector(FakeSmtp())
        delivery = deliver(
            digest,
            settings_with(
                host="relay.internal", sender="mdqm@example.internal" + CRLF + "Bcc: x@y"
            ),
            page_dir=tmp_path,
            connect=connector,
        )
        assert delivery.channel is DeliveryChannel.PAGE
        assert connector.calls == []
        assert delivery.page_path.is_file()

    def test_no_secret_in_the_message_the_page_or_the_page_text(
        self, queue: ReviewQueue, tmp_path: Path
    ) -> None:
        ten_samples(queue)
        digest = build_digest(queue, clock=FakeClock())
        client = FakeSmtp()
        delivery = deliver(
            digest,
            settings_with(
                host="relay.internal",
                sender="mdqm@example.internal",
                username="mdqm",
                password=SECRET,
            ),
            recipients=["data-team@example.internal"],
            page_dir=tmp_path,
            connect=Connector(client),
        )
        serialized = client.sent[0].as_string()
        assert SECRET not in serialized
        assert SECRET not in render_text(digest)
        assert SECRET not in render_html(digest)
        assert SECRET not in delivery.page_path.read_text(encoding="utf-8")
        assert SECRET not in delivery.reason


class TestPagePath:
    """One file per week and language, so a re-render overwrites instead of piling up."""

    def test_the_name_is_the_iso_week(self, queue: ReviewQueue, tmp_path: Path) -> None:
        digest = build_digest(queue, clock=FakeClock())
        assert page_path(digest, tmp_path).name == "digest-he-2026-W10.html"
        english = build_digest(queue, language=DigestLanguage.ENGLISH, clock=FakeClock())
        assert page_path(english, tmp_path).name == "digest-en-2026-W10.html"

    def test_next_week_is_a_new_file(self, queue: ReviewQueue, tmp_path: Path) -> None:
        clock = FakeClock()
        first = page_path(build_digest(queue, clock=clock), tmp_path)
        clock.advance(timedelta(days=7))
        second = page_path(build_digest(queue, clock=clock), tmp_path)
        assert first != second


class TestRecordAnswer:
    """A reply goes through the review queue: approval writes ``curated/``, nothing else does."""

    def test_yes_approves_through_the_queue(self, queue: ReviewQueue, root: Path) -> None:
        submitted = seed(queue, 1)
        result = record_answer(queue, submitted[0].id, Answer.YES, by="data-team")
        assert isinstance(result, AnswerResult)
        assert result.recorded is True
        assert result.status is ProposalStatus.APPROVED
        assert queue.get(submitted[0].id).status is ProposalStatus.APPROVED
        assert (root / "curated" / "value_maps.yaml").is_file()

    def test_no_rejects_and_writes_nothing(self, queue: ReviewQueue, root: Path) -> None:
        submitted = seed(queue, 1)
        result = record_answer(queue, submitted[0].id, Answer.NO, by="data-team")
        assert result.recorded is True
        assert result.status is ProposalStatus.REJECTED
        assert queue.pending() == ()
        assert not (root / "curated" / "value_maps.yaml").exists()

    def test_other_decides_nothing(self, queue: ReviewQueue, root: Path) -> None:
        submitted = seed(queue, 1)
        result = record_answer(queue, submitted[0].id, Answer.OTHER, by="data-team")
        assert result.recorded is False
        assert result.status is ProposalStatus.PENDING
        assert queue.get(submitted[0].id).status is ProposalStatus.PENDING
        assert not (root / "curated" / "value_maps.yaml").exists()
        assert "stays pending" in result.reason


class TestDeterminism:
    """Two renders of one queue are byte-identical; a naive clock is a bug, not a default."""

    def test_two_renders_are_identical(self, queue: ReviewQueue) -> None:
        seed(queue, 12)
        first = build_digest(queue, clock=FakeClock())
        second = build_digest(queue, clock=FakeClock())
        assert first.id == second.id
        assert render_html(first) == render_html(second)
        assert render_text(first) == render_text(second)

    def test_language_changes_the_id(self, queue: ReviewQueue) -> None:
        seed(queue, 3)
        hebrew = build_digest(queue, clock=FakeClock())
        english = build_digest(queue, language=DigestLanguage.ENGLISH, clock=FakeClock())
        assert hebrew.id != english.id

    def test_a_new_pending_proposal_changes_the_id(self, queue: ReviewQueue) -> None:
        seed(queue, 3)
        before = build_digest(queue, clock=FakeClock())
        queue.submit(make(99, confidence=0.99))
        after = build_digest(queue, clock=FakeClock())
        assert before.id != after.id
        assert after.items[0].confidence == 0.99

    def test_a_naive_clock_is_refused(self, queue: ReviewQueue) -> None:
        class Naive:
            def now(self) -> datetime:
                return datetime(2026, 3, 4, 9, 30)

        with pytest.raises(DigestError, match="naive"):
            build_digest(queue, clock=Naive())

    def test_a_digest_serialises_and_round_trips(self, queue: ReviewQueue) -> None:
        seed(queue, 2)
        digest = build_digest(queue, clock=FakeClock())
        assert Digest.model_validate_json(digest.model_dump_json()) == digest
