"""H2: the advertised issuer must be reachable, not ``http://0.0.0.0:8765``.

The image sets ``MDQM_SERVER__HOST=0.0.0.0`` and an edge Route/Ingress terminates TLS, so the URL
derived from the bind address is both unroutable and the wrong scheme. ``MDQM_SERVER__PUBLIC_URL``
exists to say what clients actually use; these tests pin the default, the override, and the
validation that keeps a relative or scheme-less value out.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from mdqm.config.settings import ServerSettings, Settings
from mdqm.server.app import advertised_url

#: The image's bind address, as a string this test only ever passes to a pure function: nothing
#: here binds a socket, so the "binding to all interfaces" warning does not apply.
WILDCARD = "0.0.0.0"  # noqa: S104


def test_the_default_is_derived_from_the_bind_address() -> None:
    """Still the default on purpose: a laptop needs no configuration."""
    assert advertised_url(host="127.0.0.1", port=8765) == "http://127.0.0.1:8765"


def test_the_image_bind_address_is_what_needs_the_override() -> None:
    """This is the bug: the image's own bind address is not a URL any client can follow."""
    assert advertised_url(host=WILDCARD, port=8765) == "http://0.0.0.0:8765"
    assert (
        advertised_url(host=WILDCARD, port=8765, public_url="https://mdqm.internal")
        == "https://mdqm.internal"
    )


def test_a_trailing_slash_is_dropped() -> None:
    """The issuer is built by appending `/`, so a second slash would make `https://host//`."""
    assert advertised_url(host=WILDCARD, port=8765, public_url="https://mdqm.internal/") == (
        "https://mdqm.internal"
    )


def test_the_setting_comes_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MDQM_SERVER__PUBLIC_URL", "https://mdqm.internal")
    settings = Settings.load(None)

    assert settings.server.public_url == "https://mdqm.internal"


def test_the_setting_comes_from_the_yaml_file(tmp_path: Path) -> None:
    config = tmp_path / "mdqm.yaml"
    config.write_text("server:\n  public_url: https://mdqm.internal\n", encoding="utf-8")

    assert Settings.load(config).server.public_url == "https://mdqm.internal"


@pytest.mark.parametrize(
    "value",
    ["mdqm.internal", "/mcp", "ftp://mdqm.internal", "https://", "  "],
    ids=["no-scheme", "relative", "wrong-scheme", "no-host", "blank"],
)
def test_an_unusable_public_url_is_rejected(value: str) -> None:
    """``https://`` with no host is the mistake a hand-edited env var makes."""
    if not value.strip():
        # Blank means "not configured", which is allowed and falls back to the derived URL.
        assert ServerSettings(public_url=value).public_url is None
        return
    with pytest.raises(ValidationError):
        ServerSettings(public_url=value)


def test_the_documented_env_var_name_is_the_one_that_works(monkeypatch: pytest.MonkeyPatch) -> None:
    """docs/deployment.md names MDQM_SERVER__PUBLIC_URL; a rename here would break the docs."""
    monkeypatch.setenv("MDQM_SERVER__PUBLIC_URL", "https://mdqm.internal/")
    assert Settings.load(None).server.public_url == "https://mdqm.internal"


def test_the_image_does_not_hard_code_a_public_url() -> None:
    """Every deployment has its own hostname, so the platform supplies it."""
    dockerfile = (Path(__file__).resolve().parents[2] / "Dockerfile").read_text(encoding="utf-8")

    assert "MDQM_SERVER__PUBLIC_URL=" not in dockerfile
