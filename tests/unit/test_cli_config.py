"""H1: every CLI entry point reads ``mdqm.yaml``, not just ``Settings.load()``.

The bug this pins: only :meth:`Settings.load` wires the YAML file in. ``mdqm/cli.py`` used a bare
``Settings()``, so an operator who mounted a config got the defaults -- every ``query`` failing
"unknown table" while ``mdqm probe`` looked green. The proof is end to end and through the code path
the operator actually runs (``mdqm review list``), not through ``Settings`` alone:

* with ``mdqm.yaml`` in the working directory naming a metadata DB under ``tmp_path``, the command
  must create *that* file. A silent default creates ``data/mdqm.sqlite`` instead and the assertion
  on the file fails;
* the same holds when the file is passed as ``--config``;
* the pinned file must also be visible to the modules that build ``Settings()`` themselves (the
  probe and the bootstrap pipeline do), which is what ``pin_yaml_file`` is for.

A CliRunner invocation runs in this process, so the pinned ContextVar outlives the call: the
``unpin`` fixture puts it back, otherwise one test's config would leak into the next.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from typer.testing import CliRunner

from mdqm.cli import app
from mdqm.config.settings import Settings, pin_yaml_file

RUNNER = CliRunner()


@pytest.fixture(autouse=True)
def unpin() -> Iterator[None]:
    """Leave the YAML ContextVar as it was: ``None`` is the default value."""
    yield
    pin_yaml_file(None)


def write_config(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    return path


def test_a_config_in_the_working_directory_is_honoured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The operator mounts ``mdqm.yaml`` next to the server and expects it to be read."""
    monkeypatch.chdir(tmp_path)
    database = tmp_path / "from-file.sqlite"
    write_config(tmp_path / "mdqm.yaml", f"paths:\n  metadata_db: {database.as_posix()}\n")

    result = RUNNER.invoke(app, ["review", "list"])

    assert result.exit_code == 0
    assert database.is_file(), "the CLI ignored mdqm.yaml and opened the default database"


def test_config_option_names_the_file(tmp_path: Path) -> None:
    database = tmp_path / "from-option.sqlite"
    config = write_config(tmp_path / "elsewhere.yaml", f"paths:\n  metadata_db: {database}\n")

    result = RUNNER.invoke(app, ["--config", str(config), "review", "list"])

    assert result.exit_code == 0
    assert database.is_file()


def test_a_broken_config_fails_loudly_and_names_the_file(tmp_path: Path) -> None:
    """Exit 1 with the file and the key path: a nightly job must notice, an agent must see why."""
    config = write_config(tmp_path / "mdqm.yaml", "server:\n  port: not-a-number\n")

    result = RUNNER.invoke(app, ["--config", str(config), "review", "list"])

    assert result.exit_code == 1
    assert "mdqm.yaml" in result.stderr or str(config) in result.stderr
    assert "server.port" in result.stderr


def test_a_missing_config_file_is_an_error_not_a_silent_default(tmp_path: Path) -> None:
    result = RUNNER.invoke(app, ["--config", str(tmp_path / "nope.yaml"), "review", "list"])

    assert result.exit_code == 1
    assert "nope.yaml" in result.stderr


def test_the_pinned_file_reaches_modules_that_build_settings_themselves(tmp_path: Path) -> None:
    """``mdqm probe`` and ``mdqm bootstrap`` call ``Settings()`` directly, in their own modules.

    Pinning is what makes one operator config apply to them too; without it those commands read the
    environment only, which is the same silent-ignore bug one layer down.
    """
    assert Settings().server.port == 8765, "the default must still be the default"

    config = write_config(tmp_path / "pinned.yaml", "server:\n  port: 19999\n")
    RUNNER.invoke(app, ["--config", str(config), "review", "list"])

    # After the CLI pinned it, a bare Settings() -- what probe/cli.py and pipeline/cli.py do -- sees
    # the same file. The unpin fixture puts this back for the next test.
    assert Settings().server.port == 19999


def test_the_version_flag_still_works_without_a_config() -> None:
    result = RUNNER.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert result.stdout.strip()
