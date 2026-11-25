"""Tests for mdqm.config: settings sources, secret providers, quiet windows."""

from __future__ import annotations

import os
import re
from datetime import datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
from pydantic import SecretStr, ValidationError

from mdqm.config import (
    ConfigError,
    EnvSecretProvider,
    FileSecretProvider,
    QuietWindow,
    SecretNotFound,
    SecretProvider,
    Settings,
    SourceConfig,
    check_secret_name,
    find_literal_secrets,
)

SECRET_VALUE = "throwaway-value-8f2c1d"  # noqa: S105 - throwaway, asserted to never leak


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Nothing MDQM_* from the host may leak into these tests."""
    for name in list(os.environ):
        if name.startswith("MDQM_"):
            monkeypatch.delenv(name, raising=False)


def write_yaml(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    return path


# --- acceptance: a literal password in YAML -------------------------------------------------


def test_yaml_literal_password_fails_pointing_to_secret(tmp_path: Path) -> None:
    yaml_file = write_yaml(
        tmp_path / "mdqm.yaml",
        """
sources:
  crm:
    adapter: postgresql
    password: hunter2
""",
    )
    with pytest.raises(ConfigError) as caught:
        Settings.load(yaml_file)
    message = str(caught.value)
    assert "sources.crm.password" in message
    assert "secret: <name>" in message
    assert "hunter2" not in message  # the literal never reaches the message
    assert str(yaml_file) in message


@pytest.mark.parametrize(
    "key", ["password", "passwd", "pwd", "api_key", "apikey", "token", "client_secret"]
)
def test_every_credential_spelling_is_rejected(tmp_path: Path, key: str) -> None:
    yaml_file = write_yaml(
        tmp_path / "mdqm.yaml", f"sources:\n  crm:\n    adapter: postgresql\n    {key}: x\n"
    )
    with pytest.raises(ConfigError) as caught:
        Settings.load(yaml_file)
    assert f"sources.crm.{key}" in str(caught.value)


def test_literal_secret_outside_sources_is_rejected(tmp_path: Path) -> None:
    yaml_file = write_yaml(tmp_path / "mdqm.yaml", "model:\n  api_key: sk-live-123\n")
    with pytest.raises(ConfigError) as caught:
        Settings.load(yaml_file)
    assert "model.api_key" in str(caught.value)
    assert "sk-live-123" not in str(caught.value)


def test_literal_secret_in_a_list_is_rejected_with_index(tmp_path: Path) -> None:
    yaml_file = write_yaml(
        tmp_path / "mdqm.yaml",
        "sources:\n  - name: crm\n    adapter: postgresql\n    options:\n      token: abc\n",
    )
    with pytest.raises(ConfigError) as caught:
        Settings.load(yaml_file)
    assert "sources[0].options.token" in str(caught.value)


def test_find_literal_secrets_ignores_empty_and_reference_fields() -> None:
    data = {
        "sources": {"crm": {"adapter": "postgresql", "secret": "crm_ro", "password": ""}},
        "smtp": {"password": None},
    }
    assert find_literal_secrets(data) == []


def test_secret_reference_is_accepted_and_stays_a_plain_name(tmp_path: Path) -> None:
    yaml_file = write_yaml(
        tmp_path / "mdqm.yaml",
        "sources:\n  crm:\n    adapter: postgresql\n    secret: crm_ro\n",
    )
    settings = Settings.load(yaml_file)
    (source,) = settings.sources
    assert source.secret == "crm_ro"


# --- acceptance: no secret in repr / json / str ---------------------------------------------


def test_secret_value_never_in_repr_str_or_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MDQM_MODEL__API_KEY", SECRET_VALUE)
    monkeypatch.setenv("MDQM_SMTP__PASSWORD", SECRET_VALUE)
    yaml_file = write_yaml(tmp_path / "mdqm.yaml", "smtp:\n  host: mail.internal\n")
    settings = Settings.load(yaml_file)

    assert isinstance(settings.model.api_key, SecretStr)
    assert settings.model.api_key.get_secret_value() == SECRET_VALUE
    for rendered in (
        repr(settings),
        str(settings),
        settings.model_dump_json(),
        str(settings.model_dump()),
        repr(settings.model),
        repr(settings.smtp),
        settings.model.model_dump_json(),
    ):
        assert SECRET_VALUE not in rendered
    assert settings.model_dump()["model"]["api_key"] == SecretStr(SECRET_VALUE)


def test_secret_does_not_leak_into_an_error_caused_by_another_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The value arrives from the environment (allowed) and a sibling field is broken.
    monkeypatch.setenv("MDQM_MODEL__API_KEY", SECRET_VALUE)
    monkeypatch.setenv("MDQM_SERVER__PORT", "not-a-port")
    with pytest.raises(ConfigError) as caught:
        Settings.load(None)
    assert "server.port" in str(caught.value)
    assert SECRET_VALUE not in str(caught.value)
    # No chained cause: pydantic errors can carry the whole input mapping.
    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__ is True


def test_config_error_does_not_echo_a_secret_value(tmp_path: Path) -> None:
    # A wrong *type* for a secret-free field still must not echo anything else from the file.
    yaml_file = write_yaml(
        tmp_path / "mdqm.yaml", "server:\n  port: not-a-number\nlimits:\n  max_rows: 0\n"
    )
    with pytest.raises(ConfigError) as caught:
        Settings.load(yaml_file)
    message = str(caught.value)
    assert "server.port" in message
    assert "limits.max_rows" in message
    assert "input_value" not in message


# --- acceptance: env overrides the file, unknown keys are rejected with a key path ----------


def test_env_overrides_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    yaml_file = write_yaml(
        tmp_path / "mdqm.yaml",
        "server:\n  port: 1111\npaths:\n  cache_dir: /from/file\n",
    )
    monkeypatch.setenv("MDQM_SERVER__PORT", "2222")
    monkeypatch.setenv("MDQM_PATHS__CACHE_DIR", "/from/env")
    settings = Settings.load(yaml_file)
    assert settings.server.port == 2222
    assert settings.paths.cache_dir == Path("/from/env")
    # Untouched values still come from the file.
    assert settings.server.host == "127.0.0.1"


def test_constructor_argument_beats_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MDQM_SERVER__PORT", "2222")
    assert Settings(server={"port": 3333}).server.port == 3333


def test_yaml_beats_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)  # no mdqm.yaml here, so the default applies
    yaml_file = write_yaml(tmp_path / "mdqm.yaml", "server:\n  port: 4444\n")
    assert Settings.load(yaml_file).server.port == 4444
    assert Settings.load(None).server.port == 4444


@pytest.mark.parametrize(
    ("body", "path_in_error"),
    [
        ("nope: 1\n", "nope"),
        ("server:\n  prot: 1\n", "server.prot"),
        ("sources:\n  - name: crm\n    adapter: pg\n    adapterr: pg\n", "sources.0.adapterr"),
        ("quiet_windows:\n  - tz: Asia/Jerusalem\n    start: '21:00'\n    endd: '07:00'\n",
         "quiet_windows.0.endd"),
    ],
)
def test_unknown_key_is_rejected_with_its_path(
    tmp_path: Path, body: str, path_in_error: str
) -> None:
    yaml_file = write_yaml(tmp_path / "mdqm.yaml", body)
    with pytest.raises(ConfigError) as caught:
        Settings.load(yaml_file)
    message = str(caught.value)
    assert path_in_error in message
    assert str(yaml_file) in message
    assert "Extra inputs are not permitted" in message


def test_unknown_nested_env_key_is_rejected_with_its_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MDQM_SERVER__PROT", "1")
    with pytest.raises(ConfigError) as caught:
        Settings.load(None)
    assert "server.prot" in str(caught.value)
    assert "environment variables" in str(caught.value)


def test_missing_yaml_file_names_the_file(tmp_path: Path) -> None:
    missing = tmp_path / "nope.yaml"
    with pytest.raises(ConfigError) as caught:
        Settings.load(missing)
    assert str(missing) in str(caught.value)
    assert "file not found" in str(caught.value)


def test_broken_yaml_reports_position_not_source_text(tmp_path: Path) -> None:
    yaml_file = write_yaml(tmp_path / "mdqm.yaml", "server: {port: 1\n")
    with pytest.raises(ConfigError) as caught:
        Settings.load(yaml_file)
    assert re.search(r"line \d+, column \d+", str(caught.value))
    assert "port: 1" not in str(caught.value)  # the offending source line is not echoed


def test_non_mapping_root_is_rejected(tmp_path: Path) -> None:
    yaml_file = write_yaml(tmp_path / "mdqm.yaml", "- a\n- b\n")
    with pytest.raises(ConfigError) as caught:
        Settings.load(yaml_file)
    assert "top level must be a mapping" in str(caught.value)


def test_default_yaml_file_is_picked_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    write_yaml(tmp_path / "mdqm.yaml", "server:\n  port: 5555\n")
    assert Settings.load().server.port == 5555


# --- sources --------------------------------------------------------------------------------


def test_sources_accepts_mapping_and_list_forms(tmp_path: Path) -> None:
    mapped = write_yaml(
        tmp_path / "a.yaml",
        """
sources:
  crm: {adapter: postgresql, secret: crm_ro, load: {max_concurrency: 2}}
  api_fx:
    adapter: openapi
    spec: specs/fx.yaml
    auth: {header: X-API-Key, secret: fx_project_key}
    rate_limit: {shared_pool: true, rps: 2}
""",
    )
    crm, api = Settings.load(mapped).sources
    assert (crm.name, crm.adapter, crm.secret, crm.load.max_concurrency) == (
        "crm",
        "postgresql",
        "crm_ro",
        2,
    )
    assert api.auth is not None and api.auth.secret == "fx_project_key"
    assert api.rate_limit is not None and api.rate_limit.rps == 2

    listed = write_yaml(tmp_path / "b.yaml", "sources:\n  - {name: crm, adapter: postgresql}\n")
    assert [s.name for s in Settings.load(listed).sources] == ["crm"]


def test_duplicate_source_names_are_rejected(tmp_path: Path) -> None:
    yaml_file = write_yaml(
        tmp_path / "mdqm.yaml",
        "sources:\n  - {name: crm, adapter: postgresql}\n  - {name: crm, adapter: oracle}\n",
    )
    with pytest.raises(ConfigError) as caught:
        Settings.load(yaml_file)
    assert "duplicate source name" in str(caught.value)
    assert "crm" in str(caught.value)


@pytest.mark.parametrize(
    "name", ["../etc/passwd", "a/b", "", "crm ro", "crm\ty", "crm:secret", "http://x"]
)
def test_bad_names_are_rejected(name: str) -> None:
    with pytest.raises(ValueError, match="invalid secret name"):
        check_secret_name(name)


def test_source_config_requires_an_adapter() -> None:
    with pytest.raises(ValidationError):
        SourceConfig(name="crm", adapter="  ")


# --- acceptance: quiet windows --------------------------------------------------------------


def window(start: str, end: str, tz: str = "Asia/Jerusalem") -> QuietWindow:
    return QuietWindow(tz=tz, start=time.fromisoformat(start), end=time.fromisoformat(end))


@pytest.mark.parametrize(
    ("hhmm", "expected"),
    [
        ("20:59", False),
        ("21:00", True),  # start is inclusive
        ("23:59", True),
        ("00:00", True),  # crosses midnight
        ("03:30", True),
        ("06:59", True),
        ("07:00", False),  # end is exclusive
        ("12:00", False),
    ],
)
def test_cross_midnight_window(hhmm: str, expected: bool) -> None:
    quiet = window("21:00", "07:00")
    hour, minute = (int(part) for part in hhmm.split(":"))
    now = datetime(2026, 3, 10, hour, minute, tzinfo=ZoneInfo("Asia/Jerusalem"))
    assert quiet.is_quiet(now) is expected


@pytest.mark.parametrize(
    ("hhmm", "expected"),
    [("08:59", False), ("09:00", True), ("12:00", True), ("16:59", True), ("17:00", False)],
)
def test_same_day_window(hhmm: str, expected: bool) -> None:
    quiet = window("09:00", "17:00", tz="UTC")
    hour, minute = (int(part) for part in hhmm.split(":"))
    assert quiet.is_quiet(datetime(2026, 3, 10, hour, minute, tzinfo=ZoneInfo("UTC"))) is expected


def test_aware_datetime_in_another_zone_is_converted() -> None:
    quiet = window("21:00", "07:00")  # Asia/Jerusalem, UTC+2 in March (before DST)
    # 19:30 UTC == 21:30 in Jerusalem: quiet.
    assert quiet.is_quiet(datetime(2026, 3, 10, 19, 30, tzinfo=ZoneInfo("UTC"))) is True
    # 18:30 UTC == 20:30 in Jerusalem: not quiet.
    assert quiet.is_quiet(datetime(2026, 3, 10, 18, 30, tzinfo=ZoneInfo("UTC"))) is False


def test_naive_datetime_is_read_as_local_wall_clock() -> None:
    quiet = window("21:00", "07:00")
    assert quiet.is_quiet(datetime(2026, 3, 10, 23, 0)) is True
    assert quiet.is_quiet(datetime(2026, 3, 10, 12, 0)) is False


def test_zero_length_window_is_never_quiet() -> None:
    quiet = window("00:00", "00:00", tz="UTC")
    for hour in range(24):
        assert quiet.is_quiet(datetime(2026, 3, 10, hour, 0, tzinfo=ZoneInfo("UTC"))) is False


def test_unknown_timezone_is_rejected_with_the_value() -> None:
    with pytest.raises(ValidationError) as caught:
        QuietWindow(tz="Mars/Olympus", start=time(21), end=time(7))
    assert "Mars/Olympus" in str(caught.value)


def test_time_with_offset_is_rejected() -> None:
    with pytest.raises(ValidationError):
        QuietWindow(tz="UTC", start="21:00+02:00", end="07:00")  # type: ignore[arg-type]


def test_quiet_windows_load_from_yaml(tmp_path: Path) -> None:
    yaml_file = write_yaml(
        tmp_path / "mdqm.yaml",
        "quiet_windows:\n  - {tz: Asia/Jerusalem, start: '21:00', end: '07:00'}\n",
    )
    (quiet,) = Settings.load(yaml_file).quiet_windows
    assert (quiet.tz, quiet.start, quiet.end) == ("Asia/Jerusalem", time(21), time(7))
    assert quiet.zone.key == "Asia/Jerusalem"


@pytest.mark.parametrize(
    ("hhmm", "expected"),
    [("22:00", True), ("02:00", True), ("10:00", False)],
)
def test_settings_is_quiet_any_window(hhmm: str, expected: bool) -> None:
    hour, minute = (int(part) for part in hhmm.split(":"))
    settings = Settings(
        quiet_windows=[
            {"tz": "Asia/Jerusalem", "start": "21:00", "end": "07:00"},
            {"tz": "Asia/Jerusalem", "start": "01:00", "end": "03:00"},
        ]
    )
    now = datetime(2026, 3, 10, hour, minute, tzinfo=ZoneInfo("Asia/Jerusalem"))
    assert settings.is_quiet(now) is expected


def test_no_configured_window_means_unrestricted() -> None:
    assert Settings().quiet_windows == []
    assert Settings().is_quiet(datetime(2026, 3, 10, 12, 0)) is True


# --- secrets: env provider ------------------------------------------------------------------


def test_env_provider_maps_reference_to_variable() -> None:
    provider = EnvSecretProvider(environ={"MDQM_SECRET_CRM_RO": SECRET_VALUE})
    assert provider.var_name("crm_ro") == "MDQM_SECRET_CRM_RO"
    assert provider.var_name("api-fx.prod") == "MDQM_SECRET_API_FX_PROD"
    assert provider.get("crm_ro") == SECRET_VALUE
    assert isinstance(provider, SecretProvider)
    assert SECRET_VALUE not in repr(provider)


def test_env_provider_missing_secret_names_where_it_looked() -> None:
    provider = EnvSecretProvider(environ={})
    with pytest.raises(SecretNotFound) as caught:
        provider.get("crm_ro")
    assert "crm_ro" in str(caught.value)
    assert "MDQM_SECRET_CRM_RO" in str(caught.value)


def test_env_provider_custom_prefix() -> None:
    provider = EnvSecretProvider("ACME_", environ={"ACME_ERP": SECRET_VALUE})
    assert provider.get("erp") == SECRET_VALUE


def test_env_provider_rejects_bad_reference_names() -> None:
    with pytest.raises(ValueError, match="invalid secret name"):
        EnvSecretProvider(environ={}).get("../etc/passwd")


def test_env_provider_reads_process_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MDQM_SECRET_BILLING_RO", SECRET_VALUE)
    assert Settings().secret_provider().get("billing_ro") == SECRET_VALUE


# --- secrets: file provider -----------------------------------------------------------------


def test_file_provider_reads_mounted_files(tmp_path: Path) -> None:
    (tmp_path / "crm_ro").write_text(SECRET_VALUE + "\n", encoding="utf-8")
    provider = FileSecretProvider(tmp_path)
    assert provider.get("crm_ro") == SECRET_VALUE
    assert provider.root == tmp_path
    assert SECRET_VALUE not in repr(provider)


def test_file_provider_strips_crlf(tmp_path: Path) -> None:
    (tmp_path / "crm_ro").write_bytes((SECRET_VALUE + "\r\n").encode("utf-8"))
    assert FileSecretProvider(tmp_path).get("crm_ro") == SECRET_VALUE


def test_file_provider_missing_file(tmp_path: Path) -> None:
    with pytest.raises(SecretNotFound) as caught:
        FileSecretProvider(tmp_path).get("crm_ro")
    assert "crm_ro" in str(caught.value)
    assert str(tmp_path) in str(caught.value)


def test_file_provider_empty_file(tmp_path: Path) -> None:
    (tmp_path / "crm_ro").write_text("\n", encoding="utf-8")
    with pytest.raises(SecretNotFound, match="empty"):
        FileSecretProvider(tmp_path).get("crm_ro")


def test_file_provider_error_does_not_contain_the_value(tmp_path: Path) -> None:
    provider = FileSecretProvider(tmp_path / "secrets")
    with pytest.raises(SecretNotFound) as caught:
        provider.get("crm_ro")
    assert caught.value.__cause__ is None
    assert SECRET_VALUE not in str(caught.value)


@pytest.mark.parametrize("name", ["../escape", "/etc/passwd", "a/b", ".."])
def test_file_provider_refuses_path_traversal(tmp_path: Path, name: str) -> None:
    with pytest.raises((ValueError, SecretNotFound)):
        FileSecretProvider(tmp_path).get(name)


def test_file_provider_does_not_follow_symlink_out_of_root(tmp_path: Path) -> None:
    root = tmp_path / "secrets"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text(SECRET_VALUE, encoding="utf-8")
    try:
        (root / "crm_ro").symlink_to(outside)
    except (OSError, NotImplementedError):  # pragma: no cover - Windows without developer mode
        pytest.skip("symlinks need privileges on this machine")
    assert FileSecretProvider(root).get("crm_ro") == SECRET_VALUE


def test_settings_choose_provider_by_configuration(tmp_path: Path) -> None:
    assert isinstance(Settings().secret_provider(), EnvSecretProvider)
    assert isinstance(
        Settings(secrets={"dir": str(tmp_path)}).secret_provider(), FileSecretProvider
    )


def test_secret_flows_from_file_to_settings_without_landing_in_yaml(tmp_path: Path) -> None:
    secrets_dir = tmp_path / "secrets"
    secrets_dir.mkdir()
    (secrets_dir / "crm_ro").write_text(SECRET_VALUE, encoding="utf-8")
    yaml_file = write_yaml(
        tmp_path / "mdqm.yaml",
        "sources:\n  crm:\n    adapter: postgresql\n    secret: crm_ro\n"
        f"secrets:\n  dir: {secrets_dir}\n",
    )
    settings = Settings.load(yaml_file)
    assert settings.secret_provider().get(settings.sources[0].secret or "") == SECRET_VALUE
    assert SECRET_VALUE not in yaml_file.read_text(encoding="utf-8")
    assert SECRET_VALUE not in settings.model_dump_json()


# --- limits and defaults --------------------------------------------------------------------


def test_defaults_are_conservative() -> None:
    settings = Settings()
    assert settings.server.host == "127.0.0.1"
    assert settings.limits.max_concurrency <= 8
    assert settings.limits.per_source_concurrency <= 3
    assert settings.model.max_retries == 2
    assert settings.smtp.enabled is False


def test_limits_come_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MDQM_LIMITS__MAX_ROWS", "50")
    monkeypatch.setenv("MDQM_LIMITS__PER_SOURCE_CONCURRENCY", "1")
    settings = Settings.load(None)
    assert settings.limits.max_rows == 50
    assert settings.limits.per_source_concurrency == 1


@pytest.mark.parametrize("port", [0, 70000, -1])
def test_out_of_range_port_is_rejected(port: int) -> None:
    with pytest.raises(ValidationError):
        Settings(server={"port": port})


def test_settings_are_frozen() -> None:
    settings = Settings()
    with pytest.raises(ValidationError):
        settings.server = Settings().server  # type: ignore[misc]


def test_model_base_url_must_be_http() -> None:
    with pytest.raises(ValidationError):
        Settings(model={"base_url": "glm.internal:8080"})
    assert Settings(model={"base_url": "http://glm.internal:8080/"}).model.base_url == (
        "http://glm.internal:8080"
    )
