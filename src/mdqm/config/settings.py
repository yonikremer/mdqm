"""``mdqm.config``: the server's settings and its secret references.

Settings come from, highest priority first:

1. constructor arguments (tests, embedding),
2. ``MDQM_*`` environment variables (nested sections use a double underscore:
   ``MDQM_SERVER__PORT=9000``, ``MDQM_MODEL__API_KEY=...``),
3. an optional ``mdqm.yaml`` (never contains a credential, see below),
4. the defaults declared here.

Credentials never live in YAML, code, logs or error messages. YAML names a *reference*
(``secret: crm_ro``) that a :class:`~mdqm.config.secrets.SecretProvider` resolves at connect
time. A literal credential in the YAML is rejected before validation, with the key path and a
pointer to ``secret:``.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextvars import ContextVar
from datetime import datetime, time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic.fields import FieldInfo
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict
from ruamel.yaml import YAML
from ruamel.yaml.error import MarkedYAMLError, YAMLError

from mdqm.config.quiet import QuietWindow
from mdqm.config.secrets import (
    EnvSecretProvider,
    FileSecretProvider,
    SecretProvider,
    check_secret_name,
)

__all__ = [
    "AuthSettings",
    "ConfigError",
    "LimitsSettings",
    "LoadLimits",
    "ModelSettings",
    "PathsSettings",
    "QuietWindow",
    "RateLimitSettings",
    "SecretsSettings",
    "ServerSettings",
    "Settings",
    "SmtpSettings",
    "SourceConfig",
    "StatsSettings",
    "find_literal_secrets",
    "pin_yaml_file",
    "resolve_yaml_file",
]

DEFAULT_YAML_FILE = "mdqm.yaml"

#: Keys that mean "this is a credential". ``secret`` is absent on purpose: it is the ref field.
CREDENTIAL_KEYS: frozenset[str] = frozenset(
    {
        "apikey",
        "api_key",
        "access_key",
        "access_token",
        "auth_token",
        "client_secret",
        "passwd",
        "password",
        "pwd",
        "refresh_token",
        "secret_key",
        "token",
    }
)

_FIX_HINT = (
    "put the value in an environment variable or a secret file and reference it with "
    "`secret: <name>` (SourceConfig.secret / auth.secret); literal credentials are never "
    "allowed in YAML"
)


class ConfigError(ValueError):
    """A configuration problem an operator must fix.

    ``problems`` carry the file and the key path (``server.prot``), never a value, so the
    message is safe to log and to show to an agent.
    """

    def __init__(self, source: str, problems: list[str] | tuple[str, ...]) -> None:
        self.source = source
        self.problems = tuple(problems)
        super().__init__(f"{source}: " + "; ".join(self.problems))


def find_literal_secrets(node: Any, path: str = "") -> list[str]:
    """Return the key paths that hold a literal credential.

    Walks the parsed YAML before validation so the error names the offending key instead of
    the generic "extra input" message that ``extra="forbid"`` would produce.
    """
    found: list[str] = []
    if isinstance(node, Mapping):
        for raw_key, value in node.items():
            key = str(raw_key)
            here = f"{path}.{key}" if path else key
            if key.lower() in CREDENTIAL_KEYS and value not in (None, ""):
                found.append(f"{here}: {_FIX_HINT}")
            found.extend(find_literal_secrets(value, here))
    elif isinstance(node, (list, tuple)):
        for index, item in enumerate(node):
            found.extend(find_literal_secrets(item, f"{path}[{index}]"))
    return found


# --- sections ---------------------------------------------------------------------------


class _Section(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ServerSettings(_Section):
    """Where the MCP server binds, and the URL clients are told to use. Loopback by default.

    ``public_url`` is separate from ``host``/``port`` on purpose: the image binds ``0.0.0.0`` and an
    edge proxy terminates TLS, so a derived ``http://0.0.0.0:8765`` is both unroutable and the
    wrong scheme. It is what ``/mcp``'s auth metadata advertises as the issuer.
    """

    host: str = "127.0.0.1"
    port: int = Field(default=8765, ge=1, le=65535)
    public_url: str | None = None

    @field_validator("public_url")
    @classmethod
    def _public_url(cls, value: str | None) -> str | None:
        """An absolute http(s) URL with a host, or nothing (the derived URL is used then)."""
        if value is None or not value.strip():
            return None
        candidate = value.strip().rstrip("/")
        parsed = urlsplit(candidate)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError("public_url must be an absolute http:// or https:// URL with a host")
        return candidate


class PathsSettings(_Section):
    """Filesystem locations. Relative paths resolve against the working directory."""

    metadata_db: Path = Path("data") / "mdqm.sqlite"
    semantic_dir: Path = Path("semantic")
    cache_dir: Path = Path("cache")

    @property
    def spill_dir(self) -> Path:
        """DuckDB's ``temp_directory``: inside the cache tree, never the working directory.

        DuckDB's default is ``.tmp`` relative to the process CWD, which is ``/app`` in the image
        and read-only there, so a query that spills would fail. ``TMPDIR`` does not move it.
        """
        return self.cache_dir / "spill"


class ModelSettings(_Section):
    """The on-prem, OpenAI-compatible chat endpoint used by the bootstrap pipeline."""

    base_url: str = ""
    model: str = ""
    api_key: SecretStr | None = None
    timeout_s: float = Field(default=120.0, gt=0)
    max_retries: int = Field(default=2, ge=0, le=5)

    @field_validator("base_url")
    @classmethod
    def _url_shape(cls, value: str) -> str:
        if value and not value.startswith(("http://", "https://")):
            raise ValueError("base_url must start with http:// or https://")
        return value.rstrip("/")


class SmtpSettings(_Section):
    """Internal SMTP relay for the weekly digest. ``host`` empty means "not configured"."""

    host: str = ""
    port: int = Field(default=25, ge=1, le=65535)
    sender: str = ""
    username: str = ""
    password: SecretStr | None = None
    starttls: bool = True

    @property
    def enabled(self) -> bool:
        """True when a relay is configured (the digest then uses email)."""
        return bool(self.host)


class SecretsSettings(_Section):
    """How secret references are resolved; see :meth:`Settings.secret_provider`."""

    env_prefix: str = "MDQM_SECRET_"
    dir: Path | None = None


class StatsSettings(_Section):
    """Continuous statistics collection. It is heavy, so the default window is overnight.

    ``quiet`` is 21:00-07:00 Asia/Jerusalem. A deployment that sets ``Settings.quiet_windows``
    uses those instead. ``sample_rows`` caps every sampled read.
    """

    enabled: bool = True
    sample_rows: int = Field(default=5_000, ge=1)
    top_k: int = Field(default=50, ge=1)
    max_pairs: int = Field(default=12, ge=0)
    null_warn_share: float = Field(default=0.2, ge=0.0, le=1.0)
    result_warn_rows: int = Field(default=10_000, ge=1)
    result_block_rows: int = Field(default=100_000, ge=1)
    quiet: QuietWindow = Field(
        default_factory=lambda: QuietWindow(tz="Asia/Jerusalem", start=time(21, 0), end=time(7, 0))
    )


class LimitsSettings(_Section):
    """Runtime limits; the governor enforces them."""

    max_rows: int = Field(default=1000, ge=1)
    max_concurrency: int = Field(default=8, ge=1)
    per_source_concurrency: int = Field(default=3, ge=1)
    query_timeout_s: float = Field(default=60.0, gt=0)
    max_partitions_scanned: int = Field(default=8, ge=1)


class LoadLimits(_Section):
    """Per-source load limits. Conservative: primary databases have no replicas."""

    max_concurrency: int = Field(default=3, ge=1)
    query_timeout_s: float | None = Field(default=None, gt=0)


class AuthSettings(_Section):
    """How an API source authenticates: a static header whose value is a secret reference."""

    header: str = "Authorization"
    secret: str

    @field_validator("secret")
    @classmethod
    def _ref(cls, value: str) -> str:
        return check_secret_name(value)


class RateLimitSettings(_Section):
    """Shared client budget for an API source. The real budgets are unknown: stay low."""

    shared_pool: bool = True
    rps: float = Field(default=1.0, gt=0)


class SourceConfig(_Section):
    """One data source. ``secret`` is a *name* a SecretProvider resolves, never a value."""

    name: str
    adapter: str
    secret: str | None = None
    spec: str | None = None
    load: LoadLimits = Field(default_factory=LoadLimits)
    auth: AuthSettings | None = None
    rate_limit: RateLimitSettings | None = None
    options: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def _name(cls, value: str) -> str:
        return check_secret_name(value)

    @field_validator("secret")
    @classmethod
    def _secret_ref(cls, value: str | None) -> str | None:
        return None if value is None else check_secret_name(value)

    @field_validator("adapter")
    @classmethod
    def _adapter(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("adapter must name an adapter kind, e.g. 'postgresql'")
        return value


# --- the settings object ----------------------------------------------------------------

#: YAML path of the load in progress. A ContextVar, not a module global, so two concurrent
#: loads (tests, one server per thread) cannot see each other's file.
_YAML_FILE: ContextVar[Path | None] = ContextVar("mdqm_yaml_file", default=None)


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(str(path), ["file not found"])
    # typ="safe" so a YAML file can never construct arbitrary Python objects.
    loader = YAML(typ="safe")
    try:
        data = loader.load(path.read_text(encoding="utf-8"))
    except MarkedYAMLError as exc:
        mark = exc.problem_mark
        where = f"line {mark.line + 1}, column {mark.column + 1}" if mark else "unknown position"
        # `from None`: ruamel echoes the offending source line, which a broken file might
        # have filled with a credential literal. Position + problem text is enough to fix it.
        raise ConfigError(str(path), [f"invalid YAML at {where}: {exc.problem}"]) from None
    except YAMLError as exc:
        raise ConfigError(str(path), [f"invalid YAML: {type(exc).__name__}"]) from None
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(str(path), ["the top level must be a mapping of setting names"])
    problems = find_literal_secrets(data)
    if problems:
        raise ConfigError(str(path), problems)
    return {str(key): value for key, value in data.items()}


class _YamlSettingsSource(PydanticBaseSettingsSource):
    """The ``mdqm.yaml`` file as the lowest-priority source above the field defaults."""

    def get_field_value(self, field: FieldInfo, field_name: str) -> tuple[Any, str, bool]:
        return None, field_name, False  # unused: __call__ returns the whole mapping

    def __call__(self) -> dict[str, Any]:
        path = _YAML_FILE.get()
        return {} if path is None else _read_yaml_mapping(path)


def resolve_yaml_file(yaml_file: Path | str | None = None) -> Path | None:
    """Which YAML file to read: the argument, else the pinned file, else ``./mdqm.yaml``.

    Returns ``None`` when there is none, which means "read the environment only". The order matters:
    a CLI entry point pins the operator's file once (:func:`pin_yaml_file`), and every later
    ``Settings.load()`` in that process must resolve to the same file rather than re-guessing from
    the working directory.
    """
    if yaml_file is not None:
        return Path(yaml_file)
    pinned = _YAML_FILE.get()
    if pinned is not None:
        return pinned
    return Path(DEFAULT_YAML_FILE) if Path(DEFAULT_YAML_FILE).is_file() else None


def pin_yaml_file(yaml_file: Path | str | None = None) -> Path | None:
    """Pin the YAML file for the rest of the process, and return it.

    :meth:`Settings.load` scopes the file to one call, which is what a test wants. A CLI entry
    point wants the opposite: one operator config, honoured by every module the command reaches,
    including the ones that build ``Settings()`` themselves. Pinning is not a global mutable
    setting -- it is still a ContextVar, so a second thread or test can load a different file.

    ``None`` means "the file in the working directory, if there is one", never "keep the current
    pin": that is what makes ``pin_yaml_file(None)`` a reset.
    """
    path = (
        Path(yaml_file)
        if yaml_file is not None
        else (Path(DEFAULT_YAML_FILE) if Path(DEFAULT_YAML_FILE).is_file() else None)
    )
    _YAML_FILE.set(path)
    return path


def _format_problems(exc: ValidationError) -> list[str]:
    problems: list[str] = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"]) or "<document>"
        # Only loc and msg: pydantic also records the offending input, and that is exactly
        # what must never reach a log line or an agent.
        problems.append(f"{location}: {error['msg']}")
    return problems


class Settings(BaseSettings):
    """Resolved server settings. Immutable; build one per process (and per test)."""

    model_config = SettingsConfigDict(
        env_prefix="MDQM_",
        env_nested_delimiter="__",
        extra="forbid",
        frozen=True,
    )

    server: ServerSettings = Field(default_factory=ServerSettings)
    paths: PathsSettings = Field(default_factory=PathsSettings)
    model: ModelSettings = Field(default_factory=ModelSettings)
    smtp: SmtpSettings = Field(default_factory=SmtpSettings)
    limits: LimitsSettings = Field(default_factory=LimitsSettings)
    secrets: SecretsSettings = Field(default_factory=SecretsSettings)
    quiet_windows: list[QuietWindow] = Field(default_factory=list)
    stats: StatsSettings = Field(default_factory=StatsSettings)
    sources: list[SourceConfig] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _sources_from_mapping(cls, data: Any) -> Any:
        """Accept ``sources: {crm: {...}}`` (readable) as well as the list form."""
        if not isinstance(data, dict):
            return data
        raw = data.get("sources")
        if not isinstance(raw, dict):
            return data
        items: list[Any] = []
        for name, value in raw.items():
            item = dict(value) if isinstance(value, dict) else {}
            item.setdefault("name", name)
            items.append(item)
        return {**data, "sources": items}

    @model_validator(mode="after")
    def _unique_sources(self) -> Settings:
        seen = [source.name for source in self.sources]
        duplicates = sorted({name for name in seen if seen.count(name) > 1})
        if duplicates:
            raise ValueError(f"duplicate source name(s): {', '.join(duplicates)}")
        return self

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Constructor > environment > ``mdqm.yaml`` > defaults (env overrides the file)."""
        return (
            init_settings,
            env_settings,
            _YamlSettingsSource(settings_cls),
            dotenv_settings,
            file_secret_settings,
        )

    @classmethod
    def load(cls, yaml_file: Path | str | None = None) -> Settings:
        """Load settings, raising :class:`ConfigError` with file and key path on failure.

        ``yaml_file=None`` uses ``mdqm.yaml`` in the working directory when it exists and
        behaves as if it were absent otherwise. Environment variables still win over the file.
        """
        path = resolve_yaml_file(yaml_file)
        token = _YAML_FILE.set(path)
        try:
            return cls()
        except ValidationError as exc:
            label = str(path) if path else "environment variables"
            raise ConfigError(label, _format_problems(exc)) from None
        finally:
            _YAML_FILE.reset(token)

    def is_quiet(self, now: datetime) -> bool:
        """True when heavy bootstrap work may run at ``now``.

        No configured window means "not restricted": a fresh install must be able to
        bootstrap on a laptop. Configure windows to protect primary databases.
        """
        if not self.quiet_windows:
            return True
        return any(window.is_quiet(now) for window in self.quiet_windows)

    def stats_windows(self) -> tuple[QuietWindow, ...]:
        """Windows for statistics collection: the deployment's, or the overnight default."""
        if self.quiet_windows:
            return tuple(self.quiet_windows)
        return (self.stats.quiet,)

    def stats_collection_open(self, now: datetime, *, force: bool = False) -> bool:
        """True when a statistics pass may start. ``force`` is an operator override.

        A disabled collector never runs, including under ``force``: force overrides the clock,
        not the switch.
        """
        if not self.stats.enabled:
            return False
        if force:
            return True
        return any(window.is_quiet(now) for window in self.stats_windows())

    def secret_provider(self) -> SecretProvider:
        """The provider for this deployment: secret files when a directory is set, else env."""
        if self.secrets.dir is not None:
            return FileSecretProvider(self.secrets.dir)
        return EnvSecretProvider(self.secrets.env_prefix)
