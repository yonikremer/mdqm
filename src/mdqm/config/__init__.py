"""``mdqm.config``: settings, sources and secret references.

Everything a deployment differs by lives here, never in code: server binding, paths, limits,
quiet windows, the model endpoint, SMTP, and the data sources with their secret *references*.
"""

from __future__ import annotations

from mdqm.config.quiet import QuietWindow
from mdqm.config.secrets import (
    EnvSecretProvider,
    FileSecretProvider,
    SecretNotFound,
    SecretProvider,
    check_secret_name,
)
from mdqm.config.settings import (
    AuthSettings,
    ConfigError,
    LimitsSettings,
    LoadLimits,
    ModelSettings,
    PathsSettings,
    RateLimitSettings,
    SecretsSettings,
    ServerSettings,
    Settings,
    SmtpSettings,
    SourceConfig,
    find_literal_secrets,
)

__all__ = [
    "AuthSettings",
    "ConfigError",
    "EnvSecretProvider",
    "FileSecretProvider",
    "LimitsSettings",
    "LoadLimits",
    "ModelSettings",
    "PathsSettings",
    "QuietWindow",
    "RateLimitSettings",
    "SecretNotFound",
    "SecretProvider",
    "SecretsSettings",
    "ServerSettings",
    "Settings",
    "SmtpSettings",
    "SourceConfig",
    "check_secret_name",
    "find_literal_secrets",
]
