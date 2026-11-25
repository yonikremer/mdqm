"""Secret providers: config refers to a secret by *name*, never by value.

Two providers cover the deployment shapes we know:

* :class:`EnvSecretProvider` reads an environment variable (development, the container env).
* :class:`FileSecretProvider` reads one file per secret from a directory (Docker/Kubernetes
  secret mounts, ``/run/secrets``).

Providers return the value to the caller and never log it. Errors name the secret and where
it was looked for, never what was found.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Protocol, runtime_checkable

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_ENV_SAFE_RE = re.compile(r"[^A-Za-z0-9_]")


class SecretNotFound(LookupError):
    """A referenced secret does not exist. The message names the reference, not a value."""

    def __init__(self, name: str, *, where: str) -> None:
        self.name = name
        self.where = where
        super().__init__(f"secret {name!r} not found: {where}")


def check_secret_name(name: str) -> str:
    """Validate a secret *reference* name.

    References are not values, so they may travel in YAML and logs. Restricting the alphabet
    keeps a reference from being usable as a path traversal or as a URL.
    """
    if not _NAME_RE.match(name):
        raise ValueError(
            f"invalid secret name {name!r}: use letters, digits, '_', '.', '-' "
            "(starting with a letter or digit); secret names are references, never values"
        )
    return name


@runtime_checkable
class SecretProvider(Protocol):
    """Resolve a secret name to its value. Raises :class:`SecretNotFound`."""

    def get(self, name: str) -> str: ...


class EnvSecretProvider:
    """Read secrets from environment variables: ``crm_ro`` -> ``MDQM_SECRET_CRM_RO``."""

    def __init__(
        self, prefix: str = "MDQM_SECRET_", environ: Mapping[str, str] | None = None
    ) -> None:
        self._prefix = prefix
        self._environ: Mapping[str, str] = os.environ if environ is None else environ

    def var_name(self, name: str) -> str:
        """The environment variable that holds ``name`` (uppercased, unsafe chars -> ``_``)."""
        check_secret_name(name)
        return self._prefix + _ENV_SAFE_RE.sub("_", name).upper()

    def get(self, name: str) -> str:
        var = self.var_name(name)
        try:
            value = self._environ[var]
        except KeyError:
            raise SecretNotFound(name, where=f"environment variable {var} is not set") from None
        if not value:
            raise SecretNotFound(name, where=f"environment variable {var} is empty")
        return value

    def __repr__(self) -> str:
        return f"EnvSecretProvider(prefix={self._prefix!r})"


class FileSecretProvider:
    """Read one file per secret under ``root`` (a Docker/Kubernetes secret directory)."""

    def __init__(self, root: Path | str) -> None:
        self._root = Path(root)

    @property
    def root(self) -> Path:
        """The directory the provider reads from."""
        return self._root

    def path_for(self, name: str) -> Path:
        """The file that holds ``name``; a name is a plain file name, never a path."""
        check_secret_name(name)
        if "/" in name or "\\" in name or name in {".", ".."}:
            raise ValueError(f"invalid secret name {name!r}: secret names are file names")
        return self._root / name

    def get(self, name: str) -> str:
        path = self.path_for(name)
        if not path.is_file():
            raise SecretNotFound(name, where=f"no secret file {path}")
        # Trailing newlines come from `echo` and from mounted secret files; they are not data.
        value = path.read_text(encoding="utf-8").rstrip("\r\n")
        if not value:
            raise SecretNotFound(name, where=f"secret file {path} is empty")
        return value

    def __repr__(self) -> str:
        return f"FileSecretProvider(root={str(self._root)!r})"
