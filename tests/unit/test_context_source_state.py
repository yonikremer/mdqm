"""The states a semantic directory can be in, and what the server does in each.

This is the difference between a fresh deployment and a broken one:

* no directory                      -> start, `/context` says NOT_SUPPORTED with a fix
* directory, no YAML yet            -> the same (the bootstrap pipeline has not run)
* files the loader reads, broken    -> refuse to start rather than serve guesses

The middle case is not hypothetical: it is exactly what the container sees on its first start,
where a crash loop would hide the real state from an operator who cannot read a stack trace.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from mdqm.config.settings import PathsSettings, Settings
from mdqm.semantic.errors import SemanticError
from mdqm.server.context_endpoint import context_source_from_settings

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_LAYER = REPO_ROOT / "examples" / "semantic_example.yaml"

NEWLINE = chr(10)


def _settings(tmp_path: Path, *, semantic_dir: Path) -> Settings:
    return Settings(
        paths=PathsSettings(
            metadata_db=tmp_path / "metadata.sqlite",
            semantic_dir=semantic_dir,
            cache_dir=tmp_path / "cache",
        )
    )


def test_missing_directory_starts_without_a_layer(tmp_path: Path) -> None:
    settings = _settings(tmp_path, semantic_dir=tmp_path / "absent")
    assert context_source_from_settings(settings) is None


def test_existing_but_empty_directory_starts_without_a_layer(tmp_path: Path) -> None:
    """The first-boot case: the directory is mounted, nothing has been written into it yet."""
    empty = tmp_path / "semantic"
    (empty / "generated").mkdir(parents=True)
    (empty / "curated").mkdir()

    assert context_source_from_settings(_settings(tmp_path, semantic_dir=empty)) is None


def test_a_layer_with_files_that_do_not_load_still_refuses_to_start(tmp_path: Path) -> None:
    """Absence of facts is normal; corruption is not. A broken layer must fail loudly."""
    broken = tmp_path / "semantic"
    (broken / "curated").mkdir(parents=True)
    body = NEWLINE.join(("version: 1", "this: [is not, a semantic layer]", ""))
    (broken / "curated" / "layer.yaml").write_text(body, encoding="utf-8")

    with pytest.raises(SemanticError):
        context_source_from_settings(_settings(tmp_path, semantic_dir=broken))


def test_a_valid_layer_is_served(tmp_path: Path) -> None:
    layer = tmp_path / "semantic"
    (layer / "curated").mkdir(parents=True)
    shutil.copy(EXAMPLE_LAYER, layer / "curated" / "semantic.yaml")

    source = context_source_from_settings(_settings(tmp_path, semantic_dir=layer))
    assert source is not None, "a loadable layer must be served, not ignored"
    assert source.layer.version_hash, "the served layer must carry its version hash"


def test_a_file_at_the_root_is_reported_not_loaded_and_does_not_crash(tmp_path: Path) -> None:
    """A stray YAML at the root is a deployment mistake: warn, start, do not pretend or crash."""
    layer = tmp_path / "semantic"
    layer.mkdir()
    shutil.copy(EXAMPLE_LAYER, layer / "semantic.yaml")

    assert context_source_from_settings(_settings(tmp_path, semantic_dir=layer)) is None
