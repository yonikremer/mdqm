"""Tests for the open-source mock stack: `scripts/seed_oss.py`, `docker/oss/` and the `oss` profile.

Owner request (not in TASKS.md): a local test stack that is open-source only — Postgres, MariaDB,
OpenSearch, the mock API — with seeds generated from the synthetic world of task D02.

Everything here runs without Docker. What is therefore NOT covered, and cannot be claimed as
verified by this file: that the containers start, that MariaDB runs the init scripts, that the
`_bulk` load succeeds, and that the indices/alias exist. `progress/OSS-mocks.md` lists those.
"""

from __future__ import annotations

import importlib.util
import json
import math
import re
import sys
import types
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import sqlglot
from ruamel.yaml import YAML

from mdqm.testing.world import WorldConfig, build_world, render_json

REPO_ROOT = Path(__file__).resolve().parents[2]
COMPOSE_PATH = REPO_ROOT / "docker-compose.yml"
OSS_DIR = REPO_ROOT / "docker" / "oss"
ENV_EXAMPLE_PATH = REPO_ROOT / ".env.example"

SMALL_CONFIG = WorldConfig.small(seed=1)
"""The config the committed fixtures are generated with (see scripts/seed_oss.py --small)."""

MONTHLY_INDEX = "mdqm-usage-events-"
ALIAS = "mdqm-usage-events"

PROPRIETARY_IMAGE_PREFIXES = (
    "docker.elastic.co/",
    "mcr.microsoft.com/mssql",
    "gvenzl/oracle",
    "container-registry.oracle.com/",
)


def load_seed_oss() -> types.ModuleType:
    """Import `scripts/seed_oss.py` by path: `scripts/` is not a package."""
    path = REPO_ROOT / "scripts" / "seed_oss.py"
    spec = importlib.util.spec_from_file_location("seed_oss_module", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register before exec_module: the dataclasses in the script resolve their annotations through
    # sys.modules, which fails for a module that was never installed there.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def seed_oss() -> types.ModuleType:
    return load_seed_oss()


@pytest.fixture(scope="module", params=["small", "json"], ids=["in-process", "from-json"])
def seed(seed_oss: types.ModuleType, request: pytest.FixtureRequest, tmp_path_factory: Any) -> Any:
    """The same world twice: built in process, and read back from world JSON."""
    world = build_world(SMALL_CONFIG)
    if request.param == "small":
        return seed_oss.OssSeed.from_world(world)
    path = Path(tmp_path_factory.mktemp("world")) / "world.json"
    path.write_text(render_json(world), encoding="utf-8", newline="\n")
    document = json.loads(path.read_text(encoding="utf-8"))
    return seed_oss.OssSeed.from_json(document)


# --------------------------------------------------------------------------------------
# Both input paths give the same seed
# --------------------------------------------------------------------------------------


def test_json_and_in_process_produce_identical_rows(seed_oss: types.ModuleType) -> None:
    world = build_world(SMALL_CONFIG)
    from_world = seed_oss.OssSeed.from_world(world)
    from_json = seed_oss.OssSeed.from_json(json.loads(render_json(world)))
    assert from_json == from_world
    assert from_json.counts() == from_world.counts()


def test_row_counts_are_the_worlds_own(seed: Any) -> None:
    world = build_world(SMALL_CONFIG)
    assert seed.counts() == {
        "crm_customers": len(world.crm.customers),
        "billing_accounts": len(world.billing.accounts),
        "billing_usage_events": len(world.billing.usage_events),
        "billing_accounts_hist": len(world.billing.accounts_hist),
        "erp_customers": len(world.erp.customers),
    }
    assert seed.counts()["billing_usage_events"] == world.expected.events_total


# --------------------------------------------------------------------------------------
# Determinism: same seed, byte-identical output
# --------------------------------------------------------------------------------------


def test_two_generations_are_byte_identical(seed_oss: types.ModuleType, tmp_path: Path) -> None:
    world = build_world(SMALL_CONFIG)
    first = seed_oss.write_all(seed_oss.OssSeed.from_world(world), tmp_path / "a")
    second = seed_oss.write_all(seed_oss.OssSeed.from_world(world), tmp_path / "b")
    assert [p.relative_to(tmp_path / "a") for p in first] == [
        p.relative_to(tmp_path / "b") for p in second
    ]
    for left, right in zip(first, second, strict=True):
        assert left.read_bytes() == right.read_bytes(), f"{left.name} is not deterministic"


def test_committed_fixtures_match_the_generator(seed_oss: types.ModuleType, tmp_path: Path) -> None:
    """`docker/oss` must be exactly what the committed command emits, not hand-edited."""
    seed = seed_oss.OssSeed.from_world(build_world(SMALL_CONFIG))
    generated = seed_oss.write_all(seed, tmp_path)
    for path in generated:
        committed = OSS_DIR / path.relative_to(tmp_path)
        assert committed.is_file(), f"{committed} is missing; run scripts/seed_oss.py"
        assert committed.read_bytes() == path.read_bytes(), (
            f"{committed.name} is stale: run `python scripts/seed_oss.py --small --out docker/oss/`"
        )


def test_line_endings_are_lf(seed_oss: types.ModuleType, tmp_path: Path) -> None:
    generated = seed_oss.write_all(seed_oss.OssSeed.from_world(build_world(SMALL_CONFIG)), tmp_path)
    for path in generated:
        assert b"\r\n" not in path.read_bytes(), f"{path.name} has CRLF line endings"


# --------------------------------------------------------------------------------------
# MariaDB: real MySQL dialect, parsed by sqlglot
# --------------------------------------------------------------------------------------


def test_schema_and_data_parse_as_mysql(seed_oss: types.ModuleType, seed: Any) -> None:
    for text in (seed_oss.SCHEMA_SQL, seed_oss.render_data_sql(seed)):
        statements = sqlglot.parse(text, dialect="mysql")
        assert statements, "the seed must contain at least one statement"
        assert all(statement is not None for statement in statements)


def test_schema_creates_the_five_tables_with_mysql_types(seed_oss: types.ModuleType) -> None:
    tables = sqlglot.parse(seed_oss.SCHEMA_SQL, dialect="mysql")
    statements = [statement for statement in tables if statement is not None]
    created = [statement for statement in statements if isinstance(statement, sqlglot.exp.Create)]
    names = [statement.this.this.name for statement in created]
    assert names == [
        "crm_customers",
        "billing_accounts",
        "billing_usage_events",
        "billing_accounts_hist",
        "erp_customers",
    ]
    text = seed_oss.SCHEMA_SQL
    assert "`cust_id`" in text, "the dialect must quote identifiers with backticks"
    assert "DATETIME" in text, "event and insertion times must be DATETIME"
    assert "DECIMAL(12,2)" in text, "money columns must be DECIMAL, not float"
    assert ") ENGINE=InnoDB;" in text, "the mutable/hist table must be InnoDB"


def test_schema_has_no_postgres_or_tsql_only_syntax(seed_oss: types.ModuleType) -> None:
    text = seed_oss.SCHEMA_SQL.lower()
    for forbidden in ("serial", "bytea", "nvarchar", "timestamp with time zone", "::text"):
        assert forbidden not in text


def test_insert_row_counts_match_the_world(seed_oss: types.ModuleType, seed: Any) -> None:
    statements = sqlglot.parse(seed_oss.render_data_sql(seed), dialect="mysql")
    per_table: dict[str, int] = {}
    for statement in statements:
        if statement is None or not isinstance(statement, sqlglot.exp.Insert):
            continue
        table = statement.this.this.name
        expression = statement.expression
        rows = len(expression.expressions) if hasattr(expression, "expressions") else 0
        per_table[table] = per_table.get(table, 0) + rows
    assert per_table == seed.counts()


def test_nulls_are_written_as_sql_null(seed_oss: types.ModuleType, seed: Any) -> None:
    world = build_world(SMALL_CONFIG)
    planted = len(world.expected.planted_nulls)
    assert planted > 0, "the world plants NULL credit limits; otherwise this test proves nothing"
    text = seed_oss.render_data_sql(seed)
    assert "NULL" in text, "planted NULLs must survive into MariaDB"


def test_strings_are_quoted_and_escaped(seed_oss: types.ModuleType) -> None:
    assert seed_oss._sql_str("דנה כהן") == "'דנה כהן'"
    assert seed_oss._sql_str("o'brien") == "'o''brien'"
    assert seed_oss._sql_str("a\\b") == "'a\\\\b'"


# --------------------------------------------------------------------------------------
# OpenSearch: `_bulk` NDJSON, monthly indices, alias
# --------------------------------------------------------------------------------------


def bulk_lines(seed_oss: types.ModuleType, seed: Any) -> list[str]:
    text: str = seed_oss.render_bulk_ndjson(seed)
    assert text.endswith("\n"), "a _bulk body must end with a newline"
    lines: list[str] = text.splitlines()
    return lines


def test_bulk_lines_are_action_doc_pairs(seed_oss: types.ModuleType, seed: Any) -> None:
    lines = bulk_lines(seed_oss, seed)
    assert len(lines) % 2 == 0, "every action line needs a document line"
    assert len(lines) // 2 == seed.counts()["billing_usage_events"]
    for index in range(0, len(lines), 2):
        action = json.loads(lines[index])
        assert list(action) == ["index"], "an action line holds exactly one operation"
        assert set(action["index"]) == {"_index", "_id"}
        document = json.loads(lines[index + 1])
        assert set(document) == {
            "event_id",
            "client_no",
            "start_time",
            "insertion_time",
            "units",
            "corrects_event_id",
        }


def test_bulk_index_names_are_monthly_and_ordered(seed_oss: types.ModuleType, seed: Any) -> None:
    lines = bulk_lines(seed_oss, seed)
    names = [json.loads(lines[index])["index"]["_index"] for index in range(0, len(lines), 2)]
    assert names == sorted(names), "documents must be grouped by index, then by event id"
    for name in names:
        assert name.startswith(MONTHLY_INDEX)
        assert len(name) == len(MONTHLY_INDEX) + 7
        year, month = name[len(MONTHLY_INDEX) :].split("-")
        assert year.isdigit() and month.isdigit() and 1 <= int(month) <= 12


def test_bulk_index_matches_the_documents_event_month(
    seed_oss: types.ModuleType, seed: Any
) -> None:
    lines = bulk_lines(seed_oss, seed)
    for index in range(0, len(lines), 2):
        action = json.loads(lines[index])["index"]
        document = json.loads(lines[index + 1])
        assert action["_id"] == str(document["event_id"])
        start = datetime.fromisoformat(document["start_time"])
        assert action["_index"] == f"{MONTHLY_INDEX}{start.astimezone(UTC):%Y-%m}"


def test_bulk_timestamps_are_iso8601_utc(seed_oss: types.ModuleType, seed: Any) -> None:
    lines = bulk_lines(seed_oss, seed)
    for index in range(0, len(lines), 2):
        document = json.loads(lines[index + 1])
        for field in ("start_time", "insertion_time"):
            value = document[field]
            assert value.endswith("+00:00"), f"{field}={value!r} is not UTC"
            datetime.fromisoformat(value)  # raises when it is not ISO-8601


def test_monthly_indices_cover_every_event_month(seed_oss: types.ModuleType, seed: Any) -> None:
    world = build_world(SMALL_CONFIG)
    expected = sorted(
        {
            f"{MONTHLY_INDEX}{event.start_time.astimezone(UTC):%Y-%m}"
            for event in world.billing.usage_events
        }
    )
    assert seed_oss.months(seed) == expected
    assert len(expected) == len(world.expected.events_per_month), (
        "one index per month the ground truth reports events for"
    )


def test_index_template_declares_alias_mappings_and_pattern(
    seed_oss: types.ModuleType, seed: Any
) -> None:
    template = json.loads(seed_oss.render_index_template(seed))
    assert template["index_patterns"] == [f"{MONTHLY_INDEX}*"]
    assert template["_meta"]["seed"] == SMALL_CONFIG.seed
    assert template["_meta"]["source"] == "src/mdqm/testing/world.py"
    assert ALIAS in template["template"]["aliases"]
    properties = template["template"]["mappings"]["properties"]
    assert properties["start_time"] == {"type": "date"}
    assert properties["insertion_time"] == {"type": "date"}
    assert properties["client_no"] == {"type": "keyword"}
    assert template["template"]["mappings"]["dynamic"] == "strict"
    assert template["template"]["settings"]["number_of_replicas"] == 0


def test_index_template_json_is_sorted_and_readable(seed_oss: types.ModuleType, seed: Any) -> None:
    text = seed_oss.render_index_template(seed)
    assert json.loads(text) == json.loads(seed_oss.render_index_template(seed))
    assert text == json.dumps(json.loads(text), sort_keys=True, indent=2, ensure_ascii=False) + "\n"


# --------------------------------------------------------------------------------------
# The compose profile
# --------------------------------------------------------------------------------------


def compose_services() -> dict[str, Any]:
    # typ="safe" is ruamel's safe loader: no arbitrary Python object construction.
    parser = YAML(typ="safe")
    with COMPOSE_PATH.open(encoding="utf-8") as handle:
        document = parser.load(handle)
    return dict(document["services"])


def test_oss_profile_is_exactly_the_open_source_stack() -> None:
    in_profile = {
        name for name, definition in compose_services().items() if "oss" in definition["profiles"]
    }
    assert in_profile == {"pg", "mariadb", "opensearch", "opensearch-init", "api"}


def test_oss_profile_contains_no_proprietary_image() -> None:
    for name, definition in compose_services().items():
        if "oss" not in definition["profiles"]:
            continue
        image = str(definition.get("image", ""))
        for prefix in PROPRIETARY_IMAGE_PREFIXES:
            assert not image.startswith(prefix), f"{name} belongs to oss but uses {image}"
        assert image.rsplit(":", 1)[-1] not in {"latest", "stable", "main"}, f"{name}: {image}"


def test_oss_images_are_pinned_and_open_source() -> None:
    services = compose_services()
    assert services["opensearch"]["image"] == "opensearchproject/opensearch:2.17.1"
    assert services["mariadb"]["image"] == "mariadb:11.4.4"
    assert "build" in services["api"], "the mock API stays built in-repo"
    assert services["pg"]["image"] == "postgres:16"


def test_previously_proprietary_services_keep_their_own_profile() -> None:
    services = compose_services()
    for name in ("oracle", "mssql", "es"):
        assert services[name]["profiles"] == [name], f"{name} must stay out of the oss profile"
    assert services["pg"]["profiles"] == ["pg", "oss"]
    assert services["api"]["profiles"] == ["api", "oss"]


def test_mariadb_and_opensearch_are_loopback_with_healthchecks() -> None:
    services = compose_services()
    for name, container_port, variable in (
        ("mariadb", 3306, "MDQM_MARIADB_PORT"),
        ("opensearch", 9200, "MDQM_OPENSEARCH_PORT"),
    ):
        definition = services[name]
        published = definition["ports"]
        assert len(published) == 1
        assert str(published[0]).startswith("127.0.0.1:${")
        assert str(published[0]).endswith(f"}}:{container_port}")
        assert variable in str(published[0])
        assert definition["healthcheck"]["test"]
        assert definition["volumes"], f"{name} needs a named volume for its data"


def test_oss_search_settings_are_single_node_and_secure_by_dev_only_switch() -> None:
    opensearch = compose_services()["opensearch"]
    environment = opensearch["environment"]
    assert environment["discovery.type"] == "single-node"
    assert environment["DISABLE_SECURITY_PLUGIN"] == "true"
    assert environment["bootstrap.memory_lock"] == "true"
    assert opensearch["ulimits"]["memlock"]["soft"] == -1
    assert "512m" in str(environment["OPENSEARCH_JAVA_OPTS"])


def test_mariadb_init_hooks_and_loader_are_wired() -> None:
    services = compose_services()
    mariadb_mounts = [str(volume) for volume in services["mariadb"]["volumes"]]
    assert any("docker/oss/maria:/docker-entrypoint-initdb.d:ro" in m for m in mariadb_mounts)
    init = services["opensearch-init"]
    assert init["entrypoint"] == ["sh", "/seeds/load.sh"]
    assert init["restart"] == "no"
    assert "ports" not in init
    assert init["depends_on"]["opensearch"]["condition"] == "service_healthy"


def test_seed_files_the_hooks_consume_exist() -> None:
    for relative in (
        "maria/01-schema.sql",
        "maria/02-data.sql",
        "maria/10-readonly-user.sh",
        "opensearch/index-template.json",
        "opensearch/usage-events.ndjson",
        "opensearch/load.sh",
    ):
        path = OSS_DIR / relative
        assert path.is_file(), f"{path} is missing"
        assert path.stat().st_size > 0


def test_generated_file_count_is_what_the_helpers_expect() -> None:
    """The generator must not clobber the hand-written helpers next to the seeds."""
    generated = {"01-schema.sql", "02-data.sql", "index-template.json", "usage-events.ndjson"}
    for directory in (OSS_DIR / "maria", OSS_DIR / "opensearch"):
        present = {path.name for path in directory.iterdir()}
        assert generated & present or directory.name == "maria"
        for name in present - generated:
            assert name.endswith((".sh", ".md")), f"unexpected file {directory / name}"


def test_new_env_variables_are_documented() -> None:
    values: dict[str, str] = {}
    for line in ENV_EXAMPLE_PATH.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        values[key.strip()] = value.strip()
    for key in ("MDQM_MARIADB_ROOT_PASSWORD", "MDQM_MARIADB_PORT", "MDQM_OPENSEARCH_PORT"):
        assert key in values, f"{key} is used by the oss profile but missing from .env.example"
    assert values["MDQM_MARIADB_ROOT_PASSWORD"].startswith("MqmDev")
    for port in ("MDQM_MARIADB_PORT", "MDQM_OPENSEARCH_PORT"):
        assert values[port].isdigit()
        assert 1024 <= int(values[port]) <= 65535


def test_oss_ports_do_not_collide_with_the_other_profiles() -> None:
    text = COMPOSE_PATH.read_text(encoding="utf-8")
    # "127.0.0.1:${MDQM_PG_PORT:-55432}:5432" -> 55432
    defaults = [int(match) for match in re.findall(r":-([0-9]{5})\}", text)]
    assert len(defaults) == len(set(defaults)), f"duplicate host port in: {defaults}"


def test_batch_size_keeps_statements_a_sane_size(seed_oss: types.ModuleType, seed: Any) -> None:
    statements = sqlglot.parse(seed_oss.render_data_sql(seed), dialect="mysql")
    inserts = [
        statement
        for statement in statements
        if statement is not None and isinstance(statement, sqlglot.exp.Insert)
    ]
    biggest = max(len(statement.expression.expressions) for statement in inserts)
    assert biggest <= seed_oss.INSERT_BATCH
    expected = sum(math.ceil(count / seed_oss.INSERT_BATCH) for count in seed.counts().values())
    assert len(inserts) == expected
