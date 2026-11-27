"""A09: the Elasticsearch adapter's pure logic, and its read-only transport, without a cluster.

Everything here is a table of cases over functions that do no I/O — index-name arithmetic, time
bounds out of a Query DSL tree, value conversion, catalog assembly — plus the transport proof driven
by ``httpx.MockTransport``. The live cluster proof lives in
``tests/integration/test_es_adapter.py``; this file keeps the arithmetic honest when Docker is not
running.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import httpx
import pyarrow as pa
import pytest

from mdqm.adapters.elastic import (
    _UNABLE,
    ESQL_PATH,
    ElasticAdapter,
    ElasticError,
    SearchOnlyTransport,
    _build_tables,
    _coerce,
    _esql_table,
    _flatten,
    _rows_to_table,
    _time_bounds,
    _trim_bytes,
    month_of,
)
from mdqm.config.settings import SourceConfig
from mdqm.models import Limits, NormalizedType

AUGUST = datetime(2025, 8, 1, tzinfo=UTC)


def _config(options: dict[str, Any] | None = None) -> SourceConfig:
    return SourceConfig(
        name="es",
        adapter="elastic",
        options=options or {"base_url": "http://127.0.0.1:9200", "time_field": "start_time"},
    )


def _adapter(transport: httpx.BaseTransport | None = None, **kwargs: Any) -> ElasticAdapter:
    return ElasticAdapter(_config(), transport=transport, **kwargs)


# --- index names ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("mdqm-usage-events-2025-08", AUGUST),
        ("events-2026-01", datetime(2026, 1, 1, tzinfo=UTC)),
        ("events-2025-13", None),  # a month that does not exist is not a month
        ("events-2025-08-extra", None),
        ("events", None),
        ("events-2025", None),
    ],
)
def test_month_of_reads_the_naming_convention(name: str, expected: datetime | None) -> None:
    assert month_of(name) == expected


# --- time bounds out of a query tree ---------------------------------------------------------


def test_time_bounds_finds_a_nested_range() -> None:
    query = {
        "bool": {
            "filter": [
                {"term": {"client_no": "1"}},
                {
                    "range": {
                        "start_time": {"gte": "2025-08-01T00:00:00Z", "lt": "2025-09-01T00:00:00Z"}
                    }
                },
            ]
        }
    }
    assert _time_bounds(query, "start_time") == (AUGUST, datetime(2025, 9, 1, tzinfo=UTC))


def test_time_bounds_takes_the_tightest_bound_and_ignores_other_fields() -> None:
    query = {
        "bool": {
            "must": [
                {"range": {"start_time": {"gte": "2025-01-01T00:00:00Z"}}},
                {
                    "range": {
                        "start_time": {"gte": "2025-08-01T00:00:00Z", "lte": "2025-08-31T00:00:00Z"}
                    }
                },
                {"range": {"insertion_time": {"gte": "2024-01-01T00:00:00Z"}}},
            ]
        }
    }
    assert _time_bounds(query, "start_time") == (AUGUST, datetime(2025, 8, 31, tzinfo=UTC))


@pytest.mark.parametrize(
    "query",
    [
        None,
        {"match_all": {}},
        {"range": {"start_time": {"gte": "2025-08-01T00:00:00Z"}}},  # no upper bound
        {"range": {"start_time": {"lte": "2025-08-01T00:00:00Z"}}},  # no lower bound
        {"range": {"other": {"gte": "2025-08-01", "lte": "2025-09-01"}}},
    ],
)
def test_time_bounds_is_none_without_a_closed_window(query: Any) -> None:
    """An open window cannot narrow an index set, and guessing a width would drop rows."""
    assert _time_bounds(query, "start_time") is None


# --- value conversion ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "normalized", "expected"),
    [
        (None, NormalizedType.STRING, None),
        (7, NormalizedType.INTEGER, 7),
        ("7", NormalizedType.INTEGER, 7),
        (True, NormalizedType.INTEGER, 1),
        ("not a number", NormalizedType.INTEGER, _UNABLE),
        (1.5, NormalizedType.FLOAT, 1.5),
        ("1.5", NormalizedType.DECIMAL, 1.5),
        ("true", NormalizedType.BOOLEAN, True),
        ("maybe", NormalizedType.BOOLEAN, _UNABLE),
        ("2025-08-01T00:00:00Z", NormalizedType.TIMESTAMP, AUGUST),
        ("2025-08-01T00:00:00", NormalizedType.TIMESTAMP, AUGUST),
        (1754006400000, NormalizedType.TIMESTAMP, AUGUST),
        ("yesterday", NormalizedType.TIMESTAMP, _UNABLE),
        (42, NormalizedType.STRING, "42"),
        ({"a": 1}, NormalizedType.JSON, '{"a": 1}'),
    ],
)
def test_coerce_never_raises_on_remote_data(
    value: Any, normalized: NormalizedType, expected: Any
) -> None:
    assert _coerce(value, normalized) == expected


# --- catalog assembly ----------------------------------------------------------------------


MAPPINGS: dict[str, dict[str, Any]] = {
    "usage-2025-08": {
        "properties": {
            "start_time": {"type": "date"},
            "client_no": {"type": "keyword"},
            "units": {"type": "integer"},
            "meta": {"properties": {"source": {"type": "keyword"}}},
        }
    },
    "usage-2025-09": {
        "properties": {
            "start_time": {"type": "date"},
            "client_no": {"type": "keyword"},
            "units": {"type": "integer"},
            "extra": {"type": "long"},
        }
    },
}


def test_an_alias_is_one_table_and_its_indices_are_not_tables() -> None:
    tables = _build_tables(
        {"usage": ("usage-2025-08", "usage-2025-09")},
        ("usage-2025-08", "usage-2025-09"),
        {"usage-2025-08": 3, "usage-2025-09": 4},
        MAPPINGS,
        time_field="start_time",
    )
    assert [table.name for table in tables] == ["usage"]
    table = tables[0]
    assert table.members == ("usage-2025-08", "usage-2025-09")
    assert table.docs == 7
    assert table.aliased is True
    assert table.time_field == "start_time"
    assert table.granularity is not None
    assert table.column_names == ("client_no", "extra", "meta.source", "start_time", "units")


def test_an_index_in_no_alias_is_its_own_table() -> None:
    tables = _build_tables(
        {"usage": ("usage-2025-08",)},
        ("usage-2025-08", "rates"),
        {"usage-2025-08": 3, "rates": 9},
        {"rates": {"properties": {"code": {"type": "keyword"}}}},
        time_field=None,
    )
    assert [table.name for table in tables] == ["rates", "usage"]
    assert tables[0].granularity is None, "an index with no month suffix is not partitioned"


def test_the_time_field_is_inferred_only_when_the_mapping_has_exactly_one_date() -> None:
    one_date = {"i-2025-08": {"properties": {"at": {"type": "date"}, "v": {"type": "long"}}}}
    two_dates = {
        "j-2025-08": {
            "properties": {
                "event_at": {"type": "date"},
                "ingested_at": {"type": "date"},
                "v": {"type": "long"},
            }
        }
    }
    inferred = _build_tables({"i": ("i-2025-08",)}, ("i-2025-08",), {}, one_date, time_field=None)
    ambiguous = _build_tables({"j": ("j-2025-08",)}, ("j-2025-08",), {}, two_dates, time_field=None)
    configured = _build_tables(
        {"j": ("j-2025-08",)}, ("j-2025-08",), {}, two_dates, time_field="ingested_at"
    )

    assert inferred[0].time_field == "at"
    assert ambiguous[0].time_field is None, "two date fields are never guessed at"
    assert configured[0].time_field == "ingested_at"


def test_a_mapping_is_flattened_into_dotted_field_names() -> None:
    assert list(_flatten(MAPPINGS["usage-2025-08"]["properties"]))[3][0] == "meta.source"


# --- rows to Arrow -------------------------------------------------------------------------


def test_rows_become_an_arrow_table_with_nulls_and_a_bad_value_count() -> None:
    table = _build_tables(
        {"usage": ("usage-2025-08",)},
        ("usage-2025-08",),
        {},
        MAPPINGS,
        time_field="start_time",
    )[0]
    rows: list[dict[str, Any]] = [
        {
            "start_time": "2025-08-01T00:00:00Z",
            "client_no": "1",
            "units": 5,
            "meta": {"source": "crm"},
        },
        {"start_time": "2025-08-02T00:00:00Z", "client_no": None, "units": "nonsense"},
    ]
    result, bad = _rows_to_table(rows, table.fields)

    assert result.schema.field("start_time").type == pa.timestamp("us", tz="UTC")
    assert result.schema.field("units").type == pa.int32()
    assert result.column("meta.source").to_pylist() == ["crm", None]
    assert result.column("units").to_pylist() == [5, None]
    assert bad == 1, "an uncoercible value is counted, never silently replaced"


def test_esql_values_become_an_arrow_table_and_respect_the_row_cap() -> None:
    payload = {
        "columns": [{"name": "n", "type": "long"}, {"name": "s", "type": "keyword"}],
        "values": [[1, "a"], [2, "b"], [3, "c"]],
    }
    table, dropped, bad = _esql_table(payload, 2)

    assert table.to_pydict() == {"n": [1, 2], "s": ["a", "b"]}
    assert dropped == 1
    assert bad == 0


def test_byte_cap_trims_whole_batches_and_reports_what_it_dropped() -> None:
    table = pa.table({"x": list(range(500))})
    kept, dropped = _trim_bytes(table, 1)

    assert kept.num_rows == 0
    assert dropped == 500
    untouched, none_dropped = _trim_bytes(table, 1_000_000)
    assert untouched.num_rows == 500
    assert none_dropped == 0


# --- the read-only transport ------------------------------------------------------------


def _recording_inner(status: int = 200) -> tuple[httpx.MockTransport, list[str]]:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(f"{request.method} {request.url.path}")
        return httpx.Response(status, json={"ok": True})

    return httpx.MockTransport(handler), seen


def test_the_transport_allows_reads_and_refuses_every_write() -> None:
    inner, seen = _recording_inner()
    transport = SearchOnlyTransport(inner)
    client = httpx.Client(base_url="http://es:9200", transport=transport)

    assert client.get("/usage/_search").status_code == 200
    assert client.post(ESQL_PATH, json={"query": "ROW x = 1"}).status_code == 200
    for method, path in [
        ("PUT", "/usage-2026-02"),
        ("DELETE", "/usage-2025-08"),
        ("POST", "/_bulk"),
        ("PATCH", "/usage"),
    ]:
        with pytest.raises(ElasticError) as refused:
            client.request(method, path)
        assert refused.value.code == "WRITE_BLOCKED"
    assert seen == ["GET /usage/_search", f"POST {ESQL_PATH}"], "a refused request was still sent"
    client.close()


def test_the_transport_pins_the_host() -> None:
    inner, seen = _recording_inner()
    client = httpx.Client(
        base_url="http://es:9200", transport=SearchOnlyTransport(inner, allowed_host="es")
    )
    with pytest.raises(ElasticError) as refused:
        client.get("http://elsewhere:9200/usage/_search")
    assert refused.value.code == "INVALID_SPEC"
    assert seen == []
    client.close()


# --- fragment validation -----------------------------------------------------------------


def test_an_sql_fragment_is_refused_with_the_expected_keys() -> None:
    adapter = _adapter()
    fragment = adapter.fragment("usage")
    sql_fragment = fragment.model_copy(update={"dsl": None, "sql": "SELECT 1"})
    with pytest.raises(ElasticError) as refused:
        adapter.execute(
            sql_fragment, Limits(timeout_s=1, max_rows=1, max_bytes=1000, max_partitions=1)
        )
    assert refused.value.code == "INVALID_SPEC"
    assert "dsl" in refused.value.response.problem


def test_an_unknown_dsl_key_is_refused_by_name() -> None:
    adapter = _adapter()
    fragment = adapter.fragment("usage")
    with pytest.raises(ElasticError) as refused:
        adapter._parse_request(
            fragment.model_copy(update={"dsl": {"table": "usage", "size": 10}})
        )
    assert refused.value.code == "INVALID_SPEC"
    assert "dsl.size" in refused.value.response.problem


def test_the_base_url_comes_from_options() -> None:
    assert _adapter().base_url == "http://127.0.0.1:9200"
    built = ElasticAdapter(_config({"host": "es.internal", "port": 9201, "scheme": "https"}))
    assert built.base_url == "https://es.internal:9201"
    with pytest.raises(ElasticError) as refused:
        ElasticAdapter(_config({"port": "not-a-port"}))
    assert refused.value.code == "INVALID_SPEC"
