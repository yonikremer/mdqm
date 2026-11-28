# SQL Server driver and plan-XML spike (task A08)

One page: which driver to talk to SQL Server with, what each needs on the host, and the answer to the
question AGENTS.md section 9 left open — **what SQL Server's plan XML actually carries about
partitions**.

Everything below was measured against the compose service `mssql` (SQL Server 2022, `16.0.4135.4`,
Developer Edition, `mdqm-mssql-1`, host port 51433) with the D06 world loaded and the read-only login
`mdqm_ro` (`db_datareader` + `VIEW DATABASE STATE` + `SHOWPLAN`). Reproduce with:

```
python scripts/dev.py up mssql
python -m mdqm.testing.loaders.sqlserver --small
python -m pytest tests/integration/test_mssql_adapter.py --require-services -q
```

## 1. Driver: `mssql-python` (chosen), with the others as optional extras

| Driver | Version tested | Wheel / host needs | TLS | Read-only intent | Cancellation |
|---|---|---|---|---|---|
| **`mssql-python`** (chosen) | 1.19.0 | Wheels for Windows/Linux/macOS that bundle the ODBC driver. On Debian/Ubuntu the bundled `libodbcinst.so.2` still needs `libltdl.so.7`, which the wheel does not ship; without it every connection fails with `Failed to load the driver` (`apt install libltdl7` on a host, and the server image installs it at build time). Windows and macOS wheels do not have that gap | `Encrypt=` + `TrustServerCertificate=` in the connection string (encryption on, self-signed cert accepted only where configured) | `ApplicationIntent=ReadOnly` in the connection string (honoured by an availability-group listener; ignored by a standalone server, so the account is the real guarantee) | **`cursor.cancel()` works** — interrupts a running statement and the session survives (measured); `Connection.timeout` also works (`Query timeout expired`) and the session survives |
| `pymssql` (optional extra) | not installed | Bundles FreeTDS: no ODBC driver, but its own TLS stack and a slower release cadence | FreeTDS TLS | no | server-side `KILL <spid>` only |
| `pyodbc` (optional extra) | not installed | Needs a separately installed **Microsoft ODBC Driver for SQL Server** on every host, plus `unixODBC` on Linux | driver-managed | yes | `SQLCancel` via `cursor.cancel()` |

Why `mssql-python` is the default for this product: the target is an air-gapped on-prem network where
adding a system ODBC driver and its dependencies to every host is the expensive part, and the
bundled-wheels option is the only one with no host prerequisite at all. It is also the driver D06
already loads the test world with (`src/mdqm/testing/loaders/sqlserver.py`), so the loader and the
adapter prove the same driver.

What the adapter relies on:

* `cursor.fetchmany(n)` streams; `max_rows`/`max_bytes` stop the scan mid-flight.
* `cursor.description` names the **Python class** each column produces (`int`, `str`,
  `Decimal`, `datetime`, `bytes`), which is what `arrow_type()` maps.
* `cursor.cancel()` from a watchdog thread interrupts a statement **while it is executing or
  while a fetch is blocked** (`Operation canceled`, SQLSTATE-free, native error 0). Between two
  fast batches it can land when nothing is in flight and then does nothing, so the adapter's
  `cancel()` also sets a flag the iteration loop honours — the driver break is best effort, the
  flag is authoritative.
* Naive `datetime` values are stored and read as naive UTC; Arrow carries them as
  `timestamp("us", tz="UTC")` (SQL Server has no timestamp-with-time-zone type).

## 2. `SET SHOWPLAN_XML ON`: what the XML carries (the open question, answered)

Permission: `SET SHOWPLAN_XML ON` itself is allowed to anybody, but **compiling a statement into XML
needs `SHOWPLAN`** — without it SQL Server answers `SHOWPLAN permission denied in database 'erp'`
even though the `SET` succeeded. Grant it to the read-only login (`GRANT SHOWPLAN TO [mdqm_ro]`);
the probe reports `explain_plan` as unsupported with that fix when it is missing.

### It does carry the pruned partition range, as a partition-id seek predicate

`SELECT COUNT(*), client_no FROM dbo.usage_events WHERE start_time >= CONVERT(datetime2(6),'2025-06-01',126) AND start_time < CONVERT(datetime2(6),'2025-08-01',126) GROUP BY client_no`
(verbatim shape, attributes trimmed):

```xml
<RelOp NodeId="2" PhysicalOp="Clustered Index Scan" ... Partitioned="1" TableCardinality="5001">
  <IndexScan Ordered="1" ScanDirection="FORWARD">
    <Object Database="[erp]" Schema="[dbo]" Table="[usage_events]" Index="[usage_events_pk]" IndexKind="Clustered"/>
    <SeekPredicates><SeekPredicateNew><SeekKeys>
      <StartRange ScanType="GE"><RangeColumns><ColumnReference Column="PtnId1000"/></RangeColumns>
        <RangeExpressions><ScalarOperator ScalarString="(9)"/></RangeExpressions></StartRange>
      <EndRange ScanType="LE"><RangeColumns><ColumnReference Column="PtnId1000"/></RangeColumns>
        <RangeExpressions><ScalarOperator ScalarString="(11)-(1)"/></RangeExpressions></EndRange>
    </SeekKeys></SeekPredicateNew></SeekPredicates>
  </IndexScan>
</RelOp>
```

`PtnId1000` is the computed partition-id column, `(9)` is the first partition scanned and `(11)-(1)`
is SQL Server's way of writing "10" (partition 11 minus one). So the estimated plan **does** name the
range, and the adapter counts `2` partitions from it — proven, without executing anything.

### It does not carry `PartitionsAccessed`, and that element is not a count to read from an EXPLAIN

`RunTimePartitionSummary/PartitionsAccessed` exists only in an **actual** plan (`SET STATISTICS XML
ON`, i.e. the statement really ran):

```xml
<RunTimePartitionSummary>
  <PartitionsAccessed PartitionCount="1"><PartitionRange Start="1" End="1"/></PartitionsAccessed>
</RunTimePartitionSummary>
```

and for the pruned statement above `PartitionCount="2"`, matching the estimated plan's range. The
integration test asserts exactly that agreement
(`test_the_estimated_plan_range_matches_the_actual_partition_access`).

### A parameterised value turns the range into an expression

When the key value is a parameter (auto-parameterisation does this to a `CONVERT(datetime2, @1)`
too), the same place holds a call instead of a number, so the ends are unknown before execution:

```xml
<StartRange ScanType="GE">...<ScalarOperator ScalarString="RangePartitionNew(CONVERT(datetime2(6),[@1],126),(1),'2024-11-01 00:00:00.000000','2024-12-01 00:00:00.000000',...)"/>
```

The adapter reports `pruning_proof=unproven` there, never a guessed count.

### A `varchar` literal on a `datetime2` partition key eliminates nothing

`WHERE start_time >= '2025-06-01'` puts an implicit `CONVERT` on the key: the estimated plan has
**no** `PtnId1000` predicate at all, and the actual plan reports `PartitionCount="18"` — every
partition. With `CONVERT(datetime2(6), '2025-06-01', 126)` it is 1 and 2 respectively. The compiler
must therefore render a time literal as `datetime2`, not `varchar`; the adapter reports such a scan
as `unproven` (a scan with no partition predicate reaches every partition).

## 3. What the adapter does with each finding

| Question | Answer in `src/mdqm/adapters/sqlserver.py` |
|---|---|
| partition metadata | `sys.partition_functions` + `sys.partition_range_values` + `sys.partition_schemes` + `sys.destination_data_spaces`; `RANGE RIGHT` tuples become N+1 half-open bounds; `fanout` is the partition count (`ALL TO (...)` reserves one extra destination, which is not a partition); `RANGE LEFT` and non-temporal keys raise `NotSupported` rather than drop boundary rows |
| plan | `SET SHOWPLAN_XML ON`; cost/estimate/indexes from the document, pruning from the `PtnId1000` range: literal ends → `proven` + counted, an expression → `unproven`, no partition predicate but a partitioned table → `unproven`, no partitioned table → `unknown` |
| Query Store | `sys.query_store_query_text` + `_query` + `_plan` + `_runtime_stats` (needs `VIEW DATABASE STATE`); durations are microseconds, converted to ms; `user` stays `None` because the store names no executor |
| mutability | `sys.dm_db_partition_stats` row counts + `sys.dm_db_stats_properties.modification_counter` (rows changed since the statistics were last updated). Per-table insert/update/delete counters do not exist in SQL Server, so those stay `None` |
| cancel / deadline | watchdog thread calls `cursor.cancel()`; `Connection.timeout` is a backstop `timeout_s + 5 s`; the stream's own flag guarantees the iteration stops; the session stays usable (proved twice in the integration test) |

## 4. Inside the server image (task A10)

The host spike above proved the driver; the image is a different question, because the runtime stage
has no pip and installs one system library. Both halves were measured on `mdqm/server:mssql-a10`,
built with `--build-arg MDQM_EXTRAS=pg,mssql` and run on the compose network:

```
docker build --build-arg MDQM_EXTRAS=pg,mssql -t mdqm/server:mssql-a10 .
docker run --rm --network mdqm_default --env-file .env mdqm/server:mssql-a10 python -c '…'
```

* **It connects.** `mssql_python.connect(server="mssql", …, uid="mdqm_ro")` from inside the container
  answers `('erp', 'mdqm_ro', 201)`, and `SELECT COUNT(*) FROM sys.dm_db_partition_stats` works, so the
  size reader's `VIEW DATABASE STATE` grant survives the container's packaging too.
* **The extra is registered.** `entry_points(group="mdqm.adapters")` inside the image lists
  `elastic, openapi, oracle, postgresql, sqlserver` — a build without the `mssql` extra simply has no
  `sqlserver` adapter and the probe reports it unavailable.
* **`libltdl7` is what the wheel was missing.** Installed in the runtime stage at build time; no
  `apt-get` and no `pip install` happens at container start.

### The runtime image used to ship an installer

`pip uninstall --yes pip setuptools wheel` in the **builder** stage only cleared the venv, so the base
image's own pip survived into the runtime layer and sat on the container's `PATH`:

```
pip -> /usr/local/bin/pip      # pip 25.0.1 from /usr/local/lib/python3.12/site-packages/pip
```

That contradicts "no installer in the runtime image" and defeats the air-gap rule (AGENTS.md principle
7) for anyone who can mount a wheelhouse. The runtime stage now deletes it, and the same probe answers
`pip -> None`, `import pip -> No module named 'pip'`. Re-check an image with:

```
docker run --rm <image> python -c "import shutil; print(shutil.which('pip'), shutil.which('pip3'))"
```
