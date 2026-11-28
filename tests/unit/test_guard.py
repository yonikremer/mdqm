"""Tests for the read-only SQL guard (:mod:`mdqm.sql.guard`).

The guard is security-critical, so the tables below test both directions heavily: every allowed
case must pass in every dialect, and every blocked case must be refused in every dialect, with an
``ErrorResponse`` whose ``fix`` the agent can act on. Counts are asserted so the tables cannot
shrink silently (the task requires >= 25 allowed and >= 40 blocked statements per dialect).

Known-good ``SELECT`` corpus (``tests/data/queries/*.sql``) is asserted allowed too: the guard
must never block a statement the rest of the system treats as read-only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from sqlglot import exp

from mdqm.sql.guard import (
    MULTI_STATEMENT,
    PARSE_ERROR,
    WRITE_BLOCKED,
    GuardPolicy,
    GuardResult,
    guard,
)

CORPUS_DIR = Path(__file__).resolve().parent.parent / "data" / "queries"
DIALECTS = ("postgres", "oracle", "tsql")
CODES = {PARSE_ERROR, WRITE_BLOCKED, MULTI_STATEMENT}
MIN_ALLOWED_CASES = 25
MIN_BLOCKED_CASES = 40

Case = tuple[str, str]  # (case id, sql)
#: Statement roots the guard accepts: a SELECT, a set operation (UNION/EXCEPT/INTERSECT/MINUS),
#: or a parenthesised query.
READ_ROOTS = (exp.Select, exp.SetOperation, exp.Subquery)

# --------------------------------------------------------------------------------------------------
# Statements that must be allowed. Everything here is portable across the three dialects; the
# dialect-specific extras live in ALLOWED_BY_DIALECT.
# --------------------------------------------------------------------------------------------------
ALLOWED_COMMON: tuple[Case, ...] = (
    ("simple_select", "SELECT cust_id, status, credit_limit FROM crm.customers WHERE status = 'A'"),
    ("aggregate", "SELECT status, COUNT(*) AS n FROM crm.customers GROUP BY status"),
    (
        "having_and_order",
        "SELECT status, SUM(credit_limit) AS total FROM crm.customers GROUP BY status "
        "HAVING SUM(credit_limit) > 1000 ORDER BY status",
    ),
    (
        "inner_join",
        "SELECT c.cust_id, e.client_no FROM crm.customers c "
        "JOIN billing.accounts e ON c.cust_id = e.cust_id",
    ),
    (
        "left_join_is_null",
        "SELECT c.cust_id, e.client_no FROM crm.customers c "
        "LEFT JOIN billing.accounts e ON e.cust_id = c.cust_id WHERE e.client_no IS NULL",
    ),
    (
        "correlated_scalar_subquery",
        "SELECT c.cust_id, (SELECT MAX(u.bytes) FROM billing.usage_events u "
        "WHERE u.cust_id = c.cust_id) AS max_bytes FROM crm.customers c",
    ),
    (
        "exists_subquery",
        "SELECT c.cust_id FROM crm.customers c "
        "WHERE EXISTS (SELECT 1 FROM billing.accounts e WHERE e.cust_id = c.cust_id)",
    ),
    (
        "not_exists_subquery",
        "SELECT c.cust_id FROM crm.customers c "
        "WHERE NOT EXISTS (SELECT 1 FROM billing.accounts e WHERE e.cust_id = c.cust_id)",
    ),
    (
        "in_subquery",
        "SELECT c.cust_id FROM crm.customers c "
        "WHERE c.cust_id IN (SELECT e.cust_id FROM billing.accounts e)",
    ),
    (
        "not_in_subquery",
        "SELECT c.cust_id FROM crm.customers c "
        "WHERE c.cust_id NOT IN (SELECT e.cust_id FROM billing.accounts e WHERE e.cust_id > 0)",
    ),
    (
        "with_cte_join",
        "WITH recent AS (SELECT cust_id, SUM(bytes) AS bytes FROM billing.usage_events "
        "GROUP BY cust_id) SELECT COUNT(*) AS n FROM recent r "
        "JOIN crm.customers c ON c.cust_id = r.cust_id",
    ),
    (
        "two_ctes",
        "WITH a AS (SELECT cust_id FROM crm.customers), "
        "b AS (SELECT cust_id FROM billing.accounts) "
        "SELECT COUNT(*) AS n FROM a JOIN b ON a.cust_id = b.cust_id",
    ),
    (
        "union_all",
        "SELECT 'active' AS grp, COUNT(*) AS n FROM crm.customers WHERE status = 'A' "
        "UNION ALL "
        "SELECT 'closed' AS grp, COUNT(*) AS n FROM crm.customers WHERE status = 'C'",
    ),
    (
        "intersect",
        "SELECT cust_id FROM crm.customers INTERSECT SELECT cust_id FROM billing.accounts",
    ),
    (
        "case_when",
        "SELECT cust_id, CASE WHEN credit_limit >= 100000 THEN 'large' "
        "WHEN credit_limit >= 10000 THEN 'medium' ELSE 'small' END AS band FROM crm.customers",
    ),
    (
        "window_function",
        "SELECT cust_id, ROW_NUMBER() OVER (PARTITION BY status ORDER BY credit_limit DESC) AS rn "
        "FROM crm.customers",
    ),
    (
        "divide_aggregates",
        "SELECT SUM(credit_limit) / COUNT(*) AS avg_credit FROM crm.customers WHERE status <> 'A'",
    ),
    ("coalesce", "SELECT COALESCE(credit_limit, 0) AS credit_limit FROM billing.accounts"),
    ("cast_decimal", "SELECT CAST(credit_limit AS DECIMAL(18, 2)) AS credit FROM crm.customers"),
    ("distinct_order", "SELECT DISTINCT status FROM crm.customers ORDER BY status"),
    ("between", "SELECT cust_id FROM crm.customers WHERE credit_limit BETWEEN 2500 AND 7500"),
    (
        "like_is_not_null",
        "SELECT cust_id FROM crm.customers WHERE name LIKE '%bank%' AND status IS NOT NULL",
    ),
    ("count_distinct", "SELECT COUNT(DISTINCT cust_id) AS n FROM crm.customers"),
    (
        "derived_table",
        "SELECT t.status FROM (SELECT status FROM crm.customers) t WHERE t.status = 'A'",
    ),
    (
        "multi_predicate",
        "SELECT cust_id FROM crm.customers "
        "WHERE cust_id > 0 AND status = 'A' AND credit_limit IS NOT NULL",
    ),
    ("hebrew_literal", "SELECT cust_id FROM crm.customers WHERE name = 'בנק לאומי'"),
    (
        "comment_inside",
        "SELECT /* monthly rollup */ status, COUNT(*) AS n FROM crm.customers GROUP BY status",
    ),
    (
        "boolean_grouping",
        "SELECT cust_id FROM crm.customers "
        "WHERE (status = 'A' OR status = 'S') AND credit_limit > 0",
    ),
    (
        "three_way_join",
        "SELECT c.status, COUNT(*) AS n FROM crm.customers c "
        "JOIN billing.accounts e ON e.cust_id = c.cust_id "
        "JOIN billing.usage_events u ON u.cust_id = c.cust_id GROUP BY c.status",
    ),
    (
        "order_by_two",
        "SELECT cust_id, status FROM crm.customers ORDER BY status ASC, cust_id DESC",
    ),
    ("trailing_semicolon", "SELECT cust_id FROM crm.customers;"),
    ("trailing_semicolons", "SELECT cust_id FROM crm.customers;;  "),
    # Text that only *looks* like a write must still pass: the guard reads the AST, not the letters.
    ("semicolon_in_literal", "SELECT '; DROP TABLE x' AS note FROM crm.customers"),
    (
        "write_keyword_in_literal",
        "SELECT cust_id FROM crm.customers WHERE status = 'DELETE; DROP TABLE x'",
    ),
    (
        "comment_mentions_a_write",
        "SELECT cust_id FROM crm.customers WHERE status = 'A' /* DELETE is never sent here */",
    ),
)

ALLOWED_BY_DIALECT: dict[str, tuple[Case, ...]] = {
    "postgres": (
        ("limit_offset", "SELECT cust_id FROM crm.customers ORDER BY cust_id LIMIT 10 OFFSET 5"),
        ("cast_operator", "SELECT cust_id::text AS cust_no FROM crm.customers"),
        (
            "now_interval",
            "SELECT COUNT(*) AS n FROM billing.usage_events "
            "WHERE start_time >= now() - interval '30 days'",
        ),
        (
            "aggregate_filter",
            "SELECT COUNT(*) FILTER (WHERE status = 'A') AS n FROM crm.customers",
        ),
        ("pg_helper_function", "SELECT pg_typeof(credit_limit) AS t FROM crm.customers LIMIT 1"),
        (
            "pg_catalog_schema",
            "SELECT relname FROM pg_catalog.pg_class WHERE relkind = 'r'",
        ),
        (
            "distinct_on",
            "SELECT DISTINCT ON (status) cust_id, status FROM crm.customers "
            "ORDER BY status, cust_id",
        ),
        ("lateral_join", "SELECT * FROM crm.customers t, LATERAL (SELECT 1 AS x) s"),
        (
            "recursive_cte",
            "WITH RECURSIVE r AS (SELECT 1 AS n UNION ALL SELECT n + 1 FROM r WHERE n < 5) "
            "SELECT n FROM r",
        ),
        ("tablesample", "SELECT * FROM crm.customers TABLESAMPLE SYSTEM (10)"),
        ("unnest", "SELECT * FROM UNNEST(ARRAY[1, 2]) AS u"),
        ("jsonb_agg", "SELECT JSONB_AGG(cust_id) AS ids FROM crm.customers"),
    ),
    "oracle": (
        ("rownum", "SELECT cust_id FROM crm.customers WHERE ROWNUM <= 10"),
        (
            "fetch_first",
            "SELECT cust_id FROM crm.customers ORDER BY cust_id FETCH FIRST 5 ROWS ONLY",
        ),
        (
            "sysdate_arithmetic",
            "SELECT COUNT(*) AS n FROM billing.usage_events WHERE start_time >= SYSDATE - 7",
        ),
        ("nvl", "SELECT NVL(credit_limit, 0) AS credit_limit FROM billing.accounts"),
        (
            "optimizer_hint",
            "SELECT /*+ INDEX(c IDX_CUST) */ cust_id FROM crm.customers c WHERE c.status = 'A'",
        ),
        ("from_dual", "SELECT SYSDATE AS now_ts FROM dual"),
        (
            "trunc_group_by",
            "SELECT TRUNC(start_time, 'MM') AS month, COUNT(*) AS n FROM billing.usage_events "
            "GROUP BY TRUNC(start_time, 'MM')",
        ),
        ("minus", "SELECT cust_id FROM crm.customers MINUS SELECT cust_id FROM billing.accounts"),
        (
            "outer_join_plus",
            "SELECT c.cust_id, e.client_no FROM crm.customers c, billing.accounts e "
            "WHERE LPAD(CAST(c.cust_id AS VARCHAR(10)), 10, '0') = e.client_no (+)",
        ),
        # Read-only SYS_ helpers must not be caught by a blanket prefix rule.
        (
            "sys_context",
            "SELECT SYS_CONTEXT('USERENV', 'SESSION_USER') AS u FROM dual",
        ),
        ("sys_guid", "SELECT SYS_GUID() AS g FROM dual"),
        (
            "listagg_within_group",
            "SELECT LISTAGG(cust_id, ',') WITHIN GROUP (ORDER BY cust_id) AS s FROM crm.customers",
        ),
    ),
    "tsql": (
        ("top", "SELECT TOP 10 cust_id FROM dbo.customers ORDER BY cust_id"),
        (
            "brackets",
            "SELECT [status], COUNT(*) AS [n] FROM [dbo].[customers] GROUP BY [status]",
        ),
        ("nolock", "SELECT cust_id FROM dbo.customers WITH (NOLOCK) WHERE status = 'A'"),
        ("readpast", "SELECT cust_id FROM dbo.customers WITH (READPAST, NOLOCK)"),
        (
            "dateadd_getdate",
            "SELECT COUNT(*) AS n FROM dbo.usage_events "
            "WHERE start_time >= DATEADD(day, -7, GETDATE())",
        ),
        ("isnull", "SELECT ISNULL(credit_limit, 0) AS credit_limit FROM dbo.accounts"),
        (
            "offset_fetch",
            "SELECT cust_id FROM dbo.customers ORDER BY cust_id "
            "OFFSET 10 ROWS FETCH NEXT 5 ROWS ONLY",
        ),
        ("convert", "SELECT CONVERT(varchar(10), cust_id) AS cust_no FROM dbo.customers"),
        ("except", "SELECT cust_id FROM dbo.customers EXCEPT SELECT cust_id FROM dbo.accounts"),
        (
            "eomonth",
            "SELECT COUNT(*) AS n FROM dbo.usage_events WHERE start_time >= EOMONTH(GETDATE(), -3)",
        ),
        ("national_literal", "SELECT cust_id FROM dbo.customers WHERE name = N'בנק לאומי'"),
        (
            "cross_apply",
            "SELECT * FROM dbo.customers c CROSS APPLY (SELECT TOP 1 * FROM dbo.accounts) x",
        ),
        (
            "string_agg_within_group",
            "SELECT status, STRING_AGG(cust_id, ',') WITHIN GROUP (ORDER BY cust_id) AS s "
            "FROM dbo.customers GROUP BY status",
        ),
    ),
}

# --------------------------------------------------------------------------------------------------
# Statements that must be blocked. Nothing here is portable-by-accident: each case must be refused
# in every dialect (as a write, as a parse failure, or as a batch).
# --------------------------------------------------------------------------------------------------
BLOCKED_COMMON: tuple[Case, ...] = (
    # DML
    ("insert_values", "INSERT INTO crm.customers (cust_id) VALUES (1)"),
    ("insert_select", "INSERT INTO t SELECT cust_id FROM crm.customers"),
    ("update_set", "UPDATE crm.customers SET status = 'C' WHERE cust_id = 1"),
    ("update_returning", "UPDATE crm.customers SET status = 'A' RETURNING cust_id"),
    ("delete_from", "DELETE FROM crm.customers"),
    ("delete_where", "DELETE FROM crm.customers WHERE cust_id = 1"),
    ("merge", "MERGE INTO t USING s ON t.id = s.id WHEN MATCHED THEN DELETE"),
    # DDL
    ("drop_table", "DROP TABLE crm.customers"),
    ("drop_view", "DROP VIEW v"),
    ("create_table", "CREATE TABLE t (a INT)"),
    ("create_index", "CREATE INDEX i ON crm.customers (status)"),
    ("create_view", "CREATE VIEW v AS SELECT cust_id FROM crm.customers"),
    ("alter_add_column", "ALTER TABLE crm.customers ADD COLUMN b INT"),
    ("alter_drop_column", "ALTER TABLE crm.customers DROP COLUMN status"),
    ("truncate", "TRUNCATE TABLE crm.customers"),
    ("comment_on", "COMMENT ON TABLE crm.customers IS 'x'"),
    # Privileges and session control
    ("grant", "GRANT SELECT ON crm.customers TO someone"),
    ("revoke", "REVOKE SELECT ON crm.customers FROM someone"),
    ("set_statement", "SET x = 1"),
    ("begin_transaction", "BEGIN"),
    ("commit", "COMMIT"),
    ("rollback", "ROLLBACK"),
    ("pragma", "PRAGMA table_info(crm.customers)"),
    ("show_all", "SHOW ALL"),
    ("use_database", "USE billing"),
    # Row locks and SELECT INTO
    ("select_into", "SELECT cust_id INTO newtab FROM crm.customers"),
    ("for_update", "SELECT cust_id FROM crm.customers FOR UPDATE"),
    ("for_share", "SELECT cust_id FROM crm.customers FOR SHARE"),
    ("for_no_key_update", "SELECT cust_id FROM crm.customers FOR NO KEY UPDATE"),
    # Statement smuggling
    ("two_selects", "SELECT cust_id FROM crm.customers; SELECT cust_id FROM billing.accounts"),
    ("select_then_delete", "SELECT cust_id FROM crm.customers; DELETE FROM crm.customers"),
    ("select_then_drop", "SELECT 1 AS one FROM crm.customers; DROP TABLE crm.customers"),
    (
        "comment_then_drop",
        "SELECT cust_id FROM crm.customers -- harmless note\n  ; DROP TABLE crm.customers",
    ),
    ("executable_comment", "SELECT cust_id FROM crm.customers /*! ; DROP TABLE crm.customers */"),
    # Data-modifying CTEs (the root is a SELECT; only the tree walk catches these)
    ("cte_delete", "WITH d AS (DELETE FROM crm.customers RETURNING cust_id) SELECT cust_id FROM d"),
    (
        "cte_insert",
        "WITH i AS (INSERT INTO crm.customers (cust_id) VALUES (1) RETURNING cust_id) "
        "SELECT cust_id FROM i",
    ),
    (
        "cte_update",
        "WITH u AS (UPDATE crm.customers SET status = 'C' RETURNING cust_id) SELECT cust_id FROM u",
    ),
    ("cte_then_delete", "WITH a AS (SELECT 1 AS one FROM crm.customers) DELETE FROM crm.customers"),
    # Stored procedures and server commands
    ("call_procedure", "CALL keep_alive(1)"),
    ("exec_procedure", "EXEC sp_who"),
    ("exec_dynamic_sql", "EXEC sp_executesql N'DROP TABLE crm.customers'"),
    ("explain", "EXPLAIN SELECT cust_id FROM crm.customers"),
    ("vacuum", "VACUUM"),
    ("copy_from", "COPY crm.customers FROM STDIN"),
    ("copy_to", "COPY (SELECT cust_id FROM crm.customers) TO STDOUT"),
    # Dangerous functions: files, large objects, sleep, remote, sequences
    ("pg_read_file", "SELECT pg_read_file('/etc/passwd') AS x FROM crm.customers"),
    ("pg_read_binary_file", "SELECT pg_read_binary_file('/etc/shadow') AS x FROM crm.customers"),
    ("pg_sleep", "SELECT pg_sleep(10) AS x FROM crm.customers"),
    ("lo_import", "SELECT lo_import('/etc/passwd') AS x FROM crm.customers"),
    ("lo_export", "SELECT lo_export(1, '/tmp/x') AS x FROM crm.customers"),
    ("dblink", "SELECT * FROM dblink('c', 'SELECT 1') AS t(a INT)"),
    ("dblink_exec", "SELECT dblink_exec('c', 'DROP TABLE crm.customers') AS x FROM crm.customers"),
    ("xp_cmdshell", "EXEC xp_cmdshell 'dir'"),
    ("openrowset", "SELECT * FROM OPENROWSET('SQLOLEDB', 'srv', 'x')"),
    ("openquery", "SELECT * FROM OPENQUERY(srv, 'SELECT 1')"),
    ("dbms_lock_sleep", "SELECT DBMS_LOCK.SLEEP(5) AS x FROM dual"),
    ("utl_file", "SELECT UTL_FILE.FOPEN('/tmp', 'x', 'R') AS f FROM dual"),
    ("utl_http", "SELECT UTL_HTTP.REQUEST('http://example.com') AS r FROM dual"),
    ("package_qualified_write", "SELECT SYS.DBMS_LOCK.SLEEP(1) AS x FROM dual"),
    ("nextval", "SELECT nextval('customer_seq') AS id FROM crm.customers"),
    ("setval", "SELECT setval('customer_seq', 1) AS id FROM crm.customers"),
    ("sys_exec", "SELECT SYS_EXEC('id') AS x FROM dual"),
    # A quoted or full-width spelling must not dodge the denylist.
    ("quoted_function", "SELECT \"PG_READ_FILE\"('/etc/passwd') AS x FROM crm.customers"),
    (
        "fullwidth_function",
        "SELECT \uff50\uff47\uff3f\uff52\uff45\uff41\uff44\uff3f\uff46\uff49\uff4c\uff45"
        "('/etc/passwd') AS x FROM crm.customers",
    ),
    # Locking table hints (T-SQL)
    ("updlock_hint", "SELECT cust_id FROM dbo.customers WITH (UPDLOCK)"),
    ("xlock_hint", "SELECT cust_id FROM dbo.customers WITH (XLOCK)"),
    ("tablock_hint", "SELECT cust_id FROM dbo.customers WITH (TABLOCK)"),
    ("tablockx_hint", "SELECT cust_id FROM dbo.customers WITH (TABLOCKX)"),
    ("holdlock_hint", "SELECT cust_id FROM dbo.customers WITH (HOLDLOCK)"),
    # Unicode lookalikes, control bytes, comment tricks
    ("fullwidth_semicolon", "SELECT cust_id FROM crm.customers\uff1bDROP TABLE crm.customers"),
    ("cyrillic_delete", "D\u0415LETE FROM crm.customers"),
    ("cyrillic_drop", "\u0420\u041eP TABLE crm.customers"),
    ("zero_width_in_keyword", "DR\u200bOP TABLE crm.customers"),
    ("nul_byte", "SELECT cust_id FROM crm.customers\x00"),
    # Not SQL at all
    ("empty", ""),
    ("whitespace", "   \n\t "),
    ("only_comment", "-- nothing to see here"),
    ("garbage_words", "this is not sql"),
    ("garbage_select", "SELECT FROM WHERE"),
    ("half_statement", "SELECT * FROM"),
    ("values_only", "VALUES (1)"),
    ("parenthesised_delete", "(DELETE FROM crm.customers)"),
    ("select_then_truncate", "SELECT cust_id FROM crm.customers; TRUNCATE TABLE crm.customers"),
    (
        "subquery_for_update",
        "SELECT t.cust_id FROM crm.customers t "
        "WHERE t.cust_id IN (SELECT cust_id FROM crm.customers FOR UPDATE)",
    ),
    # Client-side separators a driver or CLI would split on.
    ("psql_meta_copy", "\\copy crm.customers FROM 'x.csv'"),
    ("batch_separator_go", "SELECT cust_id FROM crm.customers\nGO\nDROP TABLE crm.customers"),
    (
        "hash_comment_smuggle",
        "SELECT cust_id FROM crm.customers # note\n; DROP TABLE crm.customers",
    ),
    ("waitfor", "WAITFOR DELAY '00:00:10'"),
    ("bulk_insert", "BULK INSERT crm.customers FROM 'x.csv'"),
    # Server administration and procedural batches
    ("declare_variable", "DECLARE @x INT"),
    ("conditional_drop", "IF 1 = 1 DROP TABLE dbo.customers"),
    ("set_nocount", "SET NOCOUNT ON"),
    ("dbcc", "DBCC CHECKDB"),
    ("backup", "BACKUP DATABASE billing TO DISK = 'x.bak'"),
    ("shutdown", "SHUTDOWN"),
    ("kill_session", "KILL 51"),
    ("analyze_table", "ANALYZE TABLE t"),
    ("lock_tables", "LOCK TABLES t WRITE"),
    ("rename_table", "RENAME TABLE a TO b"),
    ("print_message", "PRINT 'x'"),
    ("raise_error", "RAISERROR('x', 16, 1)"),
    ("configure_server", "sp_configure 'show advanced options', 1"),
)

BLOCKED_BY_DIALECT: dict[str, tuple[Case, ...]] = {
    "postgres": (
        ("for_key_share", "SELECT cust_id FROM crm.customers FOR KEY SHARE"),
        ("do_block", "DO $$ BEGIN DROP TABLE crm.customers; END $$"),
        ("vacuum_full", "VACUUM FULL"),
        ("truncate_cascade", "TRUNCATE crm.customers CASCADE"),
        ("create_schema", "CREATE SCHEMA s"),
        ("vacuum_analyze", "VACUUM ANALYZE crm.customers"),
        ("listen_channel", "LISTEN some_channel"),
    ),
    "oracle": (
        ("execute_immediate", "EXECUTE IMMEDIATE 'DROP TABLE crm.customers'"),
        ("plsql_block", "BEGIN keep_alive; END;"),
        ("for_update_nowait", "SELECT cust_id FROM crm.customers FOR UPDATE NOWAIT"),
        ("alter_system", "ALTER SYSTEM KILL SESSION '1,1'"),
        ("create_table_as", "CREATE TABLE t AS SELECT cust_id FROM crm.customers"),
        ("alter_session", "ALTER SESSION SET NLS_DATE_FORMAT = 'YYYY-MM-DD'"),
    ),
    "tsql": (
        ("exec_dynamic_string", "EXEC ('DROP TABLE dbo.customers')"),
        ("openrowset_bulk", "SELECT * FROM OPENROWSET(BULK 'c:/x.csv', SINGLE_BLOB) AS x"),
        ("insert_output", "INSERT INTO dbo.customers (cust_id) OUTPUT inserted.cust_id VALUES (1)"),
        ("drop_all", "DROP TABLE dbo.customers"),
        ("execute_as", "EXECUTE AS LOGIN = 'sa'"),
        ("identity_insert", "SET IDENTITY_INSERT dbo.customers ON"),
    ),
}


def _matrix(common: tuple[Case, ...], per_dialect: dict[str, tuple[Case, ...]]) -> list[Any]:
    """Run every portable case in all three dialects, plus the dialect-specific extras."""
    params = []
    for dialect in DIALECTS:
        for case_id, sql in (*common, *per_dialect.get(dialect, ())):
            params.append(pytest.param(dialect, sql, id=f"{dialect}-{case_id}"))
    return params


ALLOWED_PARAMS = _matrix(ALLOWED_COMMON, ALLOWED_BY_DIALECT)
BLOCKED_PARAMS = _matrix(BLOCKED_COMMON, BLOCKED_BY_DIALECT)


def _cases_for(dialect: str, params: list[pytest.ParameterSet]) -> list[Any]:
    return [param for param in params if param.values[0] == dialect]


def _corpus() -> list[tuple[str, str, str]]:
    """(dialect, expect id, sql) for every block of the golden ``tests/data/queries`` corpus."""
    found: list[tuple[str, str, str]] = []
    for path in sorted(CORPUS_DIR.glob("*.sql")):
        text = path.read_text(encoding="utf-8")
        for block in text.split("-- ---"):
            lines = block.splitlines()
            at = next(
                (
                    index
                    for index, line in enumerate(lines)
                    if line.strip().startswith("-- expect:")
                ),
                None,
            )
            if at is None:
                continue
            payload = json.loads(lines[at].strip()[len("-- expect:") :])
            sql = "\n".join(lines[at + 1 :]).strip()
            if sql:
                found.append((path.stem, str(payload["id"]), sql))
    return found


CORPUS = _corpus()
CORPUS_IDS = [f"{dialect}-{case_id}" for dialect, case_id, _ in CORPUS]


# --------------------------------------------------------------------------------------------------
# The two big tables
# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(("dialect", "sql"), ALLOWED_PARAMS)
def test_read_only_statement_is_allowed(dialect: str, sql: str) -> None:
    result = guard(sql, dialect)
    assert result.ok, f"{dialect} rejected a read-only statement: {result.error}"
    assert result.error is None
    assert isinstance(result.ast, READ_ROOTS)
    assert result.dialect == dialect


@pytest.mark.parametrize(("dialect", "sql"), BLOCKED_PARAMS)
def test_write_or_batch_or_garbage_is_blocked(dialect: str, sql: str) -> None:
    result = guard(sql, dialect)
    assert not result.ok, f"{dialect} allowed a statement that must be refused: {sql!r}"
    assert result.error is not None
    assert result.error.code in CODES
    assert result.error.problem.strip()
    assert result.error.fix.strip()
    assert result.ast is None, "a rejection must not hand an AST back to the caller"
    assert result.dialect == dialect


@pytest.mark.parametrize("dialect", DIALECTS)
def test_tables_are_big_enough(dialect: str) -> None:
    allowed = _cases_for(dialect, ALLOWED_PARAMS)
    blocked = _cases_for(dialect, BLOCKED_PARAMS)
    assert len(allowed) >= MIN_ALLOWED_CASES, f"{dialect}: only {len(allowed)} allowed cases"
    assert len(blocked) >= MIN_BLOCKED_CASES, f"{dialect}: only {len(blocked)} blocked cases"


# --------------------------------------------------------------------------------------------------
# Exact codes and the error shape
# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("dialect", DIALECTS)
@pytest.mark.parametrize(
    ("sql", "code"),
    [
        ("DELETE FROM crm.customers", WRITE_BLOCKED),
        ("DROP TABLE crm.customers", WRITE_BLOCKED),
        ("CALL keep_alive(1)", WRITE_BLOCKED),
        ("SELECT pg_read_file('/etc/passwd') AS x FROM dual", WRITE_BLOCKED),
        ("SELECT cust_id FROM crm.customers; SELECT 1 AS one FROM dual", MULTI_STATEMENT),
        ("", PARSE_ERROR),
        ("SELECT cust_id FROM crm.customers\x00", PARSE_ERROR),
    ],
)
def test_exact_code(dialect: str, sql: str, code: str) -> None:
    result = guard(sql, dialect)
    assert result.error is not None
    assert result.error.code == code, f"{sql!r} in {dialect}: {result.error}"


@pytest.mark.parametrize("dialect", ("postgres", "oracle"))
@pytest.mark.parametrize(
    ("sql", "code"),
    [
        ("SELECT cust_id FROM crm.customers FOR UPDATE", WRITE_BLOCKED),
        ("SELECT cust_id INTO newtab FROM crm.customers", WRITE_BLOCKED),
    ],
)
def test_exact_code_for_locking_and_into(dialect: str, sql: str, code: str) -> None:
    # T-SQL has no FOR UPDATE and no SELECT ... INTO of this shape: there the parser rejects it,
    # which is equally safe, so the case lives here with the dialects that express it.
    result = guard(sql, dialect)
    assert result.error is not None
    assert result.error.code == code, f"{sql!r} in {dialect}: {result.error}"


@pytest.mark.parametrize("dialect", DIALECTS)
def test_unparseable_text_is_refused(dialect: str) -> None:
    # 'this is not sql' parses as a stray alias expression, so it is refused as "not a query" rather
    # than as a parse error; either way the statement never reaches a source.
    result = guard("this is not sql", dialect)
    assert result.error is not None
    assert result.error.code in (PARSE_ERROR, WRITE_BLOCKED)


@pytest.mark.parametrize("dialect", DIALECTS)
def test_error_payload_is_agent_facing(dialect: str) -> None:
    result = guard("DELETE FROM crm.customers", dialect)
    assert result.error is not None
    dumped = result.error.model_dump()
    assert set(dumped) == {"status", "code", "problem", "fix"}
    assert dumped["status"] == "error"
    assert "DELETE" in dumped["problem"]
    assert "remove" in dumped["fix"].lower() or "replace" in dumped["fix"].lower()
    json.loads(result.error.model_dump_json())  # the agent receives JSON, never a traceback


@pytest.mark.parametrize("dialect", DIALECTS)
def test_cte_hiding_a_write_names_the_write(dialect: str) -> None:
    result = guard(
        "WITH d AS (DELETE FROM crm.customers RETURNING cust_id) SELECT cust_id FROM d", dialect
    )
    assert result.error is not None
    assert "DELETE" in result.error.problem
    assert "CTE" in result.error.problem


@pytest.mark.parametrize("dialect", DIALECTS)
def test_multi_statement_fix_says_one_statement(dialect: str) -> None:
    result = guard("SELECT 1 AS one FROM dual; DROP TABLE crm.customers", dialect)
    assert result.error is not None
    assert result.error.code == MULTI_STATEMENT
    assert "one statement" in result.error.fix.lower()


@pytest.mark.parametrize("dialect", DIALECTS)
def test_denied_function_fix_names_the_function(dialect: str) -> None:
    result = guard("SELECT pg_sleep(1) AS x FROM dual", dialect)
    assert result.error is not None
    assert "pg_sleep" in result.error.problem
    assert "pg_sleep" in result.error.fix


# --------------------------------------------------------------------------------------------------
# The golden corpus must never be blocked by the guard
# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(("dialect", "case_id", "sql"), CORPUS, ids=CORPUS_IDS or None)
def test_golden_corpus_statement_is_allowed(dialect: str, case_id: str, sql: str) -> None:
    result = guard(sql, dialect)
    assert result.ok, f"{case_id} ({dialect}) must be allowed: {result.error}"


@pytest.mark.parametrize(("dialect", "case_id", "sql"), CORPUS, ids=CORPUS_IDS or None)
def test_reserialised_corpus_ast_is_allowed(dialect: str, case_id: str, sql: str) -> None:
    # The server guards the SQL it compiled itself; regenerating SQL from the guarded AST is the
    # cheapest stand-in for "the SQL we send" (C05 wires the compiler to this guard).
    first = guard(sql, dialect)
    assert first.ok and first.ast is not None
    compiled = first.ast.sql(dialect=dialect)
    second = guard(compiled, dialect)
    assert second.ok, f"{case_id}: compiled SQL was refused: {second.error}\n{compiled}"


def test_corpus_is_not_empty() -> None:
    assert len(CORPUS) >= 40, "the golden corpus went missing; the guard test is now vacuous"


# --------------------------------------------------------------------------------------------------
# Policy: the denylist is configurable, in both directions
# --------------------------------------------------------------------------------------------------
def test_policy_can_deny_an_extra_function() -> None:
    policy = GuardPolicy(denied_functions=frozenset({"sum"}))
    result = guard("SELECT SUM(credit_limit) AS total FROM crm.customers", "postgres", policy)
    assert result.error is not None
    assert "sum" in result.error.problem.lower()


def test_policy_can_deny_an_extra_prefix() -> None:
    policy = GuardPolicy(
        denied_function_prefixes=(*GuardPolicy().denied_function_prefixes, "ratio_to_report")
    )
    result = guard(
        "SELECT RATIO_TO_REPORT(credit_limit) OVER () AS share FROM crm.customers",
        "postgres",
        policy,
    )
    assert result.error is not None
    assert "ratio_to_report" in result.error.problem.lower()


def test_policy_without_denials_allows_a_packaged_helper() -> None:
    # Loosening is possible but explicit: a deployment that trusts UTL_INADDR can say so.
    policy = GuardPolicy(denied_functions=frozenset(), denied_function_prefixes=())
    sql = "SELECT UTL_INADDR.GET_HOST_ADDRESS('x') AS h FROM dual"
    assert guard(sql, "oracle").error is not None
    assert guard(sql, "oracle", policy).ok


def test_policy_controls_table_hints() -> None:
    sql = "SELECT cust_id FROM dbo.customers WITH (NOLOCK)"
    strict = GuardPolicy(allowed_table_hints=frozenset())
    assert guard(sql, "tsql").ok
    assert guard(sql, "tsql", strict).error is not None


def test_default_policy_is_shared_not_copied() -> None:
    assert guard("SELECT cust_id FROM crm.customers", "postgres").ok
    assert GuardPolicy().denied_functions is GuardPolicy().denied_functions


# --------------------------------------------------------------------------------------------------
# Hypothesis: the guard never raises, never allows a smuggled statement, and tolerates separators
# --------------------------------------------------------------------------------------------------
SQL_ALPHABET = "SELECT FROM WHERE ;'\u0000/*!-\\\n\t()=,_0123456789dbms_pg_read_fileDELETE"

BLOCKED_SAMPLES: tuple[str, ...] = tuple(sql for _, sql in BLOCKED_COMMON if sql.strip())
ALLOWED_SAMPLES: tuple[str, ...] = tuple(sql for _, sql in ALLOWED_COMMON)

GLUE: tuple[str, ...] = ("", " ", "\n", "\t", "-- note\n", "/* note */ ", "  \n  ")


@settings(max_examples=200, deadline=None)
@given(text=st.text(alphabet=SQL_ALPHABET, max_size=60), dialect=st.sampled_from(DIALECTS))
def test_guard_never_raises_and_answers_coherently(text: str, dialect: str) -> None:
    result = guard(text, dialect)
    assert isinstance(result, GuardResult)
    if result.ok:
        assert result.error is None
        assert isinstance(result.ast, READ_ROOTS)
    else:
        assert result.error is not None
        assert result.error.code in CODES
        assert result.error.fix.strip()
        assert result.ast is None


@settings(max_examples=150, deadline=None)
@given(
    glue=st.sampled_from(GLUE),
    sql=st.sampled_from(BLOCKED_SAMPLES),
    dialect=st.sampled_from(DIALECTS),
)
def test_glue_never_smuggles_a_blocked_statement(glue: str, sql: str, dialect: str) -> None:
    assert not guard(f"{glue}{sql}", dialect).ok
    assert not guard(f"{sql}{glue}", dialect).ok


@settings(max_examples=100, deadline=None)
@given(
    sql=st.sampled_from(ALLOWED_SAMPLES),
    extra=st.integers(min_value=1, max_value=5),
    dialect=st.sampled_from(DIALECTS),
)
def test_trailing_separators_keep_a_query_allowed(sql: str, extra: int, dialect: str) -> None:
    assert guard(f"{sql}{';' * extra}", dialect).ok


@settings(max_examples=100, deadline=None)
@given(
    sql=st.sampled_from(ALLOWED_SAMPLES),
    padding=st.text(alphabet=" \n\t", min_size=1, max_size=6),
    dialect=st.sampled_from(DIALECTS),
)
def test_surrounding_whitespace_keeps_a_query_allowed(sql: str, padding: str, dialect: str) -> None:
    assert guard(f"{padding}{sql}{padding}", dialect).ok
