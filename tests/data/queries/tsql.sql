-- Golden SQL corpus: T-SQL dialect (SQL Server 2022).
--
-- Format and expect-block rules are documented in tests/unit/test_query_corpus.py.
-- Every statement is a single read-only SELECT.
-- T-SQL specifics covered here: TOP, [bracketed] identifiers, GETDATE()/GETUTCDATE()/SYSDATETIME(),
-- DATEADD/DATEDIFF/EOMONTH, CONVERT, ISNULL, WITH (NOLOCK), OFFSET ... FETCH.

-- expect: {"id": "ts-01", "tables": ["customers"], "filters": ["status"]}
SELECT TOP 100 cust_id, status, credit_limit
FROM dbo.customers
WHERE status = 'A'
ORDER BY cust_id

-- ---
-- expect: {"id": "ts-02", "tables": ["customers"], "filters": ["status"], "group_by": ["status"], "aggregations": ["count"]}
SELECT [status], COUNT(*) AS [n]
FROM [dbo].[customers]
WHERE [status] <> 'C'
GROUP BY [status]

-- ---
-- expect: {"id": "ts-03", "tables": ["usage_events"], "filters": ["start_time"], "time_columns": ["start_time"], "now_functions": ["getdate"]}
SELECT client_no, bytes
FROM dbo.usage_events
WHERE start_time >= GETDATE() - 7

-- ---
-- expect: {"id": "ts-04", "tables": ["usage_events"], "filters": ["start_time"], "time_columns": ["start_time"], "now_functions": ["getdate"]}
SELECT client_no, bytes
FROM dbo.usage_events
WHERE start_time >= DATEADD(day, -7, GETDATE())

-- ---
-- expect: {"id": "ts-05", "tables": ["usage_events"], "group_by": ["client_no"], "aggregations": ["avg"]}
SELECT client_no, AVG(DATEDIFF(day, start_time, insertion_time)) AS avg_lag_days
FROM dbo.usage_events
GROUP BY client_no

-- ---
-- expect: {"id": "ts-06", "tables": ["accounts", "customers"], "joins": [{"tables": ["customers", "accounts"], "columns": ["cust_id", "client_no"], "conversion": "convert", "conversion_expr": "CONVERT(varchar(10), c.cust_id)"}]}
SELECT c.cust_id, e.client_no
FROM dbo.customers AS c
JOIN dbo.accounts AS e
  ON CONVERT(varchar(10), c.cust_id) = e.client_no

-- ---
-- expect: {"id": "ts-07", "tables": ["accounts", "customers"], "joins": [{"tables": ["customers", "accounts"], "columns": ["cust_id", "client_no"], "conversion": "cast", "conversion_expr": "CAST(c.cust_id AS varchar(10))"}]}
SELECT c.cust_id, e.client_no, e.status
FROM dbo.customers AS c
JOIN dbo.accounts AS e
  ON CAST(c.cust_id AS varchar(10)) = e.client_no

-- ---
-- expect: {"id": "ts-08", "tables": ["customers"], "rounding": ["ROUND(credit_limit / 5000, 0) * 5000"]}
SELECT cust_id, ROUND(credit_limit / 5000, 0) * 5000 AS canonical
FROM dbo.customers

-- ---
-- expect: {"id": "ts-09", "tables": ["erp_customers"], "filters": ["cust_no"], "rounding": ["limit_k * 10"]}
SELECT cust_no, limit_k * 10 AS credit_limit
FROM dbo.erp_customers
WHERE cust_no > 0

-- ---
-- expect: {"id": "ts-10", "tables": ["accounts"], "filters": ["status"]}
SELECT client_no, ISNULL(credit_limit, 0) AS credit_limit
FROM dbo.accounts
WHERE status = 'Active'

-- ---
-- expect: {"id": "ts-11", "tables": ["usage_events"], "filters": ["insertion_time", "start_time"], "time_columns": ["insertion_time", "start_time"], "now_functions": ["getdate"]}
SELECT client_no, bytes
FROM dbo.usage_events WITH (NOLOCK)
WHERE insertion_time >= DATEADD(hour, -6, GETDATE())
  AND start_time >= DATEADD(hour, -12, GETDATE())

-- ---
-- expect: {"id": "ts-12", "tables": ["customers", "usage_events"], "joins": [{"tables": ["recent", "customers"], "columns": ["cust_id", "cust_id"]}], "filters": ["start_time"], "time_columns": ["start_time"], "group_by": ["cust_id"], "aggregations": ["count", "sum"], "now_functions": ["getdate"]}
WITH recent AS (
  SELECT cust_id, SUM(bytes) AS bytes
  FROM dbo.usage_events
  WHERE start_time >= DATEADD(month, -1, GETDATE())
  GROUP BY cust_id
)
SELECT COUNT(*) AS n
FROM recent AS r
JOIN dbo.customers AS c
  ON c.cust_id = r.cust_id

-- ---
-- expect: {"id": "ts-13", "tables": ["customers"], "filters": ["n"], "group_by": ["status"], "aggregations": ["count"]}
SELECT TOP 10 t.status, t.n
FROM (
  SELECT status, COUNT(*) AS n
  FROM dbo.customers
  GROUP BY status
) AS t
WHERE t.n > 5
ORDER BY t.n DESC

-- ---
-- expect: {"id": "ts-14", "tables": ["customers"], "filters": ["status"], "group_by": ["status"], "aggregations": ["sum"]}
SELECT status, SUM(credit_limit) AS total
FROM dbo.customers
WHERE status IN ('A', 'S')
GROUP BY status
HAVING SUM(credit_limit) > 100000

-- ---
-- expect: {"id": "ts-15", "tables": ["customers"], "filters": ["status"], "aggregations": ["count"]}
SELECT 'active' AS grp, COUNT(*) AS n FROM dbo.customers WHERE status = 'A'
UNION ALL
SELECT 'closed' AS grp, COUNT(*) AS n FROM dbo.customers WHERE status = 'C'

-- ---
-- expect: {"id": "ts-16", "tables": ["customers"], "filters": ["name"]}
SELECT cust_id, name
FROM dbo.customers
WHERE name = N'בנק לאומי'
  AND name LIKE N'%בנק%'

-- ---
-- expect: {"id": "ts-17", "tables": ["accounts_hist"], "filters": ["updated_at"], "time_columns": ["updated_at"], "now_functions": ["getdate"]}
SELECT client_no, updated_at,
       ROW_NUMBER() OVER (PARTITION BY client_no ORDER BY updated_at DESC) AS rn
FROM dbo.accounts_hist
WHERE updated_at >= DATEADD(day, -30, GETDATE())

-- ---
-- expect: {"id": "ts-18", "tables": ["customers"]}
SELECT cust_id, updated_at
FROM dbo.customers
ORDER BY updated_at DESC
OFFSET 200 ROWS FETCH NEXT 100 ROWS ONLY

-- ---
-- expect: {"id": "ts-19", "tables": ["usage_events"], "filters": ["start_time"], "time_columns": ["start_time"], "aggregations": ["count"], "now_functions": ["getdate"]}
SELECT COUNT(*) AS n
FROM dbo.usage_events
WHERE start_time >= EOMONTH(GETDATE(), -3)

-- ---
-- expect: {"id": "ts-20", "tables": ["usage_events"], "filters": ["insertion_time"], "time_columns": ["insertion_time"], "aggregations": ["max"], "now_functions": ["getutcdate"]}
SELECT MAX(insertion_time) AS last_load
FROM dbo.usage_events
WHERE insertion_time >= DATEADD(hour, -24, GETUTCDATE())

-- ---
-- expect: {"id": "ts-21", "tables": ["usage_events"], "filters": ["start_time"], "time_columns": ["start_time"], "aggregations": ["count"], "now_functions": ["sysdatetime"]}
SELECT COUNT(*) AS n
FROM dbo.usage_events
WHERE start_time >= DATEADD(day, -1, SYSDATETIME())

-- ---
-- expect: {"id": "ts-22", "tables": ["usage_events"], "filters": ["insertion_time", "start_time"], "time_columns": ["insertion_time", "start_time"], "now_functions": ["current_timestamp"]}
SELECT client_no
FROM dbo.usage_events
WHERE start_time >= DATEADD(day, -7, CURRENT_TIMESTAMP)
  AND insertion_time < CURRENT_TIMESTAMP

-- ---
-- expect: {"id": "ts-23", "tables": ["accounts"], "aggregations": ["count"]}
SELECT COUNT(DISTINCT client_no) AS distinct_clients
FROM dbo.accounts

-- ---
-- expect: {"id": "ts-24", "tables": ["customers"], "filters": ["credit_limit"]}
SELECT cust_id,
       CASE WHEN credit_limit >= 100000 THEN 'large'
            ELSE 'small' END AS band
FROM dbo.customers
WHERE credit_limit IS NOT NULL

-- ---
-- expect: {"id": "ts-25", "tables": ["accounts", "customers"], "joins": [{"tables": ["customers", "accounts"], "columns": ["cust_id", "client_no"], "conversion": "convert", "conversion_expr": "CONVERT(varchar(10), c.cust_id)"}], "filters": ["status"]}
SELECT c.cust_id, c.status
FROM dbo.customers AS c
WHERE EXISTS (
  SELECT 1
  FROM dbo.accounts AS e
  WHERE e.client_no = CONVERT(varchar(10), c.cust_id)
    AND e.status = 'Active'
)

-- ---
-- expect: {"id": "ts-26", "tables": ["accounts", "customers"], "joins": [{"tables": ["customers", "accounts"], "columns": ["cust_id", "client_no"], "conversion": "convert", "conversion_expr": "CONVERT(varchar(10), c.cust_id)"}]}
SELECT c.cust_id, ISNULL(e.credit_limit, 0) AS credit_limit
FROM dbo.customers AS c
LEFT JOIN dbo.accounts AS e
  ON CONVERT(varchar(10), c.cust_id) = e.client_no

-- ---
-- expect: {"id": "ts-27", "tables": ["accounts", "customers", "usage_events"], "joins": [{"tables": ["customers", "accounts"], "columns": ["cust_id", "client_no"], "conversion": "convert", "conversion_expr": "CONVERT(varchar(10), c.cust_id)"}, {"tables": ["accounts", "usage_events"], "columns": ["client_no", "client_no"]}], "filters": ["insertion_time", "start_time"], "time_columns": ["insertion_time", "start_time"], "group_by": ["c.status"], "aggregations": ["avg"], "rounding": ["ROUND(AVG(u.bytes) / 5000, 0) * 5000"], "now_functions": ["getdate"]}
SELECT c.status, ROUND(AVG(u.bytes) / 5000, 0) * 5000 AS bucket
FROM dbo.customers AS c
JOIN dbo.accounts AS e
  ON CONVERT(varchar(10), c.cust_id) = e.client_no
JOIN dbo.usage_events AS u
  ON u.client_no = e.client_no
WHERE u.start_time >= DATEADD(day, -30, GETDATE())
  AND u.insertion_time >= DATEADD(hour, -36, GETDATE())
GROUP BY c.status

-- ---
-- expect: {"id": "ts-28", "tables": ["customers"], "filters": ["name"]}
SELECT cust_id, name
FROM dbo.customers
WHERE name LIKE N'%בנק%'

-- ---
-- expect: {"id": "ts-29", "tables": ["usage_events"], "filters": ["bytes", "start_time"], "time_columns": ["start_time"], "aggregations": ["count"]}
SELECT COUNT(*) AS big_rows
FROM dbo.usage_events
WHERE bytes BETWEEN 1000 AND 100000
  AND start_time >= '2026-11-01T00:00:00'

-- ---
-- expect: {"id": "ts-30", "tables": ["customers", "usage_events"], "joins": [{"tables": ["customers", "usage_events"], "columns": ["cust_id", "cust_id"]}], "filters": ["start_time"], "time_columns": ["start_time"], "group_by": ["c.status"], "aggregations": ["count"], "now_functions": ["getdate"]}
SELECT c.status, COUNT(*) AS n
FROM dbo.customers AS c
JOIN dbo.usage_events AS u
  ON u.cust_id = c.cust_id
WHERE u.start_time >= DATEADD(day, -30, GETDATE())
GROUP BY c.status

-- ---
-- expect: {"id": "ts-31", "tables": ["customers"], "filters": ["status"], "aggregations": ["count", "sum"]}
SELECT SUM(credit_limit) / COUNT(*) AS avg_credit_limit
FROM dbo.customers
WHERE status = 'A'

-- ---
-- expect: {"id": "ts-32", "tables": ["usage_by_load"], "filters": ["insertion_time", "load_id"], "time_columns": ["insertion_time"], "group_by": ["load_id"], "aggregations": ["sum"], "now_functions": ["getdate"]}
SELECT load_id, SUM(bytes) AS total_bytes
FROM dbo.usage_by_load
WHERE load_id = 7
  AND insertion_time >= DATEADD(day, -1, GETDATE())
GROUP BY load_id

-- ---
-- expect: {"id": "ts-33", "tables": ["usage_events"], "filters": ["start_time"], "time_columns": ["start_time"], "group_by": ["CONVERT(date, start_time)"], "aggregations": ["count"]}
SELECT CONVERT(date, start_time) AS day, COUNT(*) AS n
FROM dbo.usage_events
WHERE start_time >= '2026-11-01'
GROUP BY CONVERT(date, start_time)

-- ---
-- expect: {"id": "ts-34", "tables": ["customers"], "rounding": ["CAST(credit_limit AS decimal(18, 2)) / 1000"]}
SELECT cust_id, CAST(credit_limit AS decimal(18, 2)) / 1000 AS credit_k
FROM dbo.customers

-- ---
-- expect: {"id": "ts-35", "tables": ["accounts"], "filters": ["status"], "group_by": ["status"], "aggregations": ["avg"]}
SELECT status, AVG(credit_limit) AS avg_credit
FROM dbo.accounts
WHERE status <> 'Closed'
GROUP BY status

-- ---
-- expect: {"id": "ts-36", "tables": ["usage_events"], "filters": ["start_time"], "time_columns": ["start_time"], "aggregations": ["sum"]}
SELECT client_no,
       SUM(bytes) OVER (PARTITION BY client_no ORDER BY start_time) AS running_bytes
FROM dbo.usage_events
WHERE start_time >= '2026-11-01'

-- ---
-- expect: {"id": "ts-37", "tables": ["customers"], "filters": ["status", "updated_at"], "time_columns": ["updated_at"], "now_functions": ["getdate"]}
-- recently touched customers, joined rows only
SELECT cust_id, status, updated_at
FROM dbo.customers WITH (NOLOCK) /* read uncommitted is fine for a profile */
WHERE status = 'A'
  AND updated_at >= DATEADD(day, -1, GETDATE())

-- ---
-- expect: {"id": "ts-38", "tables": ["usage_events"], "filters": ["client_no"], "aggregations": ["max", "min"]}
SELECT MIN(bytes) AS min_bytes, MAX(bytes) AS max_bytes
FROM dbo.usage_events
WHERE client_no = '0000000042'

-- ---
-- expect: {"id": "ts-39", "tables": ["customers"], "filters": ["status"], "group_by": ["status", "ROUND(credit_limit / 5000.0, 0) * 5000"], "aggregations": ["count"], "rounding": ["ROUND(credit_limit / 5000.0, 0) * 5000"]}
SELECT status, ROUND(credit_limit / 5000.0, 0) * 5000 AS canonical, COUNT(*) AS n
FROM dbo.customers
WHERE status = 'A'
GROUP BY status, ROUND(credit_limit / 5000.0, 0) * 5000

-- ---
-- expect: {"id": "ts-40", "tables": ["accounts", "customers"], "joins": [{"tables": ["customers", "accounts"], "columns": ["cust_id", "client_no"], "conversion": "cast", "conversion_expr": "CAST(c.cust_id AS varchar(10))"}], "filters": ["status"], "group_by": ["e.status"], "aggregations": ["count"]}
SELECT e.status, COUNT(*) AS n
FROM dbo.customers AS c
JOIN dbo.accounts AS e
  ON CAST(c.cust_id AS varchar(10)) = e.client_no
WHERE c.status = 'A'
GROUP BY e.status

-- ---
-- expect: {"id": "ts-41", "tables": ["usage_events"], "filters": ["insertion_time", "start_time"], "time_columns": ["insertion_time", "start_time"], "aggregations": ["count"], "now_functions": ["getdate"]}
SELECT COUNT(*) AS n
FROM dbo.usage_events
WHERE insertion_time >= DATEADD(minute, -360, GETDATE())
  AND start_time >= DATEADD(minute, -720, GETDATE())

-- ---
-- expect: {"id": "ts-42", "tables": ["accounts_hist"], "filters": ["status"], "group_by": ["client_no", "status"], "aggregations": ["count"]}
SELECT [client_no], [status], COUNT(*) AS [changes]
FROM [dbo].[accounts_hist]
WHERE [status] IN (N'Active', N'Hold')
GROUP BY [client_no], [status]

-- ---
-- expect: {"id": "ts-43", "tables": ["customers"], "rounding": ["ROUND(credit_limit, -3)"]}
SELECT cust_id, ROUND(credit_limit, -3) AS credit_k3
FROM dbo.customers

-- ---
-- expect: {"id": "ts-44", "tables": ["accounts", "customers"], "joins": [{"tables": ["customers", "accounts"], "columns": ["cust_id", "client_no"], "conversion": "try_cast", "conversion_expr": "TRY_CAST(e.client_no AS int)"}]}
SELECT c.cust_id, e.client_no
FROM dbo.customers AS c
JOIN dbo.accounts AS e
  ON c.cust_id = TRY_CAST(e.client_no AS int)
