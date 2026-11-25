-- Golden SQL corpus: PostgreSQL dialect.
--
-- Format (see tests/unit/test_query_corpus.py, which validates this file):
--   * statements are separated by a line that is exactly `-- ---`
--   * every statement is preceded by one `-- expect: {json}` line
--   * `-- expect` fields are documented in the test module docstring; the list holds
--     names as written in the statement, collapsed whitespace, sorted
--   * every statement is a single read-only SELECT (also used as the oracle for C01/C02/S08)
--
-- Names come from the synthetic organisation of AGENTS.md section 6
-- (crm.customers rounded to 5000, billing.accounts raw, erp.customers rounded then /10,
--  billing.usage_events partitioned monthly on start_time, lag on insertion_time).

-- expect: {"id": "pg-01", "tables": ["customers"], "filters": ["status"]}
SELECT cust_id, status, credit_limit
FROM crm.customers
WHERE status = 'A'
ORDER BY cust_id
LIMIT 50

-- ---
-- expect: {"id": "pg-02", "tables": ["customers"], "group_by": ["status"], "aggregations": ["avg"]}
SELECT status, AVG(credit_limit) AS avg_credit_limit
FROM crm.customers
GROUP BY status
ORDER BY status

-- ---
-- expect: {"id": "pg-03", "tables": ["customers"], "filters": ["status"], "group_by": ["status"], "aggregations": ["avg"], "rounding": ["ROUND(AVG(credit_limit) / 5000) * 5000"]}
SELECT status, ROUND(AVG(credit_limit) / 5000) * 5000 AS rounded_avg
FROM crm.customers
WHERE status <> 'C'
GROUP BY status

-- ---
-- expect: {"id": "pg-04", "tables": ["usage_events"], "filters": ["start_time"], "time_columns": ["start_time"], "aggregations": ["count"], "now_functions": ["now"]}
SELECT COUNT(*) AS n
FROM billing.usage_events
WHERE start_time >= now() - interval '30 days'

-- ---
-- expect: {"id": "pg-05", "tables": ["usage_events"], "filters": ["insertion_time", "start_time"], "time_columns": ["insertion_time", "start_time"], "now_functions": ["now"]}
SELECT client_no, bytes
FROM billing.usage_events
WHERE insertion_time >= now() - interval '6 hours'
  AND start_time >= now() - interval '12 hours'

-- ---
-- expect: {"id": "pg-06", "tables": ["accounts", "customers"], "joins": [{"tables": ["customers", "accounts"], "columns": ["cust_id", "client_no"], "conversion": "lpad", "conversion_expr": "LPAD(c.cust_id::varchar, 10, '0')"}], "filters": ["status"]}
SELECT c.cust_id, c.status, e.client_no
FROM crm.customers AS c
JOIN billing.accounts AS e
  ON LPAD(c.cust_id::varchar, 10, '0') = e.client_no
WHERE c.status = 'A'

-- ---
-- expect: {"id": "pg-07", "tables": ["accounts", "customers"], "joins": [{"tables": ["customers", "accounts"], "columns": ["cust_id", "client_no"], "conversion": "to_char", "conversion_expr": "TO_CHAR(c.cust_id, 'FM0000000000')"}]}
SELECT c.cust_id, e.client_no
FROM crm.customers AS c
JOIN billing.accounts AS e
  ON TO_CHAR(c.cust_id, 'FM0000000000') = e.client_no

-- ---
-- expect: {"id": "pg-08", "tables": ["accounts", "customers", "usage_events"], "joins": [{"tables": ["customers", "accounts"], "columns": ["cust_id", "client_no"], "conversion": "lpad", "conversion_expr": "LPAD(c.cust_id::varchar, 10, '0')"}, {"tables": ["accounts", "usage_events"], "columns": ["client_no", "client_no"]}], "filters": ["start_time"], "time_columns": ["start_time"], "group_by": ["c.status"], "aggregations": ["sum"], "now_functions": ["now"]}
SELECT c.status, SUM(u.bytes) AS total_bytes
FROM crm.customers AS c
JOIN billing.accounts AS e
  ON LPAD(c.cust_id::varchar, 10, '0') = e.client_no
JOIN billing.usage_events AS u
  ON u.client_no = e.client_no
WHERE u.start_time >= now() - interval '7 days'
GROUP BY c.status

-- ---
-- expect: {"id": "pg-09", "tables": ["customers", "usage_events"], "joins": [{"tables": ["recent", "customers"], "columns": ["cust_id", "cust_id"]}], "filters": ["start_time"], "time_columns": ["start_time"], "aggregations": ["count"], "now_functions": ["now"]}
WITH recent AS (
  SELECT cust_id
  FROM billing.usage_events
  WHERE start_time >= now() - interval '30 days'
)
SELECT COUNT(DISTINCT r.cust_id) AS active_customers
FROM recent AS r
JOIN crm.customers AS c
  ON c.cust_id = r.cust_id

-- ---
-- expect: {"id": "pg-10", "tables": ["customers"]}
SELECT DISTINCT ON (cust_id) cust_id, status, updated_at
FROM crm.customers
ORDER BY cust_id, updated_at DESC

-- ---
-- expect: {"id": "pg-11", "tables": ["customers"], "filters": ["status"], "aggregations": ["count"]}
SELECT COUNT(*) FILTER (WHERE status = 'A') AS active,
       COUNT(*) AS total
FROM crm.customers

-- ---
-- expect: {"id": "pg-12", "tables": ["customers"], "filters": ["n"], "group_by": ["status"], "aggregations": ["count"]}
SELECT t.status, t.n
FROM (
  SELECT status, COUNT(*) AS n
  FROM crm.customers
  GROUP BY status
) AS t
WHERE t.n > 10

-- ---
-- expect: {"id": "pg-13", "tables": ["customers"], "filters": ["deleted_at", "status"]}
SELECT cust_id, status
FROM crm.customers
WHERE status = 'פעיל'
  AND deleted_at IS NULL

-- ---
-- expect: {"id": "pg-14", "tables": ["customers"], "filters": ["status"]}
SELECT "c"."cust_id", "c"."status"
FROM "crm"."customers" AS "c"
WHERE "c"."status" <> 'C'

-- ---
-- expect: {"id": "pg-15", "tables": ["customers"], "filters": ["credit_limit", "status"]}
SELECT cust_id, credit_limit
FROM crm.customers /* every customer we still bill */
WHERE status = 'A' -- only active ones
  AND credit_limit > 0

-- ---
-- expect: {"id": "pg-16", "tables": ["customers"], "filters": ["status"]}
SELECT cust_id, ROW_NUMBER() OVER (PARTITION BY status ORDER BY cust_id) AS rn
FROM crm.customers
WHERE status IN ('A', 'S')

-- ---
-- expect: {"id": "pg-17", "tables": ["usage_events"], "filters": ["start_time"], "time_columns": ["start_time"], "group_by": ["DATE_TRUNC('month', start_time)"], "aggregations": ["count"], "now_functions": ["now"]}
SELECT DATE_TRUNC('month', start_time) AS month, COUNT(*) AS n
FROM billing.usage_events
WHERE start_time >= now() - interval '90 days'
GROUP BY DATE_TRUNC('month', start_time)
ORDER BY month

-- ---
-- expect: {"id": "pg-18", "tables": ["usage_events"], "group_by": ["EXTRACT(YEAR FROM start_time)"], "aggregations": ["max"]}
SELECT EXTRACT(YEAR FROM start_time) AS yr, MAX(insertion_time) AS last_load
FROM billing.usage_events
GROUP BY EXTRACT(YEAR FROM start_time)

-- ---
-- expect: {"id": "pg-19", "tables": ["customers"], "filters": ["status"], "group_by": ["status"], "aggregations": ["count"]}
SELECT status, COUNT(*) AS n
FROM crm.customers
WHERE status = 'A'
GROUP BY status
UNION
SELECT status, COUNT(*) AS n
FROM crm.customers
WHERE status = 'S'
GROUP BY status

-- ---
-- expect: {"id": "pg-20", "tables": ["accounts_hist"], "filters": ["updated_at"], "time_columns": ["updated_at"], "group_by": ["client_no"], "aggregations": ["count"], "now_functions": ["now"]}
SELECT client_no, COUNT(*) AS changes
FROM billing.accounts_hist
WHERE updated_at >= now() - interval '1 day'
GROUP BY client_no

-- ---
-- expect: {"id": "pg-21", "tables": ["usage_by_load"], "filters": ["insertion_time", "load_id"], "time_columns": ["insertion_time"], "group_by": ["load_id"], "aggregations": ["sum"], "now_functions": ["now"]}
SELECT load_id, SUM(bytes) AS bytes
FROM billing.usage_by_load
WHERE load_id = 42
  AND insertion_time >= now() - interval '2 hours'
GROUP BY load_id

-- ---
-- expect: {"id": "pg-22", "tables": ["usage_events"], "filters": ["start_time"], "time_columns": ["start_time"], "aggregations": ["count"], "now_functions": ["now"]}
SELECT COUNT(*) AS n
FROM billing.usage_events
WHERE start_time BETWEEN now() - interval '30 days' AND now()

-- ---
-- expect: {"id": "pg-23", "tables": ["usage_events"], "filters": ["insertion_time", "start_time"], "time_columns": ["insertion_time", "start_time"], "now_functions": ["current_date", "current_timestamp"]}
SELECT cust_id
FROM billing.usage_events
WHERE start_time >= CURRENT_DATE - interval '7 days'
  AND insertion_time < CURRENT_TIMESTAMP

-- ---
-- expect: {"id": "pg-24", "tables": ["usage_events"], "filters": ["start_time"], "time_columns": ["start_time"], "aggregations": ["count"], "now_functions": ["localtimestamp"]}
SELECT COUNT(*) AS n
FROM billing.usage_events
WHERE start_time >= LOCALTIMESTAMP - interval '1 day'

-- ---
-- expect: {"id": "pg-25", "tables": ["customers"], "rounding": ["CAST(credit_limit AS NUMERIC(12, 2)) / 1000"]}
SELECT cust_id, CAST(credit_limit AS NUMERIC(12, 2)) / 1000 AS credit_k
FROM crm.customers

-- ---
-- expect: {"id": "pg-26", "tables": ["customers"], "filters": ["status"], "rounding": ["credit_limit::numeric / 1000"]}
SELECT cust_id, credit_limit::numeric / 1000 AS credit_k
FROM crm.customers
WHERE status = 'A'

-- ---
-- expect: {"id": "pg-27", "tables": ["customers"], "filters": ["name"]}
SELECT cust_id, name
FROM crm.customers
WHERE name = 'O''Brien'

-- ---
-- expect: {"id": "pg-28", "tables": ["customers"], "filters": ["name"]}
SELECT cust_id, name
FROM crm.customers
WHERE name ILIKE '%בנק%'

-- ---
-- expect: {"id": "pg-29", "tables": ["customers", "usage_events"], "filters": ["cust_id", "start_time"], "time_columns": ["start_time"], "now_functions": ["now"]}
SELECT c.cust_id, c.status
FROM crm.customers AS c
WHERE c.cust_id NOT IN (
  SELECT u.cust_id
  FROM billing.usage_events AS u
  WHERE u.start_time >= now() - interval '30 days'
)

-- ---
-- expect: {"id": "pg-30", "tables": ["accounts", "customers"], "joins": [{"tables": ["customers", "accounts"], "columns": ["cust_id", "client_no"], "conversion": "lpad", "conversion_expr": "LPAD(c.cust_id::varchar, 10, '0')"}], "filters": ["status"]}
SELECT c.cust_id, c.status
FROM crm.customers AS c
WHERE EXISTS (
  SELECT 1
  FROM billing.accounts AS e
  WHERE e.client_no = LPAD(c.cust_id::varchar, 10, '0')
    AND e.status = 'Active'
)

-- ---
-- expect: {"id": "pg-31", "tables": ["accounts", "customers"], "joins": [{"tables": ["customers", "accounts"], "columns": ["cust_id", "client_no"], "conversion": "lpad", "conversion_expr": "LPAD(c.cust_id::varchar, 10, '0')"}]}
SELECT c.cust_id, COALESCE(e.credit_limit, 0) AS credit_limit
FROM crm.customers AS c
LEFT JOIN billing.accounts AS e
  ON e.client_no = LPAD(c.cust_id::varchar, 10, '0')

-- ---
-- expect: {"id": "pg-32", "tables": ["customers"], "group_by": ["status"], "aggregations": ["count"]}
SELECT status, COUNT(*) AS n
FROM crm.customers
GROUP BY status
HAVING COUNT(*) > 100

-- ---
-- expect: {"id": "pg-33", "tables": ["customers"], "filters": ["status"], "group_by": ["status"], "aggregations": ["sum"]}
SELECT status, SUM(credit_limit) AS total
FROM crm.customers
WHERE status <> 'C'
GROUP BY status
HAVING SUM(credit_limit) > 100000

-- ---
-- expect: {"id": "pg-34", "tables": ["customers"]}
SELECT cust_id, updated_at
FROM crm.customers
ORDER BY updated_at DESC
LIMIT 100 OFFSET 200

-- ---
-- expect: {"id": "pg-35", "tables": ["customers"], "filters": ["status"], "rounding": ["ROUND(credit_limit / 5000.0) * 5000"]}
SELECT cust_id, ROUND(credit_limit / 5000.0) * 5000 AS canonical
FROM crm.customers
WHERE status = 'A'

-- ---
-- expect: {"id": "pg-36", "tables": ["customers"], "rounding": ["FLOOR(credit_limit / 1000)"]}
SELECT cust_id, FLOOR(credit_limit / 1000) AS credit_k
FROM crm.customers

-- ---
-- expect: {"id": "pg-37", "tables": ["usage_events"], "filters": ["start_time"], "time_columns": ["start_time"], "aggregations": ["count"]}
SELECT COUNT(*) AS n
FROM billing.usage_events
WHERE start_time >= DATE '2026-01-01'
  AND start_time < DATE '2026-02-01'

-- ---
-- expect: {"id": "pg-38", "tables": ["usage_events"], "filters": ["insertion_time", "start_time"], "time_columns": ["insertion_time", "start_time"]}
SELECT cust_id, bytes
FROM billing.usage_events
WHERE insertion_time >= TIMESTAMP '2026-11-25 14:00:00'
  AND start_time >= TIMESTAMP '2026-11-25 14:00:00'

-- ---
-- expect: {"id": "pg-39", "tables": ["customers"], "filters": ["status"], "aggregations": ["count", "sum"]}
SELECT SUM(credit_limit) / COUNT(*) AS avg_credit_limit
FROM crm.customers
WHERE status = 'A'

-- ---
-- expect: {"id": "pg-40", "tables": ["customers"], "filters": ["cust_no"], "rounding": ["limit_k * 10"]}
SELECT cust_no, limit_k * 10 AS credit_limit
FROM erp.customers
WHERE cust_no > 0

-- ---
-- expect: {"id": "pg-41", "tables": ["accounts", "customers", "usage_events"], "joins": [{"tables": ["customers", "accounts"], "columns": ["cust_id", "client_no"], "conversion": "lpad", "conversion_expr": "LPAD(c.cust_id::varchar, 10, '0')"}, {"tables": ["accounts", "usage_events"], "columns": ["client_no", "client_no"]}], "filters": ["insertion_time", "start_time"], "time_columns": ["insertion_time", "start_time"], "group_by": ["c.status"], "aggregations": ["avg"], "rounding": ["ROUND(AVG(CAST(u.bytes AS NUMERIC)) / 5000) * 5000"], "now_functions": ["now"]}
SELECT c.status, ROUND(AVG(CAST(u.bytes AS NUMERIC)) / 5000) * 5000 AS bucket
FROM crm.customers AS c
JOIN billing.accounts AS e
  ON LPAD(c.cust_id::varchar, 10, '0') = e.client_no
JOIN billing.usage_events AS u
  ON u.client_no = e.client_no
WHERE u.start_time >= now() - interval '30 days'
  AND u.insertion_time >= now() - interval '36 hours'
GROUP BY c.status

-- ---
-- expect: {"id": "pg-42", "tables": ["accounts", "customers"], "joins": [{"tables": ["customers", "accounts"], "columns": ["cust_id", "client_no"], "conversion": "cast", "conversion_expr": "CAST(c.cust_id AS VARCHAR(10))"}], "filters": ["status"], "group_by": ["e.status"], "aggregations": ["count"]}
SELECT e.status, COUNT(*) AS n
FROM crm.customers AS c
JOIN billing.accounts AS e
  ON CAST(c.cust_id AS VARCHAR(10)) = e.client_no
WHERE c.status = 'A'
GROUP BY e.status
