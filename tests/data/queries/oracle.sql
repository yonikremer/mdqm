-- Golden SQL corpus: Oracle dialect (Thin mode, 12c+, no 23ai-only syntax).
--
-- Format and expect-block rules are documented in tests/unit/test_query_corpus.py.
-- Every statement is a single read-only SELECT.
-- Oracle specifics covered here: SYSDATE/SYSTIMESTAMP, ROWNUM, FETCH FIRST, DUAL,
-- TO_CHAR/TO_NUMBER/LPAD conversion joins, NVL, TRUNC, ADD_MONTHS, (+) outer join and hints.

-- expect: {"id": "or-01", "tables": ["customers"], "filters": ["status"]}
SELECT cust_id, status, credit_limit
FROM crm.customers
WHERE status = 'A'
  AND ROWNUM <= 100

-- ---
-- expect: {"id": "or-02", "tables": ["customers"], "group_by": ["status"], "aggregations": ["avg"], "rounding": ["ROUND(AVG(credit_limit) / 5000) * 5000"]}
SELECT status, ROUND(AVG(credit_limit) / 5000) * 5000 AS rounded_avg
FROM crm.customers
GROUP BY status

-- ---
-- expect: {"id": "or-03", "tables": ["usage_events"], "filters": ["start_time"], "time_columns": ["start_time"], "now_functions": ["sysdate"]}
SELECT client_no, bytes
FROM billing.usage_events
WHERE start_time >= SYSDATE - 7

-- ---
-- expect: {"id": "or-04", "tables": ["dual"], "now_functions": ["sysdate"]}
SELECT SYSDATE AS now_ts FROM dual

-- ---
-- expect: {"id": "or-05", "tables": ["accounts", "customers"], "joins": [{"tables": ["customers", "accounts"], "columns": ["cust_id", "client_no"], "conversion": "to_char", "conversion_expr": "TO_CHAR(c.cust_id, 'FM0000000000')"}]}
SELECT c.cust_id, e.client_no
FROM crm.customers c
JOIN billing.accounts e
  ON TO_CHAR(c.cust_id, 'FM0000000000') = e.client_no

-- ---
-- expect: {"id": "or-06", "tables": ["accounts", "customers"], "joins": [{"tables": ["customers", "accounts"], "columns": ["cust_id", "client_no"], "conversion": "lpad", "conversion_expr": "LPAD(CAST(c.cust_id AS VARCHAR(10)), 10, '0')"}]}
SELECT c.cust_id, e.client_no, e.status
FROM crm.customers c
JOIN billing.accounts e
  ON LPAD(CAST(c.cust_id AS VARCHAR(10)), 10, '0') = e.client_no

-- ---
-- expect: {"id": "or-07", "tables": ["usage_events"], "filters": ["start_time"], "time_columns": ["start_time"]}
SELECT client_no, start_time, bytes
FROM billing.usage_events
WHERE start_time >= TIMESTAMP '2026-11-01 00:00:00'
ORDER BY start_time DESC
FETCH FIRST 50 ROWS ONLY

-- ---
-- expect: {"id": "or-08", "tables": ["usage_events"], "filters": ["start_time"], "time_columns": ["start_time"], "group_by": ["TRUNC(start_time, 'MM')"], "aggregations": ["count"], "now_functions": ["sysdate"]}
SELECT TRUNC(start_time, 'MM') AS month, COUNT(*) AS n
FROM billing.usage_events
WHERE start_time >= TRUNC(SYSDATE, 'MM') - 90
GROUP BY TRUNC(start_time, 'MM')

-- ---
-- expect: {"id": "or-09", "tables": ["accounts"], "filters": ["status"]}
SELECT client_no, NVL(credit_limit, 0) AS credit_limit
FROM billing.accounts
WHERE status = 'Active'

-- ---
-- expect: {"id": "or-10", "tables": ["usage_events"], "filters": ["start_time"], "time_columns": ["start_time"], "aggregations": ["count"], "now_functions": ["sysdate"]}
SELECT COUNT(*) AS n
FROM billing.usage_events
WHERE start_time >= ADD_MONTHS(SYSDATE, -3)

-- ---
-- expect: {"id": "or-11", "tables": ["usage_events"], "filters": ["insertion_time", "start_time"], "time_columns": ["insertion_time", "start_time"], "now_functions": ["sysdate"]}
SELECT client_no, bytes
FROM billing.usage_events
WHERE insertion_time >= SYSDATE - INTERVAL '6' HOUR
  AND start_time >= SYSDATE - INTERVAL '12' HOUR

-- ---
-- expect: {"id": "or-12", "tables": ["usage_events"], "filters": ["client_no", "start_time"], "time_columns": ["start_time"], "group_by": ["client_no"], "aggregations": ["sum"]}
SELECT /*+ INDEX(ue IDX_UE_START_TIME) */ ue.client_no, SUM(ue.bytes) AS total
FROM billing.usage_events ue
WHERE ue.client_no = '0000000042'
  AND ue.start_time >= DATE '2026-11-01'
GROUP BY ue.client_no

-- ---
-- expect: {"id": "or-13", "tables": ["customers"], "filters": ["status"], "group_by": ["status"], "aggregations": ["count"]}
SELECT "C"."STATUS", COUNT(*) AS n
FROM "CRM"."CUSTOMERS" "C"
WHERE "C"."STATUS" = 'A'
GROUP BY "C"."STATUS"

-- ---
-- expect: {"id": "or-14", "tables": ["customers"], "filters": ["name", "status"]}
SELECT cust_id, name
FROM crm.customers
WHERE name LIKE '%בנק%'
  AND status = 'פעיל'

-- ---
-- expect: {"id": "or-15", "tables": ["usage_events", "customers"], "joins": [{"tables": ["recent", "customers"], "columns": ["cust_id", "cust_id"]}], "filters": ["start_time"], "time_columns": ["start_time"], "group_by": ["cust_id"], "aggregations": ["count", "sum"], "now_functions": ["sysdate"]}
WITH recent AS (
  SELECT cust_id, SUM(bytes) AS bytes
  FROM billing.usage_events
  WHERE start_time >= SYSDATE - 30
  GROUP BY cust_id
)
SELECT COUNT(*) AS n
FROM recent r
JOIN crm.customers c
  ON c.cust_id = r.cust_id

-- ---
-- expect: {"id": "or-16", "tables": ["customers"], "filters": ["n"], "group_by": ["status"], "aggregations": ["count"]}
SELECT t.status, t.n
FROM (
  SELECT status, COUNT(*) AS n
  FROM crm.customers
  GROUP BY status
) t
WHERE t.n > 10

-- ---
-- expect: {"id": "or-17", "tables": ["accounts", "customers"], "joins": [{"tables": ["customers", "accounts"], "columns": ["cust_id", "client_no"], "conversion": "lpad", "conversion_expr": "LPAD(CAST(c.cust_id AS VARCHAR(10)), 10, '0')"}]}
SELECT c.cust_id, e.client_no
FROM crm.customers c, billing.accounts e
WHERE e.client_no = LPAD(CAST(c.cust_id AS VARCHAR(10)), 10, '0')

-- ---
-- expect: {"id": "or-18", "tables": ["accounts"], "filters": ["status"], "group_by": ["status"], "aggregations": ["count"]}
SELECT status,
       CASE WHEN status = 'Active' THEN 'פעיל' ELSE 'לא פעיל' END AS label,
       COUNT(*) AS n
FROM billing.accounts
GROUP BY status

-- ---
-- expect: {"id": "or-19", "tables": ["customers"], "filters": ["status"], "rounding": ["ROUND(credit_limit / 5000) * 5000"]}
SELECT cust_id, ROUND(credit_limit / 5000) * 5000 AS canonical
FROM crm.customers
WHERE status = 'A'

-- ---
-- expect: {"id": "or-20", "tables": ["customers"], "filters": ["cust_no", "limit_k"], "rounding": ["limit_k * 10"]}
SELECT cust_no, limit_k * 10 AS credit_limit
FROM erp.customers
WHERE cust_no > 0
  AND limit_k IS NOT NULL

-- ---
-- expect: {"id": "or-21", "tables": ["customers"], "filters": ["status"], "group_by": ["status"], "aggregations": ["count"]}
SELECT status, RATIO_TO_REPORT(COUNT(*)) OVER () AS share
FROM crm.customers
WHERE status <> 'C'
GROUP BY status

-- ---
-- expect: {"id": "or-22", "tables": ["accounts", "customers"], "joins": [{"tables": ["customers", "accounts"], "columns": ["cust_id", "client_no"], "conversion": "to_number", "conversion_expr": "TO_NUMBER(e.client_no)"}]}
SELECT c.cust_id, e.client_no
FROM crm.customers c
JOIN billing.accounts e
  ON c.cust_id = TO_NUMBER(e.client_no)

-- ---
-- expect: {"id": "or-23", "tables": ["usage_events"], "filters": ["insertion_time"], "time_columns": ["insertion_time"], "aggregations": ["max"], "now_functions": ["systimestamp"]}
SELECT MAX(insertion_time) AS last_load
FROM billing.usage_events
WHERE insertion_time >= SYSTIMESTAMP - INTERVAL '1' DAY

-- ---
-- expect: {"id": "or-24", "tables": ["usage_events"], "filters": ["insertion_time", "start_time"], "time_columns": ["insertion_time", "start_time"], "now_functions": ["current_date", "current_timestamp"]}
SELECT client_no
FROM billing.usage_events
WHERE start_time >= CURRENT_DATE - 7
  AND insertion_time < CURRENT_TIMESTAMP

-- ---
-- expect: {"id": "or-25", "tables": ["usage_events"], "group_by": ["EXTRACT(MONTH FROM start_time)"], "aggregations": ["count"]}
SELECT EXTRACT(MONTH FROM start_time) AS m, COUNT(*) AS n
FROM billing.usage_events
GROUP BY EXTRACT(MONTH FROM start_time)

-- ---
-- expect: {"id": "or-26", "tables": ["accounts"], "filters": ["status"], "group_by": ["status"], "aggregations": ["sum"]}
SELECT status, SUM(credit_limit) AS total
FROM billing.accounts
WHERE status IN ('Active', 'Hold')
GROUP BY status
HAVING SUM(credit_limit) > 100000

-- ---
-- expect: {"id": "or-27", "tables": ["customers"], "filters": ["status"], "aggregations": ["count"]}
SELECT 'active' AS grp, COUNT(*) AS n FROM crm.customers WHERE status = 'A'
UNION ALL
SELECT 'closed' AS grp, COUNT(*) AS n FROM crm.customers WHERE status = 'C'

-- ---
-- expect: {"id": "or-28", "tables": ["usage_by_load"], "filters": ["insertion_time", "load_id"], "time_columns": ["insertion_time"], "group_by": ["load_id"], "aggregations": ["sum"], "now_functions": ["sysdate"]}
SELECT load_id, SUM(bytes) AS total_bytes
FROM billing.usage_by_load
WHERE load_id = 7
  AND insertion_time >= SYSDATE - 1
GROUP BY load_id

-- ---
-- expect: {"id": "or-29", "tables": ["accounts"], "aggregations": ["count"]}
SELECT COUNT(DISTINCT client_no) AS distinct_clients
FROM billing.accounts

-- ---
-- expect: {"id": "or-30", "tables": ["customers"], "aggregations": ["max", "min"]}
SELECT MIN(credit_limit) AS min_credit, MAX(credit_limit) AS max_credit
FROM crm.customers

-- ---
-- expect: {"id": "or-31", "tables": ["usage_events"], "filters": ["start_time"], "time_columns": ["start_time"], "aggregations": ["count"], "now_functions": ["sysdate"]}
SELECT COUNT(*) AS n
FROM billing.usage_events
WHERE start_time >= TRUNC(SYSDATE, 'MM')

-- ---
-- expect: {"id": "or-32", "tables": ["customers"], "filters": ["status"]}
SELECT cust_id, credit_limit
FROM crm.customers
WHERE status IN ('A', 'S')

-- ---
-- expect: {"id": "or-33", "tables": ["customers"], "filters": ["name"]}
SELECT cust_id, name
FROM crm.customers
WHERE name LIKE 'בנק%'

-- ---
-- expect: {"id": "or-34", "tables": ["customers"], "filters": ["credit_limit"]}
SELECT cust_id,
       CASE WHEN credit_limit >= 100000 THEN 'large'
            WHEN credit_limit >= 10000 THEN 'medium'
            ELSE 'small' END AS band
FROM crm.customers
WHERE credit_limit IS NOT NULL

-- ---
-- expect: {"id": "or-35", "tables": ["accounts_hist"], "filters": ["updated_at"], "time_columns": ["updated_at"], "now_functions": ["sysdate"]}
SELECT client_no, updated_at,
       ROW_NUMBER() OVER (PARTITION BY client_no ORDER BY updated_at DESC) AS rn
FROM billing.accounts_hist
WHERE updated_at >= SYSDATE - 30

-- ---
-- expect: {"id": "or-36", "tables": ["accounts", "customers"], "joins": [{"tables": ["customers", "accounts"], "columns": ["cust_id", "client_no"], "conversion": "lpad", "conversion_expr": "LPAD(CAST(c.cust_id AS VARCHAR(10)), 10, '0')"}], "filters": ["status"], "rounding": ["ROUND(e.credit_limit / 5000) * 5000"]}
SELECT c.cust_id, ROUND(e.credit_limit / 5000) * 5000 AS rounded
FROM crm.customers c
JOIN billing.accounts e
  ON LPAD(CAST(c.cust_id AS VARCHAR(10)), 10, '0') = e.client_no
WHERE c.status = 'A'

-- ---
-- expect: {"id": "or-37", "tables": ["customers"]}
SELECT cust_id, updated_at
FROM crm.customers
ORDER BY updated_at DESC NULLS LAST

-- ---
-- expect: {"id": "or-38", "tables": ["customers"], "filters": ["name"]}
SELECT cust_id, SUBSTR(name, 1, 20) AS short_name
FROM crm.customers
WHERE name IS NOT NULL

-- ---
-- expect: {"id": "or-39", "tables": ["usage_events"], "filters": ["start_time"], "time_columns": ["start_time"], "group_by": ["TO_CHAR(start_time, 'YYYY-MM')"], "aggregations": ["count"], "now_functions": ["sysdate"]}
SELECT TO_CHAR(start_time, 'YYYY-MM') AS month, COUNT(*) AS n
FROM billing.usage_events
WHERE start_time >= ADD_MONTHS(SYSDATE, -12)
GROUP BY TO_CHAR(start_time, 'YYYY-MM')

-- ---
-- expect: {"id": "or-40", "tables": ["accounts", "customers"], "joins": [{"tables": ["customers", "accounts"], "columns": ["cust_id", "client_no"], "conversion": "lpad", "conversion_expr": "LPAD(CAST(c.cust_id AS VARCHAR(10)), 10, '0')"}]}
SELECT c.cust_id, e.client_no
FROM crm.customers c, billing.accounts e
WHERE c.cust_id = 0
  AND LPAD(CAST(c.cust_id AS VARCHAR(10)), 10, '0') = e.client_no (+)

-- ---
-- expect: {"id": "or-41", "tables": ["accounts", "customers", "usage_events"], "joins": [{"tables": ["customers", "accounts"], "columns": ["cust_id", "client_no"], "conversion": "lpad", "conversion_expr": "LPAD(CAST(c.cust_id AS VARCHAR(10)), 10, '0')"}, {"tables": ["accounts", "usage_events"], "columns": ["client_no", "client_no"]}], "filters": ["insertion_time", "start_time"], "time_columns": ["insertion_time", "start_time"], "group_by": ["c.status"], "aggregations": ["avg"], "rounding": ["ROUND(AVG(u.bytes) / 5000) * 5000"], "now_functions": ["sysdate"]}
SELECT c.status, ROUND(AVG(u.bytes) / 5000) * 5000 AS bucket
FROM crm.customers c
JOIN billing.accounts e
  ON LPAD(CAST(c.cust_id AS VARCHAR(10)), 10, '0') = e.client_no
JOIN billing.usage_events u
  ON u.client_no = e.client_no
WHERE u.start_time >= SYSDATE - 30
  AND u.insertion_time >= SYSDATE - INTERVAL '36' HOUR
GROUP BY c.status

-- ---
-- expect: {"id": "or-42", "tables": ["usage_events"], "filters": ["start_time"], "time_columns": ["start_time"], "aggregations": ["count"], "now_functions": ["sysdate"]}
SELECT COUNT(*) AS last_hour
FROM billing.usage_events
WHERE start_time >= SYSDATE - 1 / 24

-- ---
-- expect: {"id": "or-43", "tables": ["customers"], "group_by": ["cust_id", "ROUND(credit_limit / 5000) * 5000"], "aggregations": ["count"], "rounding": ["ROUND(credit_limit / 5000) * 5000"]}
SELECT cust_id, ROUND(credit_limit / 5000) * 5000 AS bucket, COUNT(*) AS n
FROM crm.customers
GROUP BY cust_id, ROUND(credit_limit / 5000) * 5000

-- ---
-- expect: {"id": "or-44", "tables": ["accounts", "customers"], "joins": [{"tables": ["customers", "accounts"], "columns": ["cust_id", "client_no"], "conversion": "cast", "conversion_expr": "CAST(e.client_no AS NUMBER)"}], "filters": ["status"]}
SELECT c.cust_id, e.client_no
FROM crm.customers c
JOIN billing.accounts e
  ON c.cust_id = CAST(e.client_no AS NUMBER)
WHERE c.status = 'A'
