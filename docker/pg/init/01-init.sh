#!/bin/sh
# PostgreSQL seed for the `pg` compose profile (task D01). Runs once, on an empty data directory.
#
#   * creates the other two databases (crm already exists as POSTGRES_DB): billing and erp;
#   * creates the shared read-only role `mdqm_ro` and makes it read-only by default;
#   * enables pg_stat_statements in all three databases (D09 mines query stats from there);
#   * grants SELECT on everything, now and for tables created later.
#
# The read-only password comes from the container environment; no credential is written here.
set -eu

: "${MDQM_RO_PASSWORD:?MDQM_RO_PASSWORD is required; see .env.example}"

psql --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" \
    --set ON_ERROR_STOP=1 --set ro_password="$MDQM_RO_PASSWORD" <<'SQL'
-- Databases: CREATE DATABASE cannot run inside a transaction, so generate and execute it.
SELECT format('CREATE DATABASE %I OWNER postgres', db)
FROM (VALUES ('billing'), ('erp')) AS wanted (db)
WHERE NOT EXISTS (SELECT 1 FROM pg_database WHERE datname = wanted.db)
\gexec

-- The read-only account. The three layers of read-only are: this account's default
-- transaction mode, the AST guard, and the adapter's read-only transaction.
SELECT format('CREATE ROLE mdqm_ro LOGIN PASSWORD %L', :'ro_password')
WHERE NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'mdqm_ro')
\gexec
ALTER ROLE mdqm_ro WITH LOGIN PASSWORD :'ro_password';
ALTER ROLE mdqm_ro SET default_transaction_read_only = on;
-- Query-store mining (A04) needs the statement text of *every* role: pg_stat_statements masks
-- another role's text as `<insufficient privilege>` unless the reader holds this predefined role.
-- Read-only: it grants visibility of statistics, never data and never a write.
GRANT pg_read_all_stats TO mdqm_ro;
SQL

for db in crm billing erp; do
    psql --username "$POSTGRES_USER" --dbname "$db" --set ON_ERROR_STOP=1 <<'SQL'
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

-- CONNECT stays with PUBLIC (the default); what the role may read is the point.
GRANT USAGE ON SCHEMA public TO mdqm_ro;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO mdqm_ro;
GRANT SELECT ON ALL SEQUENCES IN SCHEMA public TO mdqm_ro;
ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA public GRANT SELECT ON TABLES TO mdqm_ro;
SQL
done

echo "mdqm dev seed done: crm, billing, erp + read-only role mdqm_ro"
