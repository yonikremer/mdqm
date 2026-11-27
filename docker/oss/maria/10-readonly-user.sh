#!/bin/sh
# MariaDB read-only account for the `oss` compose profile.
#
# Runs from /docker-entrypoint-initdb.d after 01-schema.sql and 02-data.sql, on the first start of
# an empty data volume. MariaDB's entrypoint either executes this file (executable bit set) or
# sources it, so it must not `exit` and must not depend on bash.
#
# The password comes from the container environment (MDQM_RO_PASSWORD, the same variable the
# Postgres and Oracle services use) and is never written into the generated SQL.
#
# Read-only, like `mdqm_ro` in Postgres: SELECT on the whole database, nothing else. Writes have to
# fail here too, otherwise a test that "passes" against MariaDB proves nothing.
set -eu

: "${MDQM_RO_PASSWORD:?MDQM_RO_PASSWORD is required; see .env.example}"
: "${MARIADB_DATABASE:?MARIADB_DATABASE is set by the mariadb service}"

echo "oss seed: creating the read-only account mdqm_ro on ${MARIADB_DATABASE}"

mariadb --protocol=socket --user=root --password="$MARIADB_ROOT_PASSWORD" <<SQL
CREATE USER IF NOT EXISTS 'mdqm_ro'@'%' IDENTIFIED BY '${MDQM_RO_PASSWORD}';
ALTER USER 'mdqm_ro'@'%' IDENTIFIED BY '${MDQM_RO_PASSWORD}';
REVOKE ALL PRIVILEGES, GRANT OPTION FROM 'mdqm_ro'@'%';
GRANT SELECT ON \`${MARIADB_DATABASE}\`.* TO 'mdqm_ro'@'%';
FLUSH PRIVILEGES;
SQL

echo "oss seed done: ${MARIADB_DATABASE} + read-only account mdqm_ro"
