#!/bin/sh
# OpenSearch loader for the `oss` compose profile. Run by the one-shot `opensearch-init` service
# (profile `oss`) once `opensearch` reports healthy; it also runs by hand:
#
#   docker compose --profile oss run --rm opensearch-init
#
# It PUTs the generated index template (which carries the alias), bulk-loads the generated NDJSON
# and then prints what exists, so the container log is the proof that the seed landed. Re-running
# is safe: the template PUT is idempotent and each document's id is its event id, so a reload
# overwrites rather than duplicating.
#
# No credential: the `oss` profile runs OpenSearch with its security plugin disabled and bound to
# 127.0.0.1 (dev only, see docs/dev-env.md).
set -eu

: "${MDQM_OPENSEARCH_URL:=http://opensearch:9200}"

echo "opensearch-init: waiting for ${MDQM_OPENSEARCH_URL}"
attempt=1
while [ "$attempt" -le 60 ]; do
    if curl -fsS "${MDQM_OPENSEARCH_URL}/_cluster/health?wait_for_status=yellow&timeout=5s" >/dev/null; then
        break
    fi
    attempt=$((attempt + 1))
    sleep 2
done
if [ "$attempt" -gt 60 ]; then
    echo "opensearch-init: ${MDQM_OPENSEARCH_URL} never became ready" >&2
    exit 1
fi

echo "opensearch-init: putting the index template"
curl -fsS -X PUT "${MDQM_OPENSEARCH_URL}/_index_template/mdqm-usage-events" \
    -H 'Content-Type: application/json' \
    --data-binary @/seeds/index-template.json >/dev/null

echo "opensearch-init: bulk-loading usage events"
curl -fsS -X POST "${MDQM_OPENSEARCH_URL}/_bulk?refresh=true" \
    -H 'Content-Type: application/x-ndjson' \
    --data-binary @/seeds/usage-events.ndjson >/dev/null

echo "opensearch-init: indices now present"
curl -fsS "${MDQM_OPENSEARCH_URL}/_cat/indices/mdqm-usage-events-*?h=index,docs.count,health&s=index"
echo "opensearch-init: alias mdqm-usage-events covers"
curl -fsS "${MDQM_OPENSEARCH_URL}/_cat/aliases/mdqm-usage-events?h=alias,index&s=index"
