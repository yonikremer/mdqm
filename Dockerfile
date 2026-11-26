# mdqm server image (task M08). Deployment notes: docs/deployment.md, DESIGN section 14.
#
# Air-gapped first: the builder installs from a wheelhouse staged in the build context, so the
# build needs no package index and nothing is downloaded at runtime. Produce the wheelhouse with
#   python scripts/check_offline_install.py --wheelhouse wheelhouse
# then
#   docker build --tag mdqm:local .
# Pinned base images (no `latest`): a Python minor and a Debian codename, never a moving tag.
# Pin the digest too once the image is staged: docker build --build-arg is deliberately unused for
# the base image so a reviewer sees the exact tag in this file.

FROM python:3.12-slim-bookworm AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_NO_INPUT=1

WORKDIR /build

# Only what the wheel needs: README.md is pyproject's readme, src/ is the package. Tests, docs and
# the dev extras never enter the image (see .dockerignore).
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Layer cache: the wheelhouse changes rarely, resolve against it without a network at all.
COPY wheelhouse/ /wheelhouse/

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# `pg` is the only extra that ships today. `oracle`/`mssql` stay out until their driver verdicts
# (task A08) land, so an image never carries a driver nobody chose.
# `--only-binary=:all:`: every dependency must be a wheel. A source tarball needs a compiler and
# an sdist means the internal mirror is incomplete.
RUN pip install --no-index --find-links /wheelhouse --only-binary=:all: ".[pg]"

# No installer in the runtime image: nothing may `pip install` once the image exists (principle 7).
RUN pip uninstall --yes pip setuptools wheel


FROM python:3.12-slim-bookworm AS runtime

# PATH first so `python` and `mdqm` are the venv's. Every other value is a location or a bind
# default from mdqm.config.settings -- no organisation fact is hard-coded here.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    MDQM_SERVER__HOST=0.0.0.0 \
    MDQM_SERVER__PORT=8765 \
    MDQM_PATHS__METADATA_DB=/data/mdqm.sqlite \
    MDQM_PATHS__SEMANTIC_DIR=/semantic \
    MDQM_PATHS__CACHE_DIR=/cache \
    MDQM_SECRETS__DIR=/run/secrets \
    MDQM_TOKENS_FILE=/run/secrets/tokens.json

# Non-root, no home, no shell. The uid/gid are fixed so a mounted volume can be chowned to them.
RUN groupadd --gid 10001 mdqm \
    && useradd --uid 10001 --gid 10001 --no-create-home --home-dir /nonexistent \
        --shell /usr/sbin/nologin mdqm \
    && mkdir -p /data /cache /semantic \
    && chown 10001:10001 /data /cache /semantic

# The venv only: no compiler, no build tools, no wheels, no pip in the runtime layer.
COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
USER 10001:10001

# Metadata DB and DuckDB spill/cache are the two writable trees. Mount /data on a volume: it holds
# the SQLite metadata DB and the audit chain, the only state that must be backed up.
VOLUME ["/data", "/cache"]

EXPOSE 8765

# /healthz is public on purpose (src/mdqm/server/app.py), so the probe needs no token. Python
# answers it, so the image needs no curl/wget.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os, urllib.request; port = os.environ.get('MDQM_SERVER__PORT', '8765'); urllib.request.urlopen('http://127.0.0.1:%s/healthz' % port, timeout=4).read()"

# Host and port come from the environment above (mdqm serve falls back to Settings), so the same
# image runs on a laptop and in the cluster.
ENTRYPOINT ["mdqm", "serve"]
