# mdqm server image (task M08). Deployment notes: docs/deployment.md, DESIGN section 14.
#
# Two supported ways to get packages into the builder. They are one build arg apart, because the
# target network is not uniformly disconnected: it has internal PyPI mirrors, and a fully offline
# site does not.
#
#   mirror (default)  install from the configured index -- PyPI, or the internal mirror named by
#                     PIP_INDEX_URL / a pip.conf baked into the base image. Nothing extra to stage.
#       docker build --build-arg PIP_INDEX_URL=https://pypi.internal/simple --tag mdqm:local .
#
#   wheelhouse        no index at all: install only from ./wheelhouse, for a disconnected builder.
#       python scripts/check_offline_install.py --wheelhouse wheelhouse
#       docker build --build-arg INSTALL_MODE=wheelhouse --tag mdqm:local .
#
# Whichever mode is used, nothing may be downloaded at runtime, and every dependency must arrive as
# a wheel: the target has no compiler. `check_offline_install.py` proves the wheelhouse closure
# (build backend included) and the recipe; it cannot prove anything about a mirror's contents.
#
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

# Always copied, even in mirror mode: an absent directory fails `COPY`, and `wheelhouse/` holds only
# a placeholder until a disconnected site stages real wheels into it (`.dockerignore` keeps it in
# the context on purpose).
COPY wheelhouse/ /wheelhouse/

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# `mirror` uses the index pip is configured with; leave PIP_INDEX_URL empty on a connected build.
# `PIP_TRUSTED_HOST` is for a mirror served over plain HTTP (an internal, air-gapped address).
# Which adapter drivers the image carries. `pg` is the default (the MVP target and the one extra
# every deployment needs); a site that reads Oracle or SQL Server builds with
# `--build-arg MDQM_EXTRAS=pg,oracle` so an image never carries a driver nobody chose. The names are
# pyproject's `[project.optional-dependencies]` extras; the SQL Server driver is `mssql-python`
# (task A08). A driver that is absent makes its adapter unavailable at runtime, not the server dead.
# `--only-binary=:all:`: every dependency must be a wheel. A source tarball needs a compiler, and
# an sdist means the internal mirror is incomplete.
# A local directory (`.[pg]`) is built from source, so pip installs the build backend from
# `build-system.requires` into an isolated environment: `hatchling` is a real dependency of this
# build, and the wheelhouse mode has to stage it like any other wheel.
ARG INSTALL_MODE=mirror
ARG MDQM_EXTRAS=pg
ARG PIP_INDEX_URL=""
ARG PIP_TRUSTED_HOST=""
RUN set -eu; \
    if [ "$INSTALL_MODE" = "wheelhouse" ]; then \
        pip install --no-index --find-links /wheelhouse --only-binary=:all: ".[$MDQM_EXTRAS]"; \
    else \
        set -- --only-binary=:all:; \
        if [ -n "$PIP_INDEX_URL" ]; then set -- "$@" --index-url "$PIP_INDEX_URL"; fi; \
        if [ -n "$PIP_TRUSTED_HOST" ]; then set -- "$@" --trusted-host "$PIP_TRUSTED_HOST"; fi; \
        pip install "$@" ".[$MDQM_EXTRAS]"; \
    fi

# No installer in the runtime image: nothing may `pip install` once the image exists (principle 7).
RUN pip uninstall --yes pip setuptools wheel


FROM python:3.12-slim-bookworm AS runtime

# PATH first so `python` and `mdqm` are the venv's. Every other value is a location or a bind
# default from mdqm.config.settings -- no organisation fact is hard-coded here.
# MDQM_SERVER__PUBLIC_URL is NOT set here on purpose: it depends on the deployment's hostname, so
# the platform supplies it (an unset value advertises http://<host>:<port>, which behind an edge
# TLS proxy is the wrong scheme and, on 0.0.0.0, an unroutable host).
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

# Non-root, no home, no shell, fixed uid/gid so a plain Docker volume has a known owner.
#
# The writable trees are group-owned by GID 0 and group-writable via `chmod g=u`, which is what
# makes the same image run under an arbitrary UID:
#   * OpenShift's `restricted-v2` SCC admits a container only if it does not demand a specific
#     uid (`MustRunAsRange`), then runs it as a random namespace UID that always sits in GID 0;
#   * the same SCC mounts Secrets as `root:<namespace-gid> 0440`, so a fixed uid 10001 could not
#     read /run/secrets/tokens.json, while a process in GID 0 can.
# So the group is the access path here, not the user. Never `chown` these trees to one uid.
RUN groupadd --gid 10001 mdqm \
    && useradd --uid 10001 --gid 10001 --no-create-home --home-dir /nonexistent \
        --shell /usr/sbin/nologin mdqm \
    && mkdir -p /data /cache /semantic \
    && chgrp -R 0 /data /cache /semantic \
    && chmod -R g=u /data /cache /semantic

# mssql-python's bundled libodbcinst.so.2 is linked against libltdl.so.7, which the wheel does
# not ship. Without it every SQL Server connection fails with "Failed to load the driver" on
# Debian/Ubuntu, including this image when it is built with the mssql extra. Installed here, at
# build time, so the runtime stage still has no pip and no package index.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libltdl7 \
    && rm -rf /var/lib/apt/lists/*

# The venv only: no compiler, no build tools, no wheels, no pip in the runtime layer.
COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
# Docker and Podman run as this uid; an OpenShift namespace overrides it with its own range, and
# the GID 0 ownership above is what keeps the writable trees usable when it does.
USER 10001:10001

# Metadata DB and DuckDB spill/cache are the two writable trees. Mount /data on a volume: it holds
# the SQLite metadata DB and the audit chain, the only state that must be backed up. DuckDB spills
# into /cache/spill (mdqm.cache / mdqm.executor set `temp_directory`), never into the working
# directory, so a read-only root filesystem is enough.
VOLUME ["/data", "/cache"]

EXPOSE 8765

# /healthz is public on purpose (src/mdqm/server/app.py), so the probe needs no token. Python
# answers it, so the image needs no curl/wget.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os, urllib.request; port = os.environ.get('MDQM_SERVER__PORT', '8765'); urllib.request.urlopen('http://127.0.0.1:%s/healthz' % port, timeout=4).read()"

# Host and port come from the environment above (mdqm serve falls back to Settings), so the same
# image runs on a laptop and in the cluster.
ENTRYPOINT ["mdqm", "serve"]
