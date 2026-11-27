# Deploying `mdqm`: container image, package sources, cluster readiness

The target network is air-gapped but **not mirror-less**: it runs internal PyPI mirrors that carry
most popular libraries, and adding a package to one is easy. Fully disconnected sites also exist.
So there are two supported ways to get packages into the builder, one build argument apart:

| Mode | When | Command |
|---|---|---|
| **mirror** (default) | The normal case: the builder can reach an internal mirror or PyPI | `docker build --build-arg PIP_INDEX_URL=https://pypi.internal/simple .` |
| **wheelhouse** | The builder has no index at all | stage `wheelhouse/`, then `docker build --build-arg INSTALL_MODE=wheelhouse .` |

Nothing may be downloaded at runtime either way (AGENTS.md principle 7), and every dependency must
arrive as a **wheel**: the target has no compiler (principle 8).

Files: `Dockerfile` (multi-stage, task M08), `.dockerignore`, `wheelhouse/`,
`scripts/check_offline_install.py`.

## 1. What the image is

| Concern | Choice |
|---|---|
| Base | `python:3.12-slim-bookworm` in both stages, pinned by minor + Debian codename, no `latest` |
| Builder stage | creates `/opt/venv` and installs `.[pg]` with `--only-binary=:all:`, from the index or from `/wheelhouse` |
| Runtime stage | copies `/opt/venv` only: no compiler, no build tools, no `pip`, no wheel files |
| User | `10001:10001` (`mdqm`), no home, no login shell: the container never runs as root |
| Writable trees | `/data`, `/cache`, `/semantic`: group-owned by GID `0` and group-writable, **not** `chown`ed to a fixed uid |
| Entrypoint | `mdqm serve`; host and port come from `MDQM_SERVER__HOST` / `MDQM_SERVER__PORT` (`0.0.0.0:8765` in the image) |
| Advertised URL | `MDQM_SERVER__PUBLIC_URL`; see section 4 |
| Healthcheck | `GET /healthz` on the configured port, answered with the image's own `python` (no `curl` needed) |
| Volumes | `/data` (SQLite metadata DB + audit chain: the only state to back up), `/cache` (DuckDB spill + query cache, disposable) |
| Secrets | files under `/run/secrets`; `MDQM_SECRETS__DIR=/run/secrets` and `MDQM_TOKENS_FILE=/run/secrets/tokens.json` |

The `pg` extra is the only extra installed. `oracle` and `mssql` join the image only after their
driver verdicts (task A08); a driver nobody chose must never ride along.

## 2. Mode A: build against an internal mirror (the normal case)

```
docker build \
  --build-arg PIP_INDEX_URL=https://pypi.internal/simple \
  --build-arg PIP_TRUSTED_HOST=pypi.internal \
  --tag mdqm:local .
```

- Leave `PIP_INDEX_URL` empty on a connected machine and pip uses whatever it is configured with
  (PyPI, or a `pip.conf` baked into the base image). `PIP_TRUSTED_HOST` is only for a mirror served
  over **plain HTTP**, which an internal address may well be.
- Both are ordinary pip environment variables, so a mirror that authenticates with a URL-embedded
  credential works — but then the credential is in the build argument, visible in `docker history`
  of a local build. Prefer an unauthenticated in-network mirror or a `pip.conf` in a derived base
  image.

**What this mode needs confirmed by you, not by this repository:** that the mirror actually carries
wheels for these distributions at the pinned versions. There is no container runtime in the
development environment, so no command here can see the mirror's contents:

```
hatchling  pyarrow  duckdb  psycopg  psycopg-binary  psycopg-pool
mcp  sqlglot  pydantic  pydantic-settings  ruamel.yaml  httpx  typer  tzdata
```

`hatchling` is on that list for a reason that is easy to miss: `.[pg]` is a **local directory**, so
pip builds this project and creates an isolated build environment, resolving `build-system.requires`
from the index. A mirror without `hatchling` fails the image build with
`Could not find a version that satisfies the requirement hatchling`.

`scripts/check_offline_install.py` verifies that the *source set* includes the build backend. It
cannot verify a mirror. Section 5 says exactly what it does prove.

## 3. Mode B: build with no index at all (fully disconnected)

One command from **any** OS, Windows included. This is the whole instruction:

```
python scripts/check_offline_install.py --wheelhouse wheelhouse
```

That builds the `mdqm` wheel, stages the whole closure for the **image's** platform
(`manylinux2014_x86_64`, `cp312`), installs `mdqm[pg]` from the wheelhouse into a fresh virtual
environment with `--no-index`, and runs a smoke import plus a real store migration.

**On Windows or macOS it stages and installs inside a container.** Not a convenience: `--platform`
only filters which *wheels* are eligible, while **environment markers are still evaluated for the
interpreter running pip**. On Windows `mcp` requires `pywin32` (`sys_platform == "win32"`), no
manylinux wheel of `pywin32` exists, and plain pip dies with `ResolutionImpossible` no matter how
the platform flags are set. So the script runs the target's own interpreter:

```
docker run --rm --volume <abs>/wheelhouse:/wheelhouse \
           --volume <abs>/wheelhouse/requirements.txt:/requirements.txt:ro \
           python:3.12-slim python -m pip download \
             --only-binary=:all: --platform manylinux2014_x86_64 \
             --implementation cp --python-version 3.12 --abi cp312 \
             --dest /wheelhouse --requirement /requirements.txt
```

- Inside the Linux container the markers resolve for Linux, so the same resolution succeeds.
- The install + smoke steps also run there, because a manylinux wheelhouse holds no `win_amd64`
  binaries — installing it into a Windows venv fails on `duckdb` and would prove nothing.
- Docker is auto-detected. `--no-docker` refuses to use it and fails loudly instead; a machine with
  no Docker and no Linux host cannot stage this wheelhouse at all, and saying so is the honest answer.
- `--docker-image` overrides the staging image (default `python:3.12-slim`). It only ever stages, it
  never ships, so point it at your registry mirror. Pull it beforehand if egress is closed:
  `docker pull <mirror>/python:3.12-slim`, or `docker load` it from a saved tarball.
- A mirror is honoured: `PIP_INDEX_URL`, `PIP_EXTRA_INDEX_URL`, `PIP_TRUSTED_HOST` and `PIP_CERT`
  from the environment are passed into the staging container.

What the check proves, and a green run means all of it: the wheelhouse holds the build backend
(`hatchling`) **plus its closure** (`packaging`, `pathspec`, `pluggy`, `trove-classifiers` — no
unmarked requirement of any staged wheel may be absent), every platform-specific wheel is
`manylinux*`/`linux*` for `x86_64`, no `*.tar.gz` is present, and nothing for another OS
(a `win_amd64` wheel counts as *absent*, never as present). A package that cannot be staged for the
target is a failure that names it.

Then carry `wheelhouse/` in and build:

```
python scripts/check_offline_install.py --no-download
```

`--no-download` is the air-gapped re-run: staged wheels only, and its container runs with
`--network none`, so an index access is impossible rather than merely avoided — including for the
build backend, which is why `hatchling` has to be in the wheelhouse.

```
docker build --build-arg INSTALL_MODE=wheelhouse --tag mdqm:local .
```

`wheelhouse/` is committed as a directory with a placeholder `.gitkeep`, because `COPY wheelhouse/`
fails outright when the directory is absent; the wheels themselves are deployment artefacts and
`.gitignore` keeps them out of the repository (mirror mode never reads their contents).

**Mode A needs none of this.** A site with an internal PyPI mirror builds the image with no
wheelhouse, no container, no `check_offline_install.py` run: `pip` inside the image build resolves
the closure from the mirror (section 2). The wheelhouse, the staging container and `--no-download`
exist only for the site that has no index to talk to.

## 4. Running it

```
docker run --rm -p 8765:8765 \
  -e MDQM_SERVER__PUBLIC_URL=https://mdqm.internal \
  --mount type=volume,src=mdqm-data,dst=/data \
  --mount type=bind,src=/etc/mdqm/tokens.json,dst=/run/secrets/tokens.json,readonly \
  mdqm:local
```

Notes that cost time if missed:

- **The token file is required.** `mdqm serve` refuses to start without it (a server with no auth is
  never served). It is a deployment secret, never a repository file; see `.env.example` for its
  shape.
- **`MDQM_SERVER__PUBLIC_URL` is what clients are told.** The image binds `0.0.0.0` and the edge
  terminates TLS, so a URL derived from the bind address would advertise `http://0.0.0.0:8765` as
  the auth issuer — unroutable, and the wrong scheme. Set it to the URL clients actually use; it
  must be an absolute `http://` or `https://` URL with a host, and the server refuses to start
  otherwise. It is deliberately not baked into the image: every deployment has its own hostname.
- **Config**: `mdqm.yaml` is read from the working directory (`/app`), or wherever `--config` points.
  Environment variables win over the file. Every command reads it (`serve`, `audit`, `bootstrap`,
  `probe`, `review`), so a mounted config cannot be silently ignored. It holds names and locations,
  never a credential.
- **File permissions on secrets**: a fixed `uid 10001` is *not* the access path. See section 6.
- **Read-only root filesystem** works: `/data`, `/cache` and `/semantic` are the only writable trees.
  DuckDB spills into `<MDQM_PATHS__CACHE_DIR>/spill` — set explicitly by the cache and the executor,
  never DuckDB's default `.tmp` in the working directory, which is read-only here. Run with
  `--read-only` if the platform prefers it.
- **Backup** = the `/data` volume plus the semantic YAML repository in git. `/cache` is disposable.

## 4a. Verified locally (Docker Desktop 29.9.3, Compose v5.7.1, Windows host)

These commands were executed, not reasoned about. The outputs are what the operator should see.

```bash
python scripts/dev.py env                          # writes .env and docker/server/tokens.json
docker compose --profile pg --profile server up -d --build
docker ps --filter name=mdqm-server-1              # Up (healthy)
docker exec mdqm-server-1 id                       # uid=10001(mdqm) gid=0(root)
curl -s localhost:58765/healthz                     # {"status":"ok","name":"mdqm",...}
docker exec mdqm-server-1 mdqm probe                # crm: 1 supported, 0 unsupported, 3 unknown
```

Three findings from that run, each fixed and pinned by a test:

1. **`MDQM_PATHS__DATA_DIR` is not a setting.** The writable paths are `paths.metadata_db`,
   `paths.semantic_dir`, `paths.cache_dir`; a source's host/port/database belong under
   `sources[].options`, not at the top of the entry. `extra="forbid"` rejects both mistakes loudly,
   which is how they were found.
2. **The container must run with GID 0.** The image makes `/data`, `/cache` and `/semantic`
   group-writable instead of chowning them to one UID, so an arbitrary UID with GID 0 can write --
   the same assumption OpenShift makes. Compose therefore sets `user: "10001:0"`; without it the
   server dies with `unable to open database file`.
3. **First boot has no semantic layer and must not crash.** An empty semantic directory is a normal
   state, not corruption: the server starts, logs a warning and `/context` answers `NOT_SUPPORTED`
   with a fix. A directory whose files fail to load still refuses to start.

The image must be rebuilt after any change to `pyproject.toml` entry points: a stale image
advertised only the `openapi` adapter until `--build` was rerun, which the probe exposed with
`AdapterNotRegistered`.

## 5. Verification status (read this before trusting the image)

Runs anywhere, no container needed:

```
python scripts/check_offline_install.py     # wheelhouse closure, fresh install, smoke, recipe
python scripts/check_offline_install.py --help
```

What it proves: the dependency closure (including the `hatchling` build backend and its own
requirements) can be installed into a clean virtual environment with `--no-index`; every staged
artefact is an installable wheel for the requested platform, and a wheel for another OS counts as
absent; the `Dockerfile` pins its base image, runs non-root, has a `HEALTHCHECK`, carries no literal
credential and installs nothing in the runtime stage. For a Linux target staged from Windows or
macOS, the closure and the install are proven **inside a Linux container**, which is the environment
the image actually has; the run that did it is recorded in `progress/M08-offline.md`.

What it does **not** prove: that `docker build` works, that the image starts, that a pod schedules,
or that a mirror carries a package. `docker build` is still not run by any command here — the
wheelhouse and the clean-venv install are, and the container stages only pip. See
`progress/M08.md` and `progress/M08-offline.md`.

**Pin the base image by digest when you mirror it.** A tag is not reproducible on an air gap:
`python:3.12-slim-bookworm` can be re-pushed, and a rebuilt image would then differ from the one that
was reviewed. Copy the mirror to an internal registry, take its digest, and use it in the `FROM`
lines:

```
FROM python@sha256:<64 hex characters> AS builder
```

`check_offline_install.py` accepts either a tag that is not `latest` or a `@sha256:` digest; the
digest is the stronger form and the only one that survives being mirrored.

## 6. OpenShift: why the writable trees are group-owned, not chowned

The default `restricted-v2` SCC uses `MustRunAsRange`: it **rejects an explicit out-of-range uid at
admission**, then runs the container as a random namespace uid. That uid is always in **GID 0**, and
the same SCC mounts Secrets as `root:<namespace-gid> 0440`.

So the image must not depend on uid `10001` owning anything:

```
chgrp -R 0 /data /cache /semantic && chmod -R g=u /data /cache /semantic
```

Group 0 is the access path; the user is whatever the namespace chooses. `USER 10001:10001` stays in
the `Dockerfile` because plain Docker and Podman honour it, and an OpenShift namespace overrides it.

Consequences for the manifests (they live in `deploy/`, owned by another task):

- do **not** hard-set `runAsUser: 10001` — admission rejects it under `restricted-v2`;
- `fsGroup` is unnecessary for the same reason: GID 0 already has the group bits;
- a mounted Secret must be readable by group 0, which the SCC's `0440 root:<ns-gid>` already is.
