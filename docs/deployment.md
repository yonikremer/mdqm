# Deploying `mdqm`: container image, air-gapped build, offline install

The target network has no internet (AGENTS.md principle 7) and takes packages only as wheels from an
internal PyPI mirror (principle 8). This note covers the one container image in DESIGN section 14 and
the two things that must exist before it can be built there: a **staged wheelhouse** and the
**secret files** the server reads at startup.

Files: `Dockerfile` (multi-stage, task M08), `.dockerignore`, `scripts/check_offline_install.py`.

## 1. What the image is

| Concern | Choice |
|---|---|
| Base | `python:3.12-slim-bookworm` in both stages, pinned by minor + Debian codename, no `latest` |
| Builder stage | creates `/opt/venv` and installs `.[pg]` from `/wheelhouse` with `--no-index --only-binary=:all:` |
| Runtime stage | copies `/opt/venv` only: no compiler, no build tools, no `pip`, no wheel files |
| User | `10001:10001` (`mdqm`), no home, no login shell: the container never runs as root |
| Entrypoint | `mdqm serve`; host and port come from `MDQM_SERVER__HOST` / `MDQM_SERVER__PORT` (`0.0.0.0:8765` in the image) |
| Healthcheck | `GET /healthz` on the configured port, answered with the image's own `python` (no `curl` needed) |
| Volumes | `/data` (SQLite metadata DB + audit chain: the only state to back up), `/cache` (DuckDB spill + query cache, disposable) |
| Secrets | files under `/run/secrets`; `MDQM_SECRETS__DIR=/run/secrets` and `MDQM_TOKENS_FILE=/run/secrets/tokens.json` |

The `pg` extra is the only extra installed. `oracle` and `mssql` join the image only after their
driver verdicts (task A08); a driver nobody chose must never ride along.

## 2. Build on the air-gapped network

Pre-stage these on a connected machine, carry them in, and keep them under version control of the
deployment (not of this repo):

1. **The wheelhouse** - every wheel for the target platform:

   ```
   python scripts/check_offline_install.py --wheelhouse wheelhouse
   ```

   That builds the `mdqm` wheel, downloads the whole dependency closure with
   `--only-binary=:all:` into `wheelhouse/`, installs `mdqm[pg]` from it into a fresh virtual
   environment with `--no-index`, and runs a smoke import plus a real store migration. It fails on a
   source archive: an sdist would need a compiler the target does not have. Print the current list
   with that same command; do not trust a number written in a document.

   Build the wheelhouse **for the target platform**, not for the machine that stages it: wheels are
   platform specific (`cp312-manylinux` for the image, `cp312-win_amd64` for a Windows dev box). On
   the connected machine use a Linux container or a matching interpreter, or stage on the target OS.

2. **The build backend wheels** - a container build installs nothing from the network *except* what
   `hatchling` needs to build this project's own wheel. Either stage `hatchling` and its
   dependencies in the wheelhouse too, or build the `mdqm` wheel on the connected machine and let
   the builder stage install wheels only. The second path is what the current `Dockerfile` does: it
   installs `.[pg]` with `--no-index`, and pip builds the project wheel from the copied `src/`.

3. **The base image** - `docker save python:3.12-slim-bookworm -o python-slim.tar` on the connected
   machine, `docker load` on the target. Pin the digest (`FROM image@sha256:...`) once the staging
   step has produced it; the tag alone only pins the Python minor and the Debian release.

Then, on the air-gapped machine:

```
python scripts/check_offline_install.py --no-download
docker build --tag mdqm:local .
```

`--no-download` proves the staged wheelhouse is complete without touching any package index.

## 3. What must be pre-staged (and what must not happen)

- **Model endpoint**: an on-prem, OpenAI-compatible API. Configure `MDQM_MODEL__BASE_URL` and
  `MDQM_MODEL__MODEL`; the key comes from a secret file. Nothing downloads model files.
- **DuckDB extensions**: never installed. No `INSTALL`/`LOAD` at runtime: the local engine runs with
  the bundled build only. A wheelhouse that expects an extension download is a bug.
- **SQLite FTS5**: comes from the Python build inside the image; the probe checks it and reports it.
  A build without FTS5 degrades name search, it does not crash.
- **Oracle**: Thin mode only, no Oracle client library in the image.
- **Timezone data**: `tzdata` is a declared dependency, so `zoneinfo` works inside the slim image.
- **TLS**: terminated at the edge, per DESIGN section 14. The image serves plain HTTP on its port.

## 4. Running it

```
docker run --rm -p 8765:8765 \
  --mount type=volume,src=mdqm-data,dst=/data \
  --mount type=bind,src=/etc/mdqm/tokens.json,dst=/run/secrets/tokens.json,readonly \
  mdqm:local
```

Notes that cost time if missed:

- **The token file is required.** `mdqm serve` refuses to start without it (a server with no auth is
  never served). It is a deployment secret, never a repository file; see `.env.example` in the repo
  for its shape.
- **File permissions on secrets**: the process runs as uid `10001`, so a bind-mounted secret must be
  readable by that uid or gid (`0440`, group `10001`). Docker Compose `secrets` accept `uid`, `gid`
  and `mode` for exactly this.
- **Read-only root filesystem** works: `/data` and `/cache` are the only writable trees. Run with
  `--read-only --tmpfs /tmp` if the platform prefers it.
- **Config**: `mdqm.yaml` is read from the working directory (`/app`) or nowhere; environment
  variables win over the file (see `mdqm.config.settings`). Mount the organisation's YAML as a
  read-only file if it is used at all; it holds names and locations, never a credential.
- **Backup** = the `/data` volume plus the semantic YAML repository in git. `/cache` is disposable.

## 5. Verification status (read this before trusting the image)

The dependency and recipe checks run anywhere:

```
python scripts/check_offline_install.py     # wheelhouse, fresh install, smoke test, recipe checks
python scripts/check_offline_install.py --help
```

They inspect `pyproject.toml` and the `Dockerfile` (pinned base image, `HEALTHCHECK`, non-root
`USER`, no literal credential, no `pip install` in the runtime stage) and prove an offline install
into a clean virtual environment. They do **not** build the image.

`docker build`, the resulting image size and a run inside the image (`GET /healthz` as the non-root
user) are unverified in this repository's development environment: it has no container runtime. See
`progress/M08.md`.
