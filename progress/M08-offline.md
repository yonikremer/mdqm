status: done
task: M08 (follow-up: staging a Linux wheelhouse from Windows)
agent: m08-offline   branch: task/M08-offline   commit: 1a8df17
tests run: `python scripts/check_offline_install.py` -> exit 0 (staged 55 wheels inside `python:3.12-slim`,
             built `mdqm-0.0.1-py3-none-any.whl`, installed `mdqm[pg]` with `--no-index` inside the
             container, "smoke ok 0.0.1", `mdqm --version` -> 0.0.1, "offline install check passed")
           `python scripts/check_offline_install.py --no-download` -> exit 0, container ran with
             `--network none` (no network interface at all) and still installed + smoked green
           `python -m pytest tests/unit/test_check_offline_install.py -q` -> 69 passed
           `python scripts/check.py` -> ruff + mypy (strict, `src`) + `pytest tests/unit` green
## Summary          The bug: `--platform manylinux2014_x86_64` filters which *wheels* are eligible, but
                     environment markers are still evaluated for the interpreter running pip. On Windows
                     `mcp` requires `pywin32` (`sys_platform == "win32"`), no manylinux wheel of it exists,
                     and pip dies with `ResolutionImpossible` - no flag combination fixes that.
                     Fix: when the target is Linux and the host is not, `pip download` runs inside
                     `docker run --rm python:3.12-slim ...`, where the markers resolve for Linux. The
                     install + smoke steps run in that container too: a manylinux wheelhouse holds no
                     `win_amd64` binaries, so a Windows venv cannot install it (it fails on `duckdb`).
                     Docker is auto-detected; `--no-docker` refuses and fails loudly; `--docker-image`
                     overrides the staging image (default `python:3.12-slim`, documented as mirrorable);
                     `PIP_INDEX_URL`/`PIP_EXTRA_INDEX_URL`/`PIP_TRUSTED_HOST`/`PIP_CERT` are forwarded so a
                     mirror-only site can stage through it.
                     New honest check: `closure_gaps()` walks every staged wheel's *unmarked* `Requires-Dist`
                     and fails when one is absent, so "hatchling is staged" now means `packaging`,
                     `pathspec`, `pluggy` and `trove-classifiers` are staged with it.
                     `docs/deployment.md` section 3 rewritten: the single operator command from Windows,
                     what the container does and why, and that mirror mode needs none of it.
## Real output     `python scripts/check_offline_install.py` from Windows (Git Bash, Docker Desktop):

```
runtime requirements (pg extras): 12 + 1 build backend
  mcp>=2,<3
  pydantic>=2.8,<3
  pydantic-settings>=2.4,<3
  sqlglot>=30
  duckdb>=1.1
  pyarrow>=17
  httpx>=0.27
  ruamel.yaml>=0.18
  typer>=0.12
  tzdata>=2024.1
  psycopg[binary]>=3.2
  psycopg-pool>=3.2
  hatchling
target platform: manylinux2014_x86_64 (cp312)
container: python:3.12-slim (target manylinux2014_x86_64 is Linux but this host is win32: pip would evaluate environment markers for this machine (win_amd64 staging insists on pywin32, which has no manylinux wheel), so the wheels must be staged inside a Linux container)
Dockerfile: ok
+ docker run --rm --volume <repo>/wheelhouse:/wheelhouse --volume <repo>/wheelhouse/requirements.txt:/requirements.txt:ro python:3.12-slim python -m pip download --disable-pip-version-check --no-cache-dir --only-binary=:all: --platform manylinux2014_x86_64 --implementation cp --python-version 3.12 --abi cp312 --dest /wheelhouse --requirement /requirements.txt
Collecting mcp<3,>=2 (from -r /requirements.txt (line 1))
...
Saved /wheelhouse/mdurl-0.1.2-py3-none-any.whl
Saved /wheelhouse/pycparser-3.12-py3-none-any.whl
Successfully downloaded mcp mcp-types pydantic pydantic-core pydantic-settings sqlglot duckdb pyarrow httpx httpcore ruamel.yaml typer tzdata psycopg psycopg-pool hatchling psycopg-binary annotated-doc annotated-types anyio httpx2 httpcore2 idna jsonschema opentelemetry-api packaging pathspec pluggy pyjwt python-dotenv python-multipart rich shellingham sse-starlette starlette tomlkit typing-extensions typing-inspection uvicorn certifi trove-classifiers attrs click cryptography h11 jsonschema-specifications markdown-it-py pygments referencing rpds-py truststore cffi mdurl pycparser
+ C:\Users\yonik\mdqm\.venv\Scripts\python.exe -m pip wheel --no-deps --no-index --find-links <repo>\wheelhouse --wheel-dir <repo>\wheelhouse <repo>
...
  Created wheel for mdqm: filename=mdqm-0.0.1-py3-none-any.whl size=531346 sha256=34ae7ccf...
Successfully built mdqm
wheelhouse C:\Users\yonik\mdqm\.worktrees\M08offline\wheelhouse: 55 installable wheel(s), 0 wrong-platform, 0 source archive(s)
closure: every unmarked requirement is staged
+ docker run --rm --volume <repo>/wheelhouse:/wheelhouse:ro --volume <tmp>/smoke.py:/work/smoke.py:ro python:3.12-slim sh -c set -eu
python -m venv /venv
/venv/bin/python -m pip install --disable-pip-version-check --no-index --find-links /wheelhouse --only-binary=:all: "mdqm[pg]"
/venv/bin/python /work/smoke.py
/venv/bin/mdqm --version
...
Processing /wheelhouse/cffi-2.1.1-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (from cryptography>=3.4.0->pyjwt[crypto]>=2.10.1->mcp<3,>=2->mdqm[pg])
Processing /wheelhouse/pycparser-3.12-py3-none-any.whl (from cffi>=2.0.0->cryptography>=3.4.0->pyjwt[crypto]>=2.10.1->mcp<3,>=2->mdqm[pg])
Installing collected packages: tzdata, typing-extensions, truststore, sqlglot, shellingham, ruamel-yaml, rpds-py, python-multipart, python-dotenv, pyjwt, pygments, pycparser, pyarrow, psycopg-binary, mdurl, idna, h11, duckdb, click, certifi, attrs, annotated-types, annotated-doc, uvicorn, typing-inspection, referencing, pydantic-core, psycopg-pool, psycopg, opentelemetry-api, markdown-it-py, httpcore2, httpcore, cffi, anyio, starlette, rich, pydantic, jsonschema-specifications, httpx2, httpx, cryptography, typer, sse-starlette, pydantic-settings, mcp-types, jsonschema, mcp, mdqm
Successfully installed annotated-doc-0.0.7 ... mdqm-0.0.1 ... uvicorn-0.54.0
smoke ok 0.0.1
0.0.1

offline install check passed
```

## Wheelhouse     Listing after that run (`wheelhouse/`, 56 files: 55 wheels + `requirements.txt`):

- platform-tag histogram (`ls *.whl` -> last `-` field): `48 any`, `4 manylinux2014_x86_64.manylinux_2_17_x86_64`,
  `3 manylinux_2_17_x86_64.manylinux2014_x86_64`. Every platform-specific wheel is `manylinux*` for `x86_64`;
  nothing else. The `any` wheels are pure Python.
- build backend present with its closure: `hatchling-1.33.0-py3-none-any.whl`, `packaging-26.4-py3-none-any.whl`,
  `pathspec-1.1.1-py3-none-any.whl`, `pluggy-1.7.0-py3-none-any.whl`, `trove_classifiers-2026.9.21.13-py3-none-any.whl`
  (and `mdqm-0.0.1-py3-none-any.whl`). `closure_gaps()` is what enforces this, not the eye.
- `pywin32`: **absent** (`ls | grep -i pywin32` -> nothing). That is the whole point.
- source archives: **none** (`ls | grep -Ei '\.(tar\.gz|tgz|zip|tar\.bz2)$'` -> nothing).
## No-download    How the "no network" claim was verified:
                 1. Structural: with `--no-download` the script runs no `pip download` at all and the install
                    container is started with `--network none`, so the container has no network interface.
                    A regression test asserts no recorded command contains `download` and that every command
                    carries `--no-index` (`test_no_download_never_runs_a_pip_download`).
                 2. Real run: `--no-download` exited 0 with `--network none` on the command line, installed
                    the closure from `/wheelhouse` and printed "smoke ok 0.0.1". An index access would have
                    failed the run, not warned.
                 The host `pip wheel` step also runs with `--no-index --find-links wheelhouse`, because pip
                 builds the project in an isolated env and would otherwise fetch `hatchling`.
## Decisions        - The install + smoke steps run in the container as well, not just the download. Reason: a
                     manylinux wheelhouse contains no `win_amd64` binaries, so `pip install --no-index` into a
                     Windows venv fails on `duckdb` - and even if it worked it would say nothing about the image.
                   - `--no-docker` refuses rather than trying harder. Without a Linux kernel there is no way to
                     get the target's markers evaluated, so a silent fallback would stage a wrong wheelhouse.
                   - The container venv lives on the container's filesystem (`/venv`), not on the bind mount:
                     a venv on a Windows bind mount means symlinks across the 9p/virtiofs layer, which is a
                     needless failure mode. Only `/wheelhouse` (read-only) and `smoke.py` are mounted.
                   - The class of the bug is cross-OS, not Windows-only, so the rule is "target is Linux and
                     host is not" (macOS is included). `--target-platform win_amd64` on Windows stays native.
                   - `closure_gaps()` checks only requirements with *no* environment marker. A marked
                     requirement depends on the target environment and pip inside the staging container is the
                     authority there; an unmarked one is required everywhere, so its absence is always a hole.
## Deviations       none from the task. `docs/deployment.md` section 5's "no container runtime" sentence was
                   stale after this change and now says what is and is not proven.
## Dependencies     none new. The container is a staging tool, never a shipped dependency; `zipfile` and
                   `email`-free METADATA parsing are stdlib.
## Not verified     - `docker build .` (the image itself) is still not run by any command here; this change
                     makes the wheelhouse Linux-correct, not the image.
                   - The `--docker-image` mirror path is only exercised against `python:3.12-slim` locally;
                     no internal registry was available to point it at.
                   - A truly air-gapped staging run was not possible here (this machine has egress). The
                     `--network none` re-run is the substitute: it is enforced by the kernel, not promised.
## Plan issues      - `pip download` with no constraints file resolves "newest that satisfies the ranges"
                     (this run: `duckdb 1.2.2`, `mcp 2.3.0`). The wheelhouse is reproducible only as a set of
                     pins the operator records; a constraints file still belongs in a follow-up.
