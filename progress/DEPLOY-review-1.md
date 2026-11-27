status: done
task: DEPLOY-review-1
agent: fixer (deployment review)   branch: task/DEPLOY-review-1   commit: this one (titled "DEPLOY: fix offline-install blockers, config wiring and cluster-readiness"; a sha cannot name the commit that carries it -- `git log -1 --format=%H -- progress/DEPLOY-review-1.md` resolves it)
tests run:
  python scripts/check.py            -> ruff ok, mypy ok (123 files), 3887 unit tests pass
  python -m pytest tests/unit/test_cli_config.py tests/unit/test_check_offline_install.py
      tests/unit/test_context_hook_diagnostics.py tests/unit/test_duckdb_spill.py
      tests/unit/test_server_public_url.py -q      -> 168 passed
  # "fails before" receipt for H1: HEAD's cli.py restored, same test file
  -> 5 failed, 1 passed   (with the fix: 6 passed)

## Summary

Fixes nine findings from the deployment review. The offline-install check no longer lies about
what it staged (build backend + target platform), the CLI reads the operator's `mdqm.yaml`,
the server advertises a routable issuer, DuckDB spills into the cache tree instead of the read-only
working directory, the plugin hook has a realistic budget and reports failures, the image is
OpenShift-friendly (GID 0, not a fixed uid) and the build context no longer ships agent transcripts.

**Owner update taken into account:** the air-gapped network has internal PyPI mirrors, so the image
now defaults to installing from an index and the wheelhouse is an alternative, explicitly selected
mode. `hatchling` stays in the closure either way, because the wheelhouse mode must be provable.

## Decisions

- **Two build modes in one Dockerfile** (`INSTALL_MODE=mirror|wheelhouse`, plus `PIP_INDEX_URL` /
  `PIP_TRUSTED_HOST`). Default is `mirror` with an empty index URL, so a connected build works with
  no arguments. `COPY wheelhouse/` is unconditional and `wheelhouse/.gitkeep` keeps that directory
  in every checkout: `COPY` of a missing directory fails the build, which would have broken the
  default mode. `wheelhouse/*` + `!wheelhouse/.gitkeep` in `.gitignore` (a 2-line change to a file
  I do not own; it was needed for the placeholder, nothing else).
- **Platform matching keys the OS family and the architecture.** `manylinux2014_x86_64` accepts
  `manylinux*`/`linux*` wheels for `x86_64` and pure-Python `any`; it rejects `musllinux`,
  `macosx`, `win_*` and aarch64. `win_amd64` is accepted only when asked for explicitly.
- **`Settings.load()` resolution order** is argument -> pinned (ContextVar) -> `./mdqm.yaml`. A CLI
  entry point pins the file once in the Typer callback, so the modules that build `Settings()`
  themselves (`probe/cli.py`, `pipeline/cli.py`, `context_endpoint`) read the same file. They are
  other lanes' files, so pinning from `cli.py` was the only way to fix them without touching them.
- **`pin_yaml_file(None)` means "the cwd file" (i.e. a reset)**, never "keep the current pin". The
  first version of `resolve_yaml_file` consulted the pin and made the reset impossible, which broke
  20 tests in `test_config.py`; that is why the two functions resolve differently on purpose.
- **Hook timeouts split in two:** `CONNECT_TIMEOUT_SECONDS = 0.45` and `TIMEOUT_SECONDS = 2.0`.
  A single 2.0 s budget regressed a real product promise — the existing
  `test_server_down_prints_nothing_exits_zero_under_a_second` measures a *dropped* connect and saw
  2.20 s. `urlopen` applies one timeout to every socket operation, so the hook now uses
  `http.client`: connect on the short budget, then `sock.settimeout(2.0)` for the round trip.
- **`requires.claudeCode` in `plugin.json`** is a declarative field of our own: Claude Code ignores
  unknown manifest keys, so the README statement is the part that actually informs a human. Neither
  can make an old CLI fail loudly — that is stated plainly in the README rather than implied.

## Deviations

- The original B1 brief said `Dockerfile:35` needs no change. Under the owner's update it does:
  `--no-index` is now only the `wheelhouse` mode, not the only path.
- `Dockerfile` writable trees use `chgrp 0` + `chmod g=u`. The `deploy/openshift` manifests still
  hard-set `runAsUser/runAsGroup/fsGroup: 10001`; **they are another agent's files and still need
  that change** for the image to schedule under `restricted-v2`. Recorded in `docs/deployment.md` §6
  and in Not verified below.

## Dependencies

None added. `tomllib` (stdlib, B1) and `http.client`/`ssl` (stdlib, hook) replace nothing external.

## Not verified

No container runtime on this machine, so **none** of the following was executed:

- `docker build` in either mode; that `INSTALL_MODE`/`PIP_INDEX_URL`/`PIP_TRUSTED_HOST` reach pip.
- That an internal mirror carries the packages. The check can only prove the *source set* and the
  wheelhouse; `docs/deployment.md` §2 lists the exact names for the operator to confirm.
- The image starting, `GET /healthz` as a non-root user, a real DuckDB spill under a read-only root
  filesystem (the spill directory is proven in-process, not in a pod).
- OpenShift admission of the image (SCC `MustRunAsRange`, GID-0 Secret readability). Reasoned from
  the documented SCC behaviour, not run.
- A real `POST /context` round trip through a Route (the hook budget is reasoned from the hop list).
- Claude Code 2.1.139 behaviour with exec-form hooks.

## Plan issues

- Reviewer claim corrected: `plugin/mdqm/hooks/hooks.json` **already** had `"timeout": 3`; only
  `TIMEOUT_SECONDS` needed changing. Half of M4 was already in place.
- The reviewer's `spill_dir` observation is exact (`git grep "spill_dir="` finds only tests). There is
  also no production constructor call site yet for `SegmentCache`/`FederatedExecutor`, so "pass it
  from both call sites" became "default it from the settings in both constructors", which also covers
  the call sites that do not exist yet.
- `H3` is only half-operable: the image is fixed, the manifests are not mine. Until they drop
  `runAsUser: 10001`, a pod under `restricted-v2` is rejected at admission regardless of this image.
- `src/mdqm/adapters/openapi.py`, `src/mdqm/sql/guard.py`, `src/mdqm/server/auth.py`,
  `src/mdqm/governor/core.py` carry open security findings that a third agent is fixing. I did not
  touch them; they are the only remaining lint/LSP blockers near my files.
