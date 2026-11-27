status: partial
task: OpenShift (owner request; not in TASKS.md)
agent: claude   branch: feature/openshift   commit: see `git log -1`
tests run: `.venv/Scripts/python.exe -m pytest tests/unit/test_deploy_manifests.py -q` -> 37 passed; `python scripts/check.py` on `feature/openshift` in an isolated worktree -> green
## Summary
OpenShift/Kubernetes deployment for the MCP server, plus the air-gapped operator guide. New files:
`deploy/openshift/{deployment,pvc,service,route,ingress,configmap,secret.example,serviceaccount,kustomization}.yaml`,
`docs/deploy.md`, `tests/unit/test_deploy_manifests.py`. The Deployment is one replica with
`strategy: Recreate` (SQLite + RWO claim), runs with a read-only root filesystem, drops all
capabilities, `seccompProfile: RuntimeDefault`, `workingDir: /tmp` and `TMPDIR=/tmp` for the DuckDB
spill (an `emptyDir`), a startupProbe plus readiness/liveness probes on the public `GET /healthz`,
and `envFrom` the ConfigMap. All durable state (SQLite DB, audit chain, cache, semantic layer) is on
one 10Gi RWO claim with a pinned `storageClassName`, mounted at `/data`. The mounted Secret at
`/run/secrets` is what `FileSecretProvider` reads, and `MDQM_TOKENS_FILE` points at its `tokens.json`
key.
## Decisions
* **No UID, GID or `fsGroup` is pinned.** The pod asks only for `runAsNonRoot: true`, so OpenShift's
  default `restricted-v2` assigns a namespace UID with GID 0 and admits the pod with no SCC grant at
  all — which works because `Dockerfile` is moving to the group-writable pattern
  (`chgrp -R 0`, `chmod -R g=u`; another agent's change). For an image that still owns `/data` as
  10001:10001 the operator grants `nonroot-v2` and adds `runAsUser/runAsGroup/fsGroup: 10001`
  (documented as the second row of `docs/deploy.md` section 3). `anyuid` would work and is
  deliberately not used.
* **`storageClassName` is pinned to `CHANGE_ME_STORAGE_CLASS`** rather than left to the cluster
  default: an air-gapped cluster's default class can carry a different reclaim policy, and a silent
  substitution is worse than a claim that does not bind until the operator names a real class.
* **Route `host` is omitted.** A guessed hostname never resolves; the router derives
  `mdqm-<namespace>.apps.<cluster>` and the docs say to read it back with
  `oc get route mdqm -o jsonpath='{.spec.host}'`.
* **Semantic layer and cache moved under `/data`** (`MDQM_PATHS__SEMANTIC_DIR`, `MDQM_PATHS__CACHE_DIR`)
  instead of the image's `/semantic` and `/cache`: `mdqm review approve` writes the semantic layer, so
  it must persist, and one claim is one backup. The image's `/cache` then goes unused.
* **DuckDB spill on `emptyDir`, not the PVC.** The spill directory is a constructor argument
  (`FederatedExecutor(spill_dir=...)`), not a Settings field, so a manifest cannot point it at the
  claim; `workingDir: /tmp` plus `TMPDIR=/tmp` is the only lever, and spill is scratch. This is the one
  place the task's "PVC for the DB and the spill" was followed for the DB and deviated from for the
  spill.
* **Sources via `MDQM_SOURCES` JSON env** rather than a mounted `mdqm.yaml`: `serve` resolves
  `Settings()` before any YAML file is read from disk (`Settings.load()` is the only path that reads
  a file), and `/app` is read-only. `SourceConfig.secret` stays a *name*, so no credential is in the
  ConfigMap.
* **`route.yaml` is the default edge**, `ingress.yaml` the commented-out alternative; a cluster takes
  one. Both raise the edge timeout (600s): an MCP stream outlives the 30–60s router default.
* **Image via kustomize `images:`**, with the shipped `newTag: "changeme"` as a stop sign so a missed
  edit fails at `ImagePullBackOff` instead of running an unknown build.
* Model key and SMTP password use `secretKeyRef` with `optional: true`: the server serves without a
  model endpoint, so the pod must start and say so instead of crash-looping.
## Deviations
* Scope was given by the owner, not by `TASKS.md`; no `owns` list was available. Only the declared
  paths were touched (`deploy/openshift/**`, `docs/deploy.md`, `tests/unit/test_deploy_manifests.py`,
  this file). `Dockerfile`, `docker-compose.yml` and `docs/dev-env.md` were read, never edited.
  `docs/deployment.md` (image build, another lane) is cross-referenced, not duplicated.
* The owner's later facts (internal PyPI mirrors exist; base image must be mirrored) changed section 2
  of `docs/deploy.md` into two build modes — mirror build (normal) and staged wheelhouse (fallback) —
  and section 3 into the two-row SCC table above. The manifests were adjusted to the no-pinned-UID
  shape at the same time.
* No NetworkPolicy, PodDisruptionBudget or HPA: the first two are cluster policy the operator owns,
  the third contradicts the single writer. Stated as limitations in `docs/deploy.md` section 10.
## Dependencies
none (ruamel.yaml and pytest were already dependencies).
## Not verified
No container runtime and no cluster on this machine (`oc` and `kubectl` are not installed), so every
runtime claim is unverified:
* the Deployment is admitted by `restricted-v2` with an arbitrary UID and GID 0, and the
  group-writable image really lets that UID write `/data`;
* the pod starts, reads `/run/secrets/tokens.json` and logs its two startup lines;
* the PVC binds to the pinned class and the SQLite DB is created on it;
* `GET /healthz` answers through the Route/Ingress and the probes flip the pod to Ready;
* `POST /mcp` without a token returns 401 and with a token does not;
* the edge timeout annotations actually keep a long stream alive;
* `oc apply -k deploy/openshift` applies cleanly (not even `--dry-run=client`, since `oc` is absent).
Not verified for the build modes either:
* no package is claimed to exist on the internal mirror; `docs/deploy.md` section 2c is an operator
  step that probes the exact names (`hatchling`, `pyarrow`, `duckdb`, `psycopg`, `psycopg-binary`,
  `mcp`, `sqlglot`, `pydantic`, `pydantic-settings`, `ruamel.yaml`, `httpx`, `typer`, `tzdata`);
* the `PIP_INDEX_URL` build mode assumes the Dockerfile's builder stage honours it. Today's merged
  Dockerfile forces `--no-index --find-links /wheelhouse`, so Mode A needs that one `RUN` line changed
  (Dockerfile lane) or a local edit; the docs say so explicitly instead of pretending it works.
Automated instead: `tests/unit/test_deploy_manifests.py` checks structure, cross references
(ConfigMap/Secret/PVC/ServiceAccount names exist, Service `targetPort` equals the container port
name, Route/Ingress point at the Service), single-replica + Recreate, the hardened security context
with no pinned UID, that credentials use `valueFrom`, that the Secret template holds only placeholders
and is not a kustomize resource, and that ConfigMap paths sit on their mounts.
## Plan issues
* This work is not in `TASKS.md`/`tasks.json`; it came directly from the owner ("make it deployable
  with docker and openshift"). The Docker half is the existing `Dockerfile` plus `docs/deployment.md`.
* The Dockerfile's builder stage cannot use an index mirror today (`--no-index` is hard-coded). If the
  target network's normal path is "build from the internal mirror", that flag needs to become a build
  argument, or every deployment falls back to the slower wheelhouse path. Worth a task in the
  Dockerfile lane.
* The advertised auth metadata URL is hard-coded to `http://{host}:{port}` in `mdqm serve`, so behind
  a Route it advertises an internal address. Fixing it needs a `public_url` setting
  (`Settings.server` has only `host`/`port`). Worth a task.
* The DuckDB spill directory has no Settings field, so it cannot be configured from a manifest. Worth
  a small task if a deployment ever needs it on a PVC.
* SQLite single-writer caps the deployment at one replica per namespace. Horizontal scale would need
  the metadata store off SQLite: a design decision, not a deployment tweak.
