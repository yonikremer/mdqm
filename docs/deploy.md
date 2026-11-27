# Deploying `mdqm` on OpenShift / Kubernetes

The air-gapped operator's path: build the image once (from the internal PyPI mirror, or from a staged
wheelhouse where there is none), put it in the internal registry, apply the manifests, check the pod
and the endpoint. Files live in `deploy/openshift/`. The image build itself is described in
`docs/deployment.md` (wheelhouse, `--no-index`, no runtime downloads); this note starts where the
image exists.

Read section 8 before you plan capacity: **one replica, ever**, and that is a correctness rule, not a
default.

## 1. What gets created

| Object | File | Why |
|---|---|---|
| `ServiceAccount mdqm` | `serviceaccount.yaml` | the name the SCC grant and the pod use; no API token is mounted |
| `PersistentVolumeClaim mdqm-data` | `pvc.yaml` | `ReadWriteOnce`, 10Gi: SQLite metadata DB + audit chain, cache, semantic layer |
| `ConfigMap mdqm-config` | `configmap.yaml` | every non-secret setting, injected with `envFrom` |
| `Secret mdqm-secrets` | `secret.example.yaml` | template only; you create the real one by hand |
| `Deployment mdqm` | `deployment.yaml` | `replicas: 1`, `Recreate`, non-root UID 10001, read-only root filesystem, `/healthz` probes |
| `Service mdqm` | `service.yaml` | ClusterIP on 8765, `targetPort: http` |
| `Route mdqm` | `route.yaml` | OpenShift edge TLS; **or** |
| `Ingress mdqm` | `ingress.yaml` | plain Kubernetes with an ingress controller |
| `Kustomization` | `kustomization.yaml` | one `oc apply -k`; holds the image override |

Inside the pod: config comes from environment variables, credentials come from files under
`/run/secrets`. Nothing needs the internet at runtime — only the configured sources, the model
endpoint on the internal network, and, at **build** time only, the PyPI mirror (AGENTS.md
principle 7).

## 2. Build the image and push it into the network

The internal network has PyPI mirrors that carry most popular libraries, and adding a missing one is
easy, so there are two build modes. Mode A is the normal one; Mode B is the fallback for a site with
no mirror at all. Either way the **base image** has to be mirrored first: nothing guarantees the
internal PyPI mirror carries container images.

### 2a. Mirror the base image, in both modes

There is no network route to Docker Hub from the target network, so put
`python:3.12-slim-bookworm` into the internal container registry and pin it by **digest**, not tag:

```
docker pull python:3.12-slim-bookworm
docker tag python:3.12-slim-bookworm image-registry.openshift-image-registry.svc:5000/library/python:3.12-slim-bookworm
docker push image-registry.openshift-image-registry.svc:5000/library/python:3.12-slim-bookworm
docker inspect --format='{{index .RepoDigests 0}}' image-registry.openshift-image-registry.svc:5000/library/python:3.12-slim-bookworm
```

Then point both `FROM` lines of the `Dockerfile` (builder and runtime) at the mirrored reference with
that digest, e.g. `FROM image-registry.../library/python@sha256:<digest>`. A digest cannot drift; a
tag can, and an air-gapped rebuild months later would silently use a different base.

### 2b. Mode A: build against the internal PyPI mirror (normal)

```
docker build \
  --build-arg PIP_INDEX_URL=http://pypi.internal.example/simple \
  --build-arg PIP_TRUSTED_HOST=pypi.internal.example \
  --tag mdqm:0.1.0 .
```

`PIP_TRUSTED_HOST` is only for a plain-HTTP mirror (no certificate); a mirror with a proper internal
CA needs `--build-arg PIP_CERT=/path/to/ca.crt` instead, and the CA file inside the build context.

**Verify this works with your Dockerfile before planning on it.** The builder stage as merged today
does `pip install --no-index --find-links /wheelhouse`, which ignores any index URL:

```
grep -n 'PIP_INDEX_URL\|no-index\|find-links' Dockerfile
```

If those args are not honoured yet, Mode A needs that one `RUN` line to accept `PIP_INDEX_URL` (a
change in the Dockerfile lane, not in these manifests), or the operator edits it locally. Mode B
below works with today's Dockerfile unchanged.

### 2c. Confirm the mirror carries every wheel before you rely on it

Do not assume any package is present: check. From a host that can reach the mirror, with no build
context involved:

```
pip download --no-deps --only-binary=:all: --dest ./mirror-probe \
  --index-url http://pypi.internal.example/simple \
  hatchling pyarrow duckdb psycopg psycopg-binary mcp sqlglot \
  pydantic pydantic-settings ruamel.yaml httpx typer tzdata
```

Those are the distributions this image needs, by exact name. `psycopg-binary` and `pyarrow` are the
two most likely to be missing on a small mirror (native wheels, one per platform), and `duckdb` the
biggest. A missing wheel is a mirror-administration request or a switch to Mode B, not a reason to
build a source tarball at the target site: `--only-binary=:all:` exists so a missing compiler never
becomes a runtime surprise.

### 2d. Mode B: fully disconnected wheelhouse build (fallback)

For a site with no usable mirror, stage the whole dependency closure on a connected machine and carry
it in:

```
python scripts/check_offline_install.py --wheelhouse wheelhouse
docker build --tag mdqm:0.1.0 .
```

That script builds the wheel, downloads the closure with `--only-binary=:all:`, installs `mdqm[pg]`
from it into a fresh environment, and runs a smoke import plus a real store migration; a source
archive fails it. The Dockerfile then installs with `--no-index --find-links /wheelhouse`, so the
build itself touches no index. Details and the wheel list: `docs/deployment.md`.

### 2e. Push the image and point the manifests at it

Either push from a machine that can see the internal registry:

```
docker tag mdqm:0.1.0 image-registry.openshift-image-registry.svc:5000/mdqm/mdqm:0.1.0
docker push image-registry.openshift-image-registry.svc:5000/mdqm/mdqm:0.1.0
```

or carry the image across the gap as a `docker save` tarball and load it on the far side with
`skopeo copy` or `docker load`, then push. Use a real version tag (`0.1.0`), never `latest`: a
cluster must be able to tell two builds apart, and `imagePullPolicy: IfNotPresent` cannot
distinguish them otherwise.

Set the reference once, in `deploy/openshift/kustomization.yaml`:

```
images:
  - name: IMAGE_PLACEHOLDER
    newName: image-registry.openshift-image-registry.svc:5000/mdqm/mdqm
    newTag: "0.1.0"
```

The shipped `newTag: "changeme"` is a deliberate stop sign: forget this step and the pod stops at
`ImagePullBackOff` instead of silently running an old build.

## 3. The security context decision (do this once)

The manifests **pin no UID and no GID**. That is deliberate: it lets the OpenShift default SCC admit
the pod, and it is what the image's group-writable layout is for.

| Image layout | SCC | What the operator does |
|---|---|---|
| group-writable: `chgrp -R 0 /data /tmp …` and `chmod -R g=u …`, no fixed `USER` needed | `restricted-v2` (the default) | **nothing extra** — the pod gets a namespace UID with GID 0 and writes `/data` through group permission |
| fixed `10001:10001` owning `/data` (the earlier Dockerfile) | `nonroot-v2` (OpenShift 4.11+) | grant it, and add `runAsUser: 10001`, `runAsGroup: 10001`, `fsGroup: 10001` to the pod security context |
| either | `anyuid` | works, broader than needed: do not request it |

So: with the group-writable image the deployment needs **no SCC grant at all**. If your image still
hard-codes UID 10001, grant the narrower-by-UID option:

```
oc adm policy add-scc-to-user nonroot-v2 -z mdqm -n mdqm
```

and add to `deployment.yaml`, under `spec.template.spec.securityContext`:

```
runAsUser: 10001
runAsGroup: 10001
fsGroup: 10001
```

A wrong combination does not fail at admission, it fails on first write. Expect a CrashLoopBackOff
with `unable to open database file` or `attempt to write a readonly database` in `oc logs`, and check
the assigned ids with `oc get pod -o jsonpath='{.spec.containers[0].securityContext}'`. No `hostPath`,
no privileged container, no extra capability is requested anywhere in these manifests, so nothing
else from the SCC needs relaxing. On plain Kubernetes SCC rules do not apply: `runAsNonRoot: true`
plus a storage driver that honours group permissions (or `fsGroup`) is enough.

## 4. Create the namespace, the Secret, then apply

Create the Secret **before** the first apply. Its `tokens.json` is the only identity mechanism
today; the server refuses to start without it, so the pod must find it on first start.

```
oc new-project mdqm
oc create secret generic mdqm-secrets \
  --from-file=tokens.json \
  --from-file=crm_ro \
  --from-file=billing_ro \
  --from-literal=MDQM_MODEL__API_KEY=CHANGEME \
  -n mdqm
oc apply -k deploy/openshift
```

What the keys are, and the token file's shape: `deploy/openshift/secret.example.yaml`. Two rules:

* **one token per person**, with `user_id` and `display_name`. `history` and the audit chain record
  the `user_id`, so a shared token destroys the point of having them. Two entries with the same token
  make the server refuse to start.
* a **source credential file is named after the `secret` reference** in that source's
  `SourceConfig` (`crm_ro`, `billing_ro`, …). The value is the password or API token; the file name
  is the reference. Check `secret.example.yaml` for the exact pairing.

On plain Kubernetes, use `kubectl` for the same commands and swap the edge: comment out
`route.yaml` and uncomment `ingress.yaml` in `kustomization.yaml`, then create the TLS Secret the
Ingress names:

```
kubectl create secret tls mdqm-tls --cert=server.crt --key=server.key -n mdqm
kubectl apply -k deploy/openshift
```

## 5. Fill in the two placeholders you cannot guess

`configmap.yaml` ships values that are placeholders, not organisation facts (AGENTS.md section 9
keeps those out of code):

| Key | Replace with |
|---|---|
| `storageClassName` in `pvc.yaml` | the class your cluster provisions; the shipped value is `CHANGE_ME_STORAGE_CLASS` and the claim does not bind until it is real |
| `MDQM_MODEL__BASE_URL`, `MDQM_MODEL__MODEL` | your on-prem OpenAI-compatible endpoint and the model name |
| `MDQM_SOURCES` | the JSON list of `SourceConfig` objects; `[]` means "no sources yet", which is honest but useless to a caller |
| `host` in `route.yaml` (or the Ingress) | optional: see section 6 for reading back the router's derived host |

`MDQM_SOURCES` is a one-line JSON array because the server reads settings before any YAML file
exists on disk. Each entry carries a `secret` **name**, never a credential:

```
[{"name":"crm","adapter":"postgresql","secret":"crm_ro","options":{"dsn":"postgresql://mdqm_ro@pg.internal:5432/crm"}}]
```

The DSN holds the user and the host and no password; the password is the secret file `crm_ro`. After
editing, `oc apply -k deploy/openshift` again and let the `Recreate` rollout finish.

## 6. Check it

```
oc get pods -n mdqm -w
oc logs deploy/mdqm -n mdqm
oc get route mdqm -n mdqm -o jsonpath='{.spec.host}'
```

`route.yaml` sets no `host`, so the router derives one (`mdqm-<namespace>.apps.<cluster>`). That is
the hostname to curl; add `spec.host` to the Route once the org's DNS name is agreed, rather than
guessing one that never resolves. Expect the pod to sit in `Running` but not `Ready` for a few
seconds while the startup probe gives the store migration time.

The startup log lines to expect (they contain no token): `loaded static tokens for N user(s)` from
the verifier, then `serving mdqm <version> on 0.0.0.0:8765 for N token(s)`. A pod that cannot read
the token file exits immediately with a message naming the path; that is the intended loud failure,
not a bug.

Liveness and readiness both probe `GET /healthz`, which is public on purpose so a probe needs no
Secret:

```
curl -sS https://mdqm.apps.PLACEHOLDER.internal/healthz
# {"status":"ok","name":"mdqm","version":"..."}
```

Then prove the auth path, which is the part a probe does not cover:

```
curl -i -sS -X POST https://mdqm.apps.PLACEHOLDER.internal/mcp \
  -H 'Content-Type: application/json' \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}'
# HTTP/1.1 401 and WWW-Authenticate: Bearer error="invalid_token"
```

and with `-H "Authorization: Bearer <a real token>"` the same call must not be 401. The MCP endpoint
is streamable HTTP on `/mcp`; the Route and the Ingress raise the edge timeout because one tool call
can hold a response stream open longer than the router's default 30–60 seconds.

## 7. Where the state lives, and how to back it up

Everything durable sits on `mdqm-data`, mounted at `/data`:

| Path | Content | Back up? |
|---|---|---|
| `/data/mdqm.sqlite` (+ `-wal`, `-shm`) | query history, audit hash chain, review queue, metadata | **yes**, this is the only irreplaceable data |
| `/data/cache` | DuckDB segment cache, LLM response cache, digest pages | no: regenerated on demand |
| `/data/semantic` | the semantic layer; `mdqm review approve` writes here | yes, it is curated work |

`/tmp` is an `emptyDir`: DuckDB spills large joins there, because the spill directory is a
constructor argument today and not a setting. Losing it costs a refetch, never data.

Copy the SQLite file with the server stopped, so the `-wal` is not moving under you:

```
oc scale deploy/mdqm --replicas=0 -n mdqm
oc rsync <backup-pod>:/data/mdqm.sqlite ./backup/
oc scale deploy/mdqm --replicas=1 -n mdqm
```

Grow the claim without touching the deployment:

```
oc patch pvc mdqm-data -n mdqm -p '{"spec":{"resources":{"requests":{"storage":"20Gi"}}}}'
```

(The storage class must allow expansion; most do.)

## 8. One replica is a hard rule

`replicas: 1` and `strategy: Recreate` are correctness, not tuning:

* The metadata DB is **SQLite in WAL mode on a ReadWriteOnce claim**. Two processes on that file
  fight over the WAL lock, and the audit chain is an append-only hash chain, so interleaved writers
  corrupt the guarantee the chain exists to provide.
* `Recreate` means the old pod is gone before the new one starts. A default `RollingUpdate` would
  briefly run two pods on the same claim: a deadlock or a corrupt file, at 3 a.m., during an
  upgrade.
* No HorizontalPodAutoscaler is shipped, and none should be added.

So an upgrade costs a short outage, seconds to a minute. Plan it in a quiet window; the query cache
survives (it is on the claim), open MCP sessions do not. Throughput comes from the governor's
limits and from caching, not from pods. A second organisation gets its own namespace, Service and
claim.

## 9. Upgrade and roll back

```
# 1. back up /data/mdqm.sqlite (section 7)
# 2. set the new tag in deploy/openshift/kustomization.yaml
oc apply -k deploy/openshift
oc rollout status deploy/mdqm -n mdqm
oc logs deploy/mdqm -n mdqm
```

`oc apply -k` changes the pod template, so the `Recreate` strategy replaces the pod. Then re-run the
`/healthz` and 401 checks from section 6.

Rollback of the image (`oc rollout undo deploy/mdqm`) restores the previous pod template. Treat it as
unsafe if the new image already migrated the store: store migrations are forward-only, so a rollback
would run old code against a newer schema. Restore the backed-up DB instead — that is the real
rollback.

## 10. Known limitations of this deployment

* **Only one instance per namespace.** See section 8.
* **The advertised auth metadata URL is wrong behind the edge.** `mdqm serve` builds its issuer from
  the bind address (`http://0.0.0.0:8765`), not from the Route hostname. Bearer tokens still work,
  because the verifier does not check the resource; an OAuth discovery client that follows the
  advertised issuer will get an internal address. Fixing it needs a `public_url` setting, which does
  not exist yet.
* **No NetworkPolicy is shipped.** Add one that admits only the router namespace plus egress to the
  configured sources and the model endpoint, if your cluster enforces them.
* **No PodDisruptionBudget, no HPA, no topology spread.** All three conflict with the single writer.
* **The manifests assume the image is group-writable for GID 0.** With an image that still owns
  `/data` as a fixed user, follow section 3's second row or the pod crash-loops on the volume.
* **The identity model is the static dev token file** (`TokenVerifier` + `MDQM_TOKENS_FILE`). A real
  directory, OAuth/OIDC or a proxy identity header is a different `TokenVerifier`, not a manifest
  change (AGENTS.md section 9).
* **Nothing here has been run against a cluster.** The manifests are checked by
  `tests/unit/test_deploy_manifests.py` (structure, cross references, no literal credential); no pod
  was scheduled, no claim bound, no Route answered, no SCC granted.
