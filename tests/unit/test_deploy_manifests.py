"""File-level checks for the OpenShift/Kubernetes manifests in ``deploy/openshift/``.

Nothing here starts a cluster: this machine has neither a container runtime nor ``oc``/``kubectl``
(checked; see ``progress/OpenShift.md``). The tests are the only automated proof the manifests are
well formed, cross-reference each other correctly, and carry no literal credential. They cannot
prove a pod schedules, a claim binds, a Route answers or an SCC is granted.
"""

from __future__ import annotations

import re
from itertools import chain
from pathlib import Path
from typing import Any

import pytest
from ruamel.yaml import YAML

REPO_ROOT = Path(__file__).resolve().parents[2]
DEPLOY_DIR = REPO_ROOT / "deploy" / "openshift"

#: Not a cluster object: kustomize reads it, so it has no metadata.name.
KUSTOMIZATION = "kustomization.yaml"

#: The template that must never be applied. Kept out of the resource list on purpose.
SECRET_TEMPLATE = "secret.example.yaml"

#: Name in the Deployment that an operator replaces with a real image reference.
IMAGE_PLACEHOLDER = "IMAGE_PLACEHOLDER"

#: Container port and Service port the probes and the Service must agree on.
SERVICE_PORT_NAME = "http"
SERVICE_PORT = 8765

#: The UID/GID the image's Dockerfile bakes in today (useradd --uid 10001 --gid 10001).
#: The manifests do NOT pin it: an OpenShift namespace assigns its own UID with GID 0, and the
#: image is expected to be group-writable for group 0. See docs/deploy.md section 3.
IMAGE_UID = 10001
IMAGE_GID = 10001

#: Env names that carry a credential, so they must come from a Secret.
CREDENTIAL_ENV = re.compile(r"(TOKEN|PASSWORD|PASSWD|API_KEY|SECRET_KEY|ACCESS_KEY)")
#: A path to a secret, not a secret: these literals are legitimate.
PATH_ENV = re.compile(r"(_FILE|_DIR)$")

#: ``Bearer <token>`` never belongs in a manifest: the header exists only at runtime.
BEARER = re.compile(r"Bearer\s+\S", re.IGNORECASE)
#: A base64-looking blob is what a leaked key looks like.
BASE64_BLOB = re.compile(r"^[A-Za-z0-9+/]{40,}={0,2}$")
#: The only values a template or placeholder may hold.
PLACEHOLDER = re.compile(r"CHANGE_?ME|PLACEHOLDER", re.IGNORECASE)


def load_raw(path: Path) -> Any:
    """Parse one YAML file. ``typ="safe"`` matches the loader the server itself uses."""
    return YAML(typ="safe").load(path.read_text(encoding="utf-8"))


def manifest_paths() -> list[Path]:
    return sorted(p for p in DEPLOY_DIR.glob("*.yaml"))


def manifests() -> dict[str, dict[str, Any]]:
    """Every cluster object in the directory, keyed by file name."""
    return {
        path.name: load_raw(path)
        for path in manifest_paths()
        if path.name != KUSTOMIZATION
    }


def by_kind(kind: str) -> dict[str, dict[str, Any]]:
    return {
        name: doc for name, doc in manifests().items() if doc.get("kind") == kind
    }


def one(kind: str) -> dict[str, Any]:
    """The only object of ``kind``; the deployment ships exactly one of each."""
    found = by_kind(kind)
    assert len(found) == 1, f"expected exactly one {kind}, found {sorted(found)}"
    return next(iter(found.values()))


def kustomization() -> dict[str, Any]:
    return load_raw(DEPLOY_DIR / KUSTOMIZATION)  # type: ignore[no-any-return]


def container() -> dict[str, Any]:
    pod = one("Deployment")["spec"]["template"]["spec"]
    containers = pod["containers"]
    assert len(containers) == 1, "the deployment is one process, not a pod of many"
    return containers[0]  # type: ignore[no-any-return]


def pod_spec() -> dict[str, Any]:
    return one("Deployment")["spec"]["template"]["spec"]  # type: ignore[no-any-return]


def scalars(node: Any, path: str = "") -> list[tuple[str, str]]:
    """Every string leaf with its dotted path, for the credential scan."""
    if isinstance(node, dict):
        return list(
            chain.from_iterable(
                scalars(value, f"{path}.{key}" if path else str(key))
                for key, value in node.items()
            )
        )
    if isinstance(node, list):
        return list(
            chain.from_iterable(
                scalars(item, f"{path}[{index}]") for index, item in enumerate(node)
            )
        )
    return [] if node is None else [(path, str(node))]


def volume_mounts() -> dict[str, str]:
    """Mount name -> mountPath, so paths in the ConfigMap can be checked against them."""
    return {mount["name"]: mount["mountPath"] for mount in container()["volumeMounts"]}


# --- every file is a cluster object -------------------------------------------------------


def test_the_directory_has_the_expected_files() -> None:
    names = {path.name for path in manifest_paths()}
    assert {
        "deployment.yaml",
        "pvc.yaml",
        "service.yaml",
        "route.yaml",
        "ingress.yaml",
        "configmap.yaml",
        SECRET_TEMPLATE,
        "serviceaccount.yaml",
        KUSTOMIZATION,
    } <= names


@pytest.mark.parametrize("path", manifest_paths(), ids=lambda path: path.name)
def test_files_are_utf8_lf_and_parse(path: Path) -> None:
    raw = path.read_bytes()
    raw.decode("utf-8")  # a stray cp1252 byte would break an editor or a diff
    assert b"\r\n" not in raw, f"{path.name}: CRLF line endings"
    assert raw.endswith(b"\n"), f"{path.name}: missing trailing newline"


def test_every_cluster_object_names_itself() -> None:
    for name, doc in manifests().items():
        assert doc.get("apiVersion"), f"{name}: no apiVersion"
        assert doc.get("kind"), f"{name}: no kind"
        metadata = doc.get("metadata") or {}
        assert metadata.get("name"), f"{name}: no metadata.name"


# --- Deployment: single writer, hardened, probed ------------------------------------------


def test_deployment_is_a_single_recreate_writer() -> None:
    """SQLite in WAL mode on a ReadWriteOnce claim tolerates exactly one pod."""
    spec = one("Deployment")["spec"]
    assert spec["replicas"] == 1, "two pods would fight over the SQLite file on the claim"
    assert spec["strategy"]["type"] == "Recreate", "RollingUpdate would run two writers"
    assert "HorizontalPodAutoscaler" not in {
        doc.get("kind") for doc in manifests().values()
    }, "an HPA would break the single-writer guarantee"


def test_deployment_requests_no_api_token() -> None:
    assert pod_spec()["automountServiceAccountToken"] is False


def test_deployment_lets_the_platform_assign_the_uid() -> None:
    """A pinned UID forces the nonroot-v2 SCC; leaving it out works with restricted-v2.

    The image is group-writable for GID 0, so an arbitrary namespace UID with GID 0 can still
    write /data and /tmp. Pinning runAsUser here would silently require a wider SCC grant.
    """
    pod_context = pod_spec()["securityContext"]
    assert pod_context["runAsNonRoot"] is True
    for pinned in ("runAsUser", "runAsGroup", "fsGroup"):
        assert pinned not in pod_context, f"{pinned} would force the nonroot-v2 SCC"
    assert pod_context["seccompProfile"]["type"] == "RuntimeDefault"


def test_container_is_hardened() -> None:
    context = container()["securityContext"]
    assert context["allowPrivilegeEscalation"] is False
    assert context["readOnlyRootFilesystem"] is True
    assert context["capabilities"]["drop"] == ["ALL"]


def test_container_writes_only_to_mounted_paths() -> None:
    """readOnlyRootFilesystem means every writable path must be a volume."""
    mounts = volume_mounts()
    assert {"data", "tmp", "secrets"} <= set(mounts)
    assert mounts["data"] == "/data", "the PVC holds the SQLite DB and the cache"
    assert mounts["tmp"] == "/tmp", "DuckDB spills into TMPDIR"  # noqa: S108 - in the pod
    assert mounts["secrets"] == "/run/secrets", "MDQM_TOKENS_FILE and MDQM_SECRETS__DIR"
    claims = {
        volume["name"]: volume["persistentVolumeClaim"]["claimName"]
        for volume in pod_spec()["volumes"]
        if "persistentVolumeClaim" in volume
    }
    assert claims == {"data": one("PersistentVolumeClaim")["metadata"]["name"]}


def test_container_declares_resources_and_the_served_port() -> None:
    resources = container()["resources"]
    assert resources["requests"]["cpu"] and resources["requests"]["memory"]
    assert resources["limits"]["cpu"] and resources["limits"]["memory"]
    ports = container()["ports"]
    assert [port["name"] for port in ports] == [SERVICE_PORT_NAME]
    assert ports[0]["containerPort"] == SERVICE_PORT


def test_container_has_a_writable_working_directory_for_duckdb() -> None:
    """DuckDB spills relative to the working directory; / is read-only, /tmp is a volume."""
    assert container()["workingDir"] == volume_mounts()["tmp"]
    assert one("ConfigMap")["data"]["TMPDIR"] == container()["workingDir"]


@pytest.mark.parametrize("probe", ["startupProbe", "readinessProbe", "livenessProbe"])
def test_probes_hit_the_public_health_endpoint(probe: str) -> None:
    """/healthz needs no token (src/mdqm/server/app.py), so a probe needs no Secret."""
    http_get = container()[probe]["httpGet"]
    assert http_get["path"] == "/healthz"
    assert http_get["port"] == SERVICE_PORT_NAME, "named port, not a duplicated number"


def test_only_the_startup_probe_covers_a_slow_first_start() -> None:
    """Liveness must not restart the pod while a fresh claim is still migrating."""
    assert container()["startupProbe"]["failureThreshold"] >= 12, "at least a minute of grace"
    assert "initialDelaySeconds" not in container()["livenessProbe"]


def test_env_comes_from_the_configmap_not_from_literals() -> None:
    config_map = one("ConfigMap")["metadata"]["name"]
    assert container()["envFrom"] == [{"configMapRef": {"name": config_map}}]


# --- no literal credential anywhere --------------------------------------------------------


def test_no_value_is_a_bearer_header_or_a_blob() -> None:
    for name, doc in manifests().items():
        for path, value in scalars(doc):
            assert not BEARER.search(value), f"{name}: {path} holds a Bearer header"
            assert not BASE64_BLOB.match(value), f"{name}: {path} looks like an encoded key"


def test_credential_env_uses_value_from_the_secret() -> None:
    for entry in container().get("env", []):
        env_name = entry["name"]
        if not CREDENTIAL_ENV.search(env_name) or PATH_ENV.search(env_name):
            continue
        assert "value" not in entry, f"{env_name} carries a literal credential"
        assert entry["valueFrom"]["secretKeyRef"]["name"] == one("Secret")["metadata"]["name"]


def test_configmap_holds_paths_and_limits_only() -> None:
    data = one("ConfigMap")["data"]
    for key, value in data.items():
        if CREDENTIAL_ENV.search(key):
            assert PATH_ENV.search(key), f"{key} looks like a credential, not a path"
            assert value.startswith("/"), f"{key} must be a path, got {value!r}"
    # The JSON source list may name secrets but must never carry one.
    assert not BEARER.search(data["MDQM_SOURCES"])
    assert data["MDQM_SOURCES"] == "[]", "an empty default: sources are organisation facts"


def test_secret_template_has_only_placeholders() -> None:
    secret = one("Secret")
    assert secret["metadata"]["name"] == "mdqm-secrets"
    values = dict(scalars(secret.get("stringData") or secret.get("data")))
    assert values, "the template must show each key the pod reads"
    for key, value in values.items():
        assert PLACEHOLDER.search(value), f"{key} must be a CHANGEME placeholder"
    assert "tokens.json" in values, "MDQM_TOKENS_FILE names this key"
    assert '"tokens"' in values["tokens.json"], "the token file shape: {\"tokens\": [...]}"


def test_the_secret_template_is_never_applied() -> None:
    resources = kustomization()["resources"]
    assert SECRET_TEMPLATE not in resources, "applying the template would deploy a public token"
    assert "secret.example.yaml" not in str(resources)


# --- cross references ----------------------------------------------------------------------


def declared_names(kind: str) -> set[str]:
    return {doc["metadata"]["name"] for doc in by_kind(kind).values()}


def test_every_referenced_object_exists_in_the_directory() -> None:
    configmaps = declared_names("ConfigMap")
    secrets = declared_names("Secret")
    pvcs = declared_names("PersistentVolumeClaim")
    accounts = declared_names("ServiceAccount")

    assert pod_spec()["serviceAccountName"] in accounts
    for ref in container()["envFrom"]:
        assert ref["configMapRef"]["name"] in configmaps
    for entry in container().get("env", []):
        assert entry["valueFrom"]["secretKeyRef"]["name"] in secrets
    for volume in pod_spec()["volumes"]:
        if "persistentVolumeClaim" in volume:
            assert volume["persistentVolumeClaim"]["claimName"] in pvcs
        if "secret" in volume:
            assert volume["secret"]["secretName"] in secrets


def test_service_selects_the_deployment_pod_and_targets_its_port() -> None:
    service = one("Service")
    labels = one("Deployment")["spec"]["template"]["metadata"]["labels"]
    assert service["spec"]["selector"].items() <= labels.items()
    port = service["spec"]["ports"][0]
    assert port["targetPort"] == SERVICE_PORT_NAME
    assert [p for p in container()["ports"] if p["name"] == port["targetPort"]]
    assert port["port"] == SERVICE_PORT


def test_route_and_ingress_both_point_at_the_service() -> None:
    service_name = one("Service")["metadata"]["name"]
    route = one("Route")
    assert route["spec"]["to"]["name"] == service_name
    assert route["spec"]["port"]["targetPort"] == SERVICE_PORT_NAME
    ingress = one("Ingress")
    backend = ingress["spec"]["rules"][0]["http"]["paths"][0]["backend"]
    assert backend["service"]["name"] == service_name
    assert backend["service"]["port"]["name"] == SERVICE_PORT_NAME


def test_route_terminates_tls_at_the_edge() -> None:
    route = one("Route")
    assert route["spec"]["tls"]["termination"] == "edge"
    assert route["spec"]["tls"]["insecureEdgeTerminationPolicy"] == "Redirect"
    # A streamable-HTTP response outlives the router's 30s default idle timeout.
    assert route["metadata"]["annotations"]["haproxy.router.openshift.io/timeout"]
    # No host is set: the router derives one the operator must read back with
    # `oc get route mdqm -o jsonpath='{.spec.host}'`. A guessed host never resolves.
    assert "host" not in route["spec"]


def test_ingress_is_the_alternative_not_a_second_edge() -> None:
    resources = kustomization()["resources"]
    assert "route.yaml" in resources, "an OpenShift cluster wants the Route"
    assert "ingress.yaml" not in resources, "on plain Kubernetes, swap the two by hand"
    assert one("Ingress")["spec"]["tls"][0]["secretName"], "the operator creates the TLS Secret"


def test_pvc_is_read_write_once_with_a_pinned_class_and_a_size() -> None:
    """A cluster default class can differ in reclaim policy: the operator pins it."""
    spec = one("PersistentVolumeClaim")["spec"]
    assert spec["accessModes"] == ["ReadWriteOnce"]
    assert spec["resources"]["requests"]["storage"]
    assert spec["storageClassName"], "pin the class instead of trusting the cluster default"
    assert PLACEHOLDER.search(spec["storageClassName"]), "a placeholder, not a guessed class"


# --- paths in the ConfigMap match the mounts -----------------------------------------------


def test_configmap_paths_live_on_the_mounted_volumes() -> None:
    data = one("ConfigMap")["data"]
    mounts = volume_mounts()
    assert data["MDQM_PATHS__METADATA_DB"].startswith(mounts["data"] + "/")
    assert data["MDQM_PATHS__CACHE_DIR"].startswith(mounts["data"] + "/")
    assert data["MDQM_PATHS__SEMANTIC_DIR"].startswith(mounts["data"] + "/")
    assert data["MDQM_SECRETS__DIR"] == mounts["secrets"]
    assert data["MDQM_TOKENS_FILE"].startswith(mounts["secrets"] + "/")
    assert data["TMPDIR"] == mounts["tmp"]
    assert data["MDQM_SERVER__PORT"] == str(SERVICE_PORT)
    # Pod network, not a host network: the Service is the only way in.
    assert data["MDQM_SERVER__HOST"] == "0.0.0.0"  # noqa: S104 - in the pod


# --- kustomize -----------------------------------------------------------------------------


def test_kustomization_replaces_the_image_placeholder() -> None:
    image = one("Deployment")["spec"]["template"]["spec"]["containers"][0]["image"]
    assert image == IMAGE_PLACEHOLDER
    overrides = {entry["name"]: entry for entry in kustomization()["images"]}
    assert IMAGE_PLACEHOLDER in overrides, "the operator sets the image once, here"
    assert overrides[IMAGE_PLACEHOLDER]["newTag"] != "latest"


def test_kustomization_lists_only_files_that_exist() -> None:
    for resource in kustomization()["resources"]:
        assert (DEPLOY_DIR / resource).is_file(), f"missing resource {resource}"
