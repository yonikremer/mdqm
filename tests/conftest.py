"""Shared pytest config.

Rules (see AGENTS.md):
* Unit tests (tests/unit) must not use the network. pytest-socket enforces it.
* Integration tests (tests/integration) need docker-compose services. Without services they are
  skipped, unless --require-services is given (then they fail). Acceptance criteria that mention
  integration tests always use --require-services.
"""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--require-services",
        action="store_true",
        default=False,
        help="Fail (instead of skip) integration tests when services are unreachable.",
    )
    parser.addoption(
        "--update-golden",
        action="store_true",
        default=False,
        help="Rewrite golden files under tests/data/golden instead of comparing.",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    for item in items:
        path = Path(str(item.fspath)).resolve()
        if ROOT / "unit" in path.parents:
            item.add_marker(pytest.mark.disable_socket)  # pytest-socket
        if ROOT / "integration" in path.parents:
            item.add_marker(pytest.mark.integration)


@pytest.fixture(scope="session")
def require_services(pytestconfig: pytest.Config) -> bool:
    return bool(pytestconfig.getoption("--require-services"))


def service_unavailable(require: bool, reason: str) -> None:
    """Call from integration fixtures when a service is unreachable."""
    if require:
        pytest.fail(f"required service unavailable: {reason}")
    pytest.skip(f"service unavailable: {reason}")


@pytest.fixture
def golden(pytestconfig: pytest.Config):  # type: ignore[no-untyped-def]
    """`golden("name.sql", actual_text)` compares with tests/data/golden/name.sql (LF, UTF-8).

    With --update-golden the file is (re)written. Review the diff in git before committing.
    """
    update = bool(pytestconfig.getoption("--update-golden"))
    gdir = ROOT / "data" / "golden"

    def check(name: str, actual: str) -> None:
        path = gdir / name
        if update or not path.exists():
            if not update:
                pytest.fail(f"golden file missing: {path} (run with --update-golden, then review)")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(actual, encoding="utf-8", newline="\n")
            return
        expected = path.read_text(encoding="utf-8")
        assert actual == expected, f"golden mismatch for {name}"

    return check
