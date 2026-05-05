from pathlib import Path

import nox

PACKAGE = "package"
PYTHON_VERSIONS = ["3.10", "3.11", "3.12", "3.13"]
LATEST_VERSION = PYTHON_VERSIONS[-1]
nox.needs_version = ">=2025.5.1"
nox.options.sessions = (
    "mypy",
    "tests",
    "coverage",
)

locations = (
    "src",
    "tests",
)

@nox.session(python=LATEST_VERSION, reuse_venv=True)
def lint(session: nox.Session) -> None:
    """Lint using ruff."""
    args = session.posargs or locations
    session.install("ruff")
    session.run("ruff", *args)


@nox.session(python=LATEST_VERSION, reuse_venv=True)
def mypy(session: nox.Session) -> None:
    """Type-check using mypy."""
    session.run_install(
        "uv",
        "sync",
        "--group=dev",
        f"--python={session.virtualenv.location}",
        env={"UV_PROJECT_ENVIRONMENT": session.virtualenv.location},
    )
    session.run("mypy", "src", *session.posargs)


@nox.session(python=LATEST_VERSION,venv_backend="uv")
def tests(session: nox.Session) -> None:
    """
    Run the unit and regular tests.
    """
    session.run_install(
        "uv",
        "sync",
        # "--extra=tests",
        "--no-default-groups",
        "--frozen",
        f"--python={session.virtualenv.location}",
        env={"UV_PROJECT_ENVIRONMENT": session.virtualenv.location},
    )
    # session.run("pytest", *session.posargs)
    session.run("coverage", "run", "-m", "pytest", *session.posargs)


@nox.session(python=LATEST_VERSION, venv_backend="uv",reuse_venv=True)
def coverage(session: nox.Session) -> None:
    """Produce the coverage report."""
    args = session.posargs or ["report"]
    session.run_install(
        "uv",
        "sync",
        # "--extra=tests",
        "--no-default-groups",
        "--frozen",
        f"--python={session.virtualenv.location}",
        env={"UV_PROJECT_ENVIRONMENT": session.virtualenv.location},
    )

    if not session.posargs and any(Path().glob(".coverage.*")):
        session.run("coverage", "combine")

    session.run("coverage", "json", "--fail-under=0")
    session.run("codecov", "coverage.json", *args)