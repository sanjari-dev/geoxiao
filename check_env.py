#!/usr/bin/env python3
"""
Geoxiao environment readiness checker.

Checks:
- Hardware sizing for Ray/NautilusTrader workloads.
- OS and Python version.
- .env presence and basic loading from .env.example.
- Installed dependencies vs pyproject.toml.
- Read-only ClickHouse ping and PostgreSQL ping.

Run from the project root:
    python check_env.py
"""
from __future__ import annotations

import asyncio
import os
import platform
import shutil
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import quote, urlparse, urlunparse

try:
    from importlib import metadata as importlib_metadata
except Exception:  # pragma: no cover
    import importlib_metadata  # type: ignore

ROOT = Path(__file__).resolve().parent
PYPROJECT = ROOT / "pyproject.toml"
ENV_FILE = ROOT / ".env"
ENV_EXAMPLE = ROOT / ".env.example"

MIN_RECOMMENDED_RAM_GB = 16
GOOD_RAY_RAM_GB = 32
MIN_RECOMMENDED_CORES = 4
DB_CONNECT_TIMEOUT_SECONDS = 5


@dataclass
class CheckState:
    warnings: int = 0
    failures: int = 0

    def ok(self, message: str) -> None:
        print(f"[OK]   {message}")

    def info(self, message: str) -> None:
        print(f"[INFO] {message}")

    def warn(self, message: str) -> None:
        self.warnings += 1
        print(f"[WARN] {message}")

    def fail(self, message: str) -> None:
        self.failures += 1
        print(f"[FAIL] {message}")


def bytes_to_gb(value: int | float | None) -> float | None:
    if value is None:
        return None
    return float(value) / (1024 ** 3)


def check_hardware(state: CheckState) -> None:
    print("\n=== Hardware ===")
    logical_cores = os.cpu_count() or 0
    if logical_cores >= MIN_RECOMMENDED_CORES:
        state.ok(f"Logical CPU cores: {logical_cores}")
    else:
        state.warn(
            f"Logical CPU cores: {logical_cores}; Ray workloads may be bottlenecked. "
            f"Recommended >= {MIN_RECOMMENDED_CORES}."
        )

    ram_gb = None
    psutil_available = False
    try:
        import psutil  # type: ignore

        psutil_available = True
        vm = psutil.virtual_memory()
        ram_gb = bytes_to_gb(vm.total)
        avail_gb = bytes_to_gb(vm.available)
        state.info(f"RAM total: {ram_gb:.2f} GiB; available now: {avail_gb:.2f} GiB")
        if ram_gb < MIN_RECOMMENDED_RAM_GB:
            state.warn(
                f"Total RAM below {MIN_RECOMMENDED_RAM_GB} GiB; high OOM risk for Ray/NautilusTrader."
            )
        elif ram_gb < GOOD_RAY_RAM_GB:
            state.warn(
                f"Total RAM below {GOOD_RAY_RAM_GB} GiB; keep Ray parallelism conservative."
            )
        else:
            state.ok("RAM capacity looks suitable for distributed backtests, subject to dataset size.")
    except ImportError:
        state.warn("psutil is not installed; RAM check is limited. Add psutil for better hardware diagnostics.")

    try:
        disk = shutil.disk_usage(ROOT)
        free_gb = bytes_to_gb(disk.free)
        total_gb = bytes_to_gb(disk.total)
        state.info(f"Disk at {ROOT.anchor or ROOT}: {free_gb:.2f} GiB free / {total_gb:.2f} GiB total")
        if free_gb is not None and free_gb < 10:
            state.warn("Disk free space is below 10 GiB; logs/checkpoints may hit I/O or space issues.")
    except Exception as exc:
        state.warn(f"Could not inspect disk usage: {exc}")

    if not psutil_available:
        state.info("Install psutil if you want accurate available-RAM readings: pip install psutil")


def check_os_python(state: CheckState) -> None:
    print("\n=== OS & Python ===")
    system = platform.system()
    state.info(f"OS: {system} {platform.release()} ({platform.platform()})")
    if system not in {"Linux", "Darwin"}:
        state.warn("Project is expected to run on Linux/macOS; Windows may have Ray/NautilusTrader limitations.")
    else:
        state.ok("OS family is Linux/macOS.")

    version = sys.version_info
    state.info(f"Python executable: {sys.executable}")
    state.info(f"Python version: {platform.python_version()}")
    if version.major == 3 and version.minor == 12:
        state.ok("Python version is 3.12.x as requested.")
    else:
        state.fail("Python version is not 3.12.x. Use Python 3.12 before running the main pipeline.")


def parse_env_file(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if " #" in value:
            value = value.split(" #", 1)[0].strip()
        if value and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        env[key] = value
    return env


def ensure_env_file(state: CheckState) -> dict[str, str]:
    print("\n=== Environment Variables ===")
    if ENV_FILE.exists():
        state.ok(f"Found {ENV_FILE.name}")
    elif ENV_EXAMPLE.exists():
        shutil.copyfile(ENV_EXAMPLE, ENV_FILE)
        state.warn(f"{ENV_FILE.name} was missing; copied from {ENV_EXAMPLE.name}. Review secrets before production use.")
    else:
        state.fail(f"Neither {ENV_FILE.name} nor {ENV_EXAMPLE.name} exists.")
        return {}

    file_env = parse_env_file(ENV_FILE)
    merged = dict(file_env)
    for key in list(file_env):
        if key in os.environ:
            merged[key] = os.environ[key]

    required = [
        "CH_HOST",
        "CH_PORT",
        "CH_DATABASE",
        "CH_USER",
        "PG_HOST",
        "PG_PORT",
        "PG_DATABASE",
        "PG_USER",
        "PG_DSN",
    ]
    missing = [key for key in required if not merged.get(key)]
    if missing:
        state.fail("Missing required env keys: " + ", ".join(missing))
    else:
        state.ok("Required ClickHouse/PostgreSQL env keys are present.")

    state.info(
        "Database targets: "
        f"ClickHouse={merged.get('CH_USER', '<missing>')}@{merged.get('CH_HOST', '<missing>')}:{merged.get('CH_PORT', '<missing>')}/"
        f"{merged.get('CH_DATABASE', '<missing>')}; "
        f"PostgreSQL={merged.get('PG_USER', '<missing>')}@{merged.get('PG_HOST', '<missing>')}:{merged.get('PG_PORT', '<missing>')}/"
        f"{merged.get('PG_DATABASE', '<missing>')}"
    )
    return merged


def normalize_distribution_name(name: str) -> str:
    return name.lower().replace("_", "-")


def read_pyproject_requirements(state: CheckState) -> list[str]:
    if not PYPROJECT.exists():
        state.fail("pyproject.toml not found.")
        return []
    with PYPROJECT.open("rb") as fh:
        data = tomllib.load(fh)
    project = data.get("project", {})
    deps = list(project.get("dependencies", []))
    state.info(f"Found {len(deps)} runtime dependencies in pyproject.toml.")
    return deps


def installed_packages() -> dict[str, str]:
    packages: dict[str, str] = {}
    for dist in importlib_metadata.distributions():
        name = dist.metadata.get("Name")
        if name:
            packages[normalize_distribution_name(name)] = dist.version
    return packages


def parse_requirement(req_text: str):
    try:
        from packaging.requirements import Requirement

        return Requirement(req_text)
    except Exception:
        return None


def fallback_requirement_name(req_text: str) -> str:
    name = req_text.split(";", 1)[0].strip()
    for sep in ["[", "<", ">", "=", "!", "~"]:
        name = name.split(sep, 1)[0].strip()
    return name


def check_dependencies(state: CheckState) -> None:
    print("\n=== Dependencies ===")
    deps = read_pyproject_requirements(state)
    installed = installed_packages()

    missing: list[str] = []
    incompatible: list[str] = []
    checked = 0
    packaging_available = True

    for dep in deps:
        req = parse_requirement(dep)
        if req is None:
            packaging_available = False
            name = fallback_requirement_name(dep)
            normalized = normalize_distribution_name(name)
            version = installed.get(normalized)
            checked += 1
            if version is None:
                missing.append(dep)
            continue

        normalized = normalize_distribution_name(req.name)
        version = installed.get(normalized)
        checked += 1
        if version is None:
            missing.append(str(req))
            continue
        try:
            if req.specifier and version not in req.specifier:
                incompatible.append(f"{req.name} installed={version}, required='{req.specifier}'")
        except Exception as exc:
            state.warn(f"Could not evaluate version for {req.name}: installed={version}, requirement={req}; {exc}")

    if not packaging_available:
        state.warn("packaging is not available; dependency version comparison was limited.")

    if missing:
        state.fail("Missing dependencies:")
        for item in missing:
            print(f"       - {item}")
    else:
        state.ok(f"No missing runtime dependencies among {checked} checked packages.")

    if incompatible:
        state.fail("Dependencies with incompatible versions:")
        for item in incompatible:
            print(f"       - {item}")
    else:
        state.ok("Installed dependency versions satisfy pyproject.toml runtime specifiers.")

    try:
        freeze = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if freeze.returncode == 0:
            count = len([line for line in freeze.stdout.splitlines() if line.strip()])
            state.info(f"pip freeze succeeded; {count} installed distributions visible in this environment.")
        else:
            state.warn(f"pip freeze failed: {freeze.stderr.strip() or freeze.stdout.strip()}")
    except Exception as exc:
        state.warn(f"Could not run pip freeze: {exc}")


def postgres_asyncpg_dsn(env: dict[str, str]) -> str:
    dsn = env.get("PG_DSN", "").strip()
    if dsn:
        # asyncpg accepts postgresql://, not SQLAlchemy's postgresql+asyncpg:// URL.
        if dsn.startswith("postgresql+asyncpg://"):
            dsn = "postgresql://" + dsn[len("postgresql+asyncpg://") :]
        elif dsn.startswith("postgres+asyncpg://"):
            dsn = "postgres://" + dsn[len("postgres+asyncpg://") :]
        return dsn

    user = quote(env.get("PG_USER", ""))
    password = quote(env.get("PG_PASSWORD", ""))
    host = env.get("PG_HOST", "localhost")
    port = env.get("PG_PORT", "5432")
    database = env.get("PG_DATABASE", "postgres")
    auth = user if not password else f"{user}:{password}"
    return f"postgresql://{auth}@{host}:{port}/{database}"


async def ping_postgres(env: dict[str, str], state: CheckState) -> None:
    print("\n=== PostgreSQL Ping ===")
    try:
        import asyncpg  # type: ignore
    except ImportError:
        state.fail("asyncpg is not installed; cannot test PostgreSQL connection.")
        return

    dsn = postgres_asyncpg_dsn(env)
    safe = urlparse(dsn)
    safe_netloc = safe.hostname or "<missing-host>"
    if safe.port:
        safe_netloc += f":{safe.port}"
    state.info(f"Connecting to PostgreSQL at {safe_netloc}{safe.path or ''}")
    conn = None
    try:
        conn = await asyncio.wait_for(asyncpg.connect(dsn=dsn), timeout=DB_CONNECT_TIMEOUT_SECONDS)
        value = await asyncio.wait_for(conn.fetchval("SELECT 1"), timeout=DB_CONNECT_TIMEOUT_SECONDS)
        if value == 1:
            state.ok("PostgreSQL SELECT 1 succeeded.")
        else:
            state.fail(f"PostgreSQL ping returned unexpected value: {value!r}")
    except Exception as exc:
        state.fail(f"PostgreSQL connection/ping failed: {type(exc).__name__}: {exc}")
    finally:
        if conn is not None:
            try:
                await conn.close()
            except Exception:
                pass


async def ping_clickhouse_with_aiohttp(env: dict[str, str]) -> str:
    import aiohttp  # type: ignore

    host = env.get("CH_HOST", "localhost")
    port = env.get("CH_PORT", "8123")
    database = env.get("CH_DATABASE", "default")
    user = env.get("CH_USER", "default")
    password = env.get("CH_PASSWORD", "")
    scheme = env.get("CH_SCHEME", "http")
    url = f"{scheme}://{host}:{port}/"
    params = {
        "database": database,
        "user": user,
        "readonly": "1",
    }
    if password:
        params["password"] = password
    timeout = aiohttp.ClientTimeout(total=DB_CONNECT_TIMEOUT_SECONDS)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(url, params=params, data="SELECT 1") as response:
            text = (await response.text()).strip()
            if response.status >= 400:
                raise RuntimeError(f"HTTP {response.status}: {text[:300]}")
            return text


async def ping_clickhouse_with_connect(env: dict[str, str]) -> str:
    import clickhouse_connect  # type: ignore

    def _sync_query() -> str:
        client = clickhouse_connect.get_client(
            host=env.get("CH_HOST", "localhost"),
            port=int(env.get("CH_PORT", "8123")),
            username=env.get("CH_USER", "default"),
            password=env.get("CH_PASSWORD", ""),
            database=env.get("CH_DATABASE", "default"),
            settings={"readonly": 1},
            connect_timeout=DB_CONNECT_TIMEOUT_SECONDS,
            send_receive_timeout=DB_CONNECT_TIMEOUT_SECONDS,
        )
        try:
            result = client.query("SELECT 1")
            return str(result.result_rows[0][0]) if result.result_rows else ""
        finally:
            try:
                client.close()
            except Exception:
                pass

    return await asyncio.wait_for(asyncio.to_thread(_sync_query), timeout=DB_CONNECT_TIMEOUT_SECONDS + 2)


async def ping_clickhouse(env: dict[str, str], state: CheckState) -> None:
    print("\n=== ClickHouse Ping (READ-ONLY) ===")
    state.info(
        f"Connecting to ClickHouse HTTP at {env.get('CH_HOST', 'localhost')}:{env.get('CH_PORT', '8123')}/"
        f"{env.get('CH_DATABASE', 'default')} with readonly=1"
    )
    try:
        try:
            text = await ping_clickhouse_with_aiohttp(env)
            method = "aiohttp"
        except ImportError:
            text = await ping_clickhouse_with_connect(env)
            method = "clickhouse-connect"
        if text.strip() == "1":
            state.ok(f"ClickHouse SELECT 1 succeeded via {method}; no write operation attempted.")
        else:
            state.fail(f"ClickHouse ping returned unexpected response via {method}: {text!r}")
    except Exception as exc:
        state.fail(f"ClickHouse connection/ping failed: {type(exc).__name__}: {exc}")


async def run_db_checks(env: dict[str, str], state: CheckState) -> None:
    await asyncio.gather(
        ping_clickhouse(env, state),
        ping_postgres(env, state),
    )


def main() -> int:
    os.chdir(ROOT)
    state = CheckState()
    print("Geoxiao v1 Environment Readiness Check")
    print(f"Project root: {ROOT}")

    check_hardware(state)
    check_os_python(state)
    env = ensure_env_file(state)
    check_dependencies(state)

    if env:
        asyncio.run(run_db_checks(env, state))
    else:
        state.fail("Skipping database pings because environment variables are unavailable.")

    print("\n=== Summary ===")
    if state.failures:
        print(f"Result: NOT READY ({state.failures} failure(s), {state.warnings} warning(s)).")
        print("Fix failures before starting Ray/NautilusTrader pipeline.")
        return 1
    if state.warnings:
        print(f"Result: READY WITH WARNINGS ({state.warnings} warning(s)).")
        print("Pipeline may run, but review warnings to reduce OOM/I/O/runtime risk.")
        return 0
    print("Result: READY (no failures or warnings).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
