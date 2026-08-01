"""Build-environment snapshotting.

A parent process cannot ``call vcvars64.bat`` (or oneAPI's ``setvars``) and
inherit the result. Instead we run them in a child shell, dump the resulting
environment, and reuse that dictionary for every cmake/ctest subprocess we
spawn. The snapshot is cached on disk keyed by the configuration that produced
it; an unchanged config skips the shell entirely.

SushiBLAS needs a real Intel oneAPI DPC++ install (icx-cl/icpx, oneMKL): on
Windows that means loading MSVC (clang-cl-flavored icx-cl still needs the MSVC
headers/libs) and then oneAPI's own environment on top; on Linux, oneAPI's
setvars.sh alone. This mirrors what scripts/win/build.bat and
scripts/linux/build.sh already do by hand.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

from . import console
from .config import Config

CACHE_NAME = ".sushi_env.json"


def _cache_key(cfg: Config) -> str:
    material = json.dumps(
        {
            "platform": cfg.platform,
            "vs_vcvars": cfg.vs_vcvars,
            "oneapi_root": cfg.resolved_oneapi_root(),
        },
        sort_keys=True,
    )
    return hashlib.sha256(material.encode()).hexdigest()


def _merge_env(base: dict[str, str], overlay: dict[str, str]) -> dict[str, str]:
    """Merge overlay into base, prepending PATH into the base's case variant.

    PATH is collapsed into the *single* case variant the base already uses (e.g.
    Windows' "Path"). Writing the overlay's "PATH" as a new key while base keeps
    "Path" leaves two conflicting entries; Windows then honours only one for child
    processes, so toolchain dirs added to the other silently vanish from DLL/exe
    resolution.
    """
    merged = dict(base)
    path_key = next((k for k in merged if k.upper() == "PATH"), None)
    for k, v in overlay.items():
        if k.upper() == "PATH":
            target = path_key or k
            existing = merged.get(target, "")
            merged[target] = v + os.pathsep + existing if existing else v
            path_key = target
        else:
            merged[k] = v
    return merged


def _parse_windows_set(output: str) -> dict[str, str]:
    env: dict[str, str] = {}
    for line in output.splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            if key:
                env[key] = value
    return env


def _snapshot_windows(cfg: Config) -> dict[str, str] | None:
    vcvars = cfg.expand(cfg.vs_vcvars)
    oneapi = cfg.resolved_oneapi_root()
    setvars = str(Path(oneapi) / "setvars.bat") if oneapi else ""

    calls = []
    if vcvars and Path(vcvars).is_file():
        console.info("Loading Visual Studio environment (vcvars64)...")
        calls.append(f'call "{vcvars}"')
    if setvars and Path(setvars).is_file():
        console.info("Loading Intel oneAPI environment (setvars)...")
        calls.append(f'call "{setvars}" intel64')
    if not calls:
        return None

    # Pass the whole command as a single string, NOT as ["cmd", "/c", script]:
    # with a list, subprocess re-quotes each element and mangles the inner quotes
    # around the (space-containing) script paths, so cmd.exe sees the quoted path
    # as one unknown token and returns non-zero — silently dropping the environment.
    #
    # Chained with plain `&`, not `&&`: oneAPI's setvars.bat commonly exits 1 even
    # on a fully successful run (it does this whenever any *optional* component it
    # probes for isn't installed), so gating the next step on its exit code drops
    # the environment it just built. `&` runs `set` regardless; success is judged
    # by whether anything was actually captured, not by the batch files' exit code.
    script = " & ".join(calls) + " & set"
    result = subprocess.run("cmd /c " + script, capture_output=True, text=True)
    env = _parse_windows_set(result.stdout)
    if not env:
        console.warn("vcvars64/setvars produced no environment; using current env.")
        return None
    return env


def _snapshot_linux(cfg: Config) -> dict[str, str] | None:
    oneapi = cfg.resolved_oneapi_root()
    setvars = str(Path(oneapi) / "setvars.sh") if oneapi else ""
    if not (setvars and Path(setvars).is_file()):
        return None
    console.info("Loading Intel oneAPI environment (setvars)...")
    script = f'source "{setvars}" --force > /dev/null 2>&1 && env'
    result = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    if result.returncode != 0:
        console.warn("setvars.sh returned non-zero; using current env.")
        return None
    return _parse_windows_set(result.stdout)  # KEY=VALUE lines; parser is shell-agnostic


def load_build_env(cfg: Config, build_dir: Path) -> dict[str, str]:
    """Return the environment for build/test/run subprocesses.

    Falls back to the current ``os.environ`` when no toolchain scripts are
    configured (e.g. when icx-cl/oneAPI is already on PATH).
    """
    build_dir.mkdir(parents=True, exist_ok=True)
    cache_file = build_dir / CACHE_NAME
    key = _cache_key(cfg)

    if cache_file.is_file():
        try:
            cached = json.loads(cache_file.read_text())
            if cached.get("key") == key:
                return _merge_env(dict(os.environ), cached["env"])
        except (json.JSONDecodeError, OSError, KeyError):
            pass  # fall through and rebuild

    snapshot = _snapshot_windows(cfg) if cfg.is_windows else _snapshot_linux(cfg)
    if not snapshot:
        return dict(os.environ)

    try:
        cache_file.write_text(json.dumps({"key": key, "env": snapshot}))
    except OSError:
        pass

    return _merge_env(dict(os.environ), snapshot)
