#!/usr/bin/env python3
"""Setup / preflight for /watch.

Modes:
  setup.py --check      Silent preflight. Exit 0 if ready, 2/3/4 on failure.
  setup.py --json       Machine-readable status for the agent to parse.
  setup.py              Installer/scaffolder.

Sandbox note:
``yt-dlp`` is intentionally an optional dependency now. Local video analysis
still needs ffmpeg/ffprobe, while public YouTube transcript-only mode can fall
back when yt-dlp is unavailable. Full remote visual frame extraction still
benefits from yt-dlp and setup reports it under ``missing_optional_binaries``.
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from config import get_config  # noqa: E402


REQUIRED_BINARIES = ["ffmpeg", "ffprobe"]
OPTIONAL_BINARIES = ["yt-dlp"]
CONFIG_DIR = Path.home() / ".config" / "watch"
CONFIG_FILE = CONFIG_DIR / ".env"
ENV_TEMPLATE = """# /watch API configuration
#
# Whisper transcription fallback — used when native captions are unavailable
# (or when you point /watch at a local file with no subtitles).
#
# Groq is preferred; OpenAI is the compatible fallback.
# Leave both blank if you do not want Whisper.

GROQ_API_KEY=
OPENAI_API_KEY=

# Public YouTube transcript-only fallback.
# Used only when native yt-dlp caption access is unavailable/failing.
# Set false to disable the third-party public transcript fallback entirely.
WATCH_PUBLIC_TRANSCRIPT_FALLBACK=true

# Default watch behavior.
# Allowed values: transcript | efficient | balanced | token-burner
# WATCH_DETAIL=balanced
"""


def _which(name: str) -> str | None:
    return shutil.which(name)


def _check_required_binaries() -> list[str]:
    return [b for b in REQUIRED_BINARIES if not _which(b)]


def _check_optional_binaries() -> list[str]:
    return [b for b in OPTIONAL_BINARIES if not _which(b)]


_PERM_WARNED: set[str] = set()


def _check_file_permissions(path: Path) -> None:
    key = str(path)
    if key in _PERM_WARNED:
        return
    try:
        mode = path.stat().st_mode
        if mode & 0o044:
            _PERM_WARNED.add(key)
            sys.stderr.write(
                f"[watch] WARNING: {path} is readable by other users. "
                f"Run: chmod 600 {path}\n"
            )
            sys.stderr.flush()
    except OSError:
        pass


def _read_env_key(name: str) -> str | None:
    value = os.environ.get(name)
    if value and value.strip():
        return value.strip()
    if not CONFIG_FILE.exists():
        return None
    _check_file_permissions(CONFIG_FILE)
    try:
        for line in CONFIG_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, raw = line.partition("=")
            if key.strip() != name:
                continue
            raw = raw.strip()
            if len(raw) >= 2 and raw[0] in ('"', "'") and raw[-1] == raw[0]:
                raw = raw[1:-1]
            return raw or None
    except OSError:
        return None
    return None


def _have_api_key() -> tuple[bool, str | None]:
    if _read_env_key("GROQ_API_KEY"):
        return True, "groq"
    if _read_env_key("OPENAI_API_KEY"):
        return True, "openai"
    return False, None


def is_first_run() -> bool:
    return _read_env_key("SETUP_COMPLETE") != "true"


def _scaffold_env() -> bool:
    if CONFIG_FILE.exists():
        return False
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(ENV_TEMPLATE, encoding="utf-8")
    try:
        CONFIG_FILE.chmod(0o600)
    except OSError:
        pass
    return True


def _set_env_value(name: str, value: str) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    if not CONFIG_FILE.exists():
        CONFIG_FILE.write_text(ENV_TEMPLATE, encoding="utf-8")
    lines = CONFIG_FILE.read_text(encoding="utf-8").splitlines()
    prefix = name + "="
    replaced = False
    out: list[str] = []
    for line in lines:
        if line.strip().startswith(prefix):
            out.append(f"{name}={value}")
            replaced = True
        else:
            out.append(line)
    if not replaced:
        out.append(f"{name}={value}")
    CONFIG_FILE.write_text("\n".join(out).rstrip() + "\n", encoding="utf-8")
    try:
        CONFIG_FILE.chmod(0o600)
    except OSError:
        pass


def _write_setup_complete() -> None:
    _set_env_value("SETUP_COMPLETE", "true")


def _brew_pkg(missing: list[str]) -> list[str]:
    pkgs: list[str] = []
    for bin_name in missing:
        if bin_name in ("ffmpeg", "ffprobe"):
            if "ffmpeg" not in pkgs:
                pkgs.append("ffmpeg")
        elif bin_name == "yt-dlp":
            if "yt-dlp" not in pkgs:
                pkgs.append("yt-dlp")
        else:
            pkgs.append(bin_name)
    return pkgs


def _install_macos(missing: list[str]) -> tuple[bool, str]:
    if _which("brew") is None:
        return False, (
            "Homebrew is not installed. Install it from https://brew.sh, then re-run setup. "
            "Or install manually: `brew install " + " ".join(_brew_pkg(missing)) + "`"
        )
    pkgs = _brew_pkg(missing)
    if not pkgs:
        return True, "nothing to install"
    cmd = ["brew", "install", *pkgs]
    print(f"[setup] running: {' '.join(cmd)}", file=sys.stderr)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        return False, f"brew install failed with exit code {result.returncode}"
    return True, f"installed via brew: {', '.join(pkgs)}"


def _required_hint_linux(missing: list[str]) -> str:
    if any(b in missing for b in ("ffmpeg", "ffprobe")):
        return "apt: `sudo apt install ffmpeg` or dnf: `sudo dnf install ffmpeg`"
    return "nothing to install"


def _required_hint_windows(missing: list[str]) -> str:
    if any(b in missing for b in ("ffmpeg", "ffprobe")):
        return "winget: `winget install Gyan.FFmpeg`"
    return "nothing to install"


def _yt_dlp_hint(system: str | None = None) -> str:
    system = system or platform.system()
    if system == "Windows":
        return "`winget install yt-dlp.yt-dlp` or `python -m pip install -U yt-dlp`"
    if system == "Darwin":
        return "`brew install yt-dlp` or `python3 -m pip install -U yt-dlp`"
    return "`pipx install yt-dlp` or `python3 -m pip install --user -U yt-dlp`"


def _status() -> dict:
    missing = _check_required_binaries()
    missing_optional = _check_optional_binaries()
    has_key, backend = _have_api_key()
    setup_complete = not is_first_run()

    if not missing and has_key:
        status = "ready"
    elif missing and not has_key:
        status = "needs_install_and_key"
    elif missing:
        status = "needs_install"
    else:
        status = "needs_key"

    can_proceed = (not missing) and (has_key or setup_complete)
    cfg = get_config()
    return {
        "status": status,
        "can_proceed": can_proceed,
        "first_run": not setup_complete,
        "setup_complete": setup_complete,
        "missing_binaries": missing,
        "missing_optional_binaries": missing_optional,
        "remote_video_frames_available": "yt-dlp" not in missing_optional,
        "whisper_backend": backend,
        "has_api_key": has_key,
        "config_file": str(CONFIG_FILE),
        "watch_detail": cfg["detail"],
        "platform": platform.system(),
    }


def cmd_check() -> int:
    s = _status()
    if s["can_proceed"]:
        return 0

    parts = []
    if s["missing_binaries"]:
        parts.append(f"missing required binaries: {', '.join(s['missing_binaries'])}")
    if not s["has_api_key"] and not s["setup_complete"]:
        parts.append("no Whisper API key (optional, but first-run choice not completed)")
    installer = Path(__file__).resolve()
    sys.stderr.write(
        f"[watch] setup incomplete ({'; '.join(parts)}). "
        f"Run: python3 {installer}\n"
    )
    sys.stderr.flush()

    if s["missing_binaries"] and not s["has_api_key"]:
        return 4
    if s["missing_binaries"]:
        return 2
    return 3


def cmd_json() -> int:
    json.dump(_status(), sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


def cmd_install() -> int:
    missing = _check_required_binaries()
    system = platform.system()
    if missing:
        if system == "Darwin":
            ok, msg = _install_macos(missing)
            print(f"[setup] {msg}", file=sys.stderr)
            if not ok:
                return 2
            still_missing = _check_required_binaries()
            if still_missing:
                print(f"[setup] still missing after install: {', '.join(still_missing)}", file=sys.stderr)
                return 2
        elif system == "Linux":
            print("[setup] required dependencies missing on Linux — please install:", file=sys.stderr)
            print("  " + _required_hint_linux(missing), file=sys.stderr)
            return 2
        elif system == "Windows":
            print("[setup] required dependencies missing on Windows — please install:", file=sys.stderr)
            print("  " + _required_hint_windows(missing), file=sys.stderr)
            return 2
        else:
            print(f"[setup] unsupported platform ({system}) for auto-install; missing: {', '.join(missing)}", file=sys.stderr)
            return 2

    created = _scaffold_env()
    if created:
        print(f"[setup] created config: {CONFIG_FILE}")
    else:
        print(f"[setup] config exists: {CONFIG_FILE}")

    missing_optional = _check_optional_binaries()
    if "yt-dlp" in missing_optional:
        print("")
        print("[setup] yt-dlp is not available in this runtime.")
        print("[setup] transcript-only fallback can still work for public YouTube videos.")
        print(f"[setup] for full remote video/frame analysis install yt-dlp with: {_yt_dlp_hint(system)}")

    has_key, backend = _have_api_key()
    _write_setup_complete()
    if has_key:
        print(f"[setup] ready. whisper backend: {backend}")
    else:
        print("")
        print("[setup] ready without Whisper API key.")
        print("[setup] native/public captions can still work; videos with no captions may be frames-only.")
        print(f"[setup] optional keys can be added later in {CONFIG_FILE}")
    return 0


def main() -> int:
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == "--check":
            return cmd_check()
        if arg == "--json":
            return cmd_json()
    return cmd_install()


if __name__ == "__main__":
    raise SystemExit(main())
