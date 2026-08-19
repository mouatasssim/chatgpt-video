#!/usr/bin/env python3
"""Download a video via yt-dlp, or resolve a local file path.

Also fetches subtitles (manual first, then auto-generated) in VTT format so
transcribe.py can parse them without needing Whisper.

Sandbox-friendly behavior:
- Uses the ``yt-dlp`` executable when present.
- Falls back to ``python -m yt_dlp`` when the Python package is installed but
  the console script is not on PATH.
- For public YouTube URLs, when yt-dlp is unavailable or fails before captions
  can be fetched, it can use a public transcript API as a transcript-only
  fallback. This keeps /watch useful in restricted ChatGPT/Codex sandboxes.
- If remote video download itself is unavailable, returns a transcript-only
  result instead of aborting the whole /watch run. Local files are unaffected.
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, quote, urlparse
from urllib.request import Request, urlopen


VIDEO_EXTS = {".mp4", ".mkv", ".webm", ".mov", ".m4v", ".avi", ".flv", ".wmv"}
PUBLIC_TRANSCRIPT_API = "https://getvideotranscript.com/api/youtube"


def is_url(source: str) -> bool:
    if source.startswith("-"):
        return False
    parsed = urlparse(source)
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)


def resolve_local(path: str) -> dict:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise SystemExit(f"File not found: {p}")
    if p.suffix.lower() not in VIDEO_EXTS:
        print(
            f"[watch] warning: {p.suffix} is not a known video extension, proceeding anyway",
            file=sys.stderr,
        )
    return {
        "video_path": str(p),
        "subtitle_path": None,
        "info": {"title": p.name, "url": str(p)},
        "downloaded": False,
    }


def _pick_subtitle(out_dir: Path) -> Path | None:
    candidates = sorted(out_dir.glob("video*.vtt"))
    if not candidates:
        return None
    preferred = [
        c for c in candidates
        if any(marker in c.name for marker in (".en.", ".en-US.", ".en-GB.", ".en-orig."))
    ]
    return preferred[0] if preferred else candidates[0]


def _pick_video(out_dir: Path) -> Path | None:
    for ext in (".mp4", ".mkv", ".webm", ".mov", ".m4a", ".mp3", ".opus"):
        for candidate in out_dir.glob(f"video*{ext}"):
            return candidate
    for candidate in out_dir.glob("video.*"):
        if candidate.suffix.lower() in VIDEO_EXTS:
            return candidate
    return None


def _yt_dlp_command() -> list[str] | None:
    """Return an executable prefix for yt-dlp without assuming PATH layout."""
    executable = shutil.which("yt-dlp")
    if executable:
        return [executable]
    try:
        if importlib.util.find_spec("yt_dlp") is not None:
            return [sys.executable, "-m", "yt_dlp"]
    except (ImportError, AttributeError, ValueError):
        pass
    return None


def _public_fallback_enabled() -> bool:
    raw = os.environ.get("WATCH_PUBLIC_TRANSCRIPT_FALLBACK", "true").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _youtube_video_id(url: str) -> str | None:
    parsed = urlparse(url)
    host = parsed.netloc.lower().split(":", 1)[0]
    if host.startswith("www."):
        host = host[4:]
    if host in {"youtu.be"}:
        candidate = parsed.path.strip("/").split("/", 1)[0]
        return candidate if len(candidate) == 11 else None
    if host not in {"youtube.com", "m.youtube.com", "music.youtube.com"}:
        return None
    if parsed.path == "/watch":
        candidate = (parse_qs(parsed.query).get("v") or [None])[0]
        return candidate if candidate and len(candidate) == 11 else None
    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) >= 2 and parts[0] in {"shorts", "embed", "live"}:
        candidate = parts[1]
        return candidate if len(candidate) == 11 else None
    return None


def _vtt_timestamp(seconds: float) -> str:
    total_ms = max(0, int(round(seconds * 1000)))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"


def _write_public_transcript_vtt(entries: list[dict], path: Path) -> float:
    lines = ["WEBVTT", ""]
    max_end = 0.0
    cue_index = 0
    for item in entries:
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        try:
            start = float(item.get("start") or 0.0)
            duration = float(item.get("duration") or 0.0)
        except (TypeError, ValueError):
            continue
        end = max(start + max(duration, 0.05), start + 0.05)
        max_end = max(max_end, end)
        cue_index += 1
        lines.extend([
            str(cue_index),
            f"{_vtt_timestamp(start)} --> {_vtt_timestamp(end)}",
            text.replace("\r", " ").replace("\n", " "),
            "",
        ])
    if cue_index == 0:
        return 0.0
    path.write_text("\n".join(lines), encoding="utf-8")
    return max_end


def _fetch_public_youtube_transcript(url: str, out_dir: Path) -> dict | None:
    """Best-effort transcript-only fallback for public YouTube URLs.

    The provider is intentionally used only after native yt-dlp access is
    unavailable/failing. Disable with WATCH_PUBLIC_TRANSCRIPT_FALLBACK=false.
    """
    if not _public_fallback_enabled():
        return None
    video_id = _youtube_video_id(url)
    if not video_id:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    endpoint = f"{PUBLIC_TRANSCRIPT_API}?video_id={quote(video_id)}"
    req = Request(endpoint, headers={"User-Agent": "chatgpt-video/0.3 (+transcript-fallback)"})
    try:
        with urlopen(req, timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        print(f"[watch] public transcript fallback unavailable: {exc}", file=sys.stderr)
        return None

    entries = payload.get("transcript") if isinstance(payload, dict) else None
    if not isinstance(entries, list) or not entries:
        return None

    subtitle_path = out_dir / "video.public.en.vtt"
    duration = _write_public_transcript_vtt(entries, subtitle_path)
    if not subtitle_path.exists():
        return None
    print("[watch] using public YouTube transcript fallback", file=sys.stderr)
    return {
        "video_path": None,
        "subtitle_path": str(subtitle_path),
        "info": {
            "title": None,
            "uploader": None,
            "duration": duration,
            "url": url,
        },
        "downloaded": False,
        "transcript_source": "public transcript API",
        "remote_video_unavailable": True,
    }


def _read_info(info_path: Path, url: str) -> dict:
    info: dict = {}
    if info_path.exists():
        try:
            raw = json.loads(info_path.read_text(encoding="utf-8"))
            info = {
                "title": raw.get("title"),
                "uploader": raw.get("uploader") or raw.get("channel"),
                "duration": raw.get("duration"),
                "url": raw.get("webpage_url") or url,
            }
        except Exception as exc:
            print(f"[watch] info.json parse failed: {exc}", file=sys.stderr)
            info = {"url": url}
    return info


def fetch_captions(url: str, out_dir: Path) -> dict:
    """Fetch metadata and best available VTT captions without downloading video."""
    out_dir.mkdir(parents=True, exist_ok=True)
    existing_subtitle = _pick_subtitle(out_dir)
    if existing_subtitle:
        return {
            "video_path": None,
            "subtitle_path": str(existing_subtitle),
            "info": _read_info(out_dir / "video.info.json", url) or {"url": url},
            "downloaded": False,
        }

    yt_dlp = _yt_dlp_command()
    if yt_dlp is None:
        fallback = _fetch_public_youtube_transcript(url, out_dir)
        if fallback:
            return fallback
        print(
            "[watch] yt-dlp is unavailable; continuing without remote captions. "
            "Install with `python -m pip install -U yt-dlp` for native YouTube access.",
            file=sys.stderr,
        )
        return {
            "video_path": None,
            "subtitle_path": None,
            "info": {"url": url},
            "downloaded": False,
            "remote_video_unavailable": True,
        }

    output_template = str(out_dir / "video.%(ext)s")
    cmd = [
        *yt_dlp,
        "--skip-download",
        "--write-info-json",
        "--write-subs",
        "--write-auto-subs",
        "--sub-langs", "en.*",
        "--sub-format", "vtt",
        "--convert-subs", "vtt",
        "--no-playlist",
        "--ignore-errors",
        "-o", output_template,
        "--",
        url,
    ]
    result = subprocess.run(cmd, stdout=sys.stderr, stderr=sys.stderr)
    subtitle = _pick_subtitle(out_dir)
    info = _read_info(out_dir / "video.info.json", url)
    if subtitle:
        return {
            "video_path": None,
            "subtitle_path": str(subtitle),
            "info": info or {"url": url},
            "downloaded": False,
            "transcript_source": "captions",
        }

    if result.returncode != 0:
        fallback = _fetch_public_youtube_transcript(url, out_dir)
        if fallback:
            if info:
                fallback["info"].update({k: v for k, v in info.items() if v is not None})
            return fallback

    return {
        "video_path": None,
        "subtitle_path": None,
        "info": info or {"url": url},
        "downloaded": False,
    }


def download_url(
    url: str,
    out_dir: Path,
    audio_only: bool = False,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    yt_dlp = _yt_dlp_command()
    if yt_dlp is None:
        fallback = _fetch_public_youtube_transcript(url, out_dir)
        if fallback:
            return fallback
        print(
            "[watch] remote video download unavailable because yt-dlp is not installed; "
            "continuing transcript/metadata-only instead of aborting.",
            file=sys.stderr,
        )
        return {
            "video_path": None,
            "subtitle_path": str(_pick_subtitle(out_dir)) if _pick_subtitle(out_dir) else None,
            "info": _read_info(out_dir / "video.info.json", url) or {"url": url},
            "downloaded": False,
            "remote_video_unavailable": True,
        }

    output_template = str(out_dir / "video.%(ext)s")
    fmt = "ba/bestaudio" if audio_only else "bv*[height<=720]+ba/b[height<=720]/bv+ba/b"
    cmd = [
        *yt_dlp,
        "-N", "8",
        "-f", fmt,
        "--merge-output-format", "mp4",
        "--write-info-json",
        "--write-subs",
        "--write-auto-subs",
        "--sub-langs", "en.*",
        "--sub-format", "vtt",
        "--convert-subs", "vtt",
        "--no-playlist",
        "--ignore-errors",
        "-o", output_template,
        "--",
        url,
    ]

    result = subprocess.run(cmd, stdout=sys.stderr, stderr=sys.stderr)
    video = _pick_video(out_dir)
    subtitle = _pick_subtitle(out_dir)
    info = _read_info(out_dir / "video.info.json", url)

    if video is None:
        if result.returncode != 0 and subtitle is None:
            fallback = _fetch_public_youtube_transcript(url, out_dir)
            if fallback:
                if info:
                    fallback["info"].update({k: v for k, v in info.items() if v is not None})
                return fallback
        print(
            f"[watch] yt-dlp produced no video file (exit {result.returncode}); "
            "continuing transcript/metadata-only instead of aborting.",
            file=sys.stderr,
        )
        return {
            "video_path": None,
            "subtitle_path": str(subtitle) if subtitle else None,
            "info": info or {"url": url},
            "downloaded": False,
            "remote_video_unavailable": True,
            "transcript_source": "captions" if subtitle else None,
        }

    return {
        "video_path": str(video),
        "subtitle_path": str(subtitle) if subtitle else None,
        "info": info or {"url": url},
        "downloaded": True,
        "transcript_source": "captions" if subtitle else None,
    }


def download(
    source: str,
    out_dir: Path,
    audio_only: bool = False,
) -> dict:
    if is_url(source):
        return download_url(source, out_dir, audio_only=audio_only)
    return resolve_local(source)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: download.py <url-or-path> <out-dir>", file=sys.stderr)
        raise SystemExit(2)
    result = download(sys.argv[1], Path(sys.argv[2]))
    print(json.dumps(result, indent=2))
