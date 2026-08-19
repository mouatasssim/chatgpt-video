"""Downloader regression tests for native and sandbox-safe paths."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "skills" / "watch" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import download  # noqa: E402

URL = "https://www.youtube.com/watch?v=rlOpbu3Enkw"


def _capture_argv(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    calls: list[list[str]] = []

    class _Result:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(cmd, *args, **kwargs):
        calls.append(list(cmd))
        return _Result()

    monkeypatch.setattr(download.subprocess, "run", fake_run)
    monkeypatch.setattr(download, "_yt_dlp_command", lambda: ["yt-dlp"])
    return calls


def _sub_langs(argv: list[str]) -> str:
    idx = argv.index("--sub-langs")
    return argv[idx + 1]


def _assert_english_only(langs: str) -> None:
    tokens = langs.split(",")
    assert "all" not in tokens, f"sub-langs must not request all languages, got {langs!r}"
    assert all(t.startswith("en") for t in tokens), f"sub-langs must be English-only, got {langs!r}"


def test_fetch_captions_requests_english_only(monkeypatch, tmp_path):
    calls = _capture_argv(monkeypatch)
    download.fetch_captions(URL, tmp_path / "download")
    _assert_english_only(_sub_langs(calls[0]))


def test_download_url_requests_english_only(monkeypatch, tmp_path):
    calls = _capture_argv(monkeypatch)
    result = download.download_url(URL, tmp_path / "download")
    _assert_english_only(_sub_langs(calls[0]))
    assert result["video_path"] is None
    assert result["remote_video_unavailable"] is True


def test_python_module_yt_dlp_fallback(monkeypatch):
    monkeypatch.setattr(download.shutil, "which", lambda _name: None)
    monkeypatch.setattr(download.importlib.util, "find_spec", lambda name: object() if name == "yt_dlp" else None)
    assert download._yt_dlp_command() == [sys.executable, "-m", "yt_dlp"]


def test_missing_yt_dlp_does_not_abort(monkeypatch, tmp_path):
    monkeypatch.setattr(download, "_yt_dlp_command", lambda: None)
    monkeypatch.setattr(download, "_fetch_public_youtube_transcript", lambda _url, _out: None)
    result = download.download_url(URL, tmp_path / "download")
    assert result["video_path"] is None
    assert result["downloaded"] is False
    assert result["remote_video_unavailable"] is True


def test_public_transcript_vtt_writer(tmp_path):
    target = tmp_path / "fallback.vtt"
    duration = download._write_public_transcript_vtt(
        [
            {"text": "hello", "start": 0.5, "duration": 1.25},
            {"text": "world", "start": 2.0, "duration": 0.5},
        ],
        target,
    )
    body = target.read_text(encoding="utf-8")
    assert body.startswith("WEBVTT")
    assert "00:00:00.500 --> 00:00:01.750" in body
    assert "hello" in body and "world" in body
    assert duration == pytest.approx(2.5)


def test_public_timestamped_markdown_parser():
    entries, title, duration = download._parse_public_markdown(
        "# Transcript: Demo video\n"
        "Language: fr · Duration: 0:10 · Words: 4\n\n"
        "[0:00] hello\n\n"
        "[0:05] world\n"
    )
    assert title == "Demo video"
    assert len(entries) == 2
    assert entries[0] == {"start": 0.0, "text": "hello", "duration": 5.0}
    assert entries[1] == {"start": 5.0, "text": "world", "duration": 5.0}
    assert duration == pytest.approx(10.0)
