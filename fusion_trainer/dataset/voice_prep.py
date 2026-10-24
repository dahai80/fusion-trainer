"""Voice dataset validation (issue #74, M-A).

Mirrors fusion-mlx's voice dataset contract (fusion_mlx.training.voice):
a directory of paired audio + transcripts in one of three layouts —
  - metadata.csv:   "file|text" per line
  - metadata.jsonl: {"audio": ..., "text": ...} per line
  - bare pairs:     <name>.wav + <name>.txt side by side
Validation is file-level only (existence, non-empty transcript, non-trivial
size); audio decoding/duration checks stay upstream where the decoder lives.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

MIN_AUDIO_BYTES = 1024


class VoiceDatasetError(ValueError):
    pass


def _pairs_from_csv(root: Path) -> list[tuple[Path, str]]:
    pairs: list[tuple[Path, str]] = []
    lines = (root / "metadata.csv").read_text(encoding="utf-8").splitlines()
    for lineno, raw in enumerate(lines, 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("|")
        if len(parts) < 2:
            raise VoiceDatasetError(
                f"metadata.csv line {lineno}: expected 'file|text', got {line!r}"
            )
        pairs.append(((root / parts[0].strip()).resolve(), parts[1].strip()))
    return pairs


def _pairs_from_jsonl(root: Path) -> list[tuple[Path, str]]:
    pairs: list[tuple[Path, str]] = []
    lines = (root / "metadata.jsonl").read_text(encoding="utf-8").splitlines()
    for lineno, raw in enumerate(lines, 1):
        line = raw.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise VoiceDatasetError(
                f"metadata.jsonl line {lineno}: invalid JSON ({exc})"
            ) from exc
        audio_rel = item.get("audio") or item.get("file") or item.get("path")
        text = item.get("text") or item.get("transcript")
        if not audio_rel or not text:
            raise VoiceDatasetError(
                f"metadata.jsonl line {lineno}: needs 'audio' and 'text'"
            )
        pairs.append(((root / str(audio_rel)).resolve(), str(text).strip()))
    return pairs


def _pairs_from_bare_wavs(root: Path) -> list[tuple[Path, str]]:
    pairs: list[tuple[Path, str]] = []
    for audio in sorted(root.glob("*.wav")):
        txt = audio.with_suffix(".txt")
        if not txt.is_file():
            logger.warning("voice dataset: %s has no transcript, skipping", audio.name)
            continue
        pairs.append((audio.resolve(), txt.read_text(encoding="utf-8").strip()))
    return pairs


def validate_voice_dataset(dataset_dir: str | Path) -> dict:
    """Validate a voice dataset directory before job submission.

    Returns {"samples": n, "total_bytes": b, "problems": []} on success.
    Raises VoiceDatasetError (fail-visible) on a missing dir, no parseable
    pairs, or any per-sample problem — matching upstream's reject-on-any-
    problem behavior so the trainer refuses before spending a job slot.
    """
    root = Path(str(dataset_dir)).expanduser()
    if not root.is_dir():
        raise VoiceDatasetError(f"dataset directory not found: {root}")

    if (root / "metadata.csv").is_file():
        pairs = _pairs_from_csv(root)
    elif (root / "metadata.jsonl").is_file():
        pairs = _pairs_from_jsonl(root)
    else:
        pairs = _pairs_from_bare_wavs(root)

    if not pairs:
        raise VoiceDatasetError(
            f"no training pairs in {root}: provide metadata.csv, metadata.jsonl "
            "or <name>.wav + <name>.txt"
        )

    problems: list[str] = []
    total_bytes = 0
    for audio_path, text in pairs:
        if not audio_path.is_file():
            problems.append(f"{audio_path}: audio file missing")
            continue
        size = audio_path.stat().st_size
        if size < MIN_AUDIO_BYTES:
            problems.append(f"{audio_path}: file is empty or truncated")
            continue
        if not text:
            problems.append(f"{audio_path}: transcript is empty")
            continue
        total_bytes += size

    if problems:
        raise VoiceDatasetError(
            "dataset rejected: "
            + "; ".join(problems[:10])
            + (f" (+{len(problems) - 10} more)" if len(problems) > 10 else "")
        )

    logger.info(
        "voice dataset ok: %d pair(s), %.1f KB from %s",
        len(pairs),
        total_bytes / 1024,
        root,
    )
    return {"samples": len(pairs), "total_bytes": total_bytes, "problems": []}
