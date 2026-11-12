"""WAV spatial-format detection.

Four channels are not treated as Ambisonics unless the user selects a convention
or the file metadata names one. Detection never rewrites samples.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path

# Speaker bits commonly used in WAVEFORMATEXTENSIBLE channel masks.
_SPEAKER_NAMES = {
    0x1: "front left",
    0x2: "front right",
    0x4: "front centre",
    0x8: "low frequency",
    0x10: "back left",
    0x20: "back right",
    0x40: "front left of centre",
    0x80: "front right of centre",
    0x100: "back centre",
    0x200: "side left",
    0x400: "side right",
    0x800: "top centre",
    0x1000: "top front left",
    0x2000: "top front centre",
    0x4000: "top front right",
    0x8000: "top back left",
    0x10000: "top back centre",
    0x20000: "top back right",
}

CONVENTION_LABELS = {
    "acn_sn3d": "ACN / SN3D (AmbiX) — channels W, Y, Z, X",
    "fuma": "FuMa (Furse-Malham) — channels W, X, Y, Z",
    "acn_n3d": "ACN / N3D — channels W, Y, Z, X, first order × √3",
}

CONVENTION_HELP = """First-order Ambisonics stores a sound field in four channels. Those channels are not loudspeaker feeds.

ACN / SN3D (AmbiX) is the usual modern interchange format. The channel order is W, Y, Z, X. W is the omnidirectional pressure component. X, Y and Z are figure-of-eight components along the microphone's forward, left and up axes. For a unit-amplitude plane wave, W = 1 and X, Y, Z are the direction cosines.

FuMa (Furse-Malham) uses the channel order W, X, Y, Z, and its W channel is 1/√2 times the SN3D W channel. Soundform scales W by √2 before estimating direction, so both conventions share one estimator.

ACN / N3D uses the same order as AmbiX, but X, Y and Z are √3 times the SN3D values. Soundform divides those three channels by √3. Leaving N3D unmarked and reading it as SN3D would bias the directional vector.

A wrong convention points the architectural response along the wrong microphone axis. None of these formats records a compass bearing. Azimuth 0° is whatever direction the microphone called forward. Use the design rotation offset to align that axis with the model.

Ordinary four-channel audio (a quad bounce, for example) can share the channel count and still contain no Ambisonic sound field. Soundform will not guess a convention when metadata is missing."""


@dataclass
class FormatDetection:
    """Result of inspecting channel count and WAV metadata."""

    spatial_status: str
    suggested_convention: str | None
    severity: str
    messages: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    speaker_mask: int | None = None

    @property
    def can_force_convention(self) -> bool:
        return self.spatial_status in {"needs_convention", "ready"}


def _iter_riff_chunks(data: bytes):
    if len(data) < 12 or data[:4] not in {b"RIFF", b"RF64"}:
        return
    pos = 12
    limit = len(data)
    while pos + 8 <= limit:
        chunk_id = data[pos : pos + 4]
        size = struct.unpack_from("<I", data, pos + 4)[0]
        start = pos + 8
        end = start + size
        if size < 0 or end > limit:
            break
        yield chunk_id, data[start:end]
        pos = end + (size & 1)


def _decode_text(payload: bytes) -> str:
    for encoding in ("utf-8", "utf-16-le", "latin-1"):
        try:
            return payload.decode(encoding)
        except UnicodeDecodeError:
            continue
    return ""


def _info_strings(payload: bytes) -> list[str]:
    """Pull ASCII fields out of a LIST/INFO chunk."""
    if len(payload) < 4 or payload[:4] != b"INFO":
        return []
    texts: list[str] = []
    pos = 4
    while pos + 8 <= len(payload):
        size = struct.unpack_from("<I", payload, pos + 4)[0]
        start = pos + 8
        end = min(len(payload), start + max(0, size))
        texts.append(_decode_text(payload[start:end]))
        pos = end + (size & 1)
    return texts


def _speaker_mask(fmt_payload: bytes) -> int | None:
    """Return dwChannelMask for WAVEFORMATEXTENSIBLE, else None."""
    if len(fmt_payload) < 40:
        return None
    tag = struct.unpack_from("<H", fmt_payload, 0)[0]
    if tag != 0xFFFE:
        return None
    return struct.unpack_from("<I", fmt_payload, 20)[0]


def _scan_metadata(path: Path) -> tuple[str, list[str], int | None]:
    size = path.stat().st_size
    # Chunk headers are small; cap the read so a huge file cannot exhaust memory,
    # but keep the window large enough for a one-minute 4-channel 24-bit WAV.
    read_limit = 80 * 1024 * 1024
    with path.open("rb") as handle:
        data = handle.read(read_limit)
    evidence: list[str] = []
    texts: list[str] = []
    mask: int | None = None
    if size > read_limit:
        evidence.append("Only the first 80 MB were scanned for metadata.")
    for chunk_id, payload in _iter_riff_chunks(data):
        name = chunk_id.decode("latin-1", errors="replace")
        if chunk_id == b"fmt ":
            mask = _speaker_mask(payload)
        elif chunk_id == b"bext" and payload:
            description = _decode_text(payload[:256])
            texts.append(description)
            evidence.append("Broadcast-WAV description was read.")
        elif chunk_id == b"axml" and payload:
            texts.append(_decode_text(payload))
            evidence.append("Found an axml metadata chunk.")
        elif chunk_id == b"chna" and payload:
            texts.append(_decode_text(payload))
            evidence.append("Found a chna channel-assignment chunk.")
        elif chunk_id == b"LIST":
            info = _info_strings(payload)
            if info:
                texts.extend(info)
                evidence.append("Found LIST/INFO text.")
        elif name.lower() in {"umid", "cue "}:
            continue
    if mask not in (None, 0):
        labels = [label for bit, label in _SPEAKER_NAMES.items() if mask & bit]
        joined = ", ".join(labels) if labels else hex(mask)
        evidence.append(f"WAVE channel mask {hex(mask)} ({joined}).")
    return "\n".join(texts), evidence, mask


def _suggest_convention(text: str) -> tuple[str | None, str | None]:
    """Return (convention id, conflict note). Conflict yields no suggestion."""
    lowered = text.lower()
    has_ambix = "ambix" in lowered
    has_sn3d = "sn3d" in lowered
    has_fuma = (
        "fuma" in lowered
        or "furse-malham" in lowered
        or "furse malham" in lowered
        or (("b-format" in lowered or "bformat" in lowered) and not has_ambix and not has_sn3d)
    )
    # "sn3d" contains the letters n3d; require a standalone N3D token.
    has_n3d = "n3d" in lowered and not has_sn3d
    if "n3d" in lowered and has_sn3d:
        # Still treat an explicit separate N3D mention as a conflict if both appear
        # outside the substring relationship. "SN3D" alone is not N3D.
        stripped = lowered.replace("sn3d", "")
        has_n3d = "n3d" in stripped
    has_wyzx = "w, y, z, x" in lowered or "wyzx" in lowered
    has_wxyz = "w, x, y, z" in lowered or "wxyz" in lowered
    votes: set[str] = set()
    if has_ambix or has_sn3d or has_wyzx:
        votes.add("acn_sn3d")
    if has_fuma or has_wxyz:
        votes.add("fuma")
    if has_n3d and ("acn" in lowered or "ambisonic" in lowered or has_wyzx):
        votes.add("acn_n3d")
    if len(votes) > 1:
        names = ", ".join(sorted(votes))
        return None, f"Metadata names more than one convention ({names})."
    if len(votes) == 1:
        return next(iter(votes)), None
    return None, None


def _looks_like_raw_aformat(text: str) -> bool:
    lowered = text.lower()
    raw = "a-format" in lowered or "aformat" in lowered
    encoded = any(token in lowered for token in ("ambix", "sn3d", "fuma", "b-format", "bformat", "acn"))
    return raw and not encoded


def detect_spatial_format(path: str | Path, channels: int) -> FormatDetection:
    """Classify a WAV by channel count and optional Ambisonic metadata."""
    file_path = Path(path)
    if channels == 1:
        return FormatDetection(
            spatial_status="unsupported",
            suggested_convention=None,
            severity="error",
            messages=[
                "This file is mono. A single channel has no directional components, "
                "so Soundform cannot reconstruct a spatial sound field from it."
            ],
        )
    if channels == 2:
        return FormatDetection(
            spatial_status="unsupported",
            suggested_convention=None,
            severity="error",
            messages=[
                "This file is two-channel. Stereo, including a binaural recording, "
                "encodes ear or speaker signals rather than a first-order Ambisonic "
                "sound field. It does not contain the height and front/back components "
                "this tool needs, and it is not treated as Ambisonics."
            ],
        )
    if channels != 4:
        return FormatDetection(
            spatial_status="unsupported",
            suggested_convention=None,
            severity="error",
            messages=[
                f"This file has {channels} channels. Soundform decodes first-order "
                "Ambisonics, which is four channels. Higher-order Ambisonics and "
                "surround layouts are not trimmed or mixed down into a substitute "
                "first-order signal."
            ],
        )

    text, evidence, mask = _scan_metadata(file_path)
    if _looks_like_raw_aformat(text):
        return FormatDetection(
            spatial_status="unsupported",
            suggested_convention=None,
            severity="error",
            messages=[
                "Metadata describes A-format capsule signals. A-format is not yet "
                "a W/X/Y/Z sound field; it depends on a microphone calibration. "
                "Soundform does not convert A-format."
            ],
            evidence=evidence,
            speaker_mask=mask,
        )

    suggestion, conflict = _suggest_convention(text)
    messages: list[str] = []
    if conflict:
        messages.append(conflict + " Choose the convention you know is correct.")
        suggestion = None
    if mask not in (None, 0) and suggestion is None:
        messages.append(
            "The WAV channel mask names loudspeaker positions. That is evidence of a "
            "speaker mix, not of Ambisonics. You may still select a convention if you "
            "know the mask is wrong and the file is really first-order Ambisonics."
        )
    if suggestion is not None:
        messages.append(
            f"File metadata indicates {CONVENTION_LABELS[suggestion]}. "
            "Confirm this before analysis. A wrong choice mirrors or swaps directions. "
            "Metadata does not record which way the microphone faced in the world; "
            "azimuth 0° stays the microphone's forward axis."
        )
        return FormatDetection(
            spatial_status="ready",
            suggested_convention=suggestion,
            severity="info",
            messages=messages,
            evidence=evidence,
            speaker_mask=mask,
        )

    messages.append(
        "Four channels were found, but the convention is not identified. A four-channel "
        "WAV is not assumed to be Ambisonics: it may be ordinary quad audio, which has "
        "no W, X, Y, Z sound field. If you know this recording is first-order Ambisonics, "
        "select ACN/SN3D, FuMa, or ACN/N3D. The wrong choice assigns the microphone axes "
        "incorrectly. The file also does not establish a compass orientation; use the "
        "rotation offset when you align microphone-forward with the design."
    )
    return FormatDetection(
        spatial_status="needs_convention",
        suggested_convention=None,
        severity="warning",
        messages=messages,
        evidence=evidence,
        speaker_mask=mask,
    )


def append_axml_chunk(path: str | Path, xml: str) -> None:
    """Append an axml chunk and update the RIFF size. Used for synthetic examples."""
    file_path = Path(path)
    data = bytearray(file_path.read_bytes())
    if data[:4] != b"RIFF":
        raise ValueError("axml can only be appended to a RIFF WAV.")
    raw = xml.encode("utf-8")
    pad = b"\x00" if len(raw) % 2 else b""
    data.extend(b"axml" + struct.pack("<I", len(raw)) + raw + pad)
    struct.pack_into("<I", data, 4, len(data) - 8)
    file_path.write_bytes(data)
