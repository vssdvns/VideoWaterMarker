"""
Phase 2: User-specific fingerprint encoding for traceability.

Encodes user ID (and optional location/device) into visible watermark text
for OTT/streaming scenarios. When a pirated copy is found, the watermark
reveals which user account it originated from.
"""

from __future__ import annotations

import hashlib
import re


def encode_fingerprint(
    user_id: str,
    location: str | None = None,
    device: str | None = None,
    *,
    use_hash: bool = False,
    prefix: str = "ID",
) -> str:
    """
    Encode a user fingerprint into visible watermark text.

    Args:
        user_id: User identifier (e.g. account ID, session token).
        location: Optional geo/location code (e.g. "US-CA", "EU").
        device: Optional device type (e.g. "web", "ios", "android").
        use_hash: If True, use short hash for privacy (requires lookup table).
                  If False and user_id is short, use it directly (demo mode).
        prefix: Prefix for the watermark text (e.g. "ID", "VID").

    Returns:
        Watermark text string, e.g. "ID:alice", "ID:a1b2c3d4", or "ID:a1b2.LA.ios".
    """
    user_id = (user_id or "").strip()
    if not user_id:
        return "VideoWaterMarker"

    location = (location or "").strip()
    device = (device or "").strip()

    if use_hash or len(user_id) > 12 or location or device:
        payload = user_id
        if location:
            payload += "." + location
        if device:
            payload += "." + device
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        short = digest[:8].lower()

        parts = [f"{prefix}:{short}"]
        if location:
            parts.append(_sanitize(location, 6))
        if device:
            parts.append(_sanitize(device, 6))
        return ".".join(parts)

    return f"{prefix}:{_sanitize(user_id, 16)}"


def _sanitize(s: str, max_len: int) -> str:
    """Keep only alphanumeric and common chars, truncate."""
    s = re.sub(r"[^a-zA-Z0-9\-_]", "", s)
    return s[:max_len] if s else ""


def decode_lookup(fingerprint_text: str, registry: dict[str, str]) -> str | None:
    """
    Look up user_id from fingerprint text using a registry.
    Used when fingerprint was created with use_hash=True.

    Args:
        fingerprint_text: The watermark text (e.g. "ID:a1b2c3d4").
        registry: Map from short_code -> user_id (populated at encode time).

    Returns:
        user_id if found, else None.
    """
    match = fingerprint_text.split(".")[0] if fingerprint_text else ""
    if ":" in match:
        short_code = match.split(":")[-1].lower()
        return registry.get(short_code)
    return None


def build_registry_entry(
    user_id: str,
    location: str | None = None,
    device: str | None = None,
) -> tuple[str, str]:
    """
    Build fingerprint text and the registry entry for lookup.
    Use when encoding with use_hash=True.

    Returns:
        (watermark_text, short_code) - store short_code -> user_id in your DB.
    """
    text = encode_fingerprint(user_id, location, device, use_hash=True)
    short = text.split(":")[-1].split(".")[0]
    return text, short
