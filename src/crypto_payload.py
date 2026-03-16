"""
Encrypted payload module for forensic watermarking (proposal-aligned).

Payloads consist of encrypted user/session IDs with:
- AES-GCM for confidentiality and integrity
- Reed-Solomon error correction to tolerate distortion (compression, cropping, etc.)

Used for both visible fingerprint text and invisible DCT embedding.
"""

from __future__ import annotations

import base64
import hashlib
import os
import secrets
from typing import Optional

# Optional: AES-GCM encryption
try:
    from Crypto.Cipher import AES
    from Crypto.Util.Padding import pad, unpad
    HAS_CRYPTO = True
except ImportError:
    HAS_CRYPTO = False

# Reed-Solomon error correction
try:
    from reedsolo import RSCodec
    HAS_REEDSOLO = True
except ImportError:
    HAS_REEDSOLO = False


# Reed-Solomon: nsym=12 allows correcting up to 6 byte errors per block
# For ~32 byte payload, we need nsym to fit. RSCodec(nsym) adds nsym bytes.
# Max message length depends on GF(2^8): 255 - nsym
RS_NSYM = 12
MAX_PAYLOAD_BYTES = 48  # session_id + metadata, before ECC


def _derive_key(secret: bytes, salt: Optional[bytes] = None) -> tuple[bytes, bytes]:
    """Derive 32-byte key and 12-byte nonce from secret using HKDF-like approach."""
    if salt is None:
        salt = secrets.token_bytes(16)
    key_material = hashlib.pbkdf2_hmac("sha256", secret, salt, 100000, dklen=48)
    return key_material[:32], salt


def encode_payload(
    user_id: str,
    session_id: Optional[str] = None,
    location: Optional[str] = None,
    device: Optional[str] = None,
    *,
    secret_key: Optional[bytes] = None,
    use_aes: bool = True,
    use_ecc: bool = True,
) -> tuple[bytes, Optional[bytes]]:
    """
    Encode user/session fingerprint into a robust payload.

    Args:
        user_id: User identifier.
        session_id: Optional session token.
        location: Optional geo code (e.g. "US-CA").
        device: Optional device type (e.g. "web", "ios").
        secret_key: 32-byte key for AES-GCM. If None, uses derived key from user_id.
        use_aes: Encrypt with AES-GCM (requires pycryptodome).
        use_ecc: Apply Reed-Solomon ECC (requires reedsolo).

    Returns:
        (payload_bytes, salt_or_none) — payload to embed; salt if AES used (needed for decrypt).
    """
    parts = [user_id]
    if session_id:
        parts.append(str(session_id))
    if location:
        parts.append(str(location))
    if device:
        parts.append(str(device))

    raw = "|".join(parts).encode("utf-8")
    if len(raw) > MAX_PAYLOAD_BYTES:
        raw = raw[:MAX_PAYLOAD_BYTES]

    salt = None
    if use_aes and HAS_CRYPTO and secret_key is not None:
        # AES-GCM: 12-byte nonce (96 bits) is recommended
        key = secret_key[:32] if len(secret_key) >= 32 else _derive_key(secret_key)[0]
        cipher = AES.new(key, AES.MODE_GCM)
        nonce = cipher.nonce
        padded = pad(raw, AES.block_size)
        ciphertext, tag = cipher.encrypt_and_digest(padded)
        # Format: nonce (12) + tag (16) + ciphertext
        raw = nonce + tag + ciphertext
        salt = nonce  # callers may store nonce separately

    if use_ecc and HAS_REEDSOLO:
        rs = RSCodec(RS_NSYM)
        raw = rs.encode(raw)

    return raw, salt


def decode_payload(
    payload: bytes,
    *,
    secret_key: Optional[bytes] = None,
    use_aes: bool = True,
    use_ecc: bool = True,
    nonce: Optional[bytes] = None,
) -> Optional[str]:
    """
    Decode payload back to user/session string.

    Args:
        payload: Raw payload bytes (possibly with ECC).
        secret_key: Same key used for encryption.
        use_ecc: Apply Reed-Solomon decode first.
        nonce: If AES was used, nonce is in payload[:12] unless passed separately.

    Returns:
        Decoded string (e.g. "user_001|sess_xyz|US-CA|ios") or None on failure.
    """
    raw = bytes(payload)
    if use_ecc and HAS_REEDSOLO:
        try:
            rs = RSCodec(RS_NSYM)
            raw, _ = rs.decode(raw)
        except Exception:
            return None

    if use_aes and HAS_CRYPTO and secret_key is not None:
        if len(raw) < 12 + 16:
            return None
        n = raw[:12]
        tag = raw[12:28]
        ct = raw[28:]
        key = secret_key[:32] if len(secret_key) >= 32 else _derive_key(secret_key)[0]
        try:
            cipher = AES.new(key, AES.MODE_GCM, nonce=n)
            padded = cipher.decrypt_and_verify(ct, tag)
            raw = unpad(padded, AES.block_size)
        except Exception:
            return None

    try:
        return raw.decode("utf-8")
    except Exception:
        return None


def payload_to_embed_string(payload: bytes, max_chars: int = 24) -> str:
    """
    Convert payload to short string for visible watermark display.
    Uses base64url, truncated. For full traceability, use decode_payload on extracted bytes.
    """
    s = base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")
    return s[:max_chars] if len(s) > max_chars else s


def embed_string_to_payload(s: str) -> Optional[bytes]:
    """Reverse of payload_to_embed_string (adds padding if needed)."""
    try:
        pad_len = 4 - (len(s) % 4)
        if pad_len != 4:
            s += "=" * pad_len
        return base64.urlsafe_b64decode(s)
    except Exception:
        return None
