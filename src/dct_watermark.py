"""
DCT-based invisible watermark embedding (proposal-aligned).

Embeds payload bits in DCT coefficients of low-attention (low-complexity) regions.
Hybrid approach: combines with visible text watermark for redundancy.
Improves robustness against compression (codec operates in DCT domain).
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Reed-Solomon is optional because the project can still embed raw payloads,
# but ECC helps the hidden payload survive noisier attacks.
try:
    from reedsolo import RSCodec
    HAS_RS = True
except ImportError:
    HAS_RS = False

RS_NSYM = 10
BLOCK_SIZE = 8
# These mid-frequency coefficients are the compromise zone: they are less
# visible than low-frequency edits and more likely to survive compression than
# very high-frequency edits.
EMBED_INDICES = [(2, 3), (3, 2), (3, 3), (4, 2), (2, 4)]  # 5 bits per block
DCT_STRENGTH = 4.0  # Higher = more robust to rounding/codec (default for embed/extract)


def _dct2_blocks(block: np.ndarray) -> np.ndarray:
    """DCT of 8x8 block."""
    return cv2.dct(block.astype(np.float64))


def _idct2_blocks(block: np.ndarray) -> np.ndarray:
    """Inverse DCT."""
    return cv2.idct(block)


def _embed_bit(coeff: float, bit: int, strength: float = 2.0) -> float:
    """
    Embed one bit by QIM: bit 0 -> quantize to k*q, bit 1 -> quantize to k*q + q/2.
    """
    # Quantization index modulation nudges one coefficient into one of two
    # predictable bins so the decoder can later read back a 0 or 1.
    q = max(1.0, strength)
    half = q / 2
    c = np.round(coeff / q) * q
    if bit == 1:
        c += half
    # else: c stays at k*q (remainder 0)
    return float(c)


def _extract_bit(coeff: float, strength: float = 2.0) -> int:
    """Extract one bit: remainder in [0,q), < q/2 -> 0, >= q/2 -> 1."""
    # Read the coefficient back by checking which half of the quantization bin
    # it landed in after compression or other distortions.
    q = max(1.0, strength)
    half = q / 2
    remainder = (coeff % q + q) % q
    return 1 if remainder >= half else 0


def _bytes_to_bits(data: bytes) -> list[int]:
    """Convert bytes to list of bits (LSB first per byte)."""
    # The DCT embedder works bit-by-bit, so payload bytes are expanded here
    # using a consistent least-significant-bit-first ordering.
    bits = []
    for b in data:
        for i in range(8):
            bits.append((b >> i) & 1)
    return bits


def _bits_to_bytes(bits: list[int]) -> Optional[bytes]:
    """Convert bits back to bytes. Returns None if invalid length."""
    if len(bits) % 8 != 0:
        return None
    # Convert extracted bits back into bytes so length parsing and optional ECC
    # decode can work on normal byte strings again.
    out = []
    for i in range(0, len(bits), 8):
        b = sum(bits[i + j] << j for j in range(8))
        out.append(b)
    return bytes(out)


def _embed_payload_in_region(
    region: np.ndarray,
    payload_bits: list[int],
    strength: float = 3.0,
) -> np.ndarray:
    """
    Embed payload bits in 8x8 DCT blocks of a region.
    region: (H,W) or (H,W,3) - will use Y/luminance.
    Returns modified region (same shape).
    """
    # Work in luminance so the hidden payload is less likely to create obvious
    # color shifts in the visible frame.
    if region.ndim == 3:
        yuv = cv2.cvtColor(region, cv2.COLOR_BGR2YCrCb)
        ch = yuv[:, :, 0]  # Y channel
    else:
        ch = region.astype(np.float64)
        yuv = None

    h, w = ch.shape
    out = ch.astype(np.float64).copy()
    bit_idx = 0

    # Walk block-by-block through the ROI and write as many payload bits as
    # the selected coefficients can hold.
    for by in range(0, h - BLOCK_SIZE, BLOCK_SIZE):
        for bx in range(0, w - BLOCK_SIZE, BLOCK_SIZE):
            if bit_idx >= len(payload_bits):
                break
            block = out[by : by + BLOCK_SIZE, bx : bx + BLOCK_SIZE]
            dct_block = _dct2_blocks(block)

            # Each 8x8 block stores several bits by modifying a small set of
            # stable mid-frequency coefficients.
            for ki, (i, j) in enumerate(EMBED_INDICES):
                if bit_idx >= len(payload_bits):
                    break
                bit = payload_bits[bit_idx]
                dct_block[i, j] = _embed_bit(float(dct_block[i, j]), bit, strength)
                bit_idx += 1

            block_out = _idct2_blocks(dct_block)
            out[by : by + BLOCK_SIZE, bx : bx + BLOCK_SIZE] = np.clip(block_out, 0, 255)

        if bit_idx >= len(payload_bits):
            break

    # Reassemble the modified luminance back into the original color frame.
    if yuv is not None:
        yuv[:, :, 0] = np.clip(out, 0, 255).astype(np.uint8)
        return cv2.cvtColor(yuv, cv2.COLOR_YCrCb2BGR)
    return np.clip(out, 0, 255).astype(np.uint8)


def _extract_payload_from_region(
    region: np.ndarray,
    num_bits: int,
    strength: float = 3.0,
) -> list[int]:
    """Extract num_bits from region. Use Y channel to match embed (BGR->YCrCb->Y)."""
    # Extraction mirrors embedding: read the same ROI, same channel, same block
    # order, and the same coefficient positions.
    if region.ndim == 3:
        yuv = cv2.cvtColor(region, cv2.COLOR_BGR2YCrCb)
        gray = yuv[:, :, 0]
    else:
        gray = region

    bits = []
    h, w = gray.shape
    gray_f = gray.astype(np.float64)

    # Read bits back in the exact same traversal order used during embedding.
    for by in range(0, h - BLOCK_SIZE, BLOCK_SIZE):
        for bx in range(0, w - BLOCK_SIZE, BLOCK_SIZE):
            if len(bits) >= num_bits:
                break
            block = gray_f[by : by + BLOCK_SIZE, bx : bx + BLOCK_SIZE]
            dct_block = cv2.dct(block)

            for i, j in EMBED_INDICES:
                if len(bits) >= num_bits:
                    break
                bits.append(_extract_bit(float(dct_block[i, j]), strength))

        if len(bits) >= num_bits:
            break

    return bits[:num_bits]


def embed_dct_watermark(
    frame: np.ndarray,
    payload: bytes,
    roi: tuple[int, int, int, int],
    *,
    strength: float = 3.0,
    use_ecc: bool = True,
) -> np.ndarray:
    """
    Embed payload invisibly in frame ROI using DCT.

    Args:
        frame: BGR frame (H,W,3).
        payload: Raw bytes to embed (will be ECC-encoded if use_ecc).
        roi: (x, y, w, h) - region to embed in (should be low-attention).
        strength: Embedding strength (higher = more robust, less invisible).
        use_ecc: Apply Reed-Solomon before embedding.

    Returns:
        Modified frame (copy).
    """
    # Optionally protect the payload first so a few bit errors later can still
    # be corrected during extraction.
    if use_ecc and HAS_RS:
        rs = RSCodec(RS_NSYM)
        payload = rs.encode(payload)

    # Store the payload length first so the extractor knows how many bits to
    # read back instead of guessing the payload size blindly.
    len_bytes = struct.pack(">H", min(len(payload) * 8, 65535))
    full = len_bytes + payload
    bits = _bytes_to_bits(full)

    x, y, w, h = roi
    region = frame[y : y + h, x : x + w].copy()
    if region.size == 0:
        return frame

    # Embed the payload inside the requested ROI only, leaving the rest of the
    # frame untouched.
    modified = _embed_payload_in_region(region, bits, strength=strength)
    out = frame.copy()
    # Feather the ROI boundary so the hidden watermark does not create a harsh
    # visual seam where the modified region meets the untouched frame.
    feat = min(4, w // 8, h // 8)
    if feat > 0 and region.ndim == 3:
        orig_roi = frame[y : y + h, x : x + w].astype(np.float32)
        mod_f = modified.astype(np.float32)
        yy = np.arange(h)[:, np.newaxis]
        xx = np.arange(w)[np.newaxis, :]
        dist_top = yy
        dist_bot = h - 1 - yy
        dist_left = xx
        dist_right = w - 1 - xx
        alpha_y = np.minimum(np.minimum(dist_top, dist_bot) / feat, 1.0)
        alpha_x = np.minimum(np.minimum(dist_left, dist_right) / feat, 1.0)
        mask_2d = np.minimum(alpha_y, alpha_x)[:, :, np.newaxis]
        blended = (mask_2d * mod_f + (1 - mask_2d) * orig_roi).clip(0, 255).astype(np.uint8)
        out[y : y + h, x : x + w] = blended
    else:
        out[y : y + h, x : x + w] = modified
    return out


def extract_dct_watermark(
    frame: np.ndarray,
    roi: tuple[int, int, int, int],
    *,
    max_bytes: int = 128,
    strength: float = 3.0,
    use_ecc: bool = True,
) -> Optional[bytes]:
    """
    Extract payload from frame ROI.

    Returns:
        Extracted bytes or None on failure.
    """
    x, y, w, h = roi
    region = frame[y : y + h, x : x + w]
    if region.size == 0:
        return None

    # First recover the stored payload length so we know how many following
    # bits should be interpreted as message data.
    len_bits = _extract_payload_from_region(region, 16, strength=strength)
    len_bytes = _bits_to_bytes(len_bits)
    if not len_bytes or len(len_bytes) < 2:
        return None
    num_bits = struct.unpack(">H", len_bytes[:2])[0]
    num_bits = min(num_bits, max_bytes * 8)

    # If the recovered bit count is slightly off because of attack noise,
    # try nearby byte-aligned sizes before giving up completely.
    if num_bits % 8 != 0 and 0 < num_bits <= max_bytes * 8:
        candidates = [num_bits, (num_bits // 8) * 8, ((num_bits // 8) + 1) * 8]
        candidates = [c for c in candidates if 8 <= c <= max_bytes * 8]
    else:
        candidates = [num_bits]

    # Try each candidate payload length until one turns into plausible bytes.
    raw = None
    for nb in candidates:
        payload_bits = _extract_payload_from_region(region, 16 + nb, strength=strength)
        payload_bits = payload_bits[16:16 + nb]
        raw = _bits_to_bytes(payload_bits)
        if raw:
            break
    if not raw:
        return None

    # If ECC was used during embedding, decode it now so the caller receives
    # the original clean payload instead of the protected codeword.
    if use_ecc and HAS_RS:
        try:
            rs = RSCodec(RS_NSYM)
            raw, _ = rs.decode(raw)
        except Exception:
            return None

    return raw
