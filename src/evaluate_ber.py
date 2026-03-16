"""
BER (Bit Error Rate) and recovery accuracy evaluation (proposal-aligned).

Metrics from proposal:
- Robustness: Bit error rate (BER), detection AUC after attack suite
- Forensic Traceability: Accuracy of recovering session IDs from leaked copies
"""

from __future__ import annotations


def compute_ber(original_bits: list[int], extracted_bits: list[int]) -> float:
    """
    Bit Error Rate = (bit errors) / (total bits compared).

    Args:
        original_bits: Ground truth bits.
        extracted_bits: Recovered bits (may be shorter).
    Returns:
        BER in [0, 1]. 0 = perfect recovery.
    """
    n = min(len(original_bits), len(extracted_bits))
    if n == 0:
        return 1.0
    errors = sum(1 for i in range(n) if original_bits[i] != extracted_bits[i])
    return errors / n


def bytes_to_bits(data: bytes) -> list[int]:
    """LSB first per byte."""
    bits = []
    for b in data:
        for i in range(8):
            bits.append((b >> i) & 1)
    return bits


def compute_payload_ber(original: bytes, extracted: bytes | None) -> float:
    """
    BER between original payload and extracted payload.
    If extracted is None or wrong length, returns 1.0.
    """
    if extracted is None or len(extracted) == 0:
        return 1.0
    ob = bytes_to_bits(original)
    eb = bytes_to_bits(extracted)
    return compute_ber(ob, eb)


def recovery_accuracy(correct: int, total: int) -> float:
    """Fraction of frames/clips where payload was correctly recovered."""
    if total == 0:
        return 0.0
    return 100.0 * correct / total
