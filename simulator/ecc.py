"""
Error Correction Code (ECC) modeling.

Models real ECC schemes used in memory systems:
- SEC (Single Error Correction) — Hamming codes
- SECDED (Single Error Correction, Double Error Detection)
- Chipkill — multi-symbol correction for server-grade memory

Tracks correction coverage, detection gaps, and failure thresholds.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ECCScheme(Enum):
    NONE = "none"
    SEC = "sec"           # Single Error Correction (Hamming)
    SECDED = "secded"     # SEC + Double Error Detection
    CHIPKILL = "chipkill"  # Multi-symbol correction


@dataclass
class ECCConfig:
    """ECC scheme parameters."""
    scheme: ECCScheme = ECCScheme.SECDED
    word_size: int = 64          # Data bits per ECC word

    @property
    def correctable_errors(self) -> int:
        """Max errors correctable per word."""
        return {
            ECCScheme.NONE: 0,
            ECCScheme.SEC: 1,
            ECCScheme.SECDED: 1,
            ECCScheme.CHIPKILL: 4,
        }[self.scheme]

    @property
    def detectable_errors(self) -> int:
        """Max errors detectable (but not correctable) per word."""
        return {
            ECCScheme.NONE: 0,
            ECCScheme.SEC: 1,       # Can correct 1, detect 1
            ECCScheme.SECDED: 2,    # Can correct 1, detect 2
            ECCScheme.CHIPKILL: 8,  # Can correct 4, detect 8
        }[self.scheme]

    @property
    def overhead_bits(self) -> int:
        """ECC check bits required per word."""
        return {
            ECCScheme.NONE: 0,
            ECCScheme.SEC: 7,       # Hamming(72,64)
            ECCScheme.SECDED: 8,    # SECDED(72,64)
            ECCScheme.CHIPKILL: 32,  # Reed-Solomon style
        }[self.scheme]

    @property
    def overhead_percent(self) -> float:
        """Storage overhead as percentage."""
        if self.word_size == 0:
            return 0.0
        return (self.overhead_bits / self.word_size) * 100

    @property
    def label(self) -> str:
        """Human-readable label."""
        return {
            ECCScheme.NONE: "No ECC",
            ECCScheme.SEC: "SEC (Hamming)",
            ECCScheme.SECDED: "SECDED",
            ECCScheme.CHIPKILL: "Chipkill",
        }[self.scheme]


@dataclass
class ECCResult:
    """Result of ECC evaluation for a single word at one time step."""
    errors_in_word: int
    corrected: bool
    detected_uncorrectable: bool
    silent_failure: bool  # Errors exceeded detection — undetected corruption

    @property
    def outcome_label(self) -> str:
        if self.errors_in_word == 0:
            return "clean"
        if self.corrected:
            return "corrected"
        if self.detected_uncorrectable:
            return "detected_uncorrectable"
        if self.silent_failure:
            return "silent_data_corruption"
        return "unknown"


class ECCEngine:
    """
    Evaluates error correction for a given ECC configuration.

    Given the number of errors in a word, determines whether they
    are correctable, detectable, or result in silent data corruption.
    """

    def __init__(self, config: ECCConfig):
        self.config = config

    def evaluate(self, errors_in_word: int) -> ECCResult:
        """Evaluate ECC outcome for a word with the given number of errors."""
        if errors_in_word == 0:
            return ECCResult(
                errors_in_word=0,
                corrected=False,
                detected_uncorrectable=False,
                silent_failure=False,
            )

        correctable = errors_in_word <= self.config.correctable_errors
        detectable = errors_in_word <= self.config.detectable_errors

        return ECCResult(
            errors_in_word=errors_in_word,
            corrected=correctable,
            detected_uncorrectable=not correctable and detectable,
            silent_failure=not correctable and not detectable,
        )

    def max_safe_errors(self) -> int:
        """Maximum errors per word before undetected corruption."""
        return self.config.detectable_errors

    def summary(self) -> dict:
        """Return ECC configuration summary."""
        return {
            "scheme": self.config.scheme.value,
            "label": self.config.label,
            "word_size": self.config.word_size,
            "correctable": self.config.correctable_errors,
            "detectable": self.config.detectable_errors,
            "overhead_bits": self.config.overhead_bits,
            "overhead_percent": self.config.overhead_percent,
        }
