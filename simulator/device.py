"""
Device model and error injection engine.

Each simulated device has a base error rate, stress environment, and
accumulates soft errors, hard failures, and degradation over time.
Error probability follows: λ_eff = λ_base × stress_factor(T, V) × degradation(t)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math

import numpy as np


class FailureType(Enum):
    SOFT_ERROR = "soft_error"          # Temporary bit flip, correctable
    HARD_FAILURE = "hard_failure"      # Permanent cell defect
    DEGRADATION = "degradation"        # Progressive wear, increases error rate


@dataclass
class StressProfile:
    """Environmental stress conditions affecting failure rates."""
    temperature_c: float = 55.0       # Operating temperature (Celsius)
    voltage_v: float = 1.2            # Supply voltage
    ref_temperature_c: float = 55.0   # Reference temperature for Arrhenius
    ref_voltage_v: float = 1.2        # Reference voltage
    activation_energy_ev: float = 0.7  # Activation energy (eV), typical for DRAM
    boltzmann_ev: float = 8.617e-5    # Boltzmann constant in eV/K
    voltage_exponent: float = 2.0     # Voltage acceleration exponent

    @property
    def temperature_factor(self) -> float:
        """Arrhenius acceleration factor for temperature stress."""
        t_op = self.temperature_c + 273.15   # Convert to Kelvin
        t_ref = self.ref_temperature_c + 273.15
        exponent = (self.activation_energy_ev / self.boltzmann_ev) * (1/t_ref - 1/t_op)
        return math.exp(exponent)

    @property
    def voltage_factor(self) -> float:
        """Voltage acceleration factor."""
        if self.ref_voltage_v == 0:
            return 1.0
        return (self.voltage_v / self.ref_voltage_v) ** self.voltage_exponent

    @property
    def combined_factor(self) -> float:
        """Total stress acceleration factor."""
        return self.temperature_factor * self.voltage_factor


@dataclass
class DeviceConfig:
    """Configuration for a simulated memory device."""
    num_bits: int = 1_048_576          # 1 Mbit device
    base_soft_error_rate: float = 1e-6  # Soft errors per bit per time step
    base_hard_failure_rate: float = 1e-8  # Hard failures per bit per time step
    degradation_rate: float = 0.001    # Rate at which error rate increases per step
    degradation_exponent: float = 1.0  # 1.0=linear, 0.25=NBTI power-law
    ecc_correctable: int = 1           # Max correctable errors per word
    ecc_detectable: int = 1            # Max detectable errors per word
    word_size: int = 64                # Bits per ECC word
    stress: StressProfile = field(default_factory=StressProfile)


class Device:
    """
    Simulated memory device that accumulates errors over time.

    Models three failure mechanisms:
    - Soft errors: random transient bit flips (cosmic rays, alpha particles)
    - Hard failures: permanent cell defects from oxide breakdown, electromigration
    - Degradation: progressive wear that increases the effective error rate

    ECC evaluation uses three-way outcome:
    - Corrected: errors <= ecc_correctable
    - Detected Uncorrectable (DUE): errors > correctable but <= detectable
    - Silent Data Corruption (SDC): errors > detectable (undetected corruption)
    """

    def __init__(self, config: DeviceConfig, rng: np.random.Generator | None = None):
        self.config = config
        self.rng = rng or np.random.default_rng()

        self.num_words = config.num_bits // config.word_size
        # Track hard faults per word (not per bit) for speed
        self.hard_faults_per_word = np.zeros(self.num_words, dtype=np.int32)
        self.total_hard_faults = 0
        self.degradation_factor = 1.0

        # Counters — three-way ECC outcome tracking
        self.total_soft_errors = 0
        self.total_hard_failures = 0
        self.corrected_count = 0          # ECC corrected successfully
        self.due_count = 0                # Detected Uncorrectable Errors
        self.sdc_count = 0                # Silent Data Corruption (worst case)
        self.time_step = 0
        self.failed = False
        self.failure_time: int | None = None
        self.failure_mode: str | None = None  # "due" or "sdc"

        # Cache stress factor (doesn't change during simulation)
        self._stress_factor = config.stress.combined_factor

    def step(self):
        """
        Advance one time step. Inject errors probabilistically and
        check if ECC can handle them.
        """
        if self.failed:
            return

        self.time_step += 1

        # Apply degradation: error rate increases over time (power-law)
        self.degradation_factor = 1.0 + self.config.degradation_rate * (
            self.time_step ** self.config.degradation_exponent)
        stress_deg = self._stress_factor * self.degradation_factor

        # --- Soft error injection ---
        expected_soft = self.config.base_soft_error_rate * self.config.num_bits * stress_deg
        num_soft = self.rng.poisson(expected_soft)

        # --- Hard failure injection ---
        healthy = self.config.num_bits - self.total_hard_faults
        expected_hard = self.config.base_hard_failure_rate * healthy * stress_deg
        num_hard = self.rng.poisson(expected_hard)

        if num_soft == 0 and num_hard == 0:
            return

        self.total_soft_errors += num_soft
        self.total_hard_failures += num_hard

        # Assign hard faults to random words
        if num_hard > 0:
            hard_words = self.rng.integers(0, self.num_words, size=num_hard)
            for w in hard_words:
                self.hard_faults_per_word[w] += 1
            self.total_hard_faults += num_hard

        # Assign soft errors to random words and check ECC
        ecc_correct = self.config.ecc_correctable
        ecc_detect = self.config.ecc_detectable

        if num_soft > 0:
            soft_words = self.rng.integers(0, self.num_words, size=num_soft)
            # Count soft errors per word this step
            word_soft_counts: dict[int, int] = {}
            for w in soft_words:
                w_int = int(w)
                word_soft_counts[w_int] = word_soft_counts.get(w_int, 0) + 1

            # Three-way ECC evaluation per word
            for word_idx, soft_count in word_soft_counts.items():
                total_errors = int(self.hard_faults_per_word[word_idx]) + soft_count

                if total_errors <= ecc_correct:
                    # ECC corrects all errors — no impact
                    self.corrected_count += soft_count
                elif total_errors <= ecc_detect:
                    # Detected but uncorrectable — system knows data is bad
                    self.due_count += soft_count
                    if not self.failed:
                        self.failed = True
                        self.failure_time = self.time_step
                        self.failure_mode = "due"
                        return
                else:
                    # Silent data corruption — errors exceed detection
                    self.sdc_count += soft_count
                    if not self.failed:
                        self.failed = True
                        self.failure_time = self.time_step
                        self.failure_mode = "sdc"
                        return

        # Also check words that got new hard faults (even without soft errors)
        if num_hard > 0 and not self.failed:
            hard_words_set = set(int(w) for w in hard_words)
            for word_idx in hard_words_set:
                hard_count = int(self.hard_faults_per_word[word_idx])
                if hard_count <= ecc_correct:
                    pass  # Correctable
                elif hard_count <= ecc_detect:
                    self.due_count += 1
                    if not self.failed:
                        self.failed = True
                        self.failure_time = self.time_step
                        self.failure_mode = "due"
                        return
                else:
                    self.sdc_count += 1
                    if not self.failed:
                        self.failed = True
                        self.failure_time = self.time_step
                        self.failure_mode = "sdc"
                        return

    def get_summary(self) -> dict:
        """Return device lifetime summary."""
        return {
            "time_steps": self.time_step,
            "failed": self.failed,
            "failure_time": self.failure_time,
            "failure_mode": self.failure_mode,
            "total_soft_errors": self.total_soft_errors,
            "total_hard_failures": self.total_hard_failures,
            "corrected_errors": self.corrected_count,
            "due_count": self.due_count,
            "sdc_count": self.sdc_count,
            "final_degradation_factor": self.degradation_factor,
            "hard_fault_count": self.total_hard_faults,
        }
