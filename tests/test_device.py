"""Tests for the Device model and error injection engine."""
from __future__ import annotations

import numpy as np
import pytest

from simulator.device import Device, DeviceConfig, StressProfile, FailureType


class TestStressProfile:
    """Test Arrhenius and voltage acceleration factors."""

    def test_reference_conditions_give_factor_one(self):
        """At reference temp/voltage, acceleration factor should be ~1.0."""
        stress = StressProfile(temperature_c=55, voltage_v=1.2,
                               ref_temperature_c=55, ref_voltage_v=1.2)
        assert abs(stress.temperature_factor - 1.0) < 1e-6
        assert abs(stress.voltage_factor - 1.0) < 1e-6
        assert abs(stress.combined_factor - 1.0) < 1e-6

    def test_higher_temperature_increases_factor(self):
        """Higher temperature should increase the Arrhenius acceleration."""
        low = StressProfile(temperature_c=35)
        high = StressProfile(temperature_c=95)
        assert high.temperature_factor > low.temperature_factor

    def test_higher_voltage_increases_factor(self):
        """Higher voltage should increase the voltage acceleration."""
        low = StressProfile(voltage_v=1.0)
        high = StressProfile(voltage_v=1.35)
        assert high.voltage_factor > low.voltage_factor

    def test_arrhenius_known_value(self):
        """Spot-check Arrhenius factor with known physics constants."""
        stress = StressProfile(temperature_c=85, ref_temperature_c=55,
                               activation_energy_ev=0.7)
        # AF should be significantly > 1 at 85C vs 55C reference
        assert stress.temperature_factor > 2.0
        assert stress.temperature_factor < 20.0


class TestDevice:
    """Test device error injection and three-way ECC evaluation."""

    def _make_device(self, ecc_correctable=1, ecc_detectable=2,
                     soft_rate=1e-4, hard_rate=0, seed=42):
        config = DeviceConfig(
            num_bits=65536,
            base_soft_error_rate=soft_rate,
            base_hard_failure_rate=hard_rate,
            degradation_rate=0.0,
            ecc_correctable=ecc_correctable,
            ecc_detectable=ecc_detectable,
            word_size=64,
        )
        rng = np.random.default_rng(seed)
        return Device(config, rng=rng)

    def test_no_ecc_fails_immediately(self):
        """With no ECC and high error rate, device should fail quickly."""
        device = self._make_device(ecc_correctable=0, ecc_detectable=0,
                                   soft_rate=1e-2)
        for _ in range(100):
            device.step()
            if device.failed:
                break
        assert device.failed
        assert device.failure_mode == "sdc"  # No ECC = silent corruption

    def test_strong_ecc_survives(self):
        """With strong ECC and low error rate, device should survive."""
        device = self._make_device(ecc_correctable=4, ecc_detectable=8,
                                   soft_rate=1e-7)
        for _ in range(1000):
            device.step()
        assert not device.failed

    def test_sec_fails_with_sdc(self):
        """SEC (correct=1, detect=1) should produce SDC on multi-bit errors."""
        device = self._make_device(ecc_correctable=1, ecc_detectable=1,
                                   soft_rate=1e-2)
        for _ in range(1000):
            device.step()
            if device.failed:
                break
        if device.failed:
            assert device.failure_mode == "sdc"

    def test_secded_fails_with_due(self):
        """SECDED (correct=1, detect=2) should produce DUE on 2-bit errors.

        Uses a low error rate so 2-bit events are rare but 3-bit events
        are essentially impossible, ensuring the first uncorrectable
        event is a 2-bit DUE, not 3-bit SDC.
        """
        device = self._make_device(ecc_correctable=1, ecc_detectable=2,
                                   soft_rate=1e-5, seed=7)
        for _ in range(50000):
            device.step()
            if device.failed:
                break
        if device.failed:
            assert device.failure_mode == "due"

    def test_degradation_increases_over_time(self):
        """Degradation factor should increase with time steps."""
        config = DeviceConfig(degradation_rate=0.01, degradation_exponent=1.0)
        device = Device(config)
        device.step()
        d1 = device.degradation_factor
        for _ in range(99):
            device.step()
        d100 = device.degradation_factor
        assert d100 > d1

    def test_power_law_degradation(self):
        """With exponent < 1, degradation should grow sublinearly."""
        config_linear = DeviceConfig(degradation_rate=0.01,
                                     degradation_exponent=1.0)
        config_nbti = DeviceConfig(degradation_rate=0.01,
                                   degradation_exponent=0.25)
        dev_linear = Device(config_linear)
        dev_nbti = Device(config_nbti)

        for _ in range(1000):
            dev_linear.step()
            dev_nbti.step()
            if dev_linear.failed or dev_nbti.failed:
                break

        # NBTI power-law should degrade slower than linear
        assert dev_nbti.degradation_factor < dev_linear.degradation_factor

    def test_summary_fields(self):
        """get_summary should return all expected keys."""
        device = self._make_device()
        device.step()
        summary = device.get_summary()
        expected_keys = {
            "time_steps", "failed", "failure_time", "failure_mode",
            "total_soft_errors", "total_hard_failures", "corrected_errors",
            "due_count", "sdc_count", "final_degradation_factor",
            "hard_fault_count",
        }
        assert expected_keys == set(summary.keys())


class TestECCDifferentiation:
    """Verify SEC and SECDED produce observably different outcomes."""

    def test_sec_vs_secded_failure_modes_differ(self):
        """Run many devices — SEC should have SDC, SECDED should have DUE."""
        n_devices = 50
        sec_sdc = 0
        secded_due = 0

        for i in range(n_devices):
            # SEC device
            cfg_sec = DeviceConfig(
                num_bits=65536, base_soft_error_rate=1e-3,
                base_hard_failure_rate=0, degradation_rate=0,
                ecc_correctable=1, ecc_detectable=1, word_size=64,
            )
            dev = Device(cfg_sec, rng=np.random.default_rng(i))
            for _ in range(500):
                dev.step()
                if dev.failed:
                    break
            if dev.failed and dev.failure_mode == "sdc":
                sec_sdc += 1

            # SECDED device
            cfg_secded = DeviceConfig(
                num_bits=65536, base_soft_error_rate=1e-3,
                base_hard_failure_rate=0, degradation_rate=0,
                ecc_correctable=1, ecc_detectable=2, word_size=64,
            )
            dev = Device(cfg_secded, rng=np.random.default_rng(i))
            for _ in range(500):
                dev.step()
                if dev.failed:
                    break
            if dev.failed and dev.failure_mode == "due":
                secded_due += 1

        # At least some SEC devices should have SDC
        assert sec_sdc > 0, "SEC should produce SDC failures"
        # At least some SECDED devices should have DUE
        assert secded_due > 0, "SECDED should produce DUE failures"
