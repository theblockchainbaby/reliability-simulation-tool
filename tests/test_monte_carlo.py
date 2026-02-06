"""Tests for the Monte Carlo simulation engine."""
from __future__ import annotations

import numpy as np
import pytest

from simulator.device import DeviceConfig, StressProfile
from simulator.monte_carlo import (
    MonteCarloEngine, SimulationConfig, SimulationResult,
    _simulate_single_device, _config_to_dict,
)


class TestSimulationConfig:
    """Test simulation configuration defaults."""

    def test_default_config(self):
        cfg = SimulationConfig()
        assert cfg.num_devices == 1000
        assert cfg.max_time_steps == 8760
        assert cfg.label == "default"

    def test_custom_config(self):
        cfg = SimulationConfig(num_devices=100, max_time_steps=500, label="test")
        assert cfg.num_devices == 100
        assert cfg.max_time_steps == 500


class TestConfigSerialization:
    """Test that DeviceConfig serializes correctly for multiprocessing."""

    def test_config_to_dict_roundtrip(self):
        stress = StressProfile(temperature_c=85, voltage_v=1.35)
        config = DeviceConfig(
            num_bits=65536,
            base_soft_error_rate=1e-5,
            base_hard_failure_rate=1e-7,
            degradation_rate=0.002,
            degradation_exponent=0.25,
            ecc_correctable=1,
            ecc_detectable=2,
            word_size=64,
            stress=stress,
        )
        d = _config_to_dict(config)
        assert d["num_bits"] == 65536
        assert d["temperature_c"] == 85
        assert d["voltage_v"] == 1.35
        assert d["ecc_correctable"] == 1
        assert d["ecc_detectable"] == 2
        assert d["degradation_exponent"] == 0.25


class TestSimulateSingleDevice:
    """Test single-device simulation function."""

    def test_returns_device_result(self):
        config = DeviceConfig()
        config_dict = _config_to_dict(config)
        result = _simulate_single_device((0, config_dict, 100, 42))
        assert hasattr(result, "failed")
        assert hasattr(result, "failure_mode")
        assert hasattr(result, "due_errors")
        assert hasattr(result, "sdc_errors")
        assert result.total_time_steps <= 100

    def test_deterministic_with_seed(self):
        """Same seed should produce same result."""
        config = DeviceConfig(base_soft_error_rate=1e-4)
        config_dict = _config_to_dict(config)
        r1 = _simulate_single_device((0, config_dict, 500, 42))
        r2 = _simulate_single_device((0, config_dict, 500, 42))
        assert r1.failed == r2.failed
        assert r1.soft_errors == r2.soft_errors
        assert r1.failure_time == r2.failure_time


class TestMonteCarloEngine:
    """Test the Monte Carlo engine end-to-end."""

    def test_basic_run(self):
        config = SimulationConfig(num_devices=10, max_time_steps=100)
        engine = MonteCarloEngine()
        result = engine.run(config)
        assert isinstance(result, SimulationResult)
        assert result.num_devices == 10
        assert result.wall_time_sec > 0

    def test_failure_rate_bounds(self):
        """Failure rate should be between 0 and 1."""
        config = SimulationConfig(num_devices=50, max_time_steps=500)
        engine = MonteCarloEngine()
        result = engine.run(config)
        assert 0.0 <= result.failure_rate <= 1.0

    def test_failure_times_sorted(self):
        """failure_times property should return sorted list."""
        config = SimulationConfig(num_devices=50, max_time_steps=500,
                                  device_config=DeviceConfig(
                                      base_soft_error_rate=1e-4))
        engine = MonteCarloEngine()
        result = engine.run(config)
        times = result.failure_times
        assert times == sorted(times)

    def test_sdc_rate_property(self):
        """sdc_rate should be calculated correctly."""
        config = SimulationConfig(
            num_devices=30, max_time_steps=200,
            device_config=DeviceConfig(
                ecc_correctable=0, ecc_detectable=0,
                base_soft_error_rate=1e-3),
            seed=42,
        )
        engine = MonteCarloEngine()
        result = engine.run(config)
        # No ECC = all failures should be SDC
        if result.num_failed > 0:
            assert result.num_sdc_failures > 0
            assert result.sdc_rate == result.num_sdc_failures / result.num_devices

    def test_scenario_comparison(self):
        """run_scenario_comparison should return results for each config."""
        configs = [
            SimulationConfig(num_devices=10, max_time_steps=50, label="A"),
            SimulationConfig(num_devices=10, max_time_steps=50, label="B"),
        ]
        engine = MonteCarloEngine()
        results = engine.run_scenario_comparison(configs)
        assert len(results) == 2
        assert results[0].label == "A"
        assert results[1].label == "B"

    def test_progress_callback(self):
        """Progress callback should be called during simulation."""
        calls = []
        def cb(current, total, **kwargs):
            calls.append((current, total))

        config = SimulationConfig(num_devices=20, max_time_steps=50)
        engine = MonteCarloEngine(progress_callback=cb)
        engine.run(config)
        assert len(calls) > 0
