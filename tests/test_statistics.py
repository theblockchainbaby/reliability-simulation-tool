"""Tests for reliability statistics analysis."""
from __future__ import annotations

import numpy as np
import pytest

from simulator.device import DeviceConfig
from simulator.monte_carlo import MonteCarloEngine, SimulationConfig
from simulator.statistics import ReliabilityAnalyzer, ReliabilityMetrics


class TestReliabilityAnalyzer:
    """Test reliability metric computation."""

    def _run_sim(self, n=100, steps=500, soft_rate=1e-4, seed=42):
        config = SimulationConfig(
            num_devices=n, max_time_steps=steps,
            device_config=DeviceConfig(base_soft_error_rate=soft_rate),
            seed=seed,
        )
        engine = MonteCarloEngine()
        return engine.run(config)

    def test_analyze_returns_metrics(self):
        result = self._run_sim()
        analyzer = ReliabilityAnalyzer()
        metrics = analyzer.analyze(result)
        assert isinstance(metrics, ReliabilityMetrics)
        assert metrics.num_devices == 100

    def test_reliability_starts_at_one(self):
        """R(0) should be 1.0 (all devices alive at t=0)."""
        result = self._run_sim()
        analyzer = ReliabilityAnalyzer()
        metrics = analyzer.analyze(result)
        assert metrics.reliability[0] == 1.0

    def test_reliability_decreasing(self):
        """R(t) should be non-increasing."""
        result = self._run_sim(soft_rate=1e-3)
        analyzer = ReliabilityAnalyzer()
        metrics = analyzer.analyze(result)
        diffs = np.diff(metrics.reliability)
        assert np.all(diffs <= 1e-10)

    def test_failure_cdf_complement(self):
        """F(t) should equal 1 - R(t)."""
        result = self._run_sim()
        analyzer = ReliabilityAnalyzer()
        metrics = analyzer.analyze(result)
        np.testing.assert_allclose(
            metrics.failure_cdf, 1.0 - metrics.reliability, atol=1e-10)

    def test_mttf_has_confidence_interval(self):
        """MTTF CI should bracket the point estimate."""
        result = self._run_sim(n=200, soft_rate=1e-3)
        analyzer = ReliabilityAnalyzer()
        metrics = analyzer.analyze(result)
        assert metrics.mttf_ci_lower <= metrics.mttf <= metrics.mttf_ci_upper

    def test_blife_ordering(self):
        """B1 <= B10 <= B50 life."""
        result = self._run_sim(n=200, soft_rate=1e-3)
        analyzer = ReliabilityAnalyzer()
        metrics = analyzer.analyze(result)
        assert metrics.b1_life <= metrics.b10_life <= metrics.b50_life

    def test_weibull_fit_with_enough_data(self):
        """Weibull parameters should be fitted when enough failures exist."""
        result = self._run_sim(n=200, steps=1000, soft_rate=1e-3)
        analyzer = ReliabilityAnalyzer()
        metrics = analyzer.analyze(result)
        if result.num_failed >= 5:
            assert metrics.weibull_shape > 0
            assert metrics.weibull_scale > 0

    def test_no_failures_mttf_equals_max_time(self):
        """With no failures, MTTF should equal simulation duration."""
        config = SimulationConfig(
            num_devices=10, max_time_steps=100,
            device_config=DeviceConfig(
                base_soft_error_rate=0, base_hard_failure_rate=0),
            seed=42,
        )
        engine = MonteCarloEngine()
        result = engine.run(config)
        analyzer = ReliabilityAnalyzer()
        metrics = analyzer.analyze(result)
        assert result.num_failed == 0
        assert metrics.mttf == 100.0

    def test_hazard_rate_nonnegative(self):
        """Hazard rate should be non-negative."""
        result = self._run_sim(soft_rate=1e-3)
        analyzer = ReliabilityAnalyzer()
        metrics = analyzer.analyze(result)
        assert np.all(metrics.hazard_rate >= 0)
