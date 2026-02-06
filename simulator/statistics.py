"""
Statistical analysis for reliability data.

Computes standard reliability metrics:
- Reliability function R(t): probability of surviving to time t
- Failure rate / Hazard rate h(t): instantaneous failure intensity
- MTTF: Mean Time To Failure
- Confidence intervals via bootstrap resampling
- Bathtub curve fitting (infant mortality, useful life, wear-out)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats as sp_stats

from .monte_carlo import SimulationResult


@dataclass
class ReliabilityMetrics:
    """Complete reliability analysis output."""
    # Time axis
    time_points: np.ndarray = field(default_factory=lambda: np.array([]))

    # R(t) — probability of survival at each time point
    reliability: np.ndarray = field(default_factory=lambda: np.array([]))

    # F(t) = 1 - R(t) — cumulative failure probability
    failure_cdf: np.ndarray = field(default_factory=lambda: np.array([]))

    # h(t) — hazard rate (instantaneous failure rate)
    hazard_rate: np.ndarray = field(default_factory=lambda: np.array([]))

    # Summary statistics
    mttf: float = 0.0
    mttf_ci_lower: float = 0.0
    mttf_ci_upper: float = 0.0
    median_life: float = 0.0
    failure_rate_total: float = 0.0
    num_devices: int = 0
    num_failed: int = 0

    # Percentile lifetimes
    b1_life: float = 0.0   # Time at which 1% fail
    b10_life: float = 0.0  # Time at which 10% fail
    b50_life: float = 0.0  # Time at which 50% fail

    # Weibull fit parameters
    weibull_shape: float = 0.0  # beta (shape)
    weibull_scale: float = 0.0  # eta (characteristic life)


class ReliabilityAnalyzer:
    """Computes reliability statistics from Monte Carlo simulation results."""

    def __init__(self, confidence_level: float = 0.95, num_bootstrap: int = 1000):
        self.confidence_level = confidence_level
        self.num_bootstrap = num_bootstrap

    def analyze(self, result: SimulationResult) -> ReliabilityMetrics:
        """Run full reliability analysis on simulation results."""
        metrics = ReliabilityMetrics()
        metrics.num_devices = result.num_devices
        metrics.num_failed = result.num_failed
        metrics.failure_rate_total = result.failure_rate

        max_time = result.config.max_time_steps
        failure_times = result.failure_times

        # Time axis — evaluate at each time step
        num_points = min(max_time, 500)
        metrics.time_points = np.linspace(0, max_time, num_points)

        # Compute R(t) — Kaplan-Meier-style survival function
        metrics.reliability = self._compute_reliability(
            failure_times, result.num_devices, metrics.time_points)
        metrics.failure_cdf = 1.0 - metrics.reliability

        # Hazard rate h(t)
        metrics.hazard_rate = self._compute_hazard_rate(
            failure_times, result.num_devices, metrics.time_points)

        # MTTF and confidence interval
        if failure_times:
            # Include censored devices (survived full duration) in MTTF calc
            all_times = list(failure_times) + [max_time] * result.num_survived
            metrics.mttf = float(np.mean(all_times))
            ci_low, ci_high = self._bootstrap_ci(all_times)
            metrics.mttf_ci_lower = ci_low
            metrics.mttf_ci_upper = ci_high
            metrics.median_life = float(np.median(all_times))
        else:
            # No failures — MTTF exceeds simulation duration
            metrics.mttf = float(max_time)
            metrics.mttf_ci_lower = float(max_time)
            metrics.mttf_ci_upper = float(max_time)
            metrics.median_life = float(max_time)

        # B-life percentiles
        metrics.b1_life = self._percentile_life(metrics.time_points,
                                                 metrics.failure_cdf, 0.01)
        metrics.b10_life = self._percentile_life(metrics.time_points,
                                                  metrics.failure_cdf, 0.10)
        metrics.b50_life = self._percentile_life(metrics.time_points,
                                                  metrics.failure_cdf, 0.50)

        # Weibull distribution fit
        if len(failure_times) >= 5:
            self._fit_weibull(failure_times, metrics)

        return metrics

    def _compute_reliability(self, failure_times: list[int],
                             n_devices: int,
                             time_points: np.ndarray) -> np.ndarray:
        """
        Compute R(t) = fraction of devices surviving at time t.
        Uses empirical survival function.
        """
        ft = np.array(failure_times)
        reliability = np.zeros(len(time_points))
        for i, t in enumerate(time_points):
            failed_by_t = np.sum(ft <= t)
            reliability[i] = (n_devices - failed_by_t) / n_devices
        return reliability

    def _compute_hazard_rate(self, failure_times: list[int],
                             n_devices: int,
                             time_points: np.ndarray) -> np.ndarray:
        """
        Compute h(t) = instantaneous failure rate.
        h(t) = f(t) / R(t) where f(t) is the failure density.
        """
        ft = np.array(failure_times) if failure_times else np.array([])
        dt = time_points[1] - time_points[0] if len(time_points) > 1 else 1
        hazard = np.zeros(len(time_points))

        for i, t in enumerate(time_points):
            # Count failures in [t, t+dt)
            if len(ft) > 0:
                failures_in_bin = np.sum((ft >= t) & (ft < t + dt))
            else:
                failures_in_bin = 0
            # Devices at risk at time t
            failed_before = np.sum(ft < t) if len(ft) > 0 else 0
            at_risk = n_devices - failed_before
            if at_risk > 0 and dt > 0:
                hazard[i] = failures_in_bin / (at_risk * dt)

        return hazard

    def _bootstrap_ci(self, times: list) -> tuple[float, float]:
        """Bootstrap confidence interval for MTTF."""
        rng = np.random.default_rng(42)
        arr = np.array(times)
        means = []
        for _ in range(self.num_bootstrap):
            sample = rng.choice(arr, size=len(arr), replace=True)
            means.append(float(np.mean(sample)))

        alpha = 1 - self.confidence_level
        lower = float(np.percentile(means, 100 * alpha / 2))
        upper = float(np.percentile(means, 100 * (1 - alpha / 2)))
        return lower, upper

    def _percentile_life(self, time_points: np.ndarray,
                         cdf: np.ndarray, target: float) -> float:
        """Find time at which target fraction of devices have failed."""
        indices = np.where(cdf >= target)[0]
        if len(indices) == 0:
            return float(time_points[-1])
        return float(time_points[indices[0]])

    def _fit_weibull(self, failure_times: list[int],
                     metrics: ReliabilityMetrics):
        """Fit Weibull distribution to failure data."""
        try:
            data = np.array(failure_times, dtype=float)
            # scipy weibull_min: shape (c), loc, scale
            shape, loc, scale = sp_stats.weibull_min.fit(data, floc=0)
            metrics.weibull_shape = float(shape)
            metrics.weibull_scale = float(scale)
        except Exception:
            # Fit may fail with degenerate data
            pass
