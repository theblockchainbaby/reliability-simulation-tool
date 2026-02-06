"""
Monte Carlo simulation engine.

Runs thousands of independent device lifetime simulations to produce
statistically significant reliability estimates. Supports parallel
execution and progress tracking.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from .device import Device, DeviceConfig, StressProfile
from .ecc import ECCScheme, ECCConfig


@dataclass
class DeviceResult:
    """Outcome of a single device lifetime simulation."""
    device_id: int
    failed: bool
    failure_time: int | None
    total_time_steps: int
    soft_errors: int
    hard_failures: int
    corrected_errors: int
    uncorrectable_errors: int
    final_degradation: float


@dataclass
class SimulationConfig:
    """Parameters for a Monte Carlo simulation run."""
    num_devices: int = 1000
    max_time_steps: int = 8760       # 1 year in hours (default time unit)
    device_config: DeviceConfig = field(default_factory=DeviceConfig)
    seed: int | None = None
    label: str = "default"


@dataclass
class SimulationResult:
    """Aggregated results from a Monte Carlo run."""
    config: SimulationConfig
    device_results: list[DeviceResult]
    wall_time_sec: float
    label: str = ""

    @property
    def num_devices(self) -> int:
        return len(self.device_results)

    @property
    def num_failed(self) -> int:
        return sum(1 for d in self.device_results if d.failed)

    @property
    def num_survived(self) -> int:
        return self.num_devices - self.num_failed

    @property
    def failure_rate(self) -> float:
        if self.num_devices == 0:
            return 0.0
        return self.num_failed / self.num_devices

    @property
    def failure_times(self) -> list[int]:
        """Sorted failure times for devices that failed."""
        return sorted(d.failure_time for d in self.device_results
                      if d.failed and d.failure_time is not None)

    @property
    def mttf(self) -> float | None:
        """Mean Time To Failure (only for failed devices)."""
        times = self.failure_times
        if not times:
            return None
        return float(np.mean(times))


def _simulate_single_device(args: tuple) -> DeviceResult:
    """Simulate one device lifetime. Used for parallel execution."""
    device_id, config_dict, max_steps, seed = args

    # Reconstruct config from dict (needed for multiprocessing serialization)
    stress = StressProfile(
        temperature_c=config_dict["temperature_c"],
        voltage_v=config_dict["voltage_v"],
        ref_temperature_c=config_dict["ref_temperature_c"],
        ref_voltage_v=config_dict["ref_voltage_v"],
        activation_energy_ev=config_dict["activation_energy_ev"],
        voltage_exponent=config_dict["voltage_exponent"],
    )
    device_cfg = DeviceConfig(
        num_bits=config_dict["num_bits"],
        base_soft_error_rate=config_dict["base_soft_error_rate"],
        base_hard_failure_rate=config_dict["base_hard_failure_rate"],
        degradation_rate=config_dict["degradation_rate"],
        ecc_capability=config_dict["ecc_capability"],
        word_size=config_dict["word_size"],
        stress=stress,
    )

    rng = np.random.default_rng(seed + device_id if seed is not None else None)
    device = Device(device_cfg, rng=rng)

    for _ in range(max_steps):
        device.step()
        if device.failed:
            break

    summary = device.get_summary()
    return DeviceResult(
        device_id=device_id,
        failed=summary["failed"],
        failure_time=summary["failure_time"],
        total_time_steps=summary["time_steps"],
        soft_errors=summary["total_soft_errors"],
        hard_failures=summary["total_hard_failures"],
        corrected_errors=summary["corrected_errors"],
        uncorrectable_errors=summary["uncorrectable_errors"],
        final_degradation=summary["final_degradation_factor"],
    )


def _config_to_dict(config: DeviceConfig) -> dict:
    """Serialize DeviceConfig for multiprocessing."""
    return {
        "num_bits": config.num_bits,
        "base_soft_error_rate": config.base_soft_error_rate,
        "base_hard_failure_rate": config.base_hard_failure_rate,
        "degradation_rate": config.degradation_rate,
        "ecc_capability": config.ecc_capability,
        "word_size": config.word_size,
        "temperature_c": config.stress.temperature_c,
        "voltage_v": config.stress.voltage_v,
        "ref_temperature_c": config.stress.ref_temperature_c,
        "ref_voltage_v": config.stress.ref_voltage_v,
        "activation_energy_ev": config.stress.activation_energy_ev,
        "voltage_exponent": config.stress.voltage_exponent,
    }


class MonteCarloEngine:
    """
    Runs Monte Carlo reliability simulations.

    Each trial simulates a complete device lifetime under specified
    stress and ECC conditions. Results are aggregated for statistical analysis.
    """

    def __init__(self, progress_callback=None):
        self.progress_callback = progress_callback

    def run(self, config: SimulationConfig) -> SimulationResult:
        """
        Execute Monte Carlo simulation.

        Runs config.num_devices independent lifetime simulations
        and returns aggregated results.
        """
        start = time.time()
        config_dict = _config_to_dict(config.device_config)

        results = []
        for i in range(config.num_devices):
            args = (i, config_dict, config.max_time_steps, config.seed)
            result = _simulate_single_device(args)
            results.append(result)

            if self.progress_callback and (i + 1) % max(1, config.num_devices // 20) == 0:
                self.progress_callback(i + 1, config.num_devices)

        elapsed = time.time() - start

        return SimulationResult(
            config=config,
            device_results=results,
            wall_time_sec=elapsed,
            label=config.label,
        )

    def run_scenario_comparison(self, configs: list[SimulationConfig]) -> list[SimulationResult]:
        """Run multiple scenarios for comparison."""
        results = []
        for config in configs:
            if self.progress_callback:
                self.progress_callback(0, config.num_devices,
                                       label=config.label)
            result = self.run(config)
            results.append(result)
        return results
