#!/usr/bin/env python3
"""
Reliability Simulation Tool
============================
Models how hardware systems accumulate failures and how error correction
mechanisms extend usable life via Monte Carlo simulation.

Usage:
    python main.py simulate                     # Default simulation
    python main.py simulate -n 5000 -t 17520    # 5000 devices, 2 years
    python main.py temperature                  # Compare temperature effects
    python main.py ecc                          # Compare ECC schemes
    python main.py voltage                      # Compare voltage stress
    python main.py full                         # All scenario comparisons
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np

from simulator.device import DeviceConfig, StressProfile
from simulator.ecc import ECCScheme, ECCConfig
from simulator.monte_carlo import MonteCarloEngine, SimulationConfig, SimulationResult
from simulator.statistics import ReliabilityAnalyzer, ReliabilityMetrics
from simulator.visualizer import ReliabilityVisualizer


# ── Terminal formatting ───────────────────────────────────────────────
BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[91m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"
RESET = "\033[0m"

BAR_FULL = "\u2588"
BAR_EMPTY = "\u2591"


def progress_bar(current: int, total: int, width: int = 30, label: str = "") -> str:
    pct = current / total if total > 0 else 0
    filled = int(pct * width)
    bar = f"{GREEN}{BAR_FULL * filled}{DIM}{BAR_EMPTY * (width - filled)}{RESET}"
    prefix = f"  {label}: " if label else "  "
    return f"\r{prefix}{bar} {current:,}/{total:,} ({pct:.0%})"


def print_progress(current, total, label=""):
    sys.stdout.write(progress_bar(current, total, label=label))
    sys.stdout.flush()
    if current >= total:
        print()


# ── Scenario builders ─────────────────────────────────────────────────
def build_temperature_scenarios(num_devices: int, max_steps: int,
                                base_config: DeviceConfig) -> list[SimulationConfig]:
    """Build scenarios comparing different operating temperatures."""
    temps = [35, 55, 75, 95]
    configs = []
    for temp in temps:
        stress = StressProfile(temperature_c=temp)
        device_cfg = DeviceConfig(
            num_bits=base_config.num_bits,
            base_soft_error_rate=base_config.base_soft_error_rate,
            base_hard_failure_rate=base_config.base_hard_failure_rate,
            degradation_rate=base_config.degradation_rate,
            ecc_capability=base_config.ecc_capability,
            word_size=base_config.word_size,
            stress=stress,
        )
        configs.append(SimulationConfig(
            num_devices=num_devices,
            max_time_steps=max_steps,
            device_config=device_cfg,
            seed=42,
            label=f"{temp}°C",
        ))
    return configs


def build_ecc_scenarios(num_devices: int, max_steps: int,
                        base_config: DeviceConfig) -> list[SimulationConfig]:
    """Build scenarios comparing ECC schemes."""
    schemes = [
        (ECCScheme.NONE, "No ECC", 0),
        (ECCScheme.SEC, "SEC (Hamming)", 1),
        (ECCScheme.SECDED, "SECDED", 1),
        (ECCScheme.CHIPKILL, "Chipkill", 4),
    ]
    configs = []
    for scheme, label, capability in schemes:
        device_cfg = DeviceConfig(
            num_bits=base_config.num_bits,
            base_soft_error_rate=base_config.base_soft_error_rate,
            base_hard_failure_rate=base_config.base_hard_failure_rate,
            degradation_rate=base_config.degradation_rate,
            ecc_capability=capability,
            word_size=base_config.word_size,
            stress=base_config.stress,
        )
        configs.append(SimulationConfig(
            num_devices=num_devices,
            max_time_steps=max_steps,
            device_config=device_cfg,
            seed=42,
            label=label,
        ))
    return configs


def build_voltage_scenarios(num_devices: int, max_steps: int,
                            base_config: DeviceConfig) -> list[SimulationConfig]:
    """Build scenarios comparing voltage stress levels."""
    voltages = [1.0, 1.1, 1.2, 1.35]
    configs = []
    for v in voltages:
        stress = StressProfile(voltage_v=v)
        device_cfg = DeviceConfig(
            num_bits=base_config.num_bits,
            base_soft_error_rate=base_config.base_soft_error_rate,
            base_hard_failure_rate=base_config.base_hard_failure_rate,
            degradation_rate=base_config.degradation_rate,
            ecc_capability=base_config.ecc_capability,
            word_size=base_config.word_size,
            stress=stress,
        )
        configs.append(SimulationConfig(
            num_devices=num_devices,
            max_time_steps=max_steps,
            device_config=device_cfg,
            seed=42,
            label=f"{v}V",
        ))
    return configs


# ── Run + analyze + print ─────────────────────────────────────────────
def run_and_analyze(configs: list[SimulationConfig],
                    output_dir: str, scenario_name: str) -> tuple[list[SimulationResult], list[ReliabilityMetrics]]:
    """Run simulations, analyze, visualize, and print results."""
    engine = MonteCarloEngine(progress_callback=print_progress)
    analyzer = ReliabilityAnalyzer()
    vis = ReliabilityVisualizer(output_dir=output_dir)

    all_results = []
    all_metrics = []

    for config in configs:
        print(f"\n  {BOLD}{MAGENTA}[Simulating] {config.label}{RESET}")
        print(f"  {DIM}{config.num_devices:,} devices × {config.max_time_steps:,} time steps{RESET}")

        result = engine.run(config)
        metrics = analyzer.analyze(result)
        all_results.append(result)
        all_metrics.append(metrics)

        print(f"  {GREEN}Done{RESET} in {result.wall_time_sec:.1f}s — "
              f"{metrics.num_failed:,} failures ({metrics.failure_rate_total:.2%}), "
              f"MTTF={metrics.mttf:,.0f}h")

    # Generate charts
    print(f"\n  {BOLD}Generating charts...{RESET}")
    prefix = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{scenario_name}"
    chart_paths = vis.generate_full_report(all_results, all_metrics, prefix=prefix)

    # Print comparison table
    print_comparison_table(all_results, all_metrics)

    # Print charts
    print(f"\n  {BOLD}Generated Charts{RESET}")
    for p in chart_paths:
        print(f"    {GREEN}{p}{RESET}")

    return all_results, all_metrics


def print_comparison_table(results: list[SimulationResult],
                           metrics: list[ReliabilityMetrics]):
    """Print formatted comparison table."""
    print(f"\n{BOLD}{CYAN}{'=' * 80}")
    print(f"  Reliability Comparison")
    print(f"{'=' * 80}{RESET}\n")

    # Header
    print(f"  {'Scenario':<20s} {'Failures':>10s} {'Rate':>10s} {'MTTF (h)':>12s} "
          f"{'B10 Life':>10s} {'Weibull β':>10s}")
    print(f"  {'─' * 20} {'─' * 10} {'─' * 10} {'─' * 12} {'─' * 10} {'─' * 10}")

    for res, met in zip(results, metrics):
        label = (res.label or "—")[:20]
        beta = f"{met.weibull_shape:.3f}" if met.weibull_shape > 0 else "—"
        print(f"  {label:<20s} {met.num_failed:>10,d} {met.failure_rate_total:>10.2%} "
              f"{met.mttf:>12,.0f} {met.b10_life:>10,.0f} {beta:>10s}")

    # Best scenario
    best = min(zip(results, metrics), key=lambda x: x[1].failure_rate_total)
    print(f"\n  {GREEN}Best: {best[0].label} — {best[1].failure_rate_total:.2%} failure rate, "
          f"MTTF {best[1].mttf:,.0f}h{RESET}")

    # Worst scenario
    worst = max(zip(results, metrics), key=lambda x: x[1].failure_rate_total)
    if worst[0].label != best[0].label:
        print(f"  {RED}Worst: {worst[0].label} — {worst[1].failure_rate_total:.2%} failure rate, "
              f"MTTF {worst[1].mttf:,.0f}h{RESET}")

    print(f"\n{CYAN}{'=' * 80}{RESET}")


# ── Export ────────────────────────────────────────────────────────────
def export_json(results: list[SimulationResult],
                metrics: list[ReliabilityMetrics],
                output_dir: str) -> str:
    """Export results to JSON."""
    data = {
        "timestamp": datetime.now().isoformat(),
        "scenarios": [],
    }
    for res, met in zip(results, metrics):
        scenario = {
            "label": res.label,
            "num_devices": met.num_devices,
            "num_failed": met.num_failed,
            "failure_rate": met.failure_rate_total,
            "mttf": met.mttf,
            "mttf_ci_95": [met.mttf_ci_lower, met.mttf_ci_upper],
            "median_life": met.median_life,
            "b1_life": met.b1_life,
            "b10_life": met.b10_life,
            "b50_life": met.b50_life,
            "weibull_shape": met.weibull_shape,
            "weibull_scale": met.weibull_scale,
            "total_soft_errors": sum(d.soft_errors for d in res.device_results),
            "total_hard_failures": sum(d.hard_failures for d in res.device_results),
            "total_corrected": sum(d.corrected_errors for d in res.device_results),
            "total_uncorrectable": sum(d.uncorrectable_errors for d in res.device_results),
            "wall_time_sec": res.wall_time_sec,
        }
        # Device config
        cfg = res.config.device_config
        scenario["config"] = {
            "num_bits": cfg.num_bits,
            "base_soft_error_rate": cfg.base_soft_error_rate,
            "base_hard_failure_rate": cfg.base_hard_failure_rate,
            "degradation_rate": cfg.degradation_rate,
            "ecc_capability": cfg.ecc_capability,
            "temperature_c": cfg.stress.temperature_c,
            "voltage_v": cfg.stress.voltage_v,
        }
        data["scenarios"].append(scenario)

    path = os.path.join(output_dir, "reliability_report.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path


# ── Main ──────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reliability Simulation Tool — Monte Carlo hardware failure modeling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", help="Command to run")

    # Common arguments
    def add_common(p):
        p.add_argument("-n", "--num-devices", type=int, default=1000,
                       help="Number of devices to simulate (default: 1000)")
        p.add_argument("-t", "--time-steps", type=int, default=8760,
                       help="Max time steps per device, in hours (default: 8760 = 1 year)")
        p.add_argument("-o", "--output", type=str, default="reports",
                       help="Output directory (default: reports)")
        p.add_argument("--soft-rate", type=float, default=1e-6,
                       help="Base soft error rate per bit per step (default: 1e-6)")
        p.add_argument("--hard-rate", type=float, default=1e-8,
                       help="Base hard failure rate per bit per step (default: 1e-8)")
        p.add_argument("--degradation", type=float, default=0.001,
                       help="Degradation rate (default: 0.001)")
        p.add_argument("--bits", type=int, default=1_048_576,
                       help="Bits per device (default: 1048576 = 1Mbit)")

    # simulate
    sim = sub.add_parser("simulate", help="Run single scenario simulation")
    add_common(sim)
    sim.add_argument("--temp", type=float, default=55,
                     help="Temperature in °C (default: 55)")
    sim.add_argument("--voltage", type=float, default=1.2,
                     help="Voltage (default: 1.2V)")
    sim.add_argument("--ecc", choices=["none", "sec", "secded", "chipkill"],
                     default="secded", help="ECC scheme (default: secded)")

    # temperature comparison
    temp = sub.add_parser("temperature", help="Compare temperature stress effects")
    add_common(temp)

    # ecc comparison
    ecc = sub.add_parser("ecc", help="Compare ECC scheme effectiveness")
    add_common(ecc)

    # voltage comparison
    volt = sub.add_parser("voltage", help="Compare voltage stress effects")
    add_common(volt)

    # full — all comparisons
    full = sub.add_parser("full", help="Run all scenario comparisons")
    add_common(full)

    return parser


def get_base_config(args) -> DeviceConfig:
    """Build base DeviceConfig from CLI args."""
    ecc_map = {"none": 0, "sec": 1, "secded": 1, "chipkill": 4}
    ecc_cap = ecc_map.get(getattr(args, "ecc", "secded"), 1)

    return DeviceConfig(
        num_bits=args.bits,
        base_soft_error_rate=args.soft_rate,
        base_hard_failure_rate=args.hard_rate,
        degradation_rate=args.degradation,
        ecc_capability=ecc_cap,
        stress=StressProfile(
            temperature_c=getattr(args, "temp", 55),
            voltage_v=getattr(args, "voltage", 1.2),
        ),
    )


def main():
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    print(f"\n{BOLD}{CYAN}  Reliability Simulation Tool{RESET}")
    print(f"  {DIM}{'─' * 40}{RESET}")

    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)
    base_config = get_base_config(args)

    all_results = []
    all_metrics = []

    if args.command == "simulate":
        label = f"{getattr(args, 'temp', 55)}°C / {getattr(args, 'voltage', 1.2)}V / {getattr(args, 'ecc', 'secded').upper()}"
        configs = [SimulationConfig(
            num_devices=args.num_devices,
            max_time_steps=args.time_steps,
            device_config=base_config,
            seed=42,
            label=label,
        )]
        all_results, all_metrics = run_and_analyze(configs, output_dir, "single")

    elif args.command == "temperature":
        configs = build_temperature_scenarios(args.num_devices, args.time_steps, base_config)
        all_results, all_metrics = run_and_analyze(configs, output_dir, "temperature")

    elif args.command == "ecc":
        configs = build_ecc_scenarios(args.num_devices, args.time_steps, base_config)
        all_results, all_metrics = run_and_analyze(configs, output_dir, "ecc")

    elif args.command == "voltage":
        configs = build_voltage_scenarios(args.num_devices, args.time_steps, base_config)
        all_results, all_metrics = run_and_analyze(configs, output_dir, "voltage")

    elif args.command == "full":
        print(f"\n  {BOLD}=== Temperature Stress Comparison ==={RESET}")
        configs = build_temperature_scenarios(args.num_devices, args.time_steps, base_config)
        r, m = run_and_analyze(configs, output_dir, "temperature")
        all_results.extend(r)
        all_metrics.extend(m)

        print(f"\n  {BOLD}=== ECC Scheme Comparison ==={RESET}")
        configs = build_ecc_scenarios(args.num_devices, args.time_steps, base_config)
        r, m = run_and_analyze(configs, output_dir, "ecc")
        all_results.extend(r)
        all_metrics.extend(m)

        print(f"\n  {BOLD}=== Voltage Stress Comparison ==={RESET}")
        configs = build_voltage_scenarios(args.num_devices, args.time_steps, base_config)
        r, m = run_and_analyze(configs, output_dir, "voltage")
        all_results.extend(r)
        all_metrics.extend(m)

    # Export JSON
    json_path = export_json(all_results, all_metrics, output_dir)
    print(f"\n  {GREEN}JSON report: {json_path}{RESET}\n")


if __name__ == "__main__":
    main()
