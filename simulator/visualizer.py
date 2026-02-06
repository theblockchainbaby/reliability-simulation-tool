"""
Visualization module — generates reliability engineering charts.

Charts:
- Survival curve R(t)
- Failure probability CDF
- Hazard rate / bathtub curve
- Temperature stress comparison
- ECC effectiveness comparison
- Weibull probability plot
"""
from __future__ import annotations

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .monte_carlo import SimulationResult
from .statistics import ReliabilityMetrics


# Dark theme matching the bottleneck analyzer
plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#c9d1d9",
    "text.color": "#c9d1d9",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "grid.color": "#21262d",
    "grid.alpha": 0.6,
    "font.family": "monospace",
    "font.size": 10,
})

COLORS = [
    "#58a6ff",  # blue
    "#f78166",  # orange
    "#7ee787",  # green
    "#d2a8ff",  # purple
    "#ff7b72",  # red
    "#ffa657",  # amber
    "#79c0ff",  # light blue
]


class ReliabilityVisualizer:
    """Generates reliability engineering charts."""

    def __init__(self, output_dir: str = "reports"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def generate_full_report(self, results: list[SimulationResult],
                             metrics: list[ReliabilityMetrics],
                             prefix: str = "reliability") -> list[str]:
        """Generate all charts and return file paths."""
        paths = []

        if len(results) == 1:
            paths.append(self.plot_single_analysis(
                results[0], metrics[0], prefix))
        else:
            paths.append(self.plot_survival_comparison(
                results, metrics, prefix))
            paths.append(self.plot_failure_cdf_comparison(
                results, metrics, prefix))
            paths.append(self.plot_hazard_comparison(
                results, metrics, prefix))

        if any(m.weibull_shape > 0 for m in metrics):
            paths.append(self.plot_weibull(results, metrics, prefix))

        paths.append(self.plot_error_breakdown(results, prefix))

        return paths

    def plot_single_analysis(self, result: SimulationResult,
                             metrics: ReliabilityMetrics,
                             prefix: str) -> str:
        """4-panel chart for a single simulation scenario."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle(f"Reliability Analysis — {result.label or 'Default'}",
                     fontsize=16, fontweight="bold", y=0.98)

        t = metrics.time_points

        # Survival curve
        ax = axes[0, 0]
        ax.fill_between(t, metrics.reliability, alpha=0.3, color=COLORS[0])
        ax.plot(t, metrics.reliability, color=COLORS[0], linewidth=2)
        ax.axhline(y=0.9, color=COLORS[4], linestyle="--", alpha=0.5, label="90% survival")
        ax.set_ylabel("R(t) — Survival Probability")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
        ax.grid(True)
        ax.set_title("Survival Function R(t)")

        # Failure CDF
        ax = axes[0, 1]
        ax.fill_between(t, metrics.failure_cdf, alpha=0.3, color=COLORS[1])
        ax.plot(t, metrics.failure_cdf, color=COLORS[1], linewidth=2)
        ax.set_ylabel("F(t) — Failure Probability")
        ax.set_ylim(0, max(0.1, min(1.05, metrics.failure_cdf[-1] * 1.5)) if len(metrics.failure_cdf) > 0 else 0.1)
        ax.grid(True)
        ax.set_title("Cumulative Failure Distribution")

        # Hazard rate
        ax = axes[1, 0]
        # Smooth hazard for display
        window = max(3, len(metrics.hazard_rate) // 50)
        if window < len(metrics.hazard_rate):
            smoothed = np.convolve(metrics.hazard_rate,
                                   np.ones(window) / window, mode="same")
        else:
            smoothed = metrics.hazard_rate
        ax.plot(t, smoothed, color=COLORS[2], linewidth=2)
        ax.fill_between(t, smoothed, alpha=0.2, color=COLORS[2])
        ax.set_ylabel("h(t) — Hazard Rate")
        ax.set_xlabel("Time (hours)")
        ax.grid(True)
        ax.set_title("Hazard Rate (Failure Intensity)")

        # Summary stats text
        ax = axes[1, 1]
        ax.axis("off")
        stats_text = (
            f"Devices Simulated:  {metrics.num_devices:,}\n"
            f"Total Failures:     {metrics.num_failed:,}\n"
            f"Failure Rate:       {metrics.failure_rate_total:.4%}\n"
            f"\n"
            f"MTTF:               {metrics.mttf:,.0f} hrs\n"
            f"  95% CI:           [{metrics.mttf_ci_lower:,.0f}, {metrics.mttf_ci_upper:,.0f}]\n"
            f"Median Life:        {metrics.median_life:,.0f} hrs\n"
            f"\n"
            f"B1 Life (1% fail):  {metrics.b1_life:,.0f} hrs\n"
            f"B10 Life (10%):     {metrics.b10_life:,.0f} hrs\n"
            f"B50 Life (50%):     {metrics.b50_life:,.0f} hrs\n"
        )
        if metrics.weibull_shape > 0:
            stats_text += (
                f"\nWeibull Shape (β):  {metrics.weibull_shape:.3f}\n"
                f"Weibull Scale (η):  {metrics.weibull_scale:,.0f} hrs\n"
            )
            if metrics.weibull_shape < 1:
                stats_text += "  → Infant mortality regime\n"
            elif metrics.weibull_shape > 1:
                stats_text += "  → Wear-out regime\n"
            else:
                stats_text += "  → Constant failure rate\n"

        ax.text(0.1, 0.95, stats_text, transform=ax.transAxes,
                fontsize=11, verticalalignment="top", fontfamily="monospace",
                color="#c9d1d9",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#161b22",
                          edgecolor="#30363d"))
        ax.set_title("Summary Statistics")

        plt.tight_layout()
        path = os.path.join(self.output_dir, f"{prefix}_single_analysis.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def plot_survival_comparison(self, results: list[SimulationResult],
                                 metrics: list[ReliabilityMetrics],
                                 prefix: str) -> str:
        """Compare survival curves across scenarios."""
        fig, ax = plt.subplots(figsize=(14, 8))
        fig.suptitle("Survival Curve Comparison", fontsize=16, fontweight="bold")

        for i, (res, met) in enumerate(zip(results, metrics)):
            color = COLORS[i % len(COLORS)]
            label = res.label or f"Scenario {i+1}"
            ax.plot(met.time_points, met.reliability, color=color,
                    linewidth=2, label=f"{label} (MTTF={met.mttf:,.0f}h)")
            ax.fill_between(met.time_points, met.reliability,
                            alpha=0.1, color=color)

        ax.axhline(y=0.9, color="#8b949e", linestyle="--", alpha=0.5,
                    label="90% survival target")
        ax.set_xlabel("Time (hours)", fontsize=12)
        ax.set_ylabel("R(t) — Survival Probability", fontsize=12)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=10, loc="lower left")
        ax.grid(True)

        plt.tight_layout()
        path = os.path.join(self.output_dir, f"{prefix}_survival_comparison.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def plot_failure_cdf_comparison(self, results: list[SimulationResult],
                                    metrics: list[ReliabilityMetrics],
                                    prefix: str) -> str:
        """Compare failure CDFs across scenarios."""
        fig, ax = plt.subplots(figsize=(14, 8))
        fig.suptitle("Failure Probability Comparison", fontsize=16, fontweight="bold")

        for i, (res, met) in enumerate(zip(results, metrics)):
            color = COLORS[i % len(COLORS)]
            label = res.label or f"Scenario {i+1}"
            ax.plot(met.time_points, met.failure_cdf * 100, color=color,
                    linewidth=2, label=label)

        ax.set_xlabel("Time (hours)", fontsize=12)
        ax.set_ylabel("Cumulative Failure %", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True)

        plt.tight_layout()
        path = os.path.join(self.output_dir, f"{prefix}_failure_cdf.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def plot_hazard_comparison(self, results: list[SimulationResult],
                               metrics: list[ReliabilityMetrics],
                               prefix: str) -> str:
        """Compare hazard rates across scenarios."""
        fig, ax = plt.subplots(figsize=(14, 8))
        fig.suptitle("Hazard Rate Comparison", fontsize=16, fontweight="bold")

        for i, (res, met) in enumerate(zip(results, metrics)):
            color = COLORS[i % len(COLORS)]
            label = res.label or f"Scenario {i+1}"
            # Smooth for readability
            window = max(3, len(met.hazard_rate) // 50)
            if window < len(met.hazard_rate):
                smoothed = np.convolve(met.hazard_rate,
                                       np.ones(window) / window, mode="same")
            else:
                smoothed = met.hazard_rate
            ax.plot(met.time_points, smoothed, color=color,
                    linewidth=2, label=label)

        ax.set_xlabel("Time (hours)", fontsize=12)
        ax.set_ylabel("h(t) — Hazard Rate", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True)

        plt.tight_layout()
        path = os.path.join(self.output_dir, f"{prefix}_hazard_rate.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def plot_weibull(self, results: list[SimulationResult],
                     metrics: list[ReliabilityMetrics],
                     prefix: str) -> str:
        """Weibull probability plot — linearized CDF for shape analysis."""
        fig, ax = plt.subplots(figsize=(14, 8))
        fig.suptitle("Weibull Probability Plot", fontsize=16, fontweight="bold")

        for i, (res, met) in enumerate(zip(results, metrics)):
            if met.weibull_shape <= 0:
                continue

            color = COLORS[i % len(COLORS)]
            label = res.label or f"Scenario {i+1}"

            failure_times = sorted(res.failure_times)
            if len(failure_times) < 3:
                continue

            # Median rank approximation for plotting positions
            n = len(failure_times)
            ranks = [(j - 0.3) / (n + 0.4) for j in range(1, n + 1)]

            # Weibull axes: ln(t) vs ln(-ln(1-F))
            x = np.log(failure_times)
            y = [np.log(-np.log(1 - r)) for r in ranks]

            ax.scatter(x, y, color=color, s=15, alpha=0.5, zorder=2)

            # Fit line
            if len(x) > 1:
                coeffs = np.polyfit(x, y, 1)
                x_fit = np.linspace(min(x), max(x), 100)
                y_fit = np.polyval(coeffs, x_fit)
                ax.plot(x_fit, y_fit, color=color, linewidth=2,
                        label=f"{label} (β={met.weibull_shape:.2f}, η={met.weibull_scale:,.0f})",
                        zorder=3)

        ax.set_xlabel("ln(Time)", fontsize=12)
        ax.set_ylabel("ln(-ln(1-F))", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True)

        plt.tight_layout()
        path = os.path.join(self.output_dir, f"{prefix}_weibull_plot.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def plot_error_breakdown(self, results: list[SimulationResult],
                             prefix: str) -> str:
        """Bar chart showing error types and correction across scenarios."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle("Error Analysis Breakdown", fontsize=16, fontweight="bold")

        labels = [r.label or f"Scenario {i+1}" for i, r in enumerate(results)]
        x = np.arange(len(labels))
        width = 0.35

        # Left: Error types
        soft = [sum(d.soft_errors for d in r.device_results) for r in results]
        hard = [sum(d.hard_failures for d in r.device_results) for r in results]

        ax1.bar(x - width/2, soft, width, color=COLORS[0], label="Soft Errors", alpha=0.8)
        ax1.bar(x + width/2, hard, width, color=COLORS[4], label="Hard Failures", alpha=0.8)
        ax1.set_xlabel("Scenario")
        ax1.set_ylabel("Total Errors (all devices)")
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
        ax1.legend(fontsize=9)
        ax1.grid(True, axis="y")
        ax1.set_title("Error Type Distribution")

        # Right: Corrected vs DUE vs SDC
        corrected = [sum(d.corrected_errors for d in r.device_results) for r in results]
        due = [sum(d.due_errors for d in r.device_results) for r in results]
        sdc = [sum(d.sdc_errors for d in r.device_results) for r in results]

        bar_w = 0.25
        ax2.bar(x - bar_w, corrected, bar_w, color=COLORS[2], label="Corrected", alpha=0.8)
        ax2.bar(x, due, bar_w, color=COLORS[5], label="DUE (Detected)", alpha=0.8)
        ax2.bar(x + bar_w, sdc, bar_w, color=COLORS[4], label="SDC (Silent)", alpha=0.8)
        ax2.set_xlabel("Scenario")
        ax2.set_ylabel("Error Count (all devices)")
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
        ax2.legend(fontsize=9)
        ax2.grid(True, axis="y")
        ax2.set_title("ECC Outcome: Corrected / DUE / SDC")

        plt.tight_layout()
        path = os.path.join(self.output_dir, f"{prefix}_error_breakdown.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path
