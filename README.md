# Reliability Simulation Tool

Monte Carlo simulation that models how hardware systems accumulate failures and how error correction mechanisms extend usable life. Directly mirrors the reliability analysis performed in the memory semiconductor industry.

## What This Does

- **Error injection** — Simulates soft errors (transient bit flips), hard failures (permanent cell defects), and progressive degradation using Poisson arrival processes
- **Stress modeling** — Arrhenius temperature acceleration and voltage stress factors based on real physics (activation energy, Boltzmann scaling)
- **ECC modeling** — Evaluates No ECC, SEC (Hamming), SECDED, and Chipkill correction schemes with realistic correction/detection thresholds
- **Monte Carlo engine** — Runs thousands of independent device lifetime simulations for statistically significant results
- **Reliability statistics** — Computes survival function R(t), hazard rate h(t), MTTF with confidence intervals, B-life percentiles, and Weibull distribution fitting
- **Scenario comparison** — Side-by-side analysis of temperature, voltage, and ECC tradeoffs

## Architecture

```
main.py                    CLI entry point + scenario builders
simulator/
  device.py                Device model + error injection engine
  ecc.py                   ECC scheme modeling (None/SEC/SECDED/Chipkill)
  monte_carlo.py           Monte Carlo simulation engine
  statistics.py            Reliability analysis (R(t), h(t), MTTF, Weibull)
  visualizer.py            Chart generation
reports/                   Generated charts and JSON reports
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Single simulation — 1000 devices over 1 year
python main.py simulate

# Compare temperature stress effects
python main.py temperature

# Compare ECC scheme effectiveness
python main.py ecc

# Compare voltage stress
python main.py voltage

# Full analysis — all comparisons with charts
python main.py full

# Custom: 5000 devices, 2 years, high temperature
python main.py simulate -n 5000 -t 17520 --temp 85

# Custom: aggressive error rates
python main.py simulate --soft-rate 5e-6 --hard-rate 1e-7 --degradation 0.005
```

## Commands

### `simulate` — Single Scenario
```
python main.py simulate [-n DEVICES] [-t STEPS] [--temp C] [--voltage V] [--ecc SCHEME]
```
Runs one simulation with specified conditions. Generates survival curve, failure CDF, hazard rate, and summary statistics.

### `temperature` — Temperature Comparison
```
python main.py temperature [-n DEVICES] [-t STEPS]
```
Compares 35°C, 55°C, 75°C, and 95°C operating conditions. Uses Arrhenius acceleration to model how heat increases failure rates.

### `ecc` — ECC Comparison
```
python main.py ecc [-n DEVICES] [-t STEPS]
```
Compares No ECC, SEC (Hamming), SECDED, and Chipkill. Shows how each scheme extends device lifetime.

### `voltage` — Voltage Comparison
```
python main.py voltage [-n DEVICES] [-t STEPS]
```
Compares 1.0V, 1.1V, 1.2V, and 1.35V supply voltages with power-law acceleration.

### `full` — All Comparisons
```
python main.py full [-n DEVICES] [-t STEPS] [-o OUTPUT_DIR]
```
Runs temperature, ECC, and voltage comparisons, generating all charts and a JSON report.

## Physics Model

### Error Injection
Each device accumulates errors via Poisson processes:
```
λ_effective = λ_base × stress_factor(T, V) × degradation(t)
```

### Temperature Acceleration (Arrhenius)
```
AF_temp = exp((Ea/k) × (1/T_ref - 1/T_op))
```
Where Ea = 0.7 eV (typical for DRAM failure mechanisms).

### Voltage Acceleration
```
AF_voltage = (V_op / V_ref)^n
```
Where n = 2 (power-law exponent for oxide stress).

### Degradation
```
degradation(t) = 1 + rate × t
```
Models progressive wear-out that increases error rates over device lifetime.

## ECC Schemes Modeled

| Scheme | Correctable | Detectable | Overhead | Use Case |
|---|---|---|---|---|
| None | 0 | 0 | 0% | Baseline comparison |
| SEC (Hamming) | 1 bit | 1 bit | 10.9% | Consumer DRAM |
| SECDED | 1 bit | 2 bits | 12.5% | Standard ECC DIMMs |
| Chipkill | 4 bits | 8 bits | 50% | Server/HPC memory |

## Generated Output

Charts saved to `reports/`:
- `*_single_analysis.png` — 4-panel analysis (survival, CDF, hazard, stats)
- `*_survival_comparison.png` — R(t) curves across scenarios
- `*_failure_cdf.png` — Cumulative failure probability comparison
- `*_hazard_rate.png` — Hazard rate comparison
- `*_weibull_plot.png` — Weibull probability plot for shape analysis
- `*_error_breakdown.png` — Error types and ECC correction rates
- `reliability_report.json` — Full machine-readable results

## Statistical Outputs

- **R(t)** — Survival probability at each time point
- **F(t)** — Cumulative failure distribution
- **h(t)** — Hazard rate (instantaneous failure intensity)
- **MTTF** — Mean Time To Failure with 95% bootstrap confidence intervals
- **B1/B10/B50 Life** — Time at which 1%, 10%, 50% of devices fail
- **Weibull fit** — Shape (beta) and scale (eta) parameters
  - beta < 1: infant mortality regime
  - beta = 1: constant (random) failure rate
  - beta > 1: wear-out regime

## Why This Matters for Memory Engineering

Large memory systems have billions of bits with non-zero error rates. Engineers must:
- **Predict failure rates** across temperature and voltage operating ranges
- **Size ECC** to meet target reliability (e.g., < 1 uncorrectable error per billion device-hours)
- **Model wear-out** to set warranty and replacement schedules
- **Characterize acceleration factors** for qualification testing

This tool models that exact workflow — from physics-based error injection through Monte Carlo simulation to reliability statistics and Weibull analysis.

## Requirements

- Python 3.9+
- numpy, matplotlib, scipy
