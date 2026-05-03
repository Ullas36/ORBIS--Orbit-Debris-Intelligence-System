# ORBIS — Orbit Debris Intelligence System

**Live demo:** [Deploy link after Streamlit Cloud setup]

A quantum-classical hybrid system for Active Debris Removal (ADR) mission sequencing, demonstrated on real IBM Quantum hardware.

## What This Is

ORBIS combines:
- **Real orbital mechanics** — SGP4 propagation of live TLE data from CelesTrak
- **QUBO optimization** — debris removal sequencing as a Quadratic Unconstrained Binary Optimization problem
- **Real quantum hardware** — QAOA circuits executed on IBM Heron r2 (ibm_fez), 17 total QPU runs
- **Classical baseline** — Simulated Annealing and Brute Force for comparison

## Key Research Results

| Problem Size | Qubits | Circuit Depth | QPU Quality (mean) | SA Quality |
|---|---|---|---|---|
| n=4 | 16 | 523 | 100.0% ± 0.0% | 100.0% |
| n=5 | 25 | 1,191 | 110.5% ± 9.7% | 100.0% |
| n=6 | 36 | 2,107 | 115.9% ± 18.7% | 100.0% |
| n=8 | 64 | ~3,500 | 195.1% ± 12.2% | 144.1% |

**Finding:** A NISQ phase transition exists between n=6 (36 qubits) and n=8 (64 qubits) on IBM Heron r2. Below this threshold, quantum hardware shows stochastic but often-optimal results. Above it, noise dominates and exceeds classical degradation.

## App Features

- **Tab 1 — Debris Globe:** Live 3D visualization of orbital debris field with ADR priority targets
- **Tab 2 — QPU Benchmark:** Complete multi-run statistical analysis from real IBM quantum hardware
- **Tab 3 — Mission Optimizer:** Interactive debris removal sequencing with classical solvers

## Deployment

### Streamlit Cloud (Recommended)

1. Fork this repo
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select this repo, branch `main`, file `app.py`
5. Click Deploy

No secrets or API keys required for the basic deployment.

### Local

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Tech Stack

- **Frontend:** Streamlit + Plotly
- **Orbital mechanics:** sgp4, numpy
- **Quantum:** Qiskit, IBM Quantum Runtime (experiments run separately in Colab)
- **Data:** CelesTrak TLE catalog, Space-Track.org

## Hardware

All QPU experiments run on **IBM Heron r2 processor (ibm_fez)**
- 156 physical qubits
- IBM Quantum Open Plan (free tier)
- 1,024 shots per circuit
- Date: 2026-05-02

## About

Built as part of a QML internship project at Ada Lovelace Software Limited.
VTU Bangalore, 8th Semester.
