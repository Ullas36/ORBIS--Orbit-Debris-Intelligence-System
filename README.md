# ORBIS — Orbit Debris Intelligence System 🛰️💥

ORBIS is a state-of-the-art orbital debris tracking, risk assessment, and trajectory remediation platform. Designed as a professional aerospace dashboard, ORBIS fetches real-time telemetry from CelesTrak, propagates satellite and debris orbital states, calculates a multi-factor **Debris Criticality Index (DCI)**, and uses quantum-inspired simulated annealing to compute optimal multi-target inspection and debris-clearing trajectories.

---

## 🌌 System Architecture & Features

ORBIS leverages a professional, high-performance UI built with **Streamlit** and styled with a custom high-end dark-themed SaaS interface. The system is divided into three key operating consoles:

```
                  ┌────────────────────────────────────────┐
                  │          CelesTrak Telemetry           │
                  └───────────────────┬────────────────────┘
                                      │ (Real-Time TLE Feed)
                                      ▼
                  ┌────────────────────────────────────────┐
                  │       SGP4 Orbital Propagator          │
                  │   (Coordinates: ECI X, Y, Z in km)     │
                  └───────────────────┬────────────────────┘
                                      │
                                      ▼
                  ┌────────────────────────────────────────┐
                  │    Debris Criticality Index (DCI)      │
                  │        Threat Classification           │
                  └───────────────────┬────────────────────┘
                                      │
                                      ▼
                  ┌────────────────────────────────────────┐
                  │    Quantum Trajectory Optimizer        │
                  │   (Hohmann + Plane Change QUBO)        │
                  └─────────────┬───────────────────┬──────┘
                                │                   │
                                ▼                   ▼
                  ┌───────────────────┐       ┌─────────────┐
                  │  Live Debris Globe│       │Risk Catalog │
                  │     (Plotly 3D)   │       │  Database   │
                  └───────────────────┘       └─────────────┘
```

### 1. Live Debris Globe (3D Visualizer)
* **Real-time Orbital Point Cloud**: Renders thousands of active objects, defunct satellites, rocket bodies, and debris fragments in Earth-Centered Inertial (ECI) coordinate space.
* **Risk Tier Color Coding**: Visually separates objects into High (Red), Medium (Orange), and Low (Blue) risk tiers.
* **Interactive Controls**: Filter objects dynamically by risk tolerance levels and adjust rendering density.

### 2. Risk Catalog Database
* **Interactive Data Grid**: Inspect detailed parameters for all propagated targets including NORAD IDs, altitudes, inclinations, speeds, and object types.
* **Debris Criticality Index (DCI)**: Renders live threat metrics.
* **Altitude Density Distributions**: Real-time histogram showcasing orbital debris concentration levels with historical collision annotations (e.g., the 2009 Iridium-Cosmos collision at 789 km).

### 3. QML Trajectory Targeting Engine
* **Mission Cost Matrix**: Dynamically computes Hohmann transfer delta-V costs between locked targets, incorporating orbital inclination plane-change penalties.
* **QUBO Solver**: Formulates the multi-target debris collection path as a Quadratic Unconstrained Binary Optimization (QUBO) problem.
* **Simulated Annealing Backend**: Solves the combinatorial optimization problem using `dimod` and D-Wave's simulated annealing, delivering high-speed execution sequences and delta-V charts.

---

## 📐 Mathematical Formulation

### 1. Debris Criticality Index (DCI)
To identify high-threat candidates for removal, ORBIS calculates the **Debris Criticality Index (DCI)** using the following multi-factor formula:

$$DCI = 0.4 \times Alt_{factor} + 0.35 \times \frac{\ln(1 + Mass_{est})}{\ln(1 + 9000)} + 0.15 \times Inc_{factor}$$

Where:
* **$Alt_{factor}$**: Evaluates the target altitude threat (prioritizing high-density LEO regimes under 1,200 km).
* **$Mass_{est}$**: Estimated mass based on object signatures (e.g., 5,000 kg for rocket bodies, 10 kg for debris fragments).
* **$Inc_{factor}$**: Evaluates the inclination angle factor ($\sin(\theta_{inc})$) to account for high-energy cross-track orbit threats.

---

### 2. Orbital Transfer Delta-V ($\Delta V$)
The transition cost matrix between targets $i$ and $j$ combines a coplanar Hohmann transfer with an orbital plane change:

$$\Delta V_{total} = \Delta V_{Hohmann} + 0.3 \times \Delta V_{Plane}$$

1. **Hohmann Transfer**:
   $$\Delta V_{Hohmann} = |v_{t1} - v_1| + |v_2 - v_{t2}|$$
   Where $v_1$ and $v_2$ are circular velocities at radii $r_1$ and $r_2$, and $v_{t1}$, $v_{t2}$ are the velocities at the periapsis and apoapsis of the elliptical transfer orbit.

2. **Plane Change**:
   $$\Delta V_{Plane} = 2 v_{mid} \sin\left(\frac{\Delta i}{2}\right)$$
   Where $v_{mid}$ is the circular velocity of the average orbit radius, and $\Delta i$ is the difference in orbital inclination.

---

### 3. QUBO Path Planning
The trajectory optimizer maps the target sequence to a Travelling Salesperson-style QUBO. Given $N$ targets and sequence steps $k \in \{0, \dots, N-1\}$, binary variables $x_{i,k}$ represent whether target $i$ is visited at step $k$.

$$\min \sum_{k=0}^{N-2} \sum_{i=0}^{N-1} \sum_{j=0}^{N-1} C_{ij} x_{i,k} x_{j,k+1} + P \sum_{i=0}^{N-1} \left( 1 - \sum_{k=0}^{N-1} x_{i,k} \right)^2 + P \sum_{k=0}^{N-1} \left( 1 - \sum_{i=0}^{N-1} x_{i,k} \right)^2$$

Where:
* $C_{ij}$ is the Delta-V cost from target $i$ to target $j$.
* $P$ is a penalty coefficient ensuring that each target is visited exactly once and each step contains exactly one target.

---

## 🛠️ Tech Stack & Dependencies

* **Language**: Python 3.11+
* **Framework**: Streamlit (SaaS dashboard interface)
* **Orbital Propagator**: SGP4 (Standard Generalized Perturbations 4)
* **Optimization Sampler**: `dimod`, `dwave-samplers` (Simulated Annealing)
* **Visualizations**: Plotly (Scatter3D, Histogram, Bar)
* **Data Processing**: Pandas, NumPy
* **Scientific Computations**: Astropy

---

## 🚀 Getting Started

### Prerequisites
Install the required packages using pip:

```bash
pip install -r ORBIS--Orbit-Debris-Intelligence-System/requirements.txt
```

### Running the Dashboard
Run the Streamlit application from the project root:

```bash
streamlit run ORBIS--Orbit-Debris-Intelligence-System/app.py
```
> **Note:** If your script file is named `app (3).py`, run:
> `streamlit run "ORBIS--Orbit-Debris-Intelligence-System/app (3).py"`
> Or rename it to `app.py` to match the default configuration.

---

## 🐳 Dev Container Support

This repository includes full VS Code Dev Container configurations (`.devcontainer/devcontainer.json`).
* **Environment**: Python 3.11 Bookworm (`mcr.microsoft.com/devcontainers/python:1-3.11-bookworm`)
* **Auto-Setup**: Installs all required Debian system packages, pip dependencies, and launches the Streamlit server automatically on port `8501`.
* **Extensions**: Pre-configured with Python and Pylance extensions for an out-of-the-box development setup.
