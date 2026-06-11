# ORBIS — Orbital Remediation Quantum Intelligence System.

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![Qiskit](https://img.shields.io/badge/Qiskit-1.x-6929C4?style=flat&logo=qiskit&logoColor=white)
![IBM Quantum](https://img.shields.io/badge/IBM%20Quantum-Heron%20r2-052FAD?style=flat&logo=ibm&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Cloud-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

**A quantum-classical hybrid pipeline for Active Debris Removal sequencing using QAOA on real IBM quantum hardware.**

[Live Dashboard](https://your-orbis-app.streamlit.app) · [Report](docs/ORBIS_Internship_Report.pdf) · [Results](#qpu-benchmark-results)

</div>

---

## Overview

Space debris is one of the most pressing engineering challenges of our time. With over **27,000 tracked objects** in Low Earth Orbit travelling at 7–8 km/s, the risk of collision cascades — the Kessler Syndrome — threatens long-term access to space.

ORBIS addresses the **Active Debris Removal (ADR) sequencing problem**: given a set of high-risk debris targets, determine the fuel-optimal removal order for a spacecraft. This is a variant of the Travelling Salesman Problem with edge weights defined by orbital transfer delta-V costs — a combinatorial optimisation problem that scales intractably for classical exhaustive search.

The system implements QAOA (Quantum Approximate Optimisation Algorithm) and validates it on **real IBM Heron r2 quantum hardware** across 17 QPU experiments, identifying an empirical NISQ phase transition between 36 and 64 qubits where quantum hardware noise begins to dominate the computation.

---

## Key Results

|
 Problem Size 
|
 Qubits 
|
 Circuit Depth 
|
 QPU Quality (Mean ± Std) 
|
 Classical SA 
|
|
:---:
|
:---:
|
:---:
|
:---:
|
:---:
|
|
 n=4 
|
 16 
|
 523 
|
**
100.0% ± 0.0%
**
|
 100.0% 
|
|
 n=5 
|
 25 
|
 1,191 
|
**
110.5% ± 9.7%
**
|
 100.0% 
|
|
 n=6 
|
 36 
|
 2,107 
|
**
115.9% ± 18.7%
**
|
 100.0% 
|
|
 n=8 
|
 64 
|
 ~3,500 
|
**
195.1% ± 12.2%
**
|
 144.1% 
|

> Quality = QPU result / optimal (brute force). 100% = optimal. Higher = worse.  
> **17 total QPU experiments on IBM Heron r2 (ibm_fez, 156 qubits). All results from real hardware — not simulators.**

**Finding:** A NISQ phase transition exists between n=6 and n=8. Below it: stochastic but often-optimal results. Above it: hardware noise completely dominates (QPU 195.1% > classical SA 144.1%).

---

## Architecture

```
Space-Track.org API
        │
        ▼
┌─────────────────────────────────────────────────────┐
│  MODULE 1 — CATALOG ENGINE                          │
│  SGP4 propagation · DCI scoring · Target selection  │
│  15,982 objects → 10 diverse ADR targets            │
│  Cost matrix variance: 0.849 km²/s²                 │
└─────────────────┬───────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────┐
│  MODULE 2 — CONJUNCTION ENGINE                      │
│  72-hour TCA propagation · Chan (2008) Pc formula   │
│  KD-Tree O(n²) → O(n log n) · RF risk classifier   │
└─────────────────┬───────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────┐
│  MODULE 3 — DELTAVNET                               │
│  563K-param physics-informed MLP · MAPE: 3.77%      │
│  Hohmann + plane-change · GELU + Softplus output    │
│  Cost matrix: 0.159 — 3.480 km/s                   │
└─────────────────┬───────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────┐
│  MODULE 4 — QUANTUM OPTIMISER                       │
│  QUBO: n² binary variables · QAOA p=1               │
│  IBM Heron r2 · 17 QPU runs · Soft decoder          │
└─────────────────┬───────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────┐
│  MODULE 5 — LIVE DASHBOARD                          │
│  Streamlit Cloud · 3D globe · QPU results · Optimizer│
└─────────────────────────────────────────────────────┘
```

---

## Project Structure

```
ORBIS/
├── notebooks/
│   ├── Module1_Catalog.ipynb          # TLE ingestion, SGP4, DCI scoring
│   ├── Module2_Conjunction.ipynb      # TCA propagation, Pc computation
│   ├── Module3_DeltaVNet.ipynb        # Neural network training
│   └── Module4_Quantum.ipynb          # QUBO + QAOA + IBM QPU
├── app.py                             # Streamlit dashboard
├── requirements.txt                   # Streamlit Cloud dependencies
├── data/
│   ├── orbis_adr_targets_v3.csv       # 10 selected ADR targets
│   ├── orbis_cost_matrix_v2.npy       # 10×10 delta-V cost matrix
│   ├── orbis_targets_v2.json          # Target metadata
│   └── orbis_final_all_runs.json      # All 17 QPU experiment results
├── models/
│   ├── deltavnet_weights_v2.pth       # Trained DeltaVNet weights
│   └── deltavnet_scaler_v2.pkl        # Feature scaler
├── docs/
│   └── ORBIS_Internship_Report.pdf    # Full technical report
└── README.md
```

---

## Modules

### Module 1 — Orbital Debris Catalog Engine

Ingests live orbital element data from **Space-Track.org** (authenticated REST API) and propagates all objects to the current epoch using SGP4.

**Data fetched:**
- Rocket bodies (1,116 objects) — highest individual mass, top ADR priority
- Active/defunct payloads (8,814 objects)
- All debris (6,155 objects)
- Total after deduplication: **15,982 unique objects**

**Debris Criticality Index:**

$$\text{DCI} = 0.28 \cdot f_{\text{mass}} + 0.28 \cdot f_{\text{alt}} + 0.20 \cdot f_{\text{inc}} + 0.14 \cdot f_{\text{decay}} + 0.10 \cdot f_{\text{ecc}}$$

Altitude factor peaks at 750–1000 km (Fengyun-1C + Iridium-Cosmos debris belt). Inclination factor peaks at sun-synchronous (96–100°) and Cosmos debris belt (62–75°) regimes.

**Diversity-aware target selection:**  
Naive top-N by DCI clusters objects in identical orbital regimes, producing a degenerate cost matrix (variance 0.0014 km²/s²) where every removal sequence has the same total delta-V. The diversity selector uses a 6-altitude × 6-inclination bin grid to ensure orbital spread, achieving:
- Cost matrix variance: **0.849 km²/s²** (520× improvement)
- Altitude spread: 376 km
- Inclination spread: 94.1°

---

### Module 2 — Conjunction Analysis Engine

Implements proper conjunction analysis — not single-epoch proximity checking.

**72-hour TCA propagation** with adaptive stepping:
- > 500 km separation: 60-second steps
- 100–500 km: 30-second steps
- 50–100 km: 10-second steps
- < 50 km: 5-second steps

**Probability of Collision (Chan 2008):**

$$P_c = \frac{\pi r_c^2}{2\pi\sigma^2} \cdot \frac{\sigma}{v_{\text{rel}}} \cdot \exp\!\left(-\frac{d^2}{2\sigma^2}\right)$$

Risk labels use NASA operational thresholds: Pc > 1/1,000 = Red, Pc > 1/10,000 = Yellow.

---

### Module 3 — DeltaVNet

A **563,457-parameter physics-informed MLP** that approximates Hohmann transfer + plane-change delta-V costs for arbitrary orbital pairs.

**Architecture:**
```
Input(8) → Linear(256)+GELU+BN+Drop → Linear(512)+GELU+BN+Drop
         → Linear(512)+GELU+BN → Linear(256)+GELU
         → Linear(128)+GELU → Linear(1)+Softplus
```

The **Softplus output** encodes the physical constraint ΔV ≥ 0 without zero-gradient issues.

**Training:** 100,000 samples, Huber loss, Adam + cosine annealing, 60 epochs, CUDA GPU.

**Performance:**

|
 Metric 
|
 Value 
|
|
---
|
---
|
|
 MAPE 
|
 3.77% 
|
|
 Median error 
|
 1.12% 
|
|
 Within 5% 
|
 92.0% 
|
|
 Within 10% 
|
 95.9% 
|

**Key bug fixed:** Original v_mid formula `sqrt(GM*(2/a - 2/a))` evaluated to 0 for all inputs, making all plane-change costs zero. Fixed to `sqrt(GM/a_t)` — the velocity at the semi-major axis of the transfer ellipse.

---

### Module 4 — Quantum Optimiser

**QUBO Formulation:**

Binary variable $x_{i,k} = 1$ if target $i$ is visited at step $k$:

$$H = \sum_{i,j,k} C[i][j] \cdot x_{i,k} \cdot x_{j,k+1} + \lambda_A \sum_k\!\left(1 - \sum_i x_{i,k}\right)^{\!2} + \lambda_B \sum_i\!\left(1 - \sum_k x_{i,k}\right)^{\!2}$$

For $n$ targets: $n^2$ binary variables (qubits).

**QAOA Circuit (p=1):**
- Initial state: $|+\rangle^{\otimes n^2}$ via Hadamard layer
- Cost unitary: Rzz gates (quadratic) + Rz gates (linear diagonal)
- Mixer: Rx gates on all qubits
- Fixed parameters: γ=0.4, β=0.3

**Soft Decoder:**  
Hard decoding rejects nearly all noisy hardware bitstrings. The soft decoder greedily reconstructs the closest valid permutation from the top-30 most-frequent measurement outcomes — extracting a usable result from every QPU run regardless of noise level.

**Real Hardware Execution:**
- Backend: IBM Heron r2 (`ibm_fez`, 156 qubits)
- Plan: IBM Quantum Open Plan (free, no credit card)
- Mode: SamplerV2, Batch (Session not available on Open Plan)
- Shots: 1,024 per circuit

**Transpiled circuit depths:**

|
 n 
|
 Qubits 
|
 Transpiled Depth 
|
 Gate Count 
|
|
---
|
---
|
---
|
---
|
|
 4 
|
 16 
|
 523 
|
 2,123 
|
|
 5 
|
 25 
|
 1,191 
|
 4,913 
|
|
 6 
|
 36 
|
 2,107 
|
 9,712 
|
|
 8 
|
 64 
|
 ~3,500 
|
 ~15,000 
|

---

## QPU Benchmark Results

**All 17 runs — IBM Heron r2 (ibm_fez):**

```
n=4 │ Runs: 100.0, 100.0, 100.0, 100.0     │ Mean: 100.0% │ Std: 0.0%
n=5 │ Runs: 121.1, 100.0, 110.3, 100.0, 121.1 │ Mean: 110.5% │ Std: 9.7%
n=6 │ Runs: 100.0, 131.4, 100.0, 140.6, 107.7 │ Mean: 115.9% │ Std: 18.7%
n=8 │ Runs: 190.5, 211.9, 182.9             │ Mean: 195.1% │ Std: 12.2%
```

**Verified IBM Quantum Job IDs:**
```
d7r4llaudops7395a2ig   n=4 run 1
d7r4lliudops7395a2j0   n=5 run 1
d7r4sq4t738s73cf5qc0   n=4 run 2
d7r4sqct738s73cf5qcg   n=5 run 2
d7r4sqkt738s73cf5qdg   n=6 run 1
```

---

## Installation

### Local Development

```bash
git clone https://github.com/Ullas36/ORBIS--Orbit-Debris-Intelligence-System.git
cd ORBIS--Orbit-Debris-Intelligence-System
pip install -r requirements.txt
streamlit run app.py
```

### Running the Notebooks

The notebooks run on Google Colab with GPU runtime. Open each notebook in order:

1. `Module1_Catalog.ipynb` — requires Space-Track.org credentials
2. `Module2_Conjunction.ipynb` — requires Module 1 output CSV
3. `Module3_DeltaVNet.ipynb` — requires Module 2 output CSV
4. `Module4_Quantum.ipynb` — requires IBM Quantum API token + Module 3 outputs

### IBM Quantum Setup

```python
# Free account — no credit card, works from India
# Register at: quantum.cloud.ibm.com

from qiskit_ibm_runtime import QiskitRuntimeService
import getpass

token = getpass.getpass("IBM Quantum API token: ")
service = QiskitRuntimeService(channel="ibm_quantum", token=token)

# Select least busy real hardware
backend = service.least_busy(operational=True, simulator=False, min_num_qubits=16)
print(f"Running on: {backend.name}")
```

### Space-Track Setup

```python
# Free account — register at space-track.org
# Approval is instant for academic use

import requests, getpass

ST_USER = getpass.getpass("Space-Track email: ")
ST_PASS = getpass.getpass("Space-Track password: ")

session = requests.Session()
session.post("https://www.space-track.org/ajaxauth/login",
             data={"identity": ST_USER, "password": ST_PASS})
```

---

## Requirements

```
streamlit
numpy
pandas
plotly
requests
sgp4
```

For notebooks (additional):
```
qiskit
qiskit-ibm-runtime
qiskit-aer
torch
scikit-learn
tqdm
scipy
dimod
mitiq
```

---

## Dashboard

Three-tab Streamlit application deployed on Streamlit Cloud:

**Tab 1 — Debris Globe**  
Live 3D Plotly globe. Clicking "Fetch Live TLE Data" pulls the current orbital catalog from CelesTrak, propagates up to 2,000 objects via SGP4, and renders by altitude regime. ADR targets shown as interactive red diamond markers.

**Tab 2 — QPU Benchmark**  
Complete 17-run experimental dataset. Box plot of quality distribution per problem size, mean degradation trend (QPU vs SA), four research findings as formatted cards, full run table with colour-coded quality indicators.

**Tab 3 — Mission Optimizer**  
Select 3–10 targets and a solver (Simulated Annealing or Brute Force). Results shown as polar orbit diagram and step-by-step mission plan with per-hop delta-V costs.

---

## Research Context

This project was developed during a **Quantum Machine Learning internship at Ada Lovelace Software Pvt. Limited**, Bengaluru (January 2026 – present).

The work represents one of the first empirical studies of NISQ hardware noise on a real-world space mission planning problem conducted using free-tier IBM Quantum access. All results are from real hardware — not simulators — and are fully reproducible via the recorded job IDs.

**Intern:** Ullas C L  
**Institution:** J.S.S. Academy of Technical Education, Bengaluru (VTU)  
**Degree:** B.E. Information Science and Engineering (CGPA: 8.82)  
**Certifications:** Familiarisation Workshop on Quantum Computing — FQCI, IISc (2026)

---

## Known Limitations and Future Work

**Current limitations:**
- ZNE mitigation incomplete — gate folding was undone by transpiler optimisation; identity-insertion noise scaling is the correct approach
- Module 2 conjunction analysis produces no results for orbitally diverse targets — KD-Tree prefilter threshold needs tuning for widely-separated orbital regimes
- p=1 QAOA with fixed parameters is the weakest possible QAOA configuration; no quantum advantage is expected or observed

**Planned next steps:**
- IonQ hardware comparison via Azure Quantum free credits — trapped-ion vs superconducting noise model
- p=3 QAOA with variational parameter optimisation to test if the phase transition shifts
- Proper ZNE with identity-insertion scaling using Mitiq
- n=6 on IonQ for cross-platform noise characterisation

---

## Citation

If you use this work, please cite:

```bibtex
@misc{orbis2026,
  author       = {Ullas C L},
  title        = {ORBIS: Orbital Remediation Quantum Intelligence System},
  year         = {2026},
  institution  = {Ada Lovelace Software Pvt. Ltd. / JSSATE Bengaluru},
  url          = {https://github.com/Ullas36/ORBIS--Orbit-Debris-Intelligence-System},
  note         = {17 real QPU experiments on IBM Heron r2. NISQ phase transition
                  identified between n=6 (36 qubits) and n=8 (64 qubits).}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgements

- **IBM Quantum** — free Open Plan access to Heron r2 hardware
- **Space-Track.org** — authenticated orbital element catalog
- **CelesTrak** — supplementary TLE data
- **Ada Lovelace Software Pvt. Limited** — internship and research support
- **IISc FQCI** — quantum computing workshop and certification

---

<div align="center">

Built with real quantum hardware · 17 QPU experiments · IBM Heron r2 · Bengaluru, India · 2026

</div>
