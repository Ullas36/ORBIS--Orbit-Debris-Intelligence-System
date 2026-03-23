
import streamlit as st
import numpy as np
import pandas as pd
import requests
import time
import pickle
import torch
import torch.nn as nn
import dimod
import itertools
from datetime import datetime, timezone
from sgp4.api import Satrec, jday
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import TEME, ITRS, CartesianRepresentation
import plotly.graph_objects as go
import plotly.express as px

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ORBIS — Orbital Debris Intelligence",
    page_icon="🛸",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main { background-color: #05091a; }
    .stMetric { background: #0d1b2e; border-radius: 8px; padding: 12px; }
    .stTabs [data-baseweb="tab-list"] { background: #0d1b2e; border-radius: 8px; }
    h1,h2,h3 { color: #90CAF9; }
</style>
""", unsafe_allow_html=True)

# ── DeltaVNet definition (must match training) ────────────────────────────────
class DeltaVNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 256), nn.GELU(), nn.BatchNorm1d(256), nn.Dropout(0.1),
            nn.Linear(256, 512), nn.GELU(), nn.BatchNorm1d(512), nn.Dropout(0.1),
            nn.Linear(512, 512), nn.GELU(), nn.BatchNorm1d(512),
            nn.Linear(512, 256), nn.GELU(),
            nn.Linear(256, 128), nn.GELU(),
            nn.Linear(128, 1), nn.Softplus()
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

# ── Helpers ───────────────────────────────────────────────────────────────────
EARTH_R = 6371.0
GM      = 398600.4418

def hohmann_delta_v(r1, r2):
    v1  = np.sqrt(GM / r1)
    v2  = np.sqrt(GM / r2)
    a_t = (r1 + r2) / 2
    vt1 = np.sqrt(GM * (2/r1 - 1/a_t))
    vt2 = np.sqrt(GM * (2/r2 - 1/a_t))
    return abs(vt1 - v1) + abs(v2 - vt2)

@st.cache_data(ttl=3600, show_spinner="Fetching live TLE data...")
def fetch_debris_catalog():
    BASE = "https://celestrak.org/NORAD/elements/gp.php"
    SOURCES = {
        "cosmos_deb":    f"{BASE}?GROUP=cosmos-deb&FORMAT=tle",
        "iridium_deb":   f"{BASE}?GROUP=iridium-33-deb&FORMAT=tle",
        "fengyun_deb":   f"{BASE}?GROUP=fengyun-1c-deb&FORMAT=tle",
        "rocket_bodies": f"{BASE}?GROUP=rocket-bodies&FORMAT=tle",
        "active":        f"{BASE}?GROUP=active&FORMAT=tle",
    }
    all_tles = {}
    for label, url in SOURCES.items():
        try:
            r = requests.get(url, timeout=20,
                             headers={"User-Agent": "ORBIS/1.0"})
            lines = r.text.strip().splitlines()
            i = 0
            while i < len(lines) - 2:
                name  = lines[i].strip()
                line1 = lines[i+1].strip()
                line2 = lines[i+2].strip()
                if line1.startswith("1 ") and line2.startswith("2 "):
                    try:
                        nid = int(line1[2:7])
                        if nid not in all_tles:
                            all_tles[nid] = (name, line1, line2)
                    except:
                        pass
                    i += 3
                else:
                    i += 1
        except:
            continue
    return list(all_tles.values())

@st.cache_data(ttl=3600, show_spinner="Propagating orbits...")
def propagate_catalog(raw_tles_tuple):
    epoch = datetime.now(timezone.utc)
    records = []
    for name, l1, l2 in raw_tles_tuple:
        try:
            sat = Satrec.twoline2rv(l1, l2)
            jd, fr = jday(epoch.year, epoch.month, epoch.day,
                          epoch.hour, epoch.minute, epoch.second)
            err, pos, vel = sat.sgp4(jd, fr)
            if err != 0 or any(np.isnan(v) for v in pos):
                continue
            mm  = float(l2[52:63])
            alt = (GM / (mm * 2 * np.pi / 86400)**2)**(1/3) - EARTH_R
            if alt < 100:
                continue
            inc = float(l2[8:16])

            # Classify
            nu = name.upper()
            if any(k in nu for k in ["R/B","ROCKET","BOOSTER"]):
                otype = "Rocket Body"
            elif any(k in nu for k in ["DEB","DEBRIS","FRAG"]):
                otype = "Debris Fragment"
            elif any(k in nu for k in ["COSMOS","IRIDIUM","FENGYUN"]):
                otype = "Defunct Satellite"
            else:
                otype = "Active/Unknown"

            # DCI score
            mass_est = 5000 if "R/B" in nu else 10 if "DEB" in nu else 700
            alt_f    = max(0.05, min(1.0, 1.0 - (alt-200)/2000)) if alt < 1200 else 0.1
            inc_f    = abs(np.sin(np.radians(inc)))
            dci      = round(0.4*alt_f + 0.35*(np.log1p(mass_est)/np.log1p(9000)) + 0.15*inc_f, 3)
            risk     = "High" if dci > 0.6 else "Medium" if dci > 0.3 else "Low"

            records.append({
                "name": name.strip(), "norad_id": int(l1[2:7]),
                "altitude_km": round(alt, 1), "inclination": round(inc, 2),
                "x_eci_km": round(pos[0], 1), "y_eci_km": round(pos[1], 1),
                "z_eci_km": round(pos[2], 1),
                "speed_kms": round(np.sqrt(sum(v**2 for v in vel)), 3),
                "object_type": otype, "dci_score": dci, "risk_tier": risk,
                "mass_kg_est": mass_est,
            })
        except:
            continue
    return pd.DataFrame(records)

def build_qubo(cost_matrix, penalty=None):
    n = len(cost_matrix)
    if penalty is None:
        penalty = 2.0 * float(np.sum(cost_matrix))
    Q = {}
    def idx(i, k): return i * n + k
    for k in range(n - 1):
        for i in range(n):
            for j in range(n):
                if i != j:
                    a, b = idx(i,k), idx(j,k+1)
                    key = (min(a,b), max(a,b))
                    Q[key] = Q.get(key, 0) + cost_matrix[i][j]
    for i in range(n):
        for k in range(n):
            a = idx(i, k)
            Q[(a,a)] = Q.get((a,a), 0) + penalty*(1-2)
            for l in range(k+1, n):
                b = idx(i, l)
                key = (min(a,b), max(a,b))
                Q[key] = Q.get(key, 0) + 2*penalty
    for k in range(n):
        for i in range(n):
            a = idx(i, k)
            Q[(a,a)] = Q.get((a,a), 0) + penalty*(1-2)
            for j in range(i+1, n):
                b = idx(j, k)
                key = (min(a,b), max(a,b))
                Q[key] = Q.get(key, 0) + 2*penalty
    return Q, n

def decode_solution(sample, n):
    matrix = np.zeros((n,n), dtype=int)
    for i in range(n):
        for k in range(n):
            matrix[i][k] = sample.get(i*n+k, 0)
    seq = []
    for k in range(n):
        col = matrix[:,k]
        if col.sum() != 1:
            return None
        seq.append(int(np.argmax(col)))
    return seq if len(set(seq)) == n else None

def sequence_cost(seq, cm):
    return round(sum(cm[seq[i]][seq[i+1]] for i in range(len(seq)-1)), 4)

def sa_solve(cost_matrix):
    Q, n = build_qubo(cost_matrix)
    bqm  = dimod.BinaryQuadraticModel.from_qubo(Q)
    try:
        from dwave.samplers import SimulatedAnnealingSampler
    except:
        SimulatedAnnealingSampler = dimod.SimulatedAnnealingSampler
    sampler = SimulatedAnnealingSampler()
    result  = sampler.sample(bqm, num_reads=500, num_sweeps=1000, seed=42)
    best_seq, best_cost = None, float("inf")
    for sample, energy in result.data(["sample","energy"]):
        seq = decode_solution(sample, n)
        if seq:
            c = sequence_cost(seq, cost_matrix)
            if c < best_cost:
                best_cost, best_seq = c, seq
    return best_seq, best_cost

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("ORBIS")
    st.caption("Orbital Remediation with Quantum Intelligence Systems")
    st.divider()
    if st.button("Refresh TLE data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.divider()
    st.caption(f"Data epoch: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    st.caption("Data: CelesTrak · Solver: D-Wave dimod")

# ── Load data ─────────────────────────────────────────────────────────────────
raw_tles = fetch_debris_catalog()
df = propagate_catalog(tuple(raw_tles))

# ── Metrics row ───────────────────────────────────────────────────────────────
st.title("ORBIS — Orbital Debris Intelligence System")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Objects tracked",    f"{len(df):,}")
c2.metric("High risk",          f"{len(df[df.risk_tier=='High']):,}")
c3.metric("Altitude range",     f"{df.altitude_km.min():.0f}–{df.altitude_km.max():.0f} km")
c4.metric("Avg DCI score",      f"{df.dci_score.mean():.3f}")

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["Live Debris Globe", "Risk Catalog", "QML Optimizer"])

# ── TAB 1: Globe ──────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Live debris field")
    col_f1, col_f2 = st.columns([2,1])
    with col_f1:
        risk_filter = st.multiselect(
            "Risk tier", ["High","Medium","Low"],
            default=["High","Medium","Low"]
        )
    with col_f2:
        max_pts = st.slider("Max points", 500, 5000, 2000, step=500)

    plot_df = df[df.risk_tier.isin(risk_filter)].sample(
        min(max_pts, len(df[df.risk_tier.isin(risk_filter)])), random_state=42
    )

    color_map = {"High":"#E24B4A","Medium":"#EF9F27","Low":"#639922"}
    fig = go.Figure()

    for tier, color in color_map.items():
        sub = plot_df[plot_df.risk_tier == tier]
        if len(sub) == 0: continue
        fig.add_trace(go.Scatter3d(
            x=sub.x_eci_km, y=sub.y_eci_km, z=sub.z_eci_km,
            mode="markers", name=f"{tier} ({len(sub):,})",
            marker=dict(size=2.5, color=color, opacity=0.7),
            text=sub.name + "<br>Alt: " + sub.altitude_km.round(0).astype(str) + " km",
            hovertemplate="%{text}<extra></extra>"
        ))

    u_ = np.linspace(0, 2*np.pi, 30)
    v_ = np.linspace(0, np.pi, 15)
    fig.add_trace(go.Surface(
        x=EARTH_R*np.outer(np.cos(u_),np.sin(v_)),
        y=EARTH_R*np.outer(np.sin(u_),np.sin(v_)),
        z=EARTH_R*np.outer(np.ones(30),np.cos(v_)),
        colorscale=[[0,"#1a3a5c"],[1,"#2d6ea8"]],
        showscale=False, opacity=0.5, hoverinfo="skip"
    ))

    fig.update_layout(
        height=600,
        scene=dict(bgcolor="rgba(2,6,15,1)",
                   xaxis_title="X (km)", yaxis_title="Y (km)", zaxis_title="Z (km)"),
        paper_bgcolor="rgba(2,6,15,1)",
        font=dict(color="white"),
        legend=dict(bgcolor="rgba(0,0,0,0.5)"),
        margin=dict(l=0,r=0,t=0,b=0)
    )
    st.plotly_chart(fig, use_container_width=True)

# ── TAB 2: Risk Catalog ───────────────────────────────────────────────────────
with tab2:
    st.subheader("Debris risk catalog")
    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        tier_sel = st.selectbox("Risk tier", ["All","High","Medium","Low"])
    with col_s2:
        type_sel = st.selectbox("Object type", ["All","Rocket Body","Debris Fragment","Defunct Satellite","Active/Unknown"])
    with col_s3:
        alt_range = st.slider("Altitude (km)", 100, 40000, (100, 2000))

    filtered = df.copy()
    if tier_sel != "All":
        filtered = filtered[filtered.risk_tier == tier_sel]
    if type_sel != "All":
        filtered = filtered[filtered.object_type == type_sel]
    filtered = filtered[
        (filtered.altitude_km >= alt_range[0]) &
        (filtered.altitude_km <= alt_range[1])
    ]

    display_cols = ["name","norad_id","altitude_km","inclination","speed_kms","object_type","dci_score","risk_tier"]
    st.dataframe(
        filtered[display_cols].sort_values("dci_score", ascending=False).head(200),
        use_container_width=True, height=420,
        column_config={
            "dci_score":  st.column_config.ProgressColumn("DCI score", min_value=0, max_value=1),
            "risk_tier":  st.column_config.TextColumn("Risk"),
            "altitude_km":st.column_config.NumberColumn("Altitude (km)", format="%.1f"),
        }
    )

    st.caption(f"Showing {min(200,len(filtered)):,} of {len(filtered):,} matching objects")

    fig2 = px.histogram(
        df[df.altitude_km < 3000], x="altitude_km",
        color="object_type", nbins=80,
        title="Debris density by altitude",
        color_discrete_map={
            "Rocket Body":"#E24B4A","Debris Fragment":"#EF9F27",
            "Defunct Satellite":"#378ADD","Active/Unknown":"#888780"
        }
    )
    fig2.add_vline(x=789, line_dash="dash", line_color="white", opacity=0.5,
                   annotation_text="Iridium-Cosmos 2009", annotation_font_color="white")
    fig2.update_layout(height=320, paper_bgcolor="rgba(2,6,15,1)",
                       plot_bgcolor="rgba(10,20,40,0.8)", font=dict(color="white"))
    st.plotly_chart(fig2, use_container_width=True)

# ── TAB 3: QML Optimizer ──────────────────────────────────────────────────────
with tab3:
    st.subheader("QML debris removal optimizer")
    st.caption("Select debris targets → build delta-V cost matrix → solve with simulated annealing (QUBO)")

    top_candidates = df.nlargest(50, "dci_score")["name"].tolist()
    selected = st.multiselect(
        "Select debris targets (pick 3–8 for best results)",
        options=top_candidates,
        default=top_candidates[:5],
        max_selections=10
    )

    if len(selected) < 2:
        st.warning("Select at least 2 debris objects to optimize.")
    else:
        n_sel = len(selected)
        st.info(f"Building {n_sel}×{n_sel} delta-V cost matrix using orbital elements...")

        np.random.seed(42)
        sel_df = df[df.name.isin(selected)].drop_duplicates("name").head(n_sel)

        # Build cost matrix from orbital elements
        feats = []
        for _, row in sel_df.iterrows():
            r = EARTH_R + row["altitude_km"]
            feats.append({
                "name": row["name"],
                "r": r,
                "alt": row["altitude_km"],
                "inc": row["inclination"],
            })

        cost_matrix_live = np.zeros((n_sel, n_sel))
        for i in range(n_sel):
            for j in range(n_sel):
                if i != j:
                    dv = hohmann_delta_v(feats[i]["r"], feats[j]["r"])
                    inc_diff = abs(feats[i]["inc"] - feats[j]["inc"])
                    v_mid    = np.sqrt(GM / ((feats[i]["r"]+feats[j]["r"])/2))
                    dv_plane = 2 * v_mid * np.sin(np.radians(inc_diff/2))
                    cost_matrix_live[i][j] = round(dv + dv_plane * 0.3, 4)

        st.write("**Delta-V cost matrix (km/s):**")
        names_short = [f[:18] for f in selected[:n_sel]]
        st.dataframe(
            pd.DataFrame(cost_matrix_live, index=names_short, columns=names_short).round(3),
            use_container_width=True
        )

        if st.button("Run QML optimizer", type="primary", use_container_width=True):
            with st.spinner("Solving with QUBO + simulated annealing..."):
                t0  = time.time()
                seq, total_dv = sa_solve(cost_matrix_live)
                elapsed = time.time() - t0

            if seq is not None:
                st.success(f"Optimal sequence found in {elapsed:.2f}s")
                st.metric("Total mission delta-V", f"{total_dv:.4f} km/s")

                st.write("**Optimal removal sequence:**")
                for step, idx in enumerate(seq):
                    name = feats[idx]["name"] if idx < len(feats) else f"Target {idx}"
                    alt  = feats[idx]["alt"]
                    dv_step = cost_matrix_live[seq[step-1]][idx] if step > 0 else 0
                    st.write(f"  Step {step+1}: **{name}**  "
                             f"(alt: {alt:.0f} km"
                             f"{f', +{dv_step:.3f} km/s' if step>0 else ' — start'})")

                # Sequence delta-V bar chart
                step_dvs = [0] + [
                    cost_matrix_live[seq[i]][seq[i+1]]
                    for i in range(len(seq)-1)
                ]
                fig3 = px.bar(
                    x=[f"Step {i+1}" for i in range(len(seq))],
                    y=step_dvs,
                    title="Delta-V per mission step",
                    labels={"x":"Step","y":"Delta-V (km/s)"},
                    color=step_dvs,
                    color_continuous_scale=["#639922","#EF9F27","#E24B4A"]
                )
                fig3.update_layout(
                    height=320,
                    paper_bgcolor="rgba(2,6,15,1)",
                    plot_bgcolor="rgba(10,20,40,0.8)",
                    font=dict(color="white"),
                    showlegend=False,
                    coloraxis_showscale=False
                )
                st.plotly_chart(fig3, use_container_width=True)
            else:
                st.error("Solver returned invalid solution. Try fewer targets or re-run.")
