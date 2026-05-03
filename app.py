import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import requests
import time
from datetime import datetime, timezone
from sgp4.api import Satrec, jday

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ORBIS — Orbit Debris Intelligence System",
    page_icon="🛸",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── CSS ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Exo+2:wght@300;400;600;700&display=swap');

:root {
    --bg:        #020810;
    --bg2:       #060f1e;
    --panel:     #0a1628;
    --border:    #0d2444;
    --cyan:      #00e5ff;
    --green:     #00ff88;
    --orange:    #ff6b35;
    --red:       #ff2d55;
    --yellow:    #ffd700;
    --text:      #c8dff5;
    --muted:     #4a6a8a;
    --font-mono: 'Share Tech Mono', monospace;
    --font-main: 'Exo 2', sans-serif;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--font-main) !important;
}

[data-testid="stHeader"] { background: transparent !important; }

/* Hide default streamlit elements */
#MainMenu, footer, [data-testid="stToolbar"] { display: none !important; }

/* ── Hero header ── */
.orbis-header {
    text-align: center;
    padding: 2.5rem 0 1.5rem;
    position: relative;
}
.orbis-title {
    font-family: var(--font-mono);
    font-size: clamp(2rem, 5vw, 3.5rem);
    letter-spacing: 0.3em;
    color: var(--cyan);
    text-shadow: 0 0 40px rgba(0,229,255,0.5), 0 0 80px rgba(0,229,255,0.2);
    margin: 0;
}
.orbis-sub {
    font-family: var(--font-mono);
    font-size: 0.75rem;
    letter-spacing: 0.25em;
    color: var(--muted);
    margin-top: 0.5rem;
    text-transform: uppercase;
}
.orbis-badge {
    display: inline-block;
    font-family: var(--font-mono);
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    color: var(--green);
    border: 1px solid var(--green);
    padding: 0.2rem 0.8rem;
    margin-top: 0.8rem;
    border-radius: 2px;
    text-transform: uppercase;
}

/* ── Tabs ── */
[data-testid="stTabs"] button {
    font-family: var(--font-mono) !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.15em !important;
    color: var(--muted) !important;
    text-transform: uppercase !important;
    border: none !important;
    background: transparent !important;
    padding: 0.6rem 1.5rem !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--cyan) !important;
    border-bottom: 2px solid var(--cyan) !important;
}
[data-testid="stTabs"] [role="tablist"] {
    border-bottom: 1px solid var(--border) !important;
    background: var(--bg2) !important;
    padding: 0 1rem !important;
}

/* ── Metric cards ── */
.metric-card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-top: 2px solid var(--cyan);
    padding: 1.2rem 1.5rem;
    border-radius: 4px;
}
.metric-label {
    font-family: var(--font-mono);
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    color: var(--muted);
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}
.metric-value {
    font-family: var(--font-mono);
    font-size: 1.8rem;
    color: var(--cyan);
    line-height: 1;
}
.metric-unit {
    font-size: 0.8rem;
    color: var(--muted);
    margin-left: 0.3rem;
}

/* ── Section headers ── */
.section-header {
    font-family: var(--font-mono);
    font-size: 0.7rem;
    letter-spacing: 0.3em;
    color: var(--muted);
    text-transform: uppercase;
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.5rem;
    margin: 2rem 0 1.2rem;
}

/* ── QPU result rows ── */
.qpu-table {
    width: 100%;
    border-collapse: collapse;
    font-family: var(--font-mono);
    font-size: 0.8rem;
}
.qpu-table th {
    color: var(--muted);
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    padding: 0.5rem 1rem;
    border-bottom: 1px solid var(--border);
    text-align: left;
}
.qpu-table td {
    padding: 0.6rem 1rem;
    border-bottom: 1px solid rgba(13,36,68,0.5);
    color: var(--text);
}
.qpu-table tr:hover td { background: rgba(0,229,255,0.03); }
.quality-optimal { color: var(--green); }
.quality-warn    { color: var(--yellow); }
.quality-bad     { color: var(--orange); }
.quality-break   { color: var(--red); }

/* ── Target list ── */
.target-item {
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 0.7rem 1rem;
    border-bottom: 1px solid var(--border);
    font-family: var(--font-mono);
    font-size: 0.78rem;
}
.target-idx {
    color: var(--cyan);
    font-size: 0.65rem;
    min-width: 2rem;
}
.target-name { color: var(--text); flex: 1; }
.target-alt  { color: var(--muted); font-size: 0.7rem; }
.target-inc  { color: var(--muted); font-size: 0.7rem; }

/* ── Finding cards ── */
.finding-card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-left: 3px solid var(--cyan);
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
    border-radius: 0 4px 4px 0;
}
.finding-number {
    font-family: var(--font-mono);
    font-size: 0.6rem;
    letter-spacing: 0.2em;
    color: var(--cyan);
    text-transform: uppercase;
    margin-bottom: 0.3rem;
}
.finding-text {
    font-size: 0.9rem;
    color: var(--text);
    line-height: 1.5;
}

/* ── Buttons ── */
.stButton button {
    background: transparent !important;
    border: 1px solid var(--cyan) !important;
    color: var(--cyan) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    padding: 0.5rem 1.5rem !important;
    border-radius: 2px !important;
    transition: all 0.2s !important;
}
.stButton button:hover {
    background: rgba(0,229,255,0.08) !important;
    box-shadow: 0 0 20px rgba(0,229,255,0.2) !important;
}

/* ── Sliders ── */
[data-testid="stSlider"] label {
    font-family: var(--font-mono) !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.1em !important;
    color: var(--muted) !important;
    text-transform: uppercase !important;
}

/* ── Selectbox ── */
[data-testid="stSelectbox"] label {
    font-family: var(--font-mono) !important;
    font-size: 0.75rem !important;
    color: var(--muted) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
}

/* ── Info/warning boxes ── */
[data-testid="stInfo"] {
    background: rgba(0,229,255,0.05) !important;
    border: 1px solid rgba(0,229,255,0.2) !important;
    border-radius: 4px !important;
    font-family: var(--font-mono) !important;
    font-size: 0.8rem !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────
EARTH_R = 6371.0
GM      = 398600.4418

# ── Hardcoded QPU results (from real IBM Heron r2 runs) ───────────────────
QPU_DATA = {
    "n4": {"qubits": 16,  "depth": 523,  "runs": [100.0, 100.0, 100.0, 100.0], "sa": 100.0},
    "n5": {"qubits": 25,  "depth": 1191, "runs": [121.1, 100.0, 110.3, 100.0, 121.1], "sa": 100.0},
    "n6": {"qubits": 36,  "depth": 2107, "runs": [100.0, 131.4, 100.0, 140.6, 107.7], "sa": 100.0},
    "n8": {"qubits": 64,  "depth": 3500, "runs": [190.5, 211.9, 182.9], "sa": 144.1},
}

# ── Hardcoded ADR targets (from real Space-Track data) ────────────────────
ADR_TARGETS = [
    {"name": "NORAD-54641", "alt": 239,  "inc": 29.0,  "type": "Rocket Body",      "dci": 0.6314},
    {"name": "NORAD-27369", "alt": 408,  "inc": 38.0,  "type": "Rocket Body",      "dci": 0.6582},
    {"name": "NORAD-49496", "alt": 280,  "inc": 64.8,  "type": "Rocket Body",      "dci": 0.6772},
    {"name": "NORAD-13778", "alt": 430,  "inc": 81.1,  "type": "Rocket Body",      "dci": 0.7354},
    {"name": "NORAD-64063", "alt": 413,  "inc": 97.4,  "type": "Rocket Body",      "dci": 0.7317},
    {"name": "NORAD-67690", "alt": 395,  "inc": 123.1, "type": "Rocket Body",      "dci": 0.5086},
    {"name": "NORAD-58015", "alt": 484,  "inc": 30.0,  "type": "Rocket Body",      "dci": 0.7803},
    {"name": "NORAD-48839", "alt": 580,  "inc": 55.0,  "type": "Debris Fragment",  "dci": 0.7447},
    {"name": "NORAD-60234", "alt": 573,  "inc": 62.0,  "type": "Debris Fragment",  "dci": 0.7721},
    {"name": "NORAD-16494", "alt": 615,  "inc": 82.5,  "type": "Defunct Satellite","dci": 0.7853},
]

# ── Helpers ────────────────────────────────────────────────────────────────
def hohmann_dv(r1, r2, i1=0.0, i2=0.0):
    v1  = np.sqrt(GM / r1); v2 = np.sqrt(GM / r2)
    a_t = (r1 + r2) / 2
    vt1 = np.sqrt(GM * (2/r1 - 1/a_t))
    vt2 = np.sqrt(GM * (2/r2 - 1/a_t))
    dv  = abs(vt1 - v1) + abs(v2 - vt2)
    vm  = np.sqrt(GM / a_t)
    dp  = 2 * vm * np.sin(np.radians(abs(i1 - i2) / 2))
    return round(dv + dp * 0.3, 4)

def build_cost_matrix(targets):
    n  = len(targets)
    cm = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                cm[i][j] = hohmann_dv(
                    EARTH_R + targets[i]["alt"],
                    EARTH_R + targets[j]["alt"],
                    targets[i]["inc"],
                    targets[j]["inc"]
                )
    return cm

def sa_solve(cm, n_reads=500):
    import random
    n    = len(cm)
    best = list(range(n))
    bc   = sum(cm[best[i]][best[i+1]] for i in range(n-1))
    for _ in range(n_reads):
        seq = list(range(n)); random.shuffle(seq)
        c   = sum(cm[seq[i]][seq[i+1]] for i in range(n-1))
        if c < bc:
            bc, best = c, seq
    return best, round(bc, 4)

def fetch_tle_catalog(max_objects=3000):
    """Fetch active catalog from CelesTrak."""
    url = "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle"
    headers = {"User-Agent": "ORBIS-Dashboard/1.0"}
    try:
        r = requests.get(url, headers=headers, timeout=30)
        r.raise_for_status()
        lines = r.text.strip().splitlines()
        tles  = []
        i = 0
        while i < len(lines) - 2 and len(tles) < max_objects:
            name  = lines[i].strip()
            line1 = lines[i+1].strip()
            line2 = lines[i+2].strip()
            if line1.startswith("1 ") and line2.startswith("2 "):
                tles.append((name, line1, line2))
                i += 3
            else:
                i += 1
        return tles
    except Exception as e:
        return []

def propagate_tles(tles):
    epoch = datetime.now(timezone.utc)
    jd, fr = jday(epoch.year, epoch.month, epoch.day,
                   epoch.hour, epoch.minute, epoch.second)
    records = []
    for name, l1, l2 in tles:
        try:
            sat = Satrec.twoline2rv(l1, l2)
            err, pos, vel = sat.sgp4(jd, fr)
            if err != 0: continue
            mm  = float(l2[52:63].strip())
            if mm <= 0: continue
            sma = (GM / (mm * 2 * np.pi / 86400)**2) ** (1/3)
            alt = sma - EARTH_R
            inc = float(l2[8:16].strip())
            if alt < 150 or alt > 50000: continue
            records.append({
                "name": name, "alt": round(alt, 1), "inc": round(inc, 2),
                "x": round(pos[0], 1), "y": round(pos[1], 1), "z": round(pos[2], 1)
            })
        except:
            continue
    return pd.DataFrame(records)

# ── Header ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="orbis-header">
    <p class="orbis-title">◈ ORBIS</p>
    <p class="orbis-sub">Orbit Debris Intelligence System</p>
    <span class="orbis-badge">⬡ IBM Heron r2 · 17 QPU Runs · Real Quantum Hardware</span>
</div>
""", unsafe_allow_html=True)

# ── Top metrics ────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
metrics = [
    ("Objects Tracked", "15,982", "catalogued"),
    ("ADR Targets", "10", "selected"),
    ("QPU Runs", "17", "experiments"),
    ("Cost Variance", "0.849", "km²/s²"),
    ("NISQ Limit", "~36q", "Heron r2"),
]
for col, (label, val, unit) in zip([c1,c2,c3,c4,c5], metrics):
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{val}<span class="metric-unit">{unit}</span></div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "◈  DEBRIS GLOBE",
    "⬡  QPU BENCHMARK",
    "◎  MISSION OPTIMIZER"
])

# ══════════════════════════════════════════════════════════════════════════
# TAB 1 — DEBRIS GLOBE
# ══════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">Live Orbital Debris Field</div>', unsafe_allow_html=True)

    col_left, col_right = st.columns([3, 1])

    with col_right:
        st.markdown('<div class="section-header">ADR Priority Targets</div>', unsafe_allow_html=True)
        for i, t in enumerate(ADR_TARGETS):
            color = "#ff2d55" if t["dci"] > 0.75 else "#ffd700" if t["dci"] > 0.65 else "#00ff88"
            st.markdown(f"""
            <div class="target-item">
                <span class="target-idx">T{i+1:02d}</span>
                <span class="target-name">{t['name']}</span>
                <span class="target-alt">{t['alt']}km</span>
                <span class="target-inc">{t['inc']}°</span>
                <span style="color:{color};font-family:var(--font-mono);font-size:0.7rem">
                    {t['dci']:.3f}
                </span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        load_live = st.button("⟳  Fetch Live TLE Data", use_container_width=True)

    with col_left:
        if load_live or "catalog_df" not in st.session_state:
            if load_live:
                with st.spinner("Fetching live TLE catalog from CelesTrak..."):
                    tles = fetch_tle_catalog(max_objects=2000)
                    if tles:
                        st.session_state["catalog_df"] = propagate_tles(tles)
                        st.session_state["catalog_ts"] = datetime.now(timezone.utc).strftime("%H:%M UTC")
                    else:
                        st.session_state["catalog_df"] = None

        df = st.session_state.get("catalog_df", None)
        ts = st.session_state.get("catalog_ts", "—")

        fig = go.Figure()

        # Earth sphere
        u_ = np.linspace(0, 2*np.pi, 50)
        v_ = np.linspace(0, np.pi, 25)
        fig.add_trace(go.Surface(
            x=EARTH_R*np.outer(np.cos(u_), np.sin(v_)),
            y=EARTH_R*np.outer(np.sin(u_), np.sin(v_)),
            z=EARTH_R*np.outer(np.ones(50), np.cos(v_)),
            colorscale=[[0,"#0a1f3d"],[0.5,"#0d3060"],[1,"#1a4a8a"]],
            showscale=False, opacity=0.85, hoverinfo="skip", name="Earth"
        ))

        # Catalog points if loaded
        if df is not None and len(df) > 0:
            sample = df.sample(min(2000, len(df)), random_state=42)
            alt_bins = pd.cut(sample["alt"], bins=[0,600,1000,2000,50000],
                              labels=["LEO<600","LEO 600-1k","LEO 1k-2k","HEO+"])
            colors = {"LEO<600":"#00e5ff","LEO 600-1k":"#00ff88",
                      "LEO 1k-2k":"#ffd700","HEO+":"#ff6b35"}
            for band, color in colors.items():
                sub = sample[alt_bins == band]
                if len(sub) == 0: continue
                fig.add_trace(go.Scatter3d(
                    x=sub["x"], y=sub["y"], z=sub["z"],
                    mode="markers", name=band,
                    marker=dict(size=1.5, color=color, opacity=0.4),
                    hovertemplate="%{customdata}<extra></extra>",
                    customdata=sub["name"]
                ))

        # ADR targets — computed positions (approximate circular orbit)
        for i, t in enumerate(ADR_TARGETS):
            r    = EARTH_R + t["alt"]
            ang  = (i / len(ADR_TARGETS)) * 2 * np.pi
            inc  = np.radians(t["inc"])
            x    = r * np.cos(ang)
            y    = r * np.sin(ang) * np.cos(inc)
            z    = r * np.sin(ang) * np.sin(inc)
            fig.add_trace(go.Scatter3d(
                x=[x], y=[y], z=[z],
                mode="markers+text",
                name=f"T{i+1} {t['name']}",
                marker=dict(size=6, color="#ff2d55",
                           symbol="diamond",
                           line=dict(color="#ffffff", width=1)),
                text=[f"T{i+1}"],
                textposition="top center",
                textfont=dict(color="#ffffff", size=9, family="Share Tech Mono"),
                hovertemplate=(
                    f"<b>T{i+1}: {t['name']}</b><br>"
                    f"Alt: {t['alt']} km<br>"
                    f"Inc: {t['inc']}°<br>"
                    f"DCI: {t['dci']:.4f}<br>"
                    f"Type: {t['type']}<extra></extra>"
                )
            ))

        fig.update_layout(
            height=580,
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor="rgba(2,8,16,0)",
            scene=dict(
                xaxis=dict(showgrid=False, showticklabels=False,
                           zeroline=False, backgroundcolor="rgba(0,0,0,0)"),
                yaxis=dict(showgrid=False, showticklabels=False,
                           zeroline=False, backgroundcolor="rgba(0,0,0,0)"),
                zaxis=dict(showgrid=False, showticklabels=False,
                           zeroline=False, backgroundcolor="rgba(0,0,0,0)"),
                bgcolor="rgba(2,8,16,1)",
                camera=dict(eye=dict(x=1.5, y=1.5, z=0.8))
            ),
            legend=dict(
                font=dict(family="Share Tech Mono", size=10, color="#4a6a8a"),
                bgcolor="rgba(6,15,30,0.8)",
                bordercolor="#0d2444", borderwidth=1,
                x=0.01, y=0.99
            )
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        if ts != "—":
            st.markdown(f'<p style="font-family:var(--font-mono);font-size:0.65rem;'
                        f'color:var(--muted);text-align:center">'
                        f'Last fetched: {ts} · {len(df):,} objects propagated</p>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<p style="font-family:var(--font-mono);font-size:0.65rem;'
                        'color:var(--muted);text-align:center">'
                        'Click "Fetch Live TLE Data" to load real-time orbital catalog</p>',
                        unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# TAB 2 — QPU BENCHMARK
# ══════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">Real Quantum Hardware Results — IBM Heron r2 (ibm_fez)</div>',
                unsafe_allow_html=True)

    # Key findings
    findings = [
        ("FINDING 01 — STABLE REGIME",
         "n=4 (16 qubits, circuit depth 523): QAOA on real IBM Heron r2 achieved 100% optimal "
         "across all 4 runs. Hardware noise at this scale is fully manageable."),
        ("FINDING 02 — STOCHASTIC DEGRADATION",
         "n=5 and n=6 (25–36 qubits, depth 1191–2107): Run-to-run quality ranged from 100% to 140.6% "
         "of optimal. Same circuit, same hardware, different results — demonstrating NISQ-era "
         "non-determinism empirically across 9 total runs."),
        ("FINDING 03 — NISQ PHASE TRANSITION",
         "Between n=6 and n=8, mean degradation jumps from ~16% to ~95% above optimal. "
         "This identifies a practical NISQ limit of ~36–64 qubits for p=1 QAOA on current hardware."),
        ("FINDING 04 — QUANTUM vs CLASSICAL SCALING",
         "At n=8 (64 qubits), QPU mean quality (195.1%) exceeds classical SA degradation (144.1%), "
         "confirming that NISQ noise outweighs quantum advantage at this problem scale. "
         "Classical methods remain superior until fault-tolerant QC."),
    ]
    for num, text in findings:
        st.markdown(f"""
        <div class="finding-card">
            <div class="finding-number">{num}</div>
            <div class="finding-text">{text}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b = st.columns([3, 2])

    with col_a:
        st.markdown('<div class="section-header">Complete Multi-Run Distribution</div>',
                    unsafe_allow_html=True)

        rows = []
        for key, d in QPU_DATA.items():
            n    = int(key[1:])
            runs = d["runs"]
            mean = np.mean(runs)
            std  = np.std(runs)
            for r_idx, quality in enumerate(runs):
                rows.append({
                    "n": n, "Run": f"Run {r_idx+1}",
                    "Qubits": d["qubits"], "Depth": d["depth"],
                    "Quality (%)": quality, "SA (%)": d["sa"],
                    "Mean": round(mean, 1), "Std": round(std, 1)
                })

        def quality_color(q):
            if q <= 100.5:   return "quality-optimal"
            elif q <= 115.0: return "quality-warn"
            elif q <= 150.0: return "quality-bad"
            return "quality-break"

        html = """
        <table class="qpu-table">
        <thead><tr>
            <th>n</th><th>Qubits</th><th>Depth</th>
            <th>Run</th><th>QPU Quality</th><th>SA Quality</th>
        </tr></thead><tbody>
        """
        prev_n = None
        for r in rows:
            border = "border-top: 1px solid #0d2444;" if r["n"] != prev_n and prev_n else ""
            cls    = quality_color(r["Quality (%)"])
            sa_cls = quality_color(r["SA (%)"])
            html += f"""
            <tr style="{border}">
                <td style="color:var(--cyan)">{r['n'] if r['n'] != prev_n else ''}</td>
                <td>{r['Qubits'] if r['n'] != prev_n else ''}</td>
                <td style="color:var(--muted)">{r['Depth'] if r['n'] != prev_n else ''}</td>
                <td style="color:var(--muted)">{r['Run']}</td>
                <td class="{cls}">{r['Quality (%)']:.1f}%</td>
                <td class="{sa_cls}">{r['SA (%)']:.1f}%</td>
            </tr>"""
            prev_n = r["n"]
        html += "</tbody></table>"
        st.markdown(html, unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="section-header">Noise vs Circuit Depth</div>',
                    unsafe_allow_html=True)

        # Box plot of distributions
        fig2 = go.Figure()
        colors = {"n4": "#00ff88", "n5": "#ffd700", "n6": "#ff6b35", "n8": "#ff2d55"}
        labels = {"n4": "n=4 (16q)", "n5": "n=5 (25q)",
                  "n6": "n=6 (36q)", "n8": "n=8 (64q)"}
        for key, d in QPU_DATA.items():
            fig2.add_trace(go.Box(
                y=d["runs"],
                name=labels[key],
                marker_color=colors[key],
                line_color=colors[key],
                fillcolor=colors[key].replace(")", ",0.1)").replace("rgb", "rgba")
                          if "rgb" in colors[key] else colors[key],
                boxmean=True,
                hovertemplate=f"{labels[key]}<br>Quality: %{{y:.1f}}%<extra></extra>"
            ))

        fig2.add_hline(y=100, line_dash="dash", line_color="#4a6a8a",
                       annotation_text="Optimal (100%)",
                       annotation_font=dict(color="#4a6a8a", size=10,
                                           family="Share Tech Mono"))
        fig2.update_layout(
            height=320,
            paper_bgcolor="rgba(6,15,30,0)",
            plot_bgcolor="rgba(6,15,30,0.5)",
            font=dict(family="Share Tech Mono", color="#4a6a8a", size=10),
            yaxis=dict(title="QPU Quality (%)", gridcolor="#0d2444",
                       color="#4a6a8a", zeroline=False),
            xaxis=dict(gridcolor="#0d2444", color="#4a6a8a"),
            showlegend=False,
            margin=dict(l=40, r=10, t=10, b=10)
        )
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

        # Mean degradation line chart
        st.markdown('<div class="section-header">Mean Degradation vs Scale</div>',
                    unsafe_allow_html=True)
        ns    = [4, 5, 6, 8]
        means = [np.mean(QPU_DATA[f"n{n}"]["runs"]) for n in ns]
        sas   = [QPU_DATA[f"n{n}"]["sa"] for n in ns]
        depths= [QPU_DATA[f"n{n}"]["depth"] for n in ns]

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=ns, y=means, name="QPU Mean",
            mode="lines+markers",
            line=dict(color="#00e5ff", width=2),
            marker=dict(size=8, color="#00e5ff"),
            hovertemplate="n=%{x}<br>QPU Mean: %{y:.1f}%<extra></extra>"
        ))
        fig3.add_trace(go.Scatter(
            x=ns, y=sas, name="SA Classical",
            mode="lines+markers",
            line=dict(color="#ff6b35", width=2, dash="dot"),
            marker=dict(size=8, color="#ff6b35"),
            hovertemplate="n=%{x}<br>SA: %{y:.1f}%<extra></extra>"
        ))
        fig3.add_hline(y=100, line_dash="dash", line_color="#4a6a8a")
        fig3.update_layout(
            height=220,
            paper_bgcolor="rgba(6,15,30,0)",
            plot_bgcolor="rgba(6,15,30,0.5)",
            font=dict(family="Share Tech Mono", color="#4a6a8a", size=10),
            xaxis=dict(title="Problem size n", gridcolor="#0d2444",
                       color="#4a6a8a", tickvals=ns),
            yaxis=dict(title="Quality (%)", gridcolor="#0d2444", color="#4a6a8a"),
            legend=dict(font=dict(size=9), bgcolor="rgba(0,0,0,0)",
                        borderwidth=0),
            margin=dict(l=40, r=10, t=10, b=40)
        )
        st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

    # Hardware info footer
    st.markdown("""
    <div style="margin-top:2rem;padding:1rem;background:var(--panel);
                border:1px solid var(--border);border-radius:4px;
                font-family:var(--font-mono);font-size:0.72rem;color:var(--muted)">
        <span style="color:var(--cyan)">Hardware:</span> IBM Heron r2 processor (ibm_fez) ·
        156 physical qubits · Falcon-class connectivity ·
        <span style="color:var(--cyan)">Total QPU runs:</span> 17 ·
        <span style="color:var(--cyan)">Date:</span> 2026-05-02 ·
        <span style="color:var(--cyan)">Backend:</span> IBM Quantum Open Plan (free tier) ·
        <span style="color:var(--cyan)">Shots per run:</span> 1,024
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# TAB 3 — MISSION OPTIMIZER
# ══════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">Active Debris Removal — Mission Sequence Optimizer</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown('<div class="section-header">Mission Parameters</div>', unsafe_allow_html=True)

        n_sel = st.slider("Number of targets to sequence", 3, 10, 5)
        solver_choice = st.selectbox(
            "Optimization method",
            ["Simulated Annealing (Classical)", "Brute Force (Exact, n≤8)"]
        )

        st.markdown("<br>", unsafe_allow_html=True)
        run_opt = st.button("◎  Compute Optimal Sequence", use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div style="background:var(--panel);border:1px solid var(--border);
                    border-left:3px solid rgba(0,229,255,0.3);
                    padding:1rem;border-radius:0 4px 4px 0;
                    font-family:var(--font-mono);font-size:0.72rem;color:var(--muted)">
            <div style="color:var(--cyan);margin-bottom:0.5rem;font-size:0.65rem;
                        letter-spacing:0.15em">NOTE ON QUANTUM RESULTS</div>
            QAOA results on IBM Heron r2 are shown in the QPU Benchmark tab.
            17 real hardware experiments were conducted across n=4 to n=8.
            Live QPU execution is not available in this demo due to
            IBM Quantum API rate limits (10 min/month free tier).
        </div>
        """, unsafe_allow_html=True)

    with col2:
        targets_sel = ADR_TARGETS[:n_sel]
        cm_sel      = build_cost_matrix(targets_sel)

        if run_opt:
            with st.spinner("Computing optimal removal sequence..."):
                if solver_choice.startswith("Brute") and n_sel <= 8:
                    from itertools import permutations
                    best_seq, best_cost = None, float("inf")
                    for perm in permutations(range(n_sel)):
                        c = sum(cm_sel[perm[i]][perm[i+1]] for i in range(n_sel-1))
                        if c < best_cost:
                            best_cost, best_seq = c, list(perm)
                    method = "Brute Force (Exact)"
                else:
                    best_seq, best_cost = sa_solve(cm_sel, n_reads=2000)
                    method = "Simulated Annealing"
                st.session_state["opt_result"] = (best_seq, best_cost, method, targets_sel, cm_sel)

        if "opt_result" in st.session_state:
            seq, cost, method, tgts, cm = st.session_state["opt_result"]
            if len(tgts) == n_sel:

                st.markdown('<div class="section-header">Optimal Removal Sequence</div>',
                            unsafe_allow_html=True)

                # Result header
                st.markdown(f"""
                <div style="display:flex;gap:2rem;margin-bottom:1.5rem">
                    <div class="metric-card" style="flex:1">
                        <div class="metric-label">Total ΔV Cost</div>
                        <div class="metric-value">{cost:.4f}<span class="metric-unit">km/s</span></div>
                    </div>
                    <div class="metric-card" style="flex:1">
                        <div class="metric-label">Method</div>
                        <div class="metric-value" style="font-size:1.1rem">{method}</div>
                    </div>
                    <div class="metric-card" style="flex:1">
                        <div class="metric-label">Targets Sequenced</div>
                        <div class="metric-value">{n_sel}<span class="metric-unit">objects</span></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Sequence visualization
                fig4 = go.Figure()

                # Add orbital shells
                for alt, color, label in [
                    (400, "#00e5ff", "400km"),
                    (500, "#00ff88", "500km"),
                    (600, "#ffd700", "600km")
                ]:
                    theta = np.linspace(0, 2*np.pi, 100)
                    r = EARTH_R + alt
                    fig4.add_trace(go.Scatterpolar(
                        r=[r]*100, theta=np.degrees(theta),
                        mode="lines",
                        line=dict(color=color, width=0.5, dash="dot"),
                        opacity=0.2, showlegend=False, hoverinfo="skip"
                    ))

                # Plot targets
                for i, t_idx in enumerate(seq):
                    t = tgts[t_idx]
                    r = EARTH_R + t["alt"]
                    angle = (t_idx / n_sel) * 360
                    color = "#ff2d55" if i == 0 else "#00e5ff" if i == len(seq)-1 else "#ffd700"
                    size  = 14 if i in [0, len(seq)-1] else 10
                    fig4.add_trace(go.Scatterpolar(
                        r=[r], theta=[angle],
                        mode="markers+text",
                        marker=dict(size=size, color=color,
                                   symbol="diamond" if i == 0 else "circle"),
                        text=[f"Step {i+1}: {t['name']}"],
                        textposition="top center",
                        textfont=dict(size=9, color=color,
                                     family="Share Tech Mono"),
                        showlegend=False,
                        hovertemplate=(
                            f"<b>Step {i+1}: {t['name']}</b><br>"
                            f"Alt: {t['alt']} km<br>"
                            f"Inc: {t['inc']}°<br>"
                            f"DCI: {t['dci']:.4f}<extra></extra>"
                        )
                    ))

                # Draw sequence path
                seq_r     = [EARTH_R + tgts[seq[i]]["alt"] for i in range(len(seq))]
                seq_theta = [(tgts[seq[i]]["inc"] / 180 * 360) % 360 for i in range(len(seq))]
                seq_r.append(seq_r[0]);  seq_theta.append(seq_theta[0])
                fig4.add_trace(go.Scatterpolar(
                    r=seq_r, theta=seq_theta,
                    mode="lines",
                    line=dict(color="rgba(0,229,255,0.3)", width=1.5, dash="dot"),
                    showlegend=False, hoverinfo="skip"
                ))

                fig4.update_layout(
                    height=400,
                    paper_bgcolor="rgba(2,8,16,0)",
                    polar=dict(
                        bgcolor="rgba(6,15,30,0.8)",
                        radialaxis=dict(
                            showgrid=True, gridcolor="#0d2444",
                            color="#4a6a8a", showticklabels=True,
                            tickfont=dict(size=8, family="Share Tech Mono"),
                            range=[6200, 7200]
                        ),
                        angularaxis=dict(
                            showgrid=True, gridcolor="#0d2444",
                            color="#4a6a8a",
                            tickfont=dict(size=8, family="Share Tech Mono")
                        )
                    ),
                    margin=dict(l=40, r=40, t=20, b=20)
                )
                st.plotly_chart(fig4, use_container_width=True,
                                config={"displayModeBar": False})

                # Step-by-step table
                st.markdown('<div class="section-header">Step-by-Step Mission Plan</div>',
                            unsafe_allow_html=True)
                html5 = """
                <table class="qpu-table">
                <thead><tr>
                    <th>Step</th><th>Target</th><th>Alt</th>
                    <th>Inc</th><th>Type</th><th>ΔV to Next</th><th>DCI</th>
                </tr></thead><tbody>
                """
                cumulative = 0.0
                for step_i, t_idx in enumerate(seq):
                    t   = tgts[t_idx]
                    dv  = cm[seq[step_i]][seq[step_i+1]] if step_i < len(seq)-1 else 0.0
                    cumulative += dv
                    dv_str = f"{dv:.4f} km/s" if step_i < len(seq)-1 else "—"
                    dci_color = "#ff2d55" if t["dci"] > 0.75 else "#ffd700" if t["dci"] > 0.65 else "#00ff88"
                    html5 += f"""
                    <tr>
                        <td style="color:var(--cyan)">{step_i+1}</td>
                        <td>{t['name']}</td>
                        <td style="color:var(--muted)">{t['alt']} km</td>
                        <td style="color:var(--muted)">{t['inc']}°</td>
                        <td style="color:var(--muted)">{t['type']}</td>
                        <td style="color:var(--text)">{dv_str}</td>
                        <td style="color:{dci_color}">{t['dci']:.4f}</td>
                    </tr>"""
                html5 += f"""
                <tr style="border-top:1px solid var(--cyan)">
                    <td colspan="5" style="color:var(--muted);text-align:right">
                        Total mission ΔV:
                    </td>
                    <td style="color:var(--cyan);font-weight:bold">{cost:.4f} km/s</td>
                    <td></td>
                </tr>
                </tbody></table>"""
                st.markdown(html5, unsafe_allow_html=True)

        else:
            st.markdown("""
            <div style="height:300px;display:flex;align-items:center;justify-content:center;
                        border:1px solid var(--border);border-radius:4px;
                        background:var(--panel)">
                <div style="text-align:center;font-family:var(--font-mono)">
                    <div style="font-size:2rem;color:var(--border);margin-bottom:1rem">◎</div>
                    <div style="color:var(--muted);font-size:0.75rem;letter-spacing:0.2em">
                        SELECT PARAMETERS AND RUN OPTIMIZER
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:3rem;padding:1.5rem;text-align:center;
            border-top:1px solid var(--border)">
    <p style="font-family:var(--font-mono);font-size:0.65rem;
              color:var(--muted);letter-spacing:0.2em;margin:0">
        ORBIS · ORBIT DEBRIS INTELLIGENCE SYSTEM ·
        VTU BANGALORE · IBM QUANTUM OPEN PLAN ·
        17 REAL QPU EXPERIMENTS ON IBM HERON R2
    </p>
</div>
""", unsafe_allow_html=True)
