import streamlit as st
import numpy as np
import pandas as pd
import requests
import time
import dimod
from datetime import datetime, timezone
from sgp4.api import Satrec, jday
import plotly.graph_objects as go
import plotly.express as px
import os

st.set_page_config(page_title="ORBIS", layout="wide", initial_sidebar_state="expanded")

# ----------------- CSS INJECTION (SAAS PROFESSIONAL) -----------------
def inject_custom_css():
    st.markdown("""
<style>
    /* ----- FUNDAMENTAL VARIABLES ----- */
    :root {
        --primary-red: #E24B4A;
        --primary-red-hover: #D13A39;
        --secondary-muted: #64748b;
        --app-bg: #0A0D14;
        --panel-bg: #131722;
        --border-color: rgba(255, 255, 255, 0.08);
        --ease-elite: cubic-bezier(0.23, 1, 0.32, 1);
        --text-color: #e2e8f0;
    }

    /* ----- CORE APP BACKGROUND ----- */
    .stApp {
        background-color: var(--app-bg) !important;
        font-family: 'Inter', -apple-system, sans-serif !important;
        color: var(--text-color);
    }

    /* ----- SIDEBAR ----- */
    [data-testid="stSidebar"] {
        background: #0D111A !important;
        border-right: 1px solid var(--border-color) !important;
    }

    /* ----- HEADER / HERO STYLING ----- */
    .hero-title {
        font-size: 3rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 0px;
    }
    .hero-subtitle {
        font-size: 1.1rem;
        color: #94a3b8;
        font-weight: 400;
        margin-bottom: 2rem;
    }

    /* ----- METRIC CARDS ----- */
    [data-testid="stMetric"] {
        background: var(--panel-bg);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 1.25rem;
        transition: all 200ms var(--ease-elite);
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        border-color: rgba(255, 255, 255, 0.2);
    }
    [data-testid="stMetricValue"] {
        font-size: 2.2rem !important;
        font-weight: 600 !important;
        color: #ffffff !important;
    }
    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-size: 0.85rem !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 500;
    }

    /* ----- BUTTONS ----- */
    div.stButton > button {
        background: #1A202C !important;
        border: 1px solid var(--border-color) !important;
        color: #e2e8f0 !important;
        border-radius: 6px !important;
        font-weight: 500 !important;
        transition: all 200ms var(--ease-elite) !important;
        padding: 0.5rem 1rem !important;
    }
    div.stButton > button:hover {
        background: #2D3748 !important;
        border-color: rgba(255, 255, 255, 0.2) !important;
        color: #ffffff !important;
    }
    div.stButton > button:active {
        transform: scale(0.98) !important;
    }

    /* Primary execute button overriding */
    button[kind="primary"] {
        background: var(--primary-red) !important;
        border: 1px solid var(--primary-red) !important;
        color: white !important;
        box-shadow: none !important;
    }
    button[kind="primary"]:hover {
        background: var(--primary-red-hover) !important;
        border-color: var(--primary-red-hover) !important;
    }

    /* ----- TABS PROFESSIONAL DESIGN ----- */
    [data-testid="stTabs"] button {
        background: transparent !important;
        border: none !important;
        border-bottom: 2px solid transparent !important;
        color: var(--secondary-muted) !important;
        font-weight: 500 !important;
        transition: all 200ms var(--ease-elite) !important;
        padding-bottom: 0.5rem !important;
        border-radius: 0px !important;
    }
    [data-testid="stTabs"] button[aria-selected="true"] {
        color: white !important;
        background: transparent !important;
        border-bottom: 2px solid var(--primary-red) !important;
        text-shadow: none !important;
    }

    /* ----- DATAFRAME OVERHAUL ----- */
    [data-testid="stDataFrame"] {
        border-radius: 8px !important;
        overflow: hidden !important;
        border: 1px solid var(--border-color) !important;
        background: var(--panel-bg) !important;
    }
    
    /* Expander / Containers Customization */
    [data-testid="stExpander"] {
        border: 1px solid var(--border-color) !important;
        border-radius: 8px !important;
        background: var(--panel-bg) !important;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- CONSTANTS & UTILS -----------------
EARTH_R = 6371.0
GM = 398600.4418

def hohmann_delta_v(r1, r2):
    v1=np.sqrt(GM/r1); v2=np.sqrt(GM/r2)
    a_t=(r1+r2)/2
    vt1=np.sqrt(GM*(2/r1-1/a_t)); vt2=np.sqrt(GM*(2/r2-1/a_t))
    return abs(vt1-v1)+abs(v2-vt2)

@st.cache_data(ttl=3600)
def load_data():
    snapshot_path = os.path.join(os.path.dirname(__file__), "debris_snapshot.csv")
    df_snapshot = pd.DataFrame()
    if os.path.exists(snapshot_path):
        try: df_snapshot = pd.read_csv(snapshot_path)
        except: pass

    BASE = "https://celestrak.org/NORAD/elements/gp.php"
    GROUPS = ["cosmos-deb","iridium-33-deb","fengyun-1c-deb","rocket-bodies","active"]
    all_tles = {}
    for group in GROUPS:
        try:
            r = requests.get(f"{BASE}?GROUP={group}&FORMAT=tle", timeout=15, headers={"User-Agent":"ORBIS/1.0"})
            if r.status_code != 200: continue
            lines = r.text.strip().splitlines()
            i = 0
            while i < len(lines)-2:
                name=lines[i].strip(); l1=lines[i+1].strip(); l2=lines[i+2].strip()
                if l1.startswith("1 ") and l2.startswith("2 "):
                    try:
                        nid=int(l1[2:7])
                        if nid not in all_tles: all_tles[nid]=(name,l1,l2)
                    except: pass
                    i+=3
                else: i+=1
        except: continue

    if all_tles:
        epoch=datetime.now(timezone.utc)
        records=[]
        for nid,(name,l1,l2) in all_tles.items():
            try:
                sat=Satrec.twoline2rv(l1,l2)
                jd,fr=jday(epoch.year,epoch.month,epoch.day, epoch.hour,epoch.minute,epoch.second)
                err,pos,vel=sat.sgp4(jd,fr)
                if err!=0 or any(np.isnan(v) for v in pos): continue
                mm=float(l2[52:63])
                if mm<=0: continue
                alt=(GM/(mm*2*np.pi/86400)**2)**(1/3)-EARTH_R
                if alt<100 or alt>50000: continue
                inc=float(l2[8:16])
                nu=name.upper()
                if any(k in nu for k in ["R/B","ROCKET","BOOSTER"]): otype="Rocket Body"
                elif any(k in nu for k in ["DEB","DEBRIS","FRAG"]): otype="Debris Fragment"
                elif any(k in nu for k in ["COSMOS","IRIDIUM","FENGYUN"]): otype="Defunct Satellite"
                else: otype="Active/Unknown"
                mass=5000 if "R/B" in nu else 10 if "DEB" in nu else 700
                alt_f=max(0.05,min(1.0,1.0-(alt-200)/2000)) if alt<1200 else 0.1
                inc_f=abs(np.sin(np.radians(inc)))
                dci=round(0.4*alt_f+0.35*(np.log1p(mass)/np.log1p(9000))+0.15*inc_f,3)
                risk="High" if dci>0.6 else "Medium" if dci>0.3 else "Low"
                records.append({
                    "name":name.strip(),"norad_id":nid,"altitude_km":round(alt,1),"inclination":round(inc,2),
                    "x_eci_km":round(pos[0],1),"y_eci_km":round(pos[1],1),"z_eci_km":round(pos[2],1),
                    "speed_kms":round(np.sqrt(sum(v**2 for v in vel)),3),
                    "object_type":otype,"dci_score":dci,"risk_tier":risk,"mass_kg_est":mass,
                })
            except: continue
        if records:
            return pd.DataFrame(records)

    if not df_snapshot.empty:
        return df_snapshot
    return pd.DataFrame()

def build_qubo(cm, penalty=None):
    n=len(cm)
    if penalty is None: penalty=2.0*float(np.sum(cm))
    Q={}
    def idx(i,k): return i*n+k
    for k in range(n-1):
        for i in range(n):
            for j in range(n):
                if i!=j:
                    a,b=idx(i,k),idx(j,k+1)
                    key=(min(a,b),max(a,b))
                    Q[key]=Q.get(key,0)+cm[i][j]
    for i in range(n):
        for k in range(n):
            a=idx(i,k)
            Q[(a,a)]=Q.get((a,a),0)+penalty*(1-2)
            for l in range(k+1,n):
                b=idx(i,l); key=(min(a,b),max(a,b))
                Q[key]=Q.get(key,0)+2*penalty
    for k in range(n):
        for i in range(n):
            a=idx(i,k)
            Q[(a,a)]=Q.get((a,a),0)+penalty*(1-2)
            for j in range(i+1,n):
                b=idx(j,k); key=(min(a,b),max(a,b))
                Q[key]=Q.get(key,0)+2*penalty
    return Q,n

def decode_sol(sample,n):
    matrix=np.zeros((n,n),dtype=int)
    for i in range(n):
        for k in range(n): matrix[i][k]=sample.get(i*n+k,0)
    seq=[]
    for k in range(n):
        col=matrix[:,k]
        if col.sum()!=1: return None
        seq.append(int(np.argmax(col)))
    return seq if len(set(seq))==n else None

def seq_cost(seq,cm):
    return round(sum(cm[seq[i]][seq[i+1]] for i in range(len(seq)-1)),4)

def sa_solve(cm):
    Q,n=build_qubo(cm)
    bqm=dimod.BinaryQuadraticModel.from_qubo(Q)
    try: from dwave.samplers import SimulatedAnnealingSampler
    except: SimulatedAnnealingSampler=dimod.SimulatedAnnealingSampler
    sampler=SimulatedAnnealingSampler()
    result=sampler.sample(bqm,num_reads=500,num_sweeps=1000,seed=42)
    best_seq,best_cost=None,float("inf")
    for sample,energy in result.data(["sample","energy"]):
        seq=decode_sol(sample,n)
        if seq:
            c=seq_cost(seq,cm)
            if c<best_cost: best_cost,best_seq=c,seq
    return best_seq,best_cost

# ----------------- UI COMPONENTS -----------------
def render_sidebar():
    with st.sidebar:
        st.markdown("<h1 style='color:#E24B4A; font-weight:700; font-size:2rem; margin-bottom: 0px;'>ORBIS Platform</h1>", unsafe_allow_html=True)
        st.caption("v2.4.1 // System Architecture")
        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.button("INITIATE TELEMETRY SYNC", use_container_width=True):
            st.cache_data.clear(); st.rerun()
            
        st.divider()
        st.markdown("<h4 style='color:#ffffff; font-size:1rem; font-weight:500;'>SYSTEM STATUS</h4>", unsafe_allow_html=True)
        st.caption(f"**EPOCH**<br>{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}", unsafe_allow_html=True)
        st.caption("**QPU BACKEND**<br>D-Wave dimod active", unsafe_allow_html=True)
        st.caption("**UPLINK FEED**<br>CelesTrak Secured", unsafe_allow_html=True)

def render_hero():
    st.markdown('''
    <div style="padding-top: 1rem;">
        <h1 class="hero-title">Orbit Debris Intelligence System</h1>
        <p class="hero-subtitle">Real-time quantum-optimized orbital tracking and trajectory remediation board.</p>
    </div>
    ''', unsafe_allow_html=True)

def render_kpi_row(df):
    if df.empty: return
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Objects Tracked", f"{len(df):,}")
    c2.metric("Critical Risk", f"{len(df[df['risk_tier']=='High']):,}")
    c3.metric("Altitude Band", f"{df['altitude_km'].min():.0f}-{df['altitude_km'].max():.0f} km")
    c4.metric("Avg DCI (Threat)", f"{df['dci_score'].mean():.3f}")
    st.markdown("<br>", unsafe_allow_html=True)

def render_globe_view(df):
    col1, col2 = st.columns([1, 4])
    with col1:
        st.markdown("<h3 style='color:#e2e8f0; font-size: 1.1rem;'>Filter Parameters</h3>", unsafe_allow_html=True)
        risk_filter = st.multiselect("Risk Tier Tolerance", ["High", "Medium", "Low"], default=["High", "Medium", "Low"])
        max_pts = st.slider("Target Density Count", 500, 5000, 2000, step=500)
    with col2:
        fdf = df[df["risk_tier"].isin(risk_filter)]
        plot_df = fdf.sample(min(max_pts, len(fdf)), random_state=42) if len(fdf) > 0 else fdf
        
        # Professional Custom Palette
        color_map = {"High": "#E24B4A", "Medium": "#EF9F27", "Low": "#378ADD"}
        fig = go.Figure()
        
        for tier, color in color_map.items():
            sub = plot_df[plot_df["risk_tier"] == tier]
            if len(sub) == 0: continue
            fig.add_trace(go.Scatter3d(
                x=sub["x_eci_km"], y=sub["y_eci_km"], z=sub["z_eci_km"],
                mode="markers", name=f"{tier} ({len(sub):,})",
                marker=dict(size=2.5, color=color, opacity=0.8, line=dict(width=0)),
                text=sub["name"]+"<br>Alt: "+sub["altitude_km"].round(0).astype(str)+" km",
                hovertemplate="%{text}<extra></extra>"
            ))
            
        u_ = np.linspace(0, 2*np.pi, 30); v_ = np.linspace(0, np.pi, 15)
        # Transparent Earth sphere with subtle grey/blue tint
        fig.add_trace(go.Surface(
            x=EARTH_R*np.outer(np.cos(u_), np.sin(v_)),
            y=EARTH_R*np.outer(np.sin(u_), np.sin(v_)),
            z=EARTH_R*np.outer(np.ones(30), np.cos(v_)),
            colorscale=[[0, "rgba(55,138,221,0.05)"], [1, "rgba(55,138,221,0.1)"]],
            showscale=False, opacity=0.2, hoverinfo="skip"
        ))
        
        fig.update_layout(
            height=650,
            scene=dict(
                bgcolor="rgba(0,0,0,0)",
                xaxis=dict(title="X (km)", showgrid=True, gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.1)", showbackground=False),
                yaxis=dict(title="Y (km)", showgrid=True, gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.1)", showbackground=False),
                zaxis=dict(title="Z (km)", showgrid=True, gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.1)", showbackground=False)
            ),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8", family="Inter, sans-serif"),
            legend=dict(bgcolor="rgba(19,23,34,0.8)", bordercolor="rgba(255,255,255,0.1)", borderwidth=1),
            margin=dict(l=0, r=0, t=10, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)

def render_catalog_view(df):
    c1, c2, c3 = st.columns(3)
    with c1: tier_sel = st.selectbox("Classification Filter", ["All", "High", "Medium", "Low"])
    with c2: type_sel = st.selectbox("Debris Signature", ["All", "Rocket Body", "Debris Fragment", "Defunct Satellite", "Active/Unknown"])
    with c3: alt_range = st.slider("Orbital Altitude (km)", 100, 40000, (100, 2000))
    
    filtered = df.copy()
    if tier_sel != "All": filtered = filtered[filtered["risk_tier"] == tier_sel]
    if type_sel != "All": filtered = filtered[filtered["object_type"] == type_sel]
    filtered = filtered[(filtered["altitude_km"] >= alt_range[0]) & (filtered["altitude_km"] <= alt_range[1])]
    
    st.dataframe(
        filtered[["name", "norad_id", "altitude_km", "inclination", "speed_kms", "object_type", "dci_score", "risk_tier"]].sort_values("dci_score", ascending=False).head(200),
        use_container_width=True, height=350,
        column_config={
            "dci_score": st.column_config.ProgressColumn("DCI score", min_value=0, max_value=1),
            "altitude_km": st.column_config.NumberColumn("Altitude (km)", format="%.1f"),
        }
    )
    st.caption(f"Rendering {min(200, len(filtered)):,} of {len(filtered):,} valid targets.")
    
    # Histogram
    fig2 = px.histogram(df[df["altitude_km"] < 3000], x="altitude_km", color="object_type", nbins=80,
        title="Altitude Density Distributions",
        color_discrete_map={"Rocket Body": "#E24B4A", "Debris Fragment": "#EF9F27", "Defunct Satellite": "#64748b", "Active/Unknown": "#378ADD"})
    fig2.add_vline(x=789, line_dash="dash", line_color="#94a3b8", opacity=0.8, annotation_text="Iridium-Cosmos 2009 Collision", annotation_font_color="#94a3b8")
    fig2.update_layout(
        height=320, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8", family="Inter, sans-serif"),
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
        title_font_color="#ffffff",
        title_font_size=14
    )
    st.plotly_chart(fig2, use_container_width=True)

def render_optimizer_view(df):
    st.markdown("<h3 style='color:#e2e8f0; font-size: 1.1rem;'>QML Trajectory Targeting Matrix</h3>", unsafe_allow_html=True)
    st.caption("Designate prime targets to synthesize Hohmann delta-v matrices. Solving via simulated annealing.")
    
    top_candidates = df.nlargest(50, "dci_score")["name"].tolist()
    selected = st.multiselect("Lock Targets (3-8 for optimal execution)", options=top_candidates, default=top_candidates[:5], max_selections=10)
    
    if len(selected) < 2:
        st.warning("SYSTEM FAULT: Two or more targets required for a viable sequence.")
    else:
        n_sel = len(selected)
        sel_df = df[df["name"].isin(selected)].drop_duplicates("name").head(n_sel)
        feats = [{"name": row["name"], "r": EARTH_R+row["altitude_km"], "alt": row["altitude_km"], "inc": row["inclination"]} for _, row in sel_df.iterrows()]
        
        cm = np.zeros((n_sel, n_sel))
        for i in range(n_sel):
            for j in range(n_sel):
                if i != j:
                    dv = hohmann_delta_v(feats[i]["r"], feats[j]["r"])
                    inc_diff = abs(feats[i]["inc"] - feats[j]["inc"])
                    v_mid = np.sqrt(GM/((feats[i]["r"]+feats[j]["r"])/2))
                    dv_plane = 2*v_mid*np.sin(np.radians(inc_diff/2))
                    cm[i][j] = round(dv + dv_plane*0.3, 4)
        
        names_short = [f[:18] for f in selected[:n_sel]]
        st.write("**DELTA-V COST MATRIX (km/s):**")
        st.dataframe(pd.DataFrame(cm, index=names_short, columns=names_short).round(3), use_container_width=True)
        
        if st.button("EXECUTE QML SOLVER", type="primary", use_container_width=True):
            with st.spinner("Processing Quantum Annealing Parameters..."):
                t0 = time.time()
                seq, total_dv = sa_solve(cm)
                elapsed = time.time() - t0
            
            if seq is not None:
                st.success(f"Trajectory Compiled Algorithmicly in {elapsed:.2f}s")
                st.metric("Total Mission Delta-V Required", f"{total_dv:.4f} km/s")
                st.markdown("#### Execution Sequence:")
                for step, sidx in enumerate(seq):
                    nm = feats[sidx]["name"] if sidx < len(feats) else f"T{sidx}"
                    alt = feats[sidx]["alt"]
                    dv_s = cm[seq[step-1]][sidx] if step > 0 else 0
                    label = f" // Increase +{dv_s:.3f} km/s" if step > 0 else " // Origin"
                    st.markdown(f"<span style='color:#E24B4A; font-family:monospace; font-weight: 500;'>Step {step+1}: </span> <span style='color:#e2e8f0; font-weight:600;'>{nm}</span> (Alt: {alt:.0f} km{label})", unsafe_allow_html=True)
                
                step_dvs = [0] + [cm[seq[i]][seq[i+1]] for i in range(len(seq)-1)]
                fig3 = px.bar(x=[f"Vector {i+1}" for i in range(len(seq))], y=step_dvs,
                              title="Thrust Consumption Per Vector", labels={"x": "Sequence Vector", "y": "Delta-V (km/s)"},
                              color=step_dvs, color_continuous_scale=["#378ADD", "#EF9F27", "#E24B4A"])
                fig3.update_layout(
                    height=320, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#94a3b8", family="Inter, sans-serif"),
                    xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
                    showlegend=False, coloraxis_showscale=False, title_font_color="#ffffff"
                )
                st.plotly_chart(fig3, use_container_width=True)
            else:
                st.error("CALCULATION FAULT: Solution mathematically invalid. Restrict targets.")

# ----------------- MAIN EXECUTION LOOP -----------------
def main():
    inject_custom_css()
    render_sidebar()
    render_hero()

    with st.spinner("Connecting to Orbital Intelligence Network..."):
        df = load_data()

    if df.empty or "risk_tier" not in df.columns:
        st.error("DATA LINK SEVERED.")
        st.info("System mandates valid temporal signature database (`debris_snapshot.csv`). Restrict upload protocols.")
    else:
        render_kpi_row(df)
        
        tab1, tab2, tab3 = st.tabs(["Live Debris Globe", "Risk Catalog Database", "QML Optimizer Engine"])
        
        with tab1: render_globe_view(df)
        with tab2: render_catalog_view(df)
        with tab3: render_optimizer_view(df)

if __name__ == "__main__":
    main()
