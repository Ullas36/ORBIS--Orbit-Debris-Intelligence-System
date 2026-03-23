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

st.set_page_config(page_title="ORBIS", page_icon="🛸", layout="wide")

EARTH_R = 6371.0
GM = 398600.4418

def hohmann_delta_v(r1, r2):
    v1=np.sqrt(GM/r1); v2=np.sqrt(GM/r2)
    a_t=(r1+r2)/2
    vt1=np.sqrt(GM*(2/r1-1/a_t)); vt2=np.sqrt(GM*(2/r2-1/a_t))
    return abs(vt1-v1)+abs(v2-vt2)

@st.cache_data(ttl=3600)
def load_data():
    # First try: load bundled snapshot (always works)
    snapshot_path = os.path.join(os.path.dirname(__file__), "debris_snapshot.csv")
    df_snapshot = pd.DataFrame()
    if os.path.exists(snapshot_path):
        try:
            df_snapshot = pd.read_csv(snapshot_path)
            st.sidebar.success(f"Snapshot loaded: {len(df_snapshot):,} objects")
        except: pass

    # Second try: fetch live data (may be blocked on cloud)
    BASE = "https://celestrak.org/NORAD/elements/gp.php"
    GROUPS = ["cosmos-deb","iridium-33-deb","fengyun-1c-deb","rocket-bodies","active"]
    all_tles = {}
    for group in GROUPS:
        try:
            r = requests.get(f"{BASE}?GROUP={group}&FORMAT=tle",
                timeout=15, headers={"User-Agent":"ORBIS/1.0"})
            if r.status_code != 200: continue
            lines = r.text.strip().splitlines()
            i = 0
            while i < len(lines)-2:
                name=lines[i].strip(); l1=lines[i+1].strip(); l2=lines[i+2].strip()
                if l1.startswith("1 ") and l2.startswith("2 "):
                    try:
                        nid=int(l1[2:7])
                        if nid not in all_tles:
                            all_tles[nid]=(name,l1,l2)
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
                jd,fr=jday(epoch.year,epoch.month,epoch.day,
                           epoch.hour,epoch.minute,epoch.second)
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
                    "name":name.strip(),"norad_id":nid,
                    "altitude_km":round(alt,1),"inclination":round(inc,2),
                    "x_eci_km":round(pos[0],1),"y_eci_km":round(pos[1],1),
                    "z_eci_km":round(pos[2],1),
                    "speed_kms":round(np.sqrt(sum(v**2 for v in vel)),3),
                    "object_type":otype,"dci_score":dci,"risk_tier":risk,
                    "mass_kg_est":mass,
                })
            except: continue
        if records:
            st.sidebar.info("Live TLE data loaded.")
            return pd.DataFrame(records)

    # Return snapshot if live fetch failed
    if not df_snapshot.empty:
        st.sidebar.warning("Using cached snapshot (live fetch unavailable).")
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
        for k in range(n):
            matrix[i][k]=sample.get(i*n+k,0)
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
    try:
        from dwave.samplers import SimulatedAnnealingSampler
    except:
        SimulatedAnnealingSampler=dimod.SimulatedAnnealingSampler
    sampler=SimulatedAnnealingSampler()
    result=sampler.sample(bqm,num_reads=500,num_sweeps=1000,seed=42)
    best_seq,best_cost=None,float("inf")
    for sample,energy in result.data(["sample","energy"]):
        seq=decode_sol(sample,n)
        if seq:
            c=seq_cost(seq,cm)
            if c<best_cost: best_cost,best_seq=c,seq
    return best_seq,best_cost

with st.sidebar:
    st.title("ORBIS")
    st.caption("Orbital Remediation with Quantum Intelligence Systems")
    st.divider()
    if st.button("Refresh TLE data", use_container_width=True):
        st.cache_data.clear(); st.rerun()
    st.divider()
    st.caption(f"Epoch: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    st.caption("Data: CelesTrak | Solver: D-Wave dimod")

st.title("ORBIS - Orbital Debris Intelligence System")

with st.spinner("Loading debris catalog..."):
    df = load_data()

if df.empty or "risk_tier" not in df.columns:
    st.error("No data available.")
    st.info("Please re-upload debris_snapshot.csv to the GitHub repo.")
else:
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Objects tracked",f"{len(df):,}")
    c2.metric("High risk",f"{len(df[df['risk_tier']=='High']):,}")
    c3.metric("Altitude range",f"{df['altitude_km'].min():.0f}-{df['altitude_km'].max():.0f} km")
    c4.metric("Avg DCI score",f"{df['dci_score'].mean():.3f}")
    st.divider()

    tab1,tab2,tab3=st.tabs(["Live Debris Globe","Risk Catalog","QML Optimizer"])

    with tab1:
        st.subheader("Live debris field")
        col1,col2=st.columns([2,1])
        with col1:
            risk_filter=st.multiselect("Risk tier",["High","Medium","Low"],default=["High","Medium","Low"])
        with col2:
            max_pts=st.slider("Max points",500,5000,2000,step=500)
        fdf=df[df["risk_tier"].isin(risk_filter)]
        plot_df=fdf.sample(min(max_pts,len(fdf)),random_state=42) if len(fdf)>0 else fdf
        color_map={"High":"#E24B4A","Medium":"#EF9F27","Low":"#639922"}
        fig=go.Figure()
        for tier,color in color_map.items():
            sub=plot_df[plot_df["risk_tier"]==tier]
            if len(sub)==0: continue
            fig.add_trace(go.Scatter3d(
                x=sub["x_eci_km"],y=sub["y_eci_km"],z=sub["z_eci_km"],
                mode="markers",name=f"{tier} ({len(sub):,})",
                marker=dict(size=2.5,color=color,opacity=0.7),
                text=sub["name"]+"<br>Alt: "+sub["altitude_km"].round(0).astype(str)+" km",
                hovertemplate="%{text}<extra></extra>"
            ))
        u_=np.linspace(0,2*np.pi,30); v_=np.linspace(0,np.pi,15)
        fig.add_trace(go.Surface(
            x=EARTH_R*np.outer(np.cos(u_),np.sin(v_)),
            y=EARTH_R*np.outer(np.sin(u_),np.sin(v_)),
            z=EARTH_R*np.outer(np.ones(30),np.cos(v_)),
            colorscale=[[0,"#1a3a5c"],[1,"#2d6ea8"]],
            showscale=False,opacity=0.5,hoverinfo="skip"
        ))
        fig.update_layout(height=600,
            scene=dict(bgcolor="rgba(2,6,15,1)",xaxis_title="X (km)",
                       yaxis_title="Y (km)",zaxis_title="Z (km)"),
            paper_bgcolor="rgba(2,6,15,1)",font=dict(color="white"),
            legend=dict(bgcolor="rgba(0,0,0,0.5)"),margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig,use_container_width=True)

    with tab2:
        st.subheader("Debris risk catalog")
        c1,c2,c3=st.columns(3)
        with c1: tier_sel=st.selectbox("Risk tier",["All","High","Medium","Low"])
        with c2: type_sel=st.selectbox("Object type",["All","Rocket Body","Debris Fragment","Defunct Satellite","Active/Unknown"])
        with c3: alt_range=st.slider("Altitude (km)",100,40000,(100,2000))
        filtered=df.copy()
        if tier_sel!="All": filtered=filtered[filtered["risk_tier"]==tier_sel]
        if type_sel!="All": filtered=filtered[filtered["object_type"]==type_sel]
        filtered=filtered[(filtered["altitude_km"]>=alt_range[0])&(filtered["altitude_km"]<=alt_range[1])]
        st.dataframe(
            filtered[["name","norad_id","altitude_km","inclination","speed_kms",
                      "object_type","dci_score","risk_tier"]].sort_values("dci_score",ascending=False).head(200),
            use_container_width=True,height=400,
            column_config={
                "dci_score":st.column_config.ProgressColumn("DCI score",min_value=0,max_value=1),
                "altitude_km":st.column_config.NumberColumn("Altitude (km)",format="%.1f"),
            }
        )
        st.caption(f"Showing {min(200,len(filtered)):,} of {len(filtered):,} objects")
        fig2=px.histogram(df[df["altitude_km"]<3000],x="altitude_km",color="object_type",nbins=80,
            title="Debris density by altitude",
            color_discrete_map={"Rocket Body":"#E24B4A","Debris Fragment":"#EF9F27",
                                "Defunct Satellite":"#378ADD","Active/Unknown":"#888780"})
        fig2.add_vline(x=789,line_dash="dash",line_color="white",opacity=0.5,
                       annotation_text="Iridium-Cosmos 2009",annotation_font_color="white")
        fig2.update_layout(height=320,paper_bgcolor="rgba(2,6,15,1)",
                           plot_bgcolor="rgba(10,20,40,0.8)",font=dict(color="white"))
        st.plotly_chart(fig2,use_container_width=True)

    with tab3:
        st.subheader("QML debris removal optimizer")
        st.caption("Select targets -> build delta-V cost matrix -> solve with QUBO + simulated annealing")
        top_candidates=df.nlargest(50,"dci_score")["name"].tolist()
        selected=st.multiselect("Select debris targets (3-8 recommended)",
            options=top_candidates,default=top_candidates[:5],max_selections=10)
        if len(selected)<2:
            st.warning("Select at least 2 targets.")
        else:
            n_sel=len(selected)
            sel_df=df[df["name"].isin(selected)].drop_duplicates("name").head(n_sel)
            feats=[]
            for _,row in sel_df.iterrows():
                feats.append({"name":row["name"],"r":EARTH_R+row["altitude_km"],
                              "alt":row["altitude_km"],"inc":row["inclination"]})
            cm=np.zeros((n_sel,n_sel))
            for i in range(n_sel):
                for j in range(n_sel):
                    if i!=j:
                        dv=hohmann_delta_v(feats[i]["r"],feats[j]["r"])
                        inc_diff=abs(feats[i]["inc"]-feats[j]["inc"])
                        v_mid=np.sqrt(GM/((feats[i]["r"]+feats[j]["r"])/2))
                        dv_plane=2*v_mid*np.sin(np.radians(inc_diff/2))
                        cm[i][j]=round(dv+dv_plane*0.3,4)
            names_short=[f[:18] for f in selected[:n_sel]]
            st.write("**Delta-V cost matrix (km/s):**")
            st.dataframe(pd.DataFrame(cm,index=names_short,columns=names_short).round(3),
                        use_container_width=True)
            if st.button("Run QML optimizer",type="primary",use_container_width=True):
                with st.spinner("Solving QUBO..."):
                    t0=time.time()
                    seq,total_dv=sa_solve(cm)
                    elapsed=time.time()-t0
                if seq is not None:
                    st.success(f"Optimal sequence found in {elapsed:.2f}s")
                    st.metric("Total mission delta-V",f"{total_dv:.4f} km/s")
                    st.write("**Optimal removal sequence:**")
                    for step,sidx in enumerate(seq):
                        nm=feats[sidx]["name"] if sidx<len(feats) else f"T{sidx}"
                        alt=feats[sidx]["alt"]
                        dv_s=cm[seq[step-1]][sidx] if step>0 else 0
                        label=f", +{dv_s:.3f} km/s" if step>0 else " - start"
                        st.write(f"Step {step+1}: **{nm}** (alt: {alt:.0f} km{label})")
                    step_dvs=[0]+[cm[seq[i]][seq[i+1]] for i in range(len(seq)-1)]
                    fig3=px.bar(x=[f"Step {i+1}" for i in range(len(seq))],y=step_dvs,
                        title="Delta-V per mission step",labels={"x":"Step","y":"Delta-V (km/s)"},
                        color=step_dvs,color_continuous_scale=["#639922","#EF9F27","#E24B4A"])
                    fig3.update_layout(height=320,paper_bgcolor="rgba(2,6,15,1)",
                        plot_bgcolor="rgba(10,20,40,0.8)",font=dict(color="white"),
                        showlegend=False,coloraxis_showscale=False)
                    st.plotly_chart(fig3,use_container_width=True)
                else:
                    st.error("Invalid solution. Try fewer targets.")
