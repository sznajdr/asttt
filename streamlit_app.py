# =============================================================================
# GOALSCORER ODDS - PROFESSIONAL EDITION
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path

st.set_page_config(page_title=" ", page_icon=" ", layout="wide", initial_sidebar_state="collapsed")

# Professional CSS
st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    .block-container { padding: 1rem 2rem; }
    .main-header { font-size: 1.4rem; font-weight: 600; color: #e0e0e0; margin-bottom: 0.5rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 0; background-color: #1a1d24; border-radius: 4px; padding: 2px; }
    .stTabs [data-baseweb="tab"] { padding: 8px 20px; background-color: transparent; border-radius: 3px; color: #909090; font-size: 0.85rem; }
    .stTabs [aria-selected="true"] { background-color: #2a2d34; color: #e0e0e0; }
    .stButton > button { background-color: #2a2d34; color: #e0e0e0; border: none; font-size: 0.8rem; padding: 0.4rem 1rem; }
    .stButton > button:hover { background-color: #3a3d44; }
    .stButton > button[kind="primary"] { background-color: #1e3a5f; }
    .stCheckbox label, .stRadio label { font-size: 0.8rem; color: #a0a0a0; }
    [data-testid="metric-container"] { background-color: #1a1d24; padding: 0.5rem 1rem; border-radius: 4px; }
    [data-testid="metric-container"] label { color: #606060 !important; font-size: 0.7rem !important; }
    [data-testid="metric-container"] [data-testid="stMetricValue"] { color: #e0e0e0 !important; font-size: 1.1rem !important; }
    #MainMenu, footer, header { visibility: hidden; }
    .section-divider { border-top: 1px solid #2a2d34; margin: 1rem 0; }
</style>
""", unsafe_allow_html=True)

# Constants
POS_GOAL_WEIGHT = {'striker': 1.0, 'centerforward': 1.0, 'centerattackingmidfielder': 0.73, 'leftwinger': 0.68, 'rightwinger': 0.67, 'right_wing_back': 0.47, 'left_wing_back': 0.09, 'centermidfielder': 0.33, 'centerdefensivemidfielder': 0.26, 'rightmidfielder': 0.23, 'leftmidfielder': 0.22, 'leftback': 0.21, 'rightback': 0.20, 'centerback': 0.21, 'keeper': 0.0, 'keeper_long': 0.0}
POS_ASSIST_WEIGHT = {'leftwinger': 1.0, 'right_wing_back': 0.65, 'rightwinger': 0.93, 'centerattackingmidfielder': 0.85, 'left_wing_back': 0.65, 'centermidfielder': 0.7, 'rightback': 0.5, 'leftback': 0.5, 'centerdefensivemidfielder': 0.52, 'leftmidfielder': 0.55, 'striker': 0.81, 'centerforward': 0.81, 'rightmidfielder': 0.55, 'centerback': 0.24, 'keeper': 0.02, 'keeper_long': 0.02}
POS_ABBREV = {'striker': 'ST', 'centerforward': 'CF', 'leftwinger': 'LW', 'rightwinger': 'RW', 'centerattackingmidfielder': 'CAM', 'leftmidfielder': 'LM', 'rightmidfielder': 'RM', 'centermidfielder': 'CM', 'centerdefensivemidfielder': 'CDM', 'leftback': 'LB', 'rightback': 'RB', 'left_wing_back': 'LWB', 'right_wing_back': 'RWB', 'centerback': 'CB', 'keeper': 'GK', 'keeper_long': 'GK', 'goalkeeper': 'GK'}
POS_BOUNDS = {'goal': {'striker': (1.5, 50), 'centerforward': (1.5, 50), 'leftwinger': (2, 80), 'rightwinger': (2, 80), 'centerattackingmidfielder': (2.5, 100), 'centermidfielder': (5, 150), 'centerdefensivemidfielder': (8, 200), 'leftback': (10, 200), 'rightback': (10, 200), 'centerback': (10, 200), 'left_wing_back': (6, 150), 'right_wing_back': (6, 150), 'keeper': (100, 500), 'keeper_long': (100, 500)}, 'assist': {'striker': (3, 80), 'centerforward': (3, 80), 'leftwinger': (2, 50), 'rightwinger': (2, 50), 'centerattackingmidfielder': (2, 50), 'centermidfielder': (3, 80), 'centerdefensivemidfielder': (5, 150), 'leftback': (5, 150), 'rightback': (5, 150), 'centerback': (15, 300), 'left_wing_back': (3, 100), 'right_wing_back': (3, 100), 'keeper': (150, 500), 'keeper_long': (150, 500)}}

class CalibratedModel:
    def __init__(self, model, calibrator):
        self.model, self.calibrator = model, calibrator
    def predict_proba(self, X):
        raw = self.model.predict_proba(X)[:, 1]
        cal = self.calibrator.predict(raw)
        return np.column_stack([1 - cal, cal])

@st.cache_resource
def load_models():
    d = Path("model_artifacts")
    gb = joblib.load(d / "goal_model_bundle.pkl")
    ab = joblib.load(d / "assist_model_bundle.pkl")
    pp = pd.read_pickle(d / "player_profile.pkl")
    with open(d / "model_metadata.json") as f: meta = json.load(f)
    tags = {}
    if (d / "player_tags.json").exists():
        with open(d / "player_tags.json") as f: tags = {int(k): v for k, v in json.load(f).items()}
    return CalibratedModel(gb['model'], gb['calibrator']), CalibratedModel(ab['model'], ab['calibrator']), gb['features'], ab['features'], pp, meta, tags

@st.cache_data
def load_lineups():
    p = Path("model_artifacts/recent_lineups.csv")
    if p.exists():
        df = pd.read_csv(p)
        if 'team_name' in df.columns: df = df.rename(columns={'team_name': 'team'})
        return df
    return None

def get_pos_abbrev(pos): return POS_ABBREV.get(str(pos).lower().strip(), 'MF')
def prob_to_odds(p): return 1/p if p > 0 else 500.0
def apply_position_bounds(odds, pos, mkt='goal'):
    b = POS_BOUNDS.get(mkt, {}).get(str(pos).lower().strip(), (1.5, 200))
    return max(b[0], min(b[1], odds))

def fuzzy_find_team(q, teams):
    if not isinstance(q, str): return None
    ql = q.lower().strip()
    
    # 1. Manual Overrides for common short names
    overrides = {
        "inter": "Inter",
        "milan": "Milan",
        "bayern": "Bayern München",
        "dortmund": "Borussia Dortmund",
        "city": "Man City",
        "united": "Man Utd",
        "bologna": "Bologna"
    }
    if ql in overrides:
        target = overrides[ql].lower()
        for t in teams:
            if str(t).lower() == target: return t

    # 2. Exact Match (The most important check)
    for t in teams:
        if str(t).lower() == ql: return t
            
    # 3. Starts-With (Shortest string wins)
    starts = [t for t in teams if str(t).lower().startswith(ql)]
    if starts: return min(starts, key=len)
        
    # 4. Last Resort: Substring (Only if query is long enough to be specific)
    if len(ql) > 4:
        subs = [t for t in teams if ql in str(t).lower()]
        if subs: return min(subs, key=len)

    return None

def format_tags(tags, n=4):
    if not tags: return ""
    pri = ['PK', 'FK_TAKER', 'FORM_HOT', 'FORM_COLD', 'ATG_UNDERP', 'ATG_OVERP', 'STARTER', 'POACHER', 'PLAYMAKER']
    return ", ".join(sorted(tags, key=lambda x: pri.index(x) if x in pri else 99)[:n])

def apply_market_scaling(df, xg):
    if df is None or len(df) == 0: return df
    df = df.copy()
    gp = df['goal_prob'] / 100
    gl = -np.log(1 - gp.clip(upper=0.99))
    ig = gl.sum()
    if ig > 0 and xg > ig:
        w = (df['xg_per90'].fillna(0) + 0.05); w = w / w.sum()
        gl = gl * (xg / ig) * (1 + (w * len(df) - 1) * 0.2)
        df['goal_prob'] = (1 - np.exp(-gl)) * 100
    ap = df['assist_prob'] / 100
    al = -np.log(1 - ap.clip(upper=0.99))
    ia = al.sum(); ta = xg * 0.88
    if ia > 0 and ta > ia:
        w = (df['xa_per90'].fillna(0) + 0.02); w = w / w.sum()
        al = al * (ta / ia) * (1 + (w * len(df) - 1) * 0.3)
        df['assist_prob'] = (1 - np.exp(-al)) * 100
    df['goal_odds'] = df.apply(lambda x: apply_position_bounds(prob_to_odds(x['goal_prob']/100), x['position'], 'goal'), axis=1)
    df['assist_odds'] = df.apply(lambda x: apply_position_bounds(prob_to_odds(x['assist_prob']/100), x['position'], 'assist'), axis=1)
    return df

def get_team_odds(team_name, xg, gm, am, gf, af, pp, tags, min_min=50, lineups=None):
    teams = pp['pf_team'].dropna().unique()
    matched = fuzzy_find_team(team_name, teams)
    
    if not matched: 
        return None, f"Team '{team_name}' not found"
    
    tp = pp[pp['pf_team'] == matched].copy()
    if 'sm_total_minutes' in tp.columns:
        tp = tp[tp['sm_total_minutes'] > 0]
    
    if len(tp) == 0: return None, f"No active players for {matched}"
    
    ls = {}
    if lineups is not None and not lineups.empty:
        tl = lineups[lineups['team'] == matched]
        if tl.empty:
            tl = lineups[lineups['team'].str.contains(matched, case=False, na=False)]
            
        if not tl.empty:
            rm = tl['match_id'].unique()[-10:]
            rec = tl[tl['match_id'].isin(rm)]
            for pid in tp['player_id'].unique():
                pr = rec[rec['player_id'] == pid]
                if len(pr) > 0:
                    lm = pr.sort_values('match_id').iloc[-1]
                    ls[pid] = {
                        'starts': int(pr['is_starter'].sum()), 
                        'apps': len(pr), 
                        'started_last': bool(lm['is_starter'])
                    }
    
    results = []
    for _, p in tp.iterrows():
        pos = p.get('pf_position', 'midfielder')
        if str(pos) in ['0', 'nan', '', 'None', '0.0']: continue
        
        gr = {f: p.get(f, 0) if f in p.index else 0 for f in gf}
        gr['match_is_starter'] = 1; gr['match_pos_goal_weight'] = POS_GOAL_WEIGHT.get(str(pos).lower(), 0.3); gr['pos_goal_weight'] = gr['match_pos_goal_weight']
        ar = {f: p.get(f, 0) if f in p.index else 0 for f in af}
        ar['match_is_starter'] = 1; ar['match_pos_assist_weight'] = POS_ASSIST_WEIGHT.get(str(pos).lower(), 0.3); ar['pos_assist_weight'] = ar['match_pos_assist_weight']
        
        try:
            Xg, Xa = pd.DataFrame([gr])[gf].fillna(0), pd.DataFrame([ar])[af].fillna(0)
            gp, ap = gm.predict_proba(Xg)[0,1], am.predict_proba(Xa)[0,1]
            go, ao = apply_position_bounds(prob_to_odds(gp), pos, 'goal'), apply_position_bounds(prob_to_odds(ap), pos, 'assist')
            
            pid = p.get('player_id', 0)
            l = ls.get(pid, {'starts': 0, 'apps': 0, 'started_last': False})
            
            # Combine starts and apps into the 'st/app' column the Slate expects
            star_icon = "*" if l['started_last'] else ""
            st_app_str = f"{l['starts']}{star_icon}/{l['apps']}"

            results.append({
                'player_id': pid, 'name': p.get('pf_name', '?'), 'position': pos, 'pos': get_pos_abbrev(pos),
                'goal_prob': round(gp*100, 2), 'goal_odds': round(go, 2), 'assist_prob': round(ap*100, 2), 'assist_odds': round(ao, 2),
                'xg_per90': round(p.get('sm_xg_per90', 0) or 0, 2), 'xa_per90': round(p.get('sm_xa_per90', 0) or 0, 2),
                'st/app': st_app_str, 'started_last': l['started_last'], 'tags': tags.get(int(pid), [])
            })
        except: continue

    if not results: return None, f"No valid players for {matched}"
    df = pd.DataFrame(results)
    df = apply_market_scaling(df, xg)
    
    # Calculate Value Gap for the Slate logic
    df['rank_model'] = df['goal_odds'].rank()
    pr_rank = {'striker': 1, 'centerforward': 1, 'leftwinger': 2, 'rightwinger': 2, 'centerattackingmidfielder': 3}
    df['rank_pos'] = df['position'].apply(lambda x: pr_rank.get(str(x).lower().strip(), 8))
    df['value_gap'] = df['rank_pos'] - df['rank_model']
    
    return df.sort_values('goal_odds'), matched

# =============================================================================
# TABS
# =============================================================================
def render_slate(gm, am, gf, af, pp, tags, lineups):
    default = "Bologna, Inter, 1.55, 1.75"
    c1, c2 = st.columns([3, 1])
    with c1: 
        slate_input = st.text_area("Matches", value=default, height=100, label_visibility="collapsed")
    with c2:
        fname = st.text_input("Filename", value="slate_export")
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    cols = st.columns([1, 1.5, 1, 1, 1])
    with cols[0]: mkt = st.radio("Mkt", ["Goal", "Assist"], horizontal=True)
    with cols[1]: maxo = st.slider("Max Odds", 1.5, 50.0, 15.0)
    with cols[2]: pk_only = st.checkbox("PK Only")
    with cols[3]: val_only = st.checkbox("Value ≥2")
    with cols[4]: scan_btn = st.button("SCAN SLATE", type="primary", use_container_width=True)

    if scan_btn:
        all_results = []
        lines = [l.strip() for l in slate_input.strip().split('\n') if l.strip()]
        
        for line in lines:
            parts = [p.strip() for p in line.split(',')]
            if len(parts) < 4: continue
            
            try:
                h_team, a_team, h_xg, a_xg = parts[0], parts[1], float(parts[2]), float(parts[3])
            except ValueError:
                continue
            
            # Process Home
            h_df, h_name = get_team_odds(h_team, h_xg, gm, am, gf, af, pp, tags, lineups=lineups)
            if h_df is not None:
                h_df['matchup'] = f"{h_name} vs {a_team}"
                h_df['team_xg'] = h_xg
                all_results.append(h_df)
                
            # Process Away
            a_df, a_name = get_team_odds(a_team, a_xg, gm, am, gf, af, pp, tags, lineups=lineups)
            if a_df is not None:
                a_df['matchup'] = f"{h_team} vs {a_name}"
                a_df['team_xg'] = a_xg
                all_results.append(a_df)

        if all_results:
            master_df = pd.concat(all_results, ignore_index=True)
            target_col = 'goal_odds' if mkt == "Goal" else 'assist_odds'
            
            # Apply Filters
            filt = master_df[master_df[target_col] <= maxo].copy()
            if pk_only:
                filt = filt[filt['tags'].apply(lambda x: 'PK' in x)]
            if val_only:
                filt = filt[filt['value_gap'] >= 2]
                
            filt = filt.sort_values(target_col)
            
            if not filt.empty:
                st.write(f"Found {len(filt)} plays")
                # Format Tags for display
                filt['Tags'] = filt['tags'].apply(lambda x: ", ".join(x[:3]))
                
                display_cols = ['matchup', 'name', 'pos', 'team_xg', 'st/app', 'xg_per90', 'xa_per90', target_col, 'Tags']
                st.dataframe(
                    filt[display_cols].rename(columns={
                        'matchup': 'Match', 
                        'name': 'Player', 
                        'team_xg': 'txG', 
                        target_col: 'Odds'
                    }),
                    hide_index=True, use_container_width=True
                )
            else:
                st.warning("No plays found with the current filters.")
        else:
            st.error("No valid teams found. Please check your spelling.")

def render_h2h(gm, am, gf, af, pp, tags, teams, lineups):
    c1, c2, c3, c4 = st.columns([2, 1, 2, 1])
    with c1: ht = st.selectbox("Home", teams, label_visibility="collapsed")
    with c2: hx = st.number_input("xG", 0.5, 4.0, 1.85, 0.05, key="hx", label_visibility="collapsed")
    with c3: at = st.selectbox("Away", teams, index=min(1, len(teams)-1), label_visibility="collapsed")
    with c4: ax = st.number_input("xG", 0.5, 4.0, 1.45, 0.05, key="ax", label_visibility="collapsed")
    if st.button("Generate", type="primary", key="h2h"):
        hdf, hm = get_team_odds(ht, hx, gm, am, gf, af, pp, tags, lineups=lineups)
        adf, am_ = get_team_odds(at, ax, gm, am, gf, af, pp, tags, lineups=lineups)
        if hdf is None or adf is None: st.error("Could not load teams"); return
        st.markdown(f'<p style="color:#e0e0e0;font-size:1rem;margin:1rem 0;">{hm} ({hx}) vs {am_} ({ax})</p>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f'<p style="color:#808080;font-size:0.8rem;">GOAL - {hm}</p>', unsafe_allow_html=True)
            st.dataframe(hdf.nsmallest(10, 'goal_odds')[['name', 'pos', 'goal_odds', 'xg_per90']].rename(columns={'name': 'Player', 'pos': 'Pos', 'goal_odds': 'Odds', 'xg_per90': 'xG90'}), hide_index=True, use_container_width=True, height=300)
        with c2:
            st.markdown(f'<p style="color:#808080;font-size:0.8rem;">GOAL - {am_}</p>', unsafe_allow_html=True)
            st.dataframe(adf.nsmallest(10, 'goal_odds')[['name', 'pos', 'goal_odds', 'xg_per90']].rename(columns={'name': 'Player', 'pos': 'Pos', 'goal_odds': 'Odds', 'xg_per90': 'xG90'}), hide_index=True, use_container_width=True, height=300)
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f'<p style="color:#808080;font-size:0.8rem;">ASSIST - {hm}</p>', unsafe_allow_html=True)
            st.dataframe(hdf.nsmallest(10, 'assist_odds')[['name', 'pos', 'assist_odds', 'xa_per90']].rename(columns={'name': 'Player', 'pos': 'Pos', 'assist_odds': 'Odds', 'xa_per90': 'xA90'}), hide_index=True, use_container_width=True, height=300)
        with c2:
            st.markdown(f'<p style="color:#808080;font-size:0.8rem;">ASSIST - {am_}</p>', unsafe_allow_html=True)
            st.dataframe(adf.nsmallest(10, 'assist_odds')[['name', 'pos', 'assist_odds', 'xa_per90']].rename(columns={'name': 'Player', 'pos': 'Pos', 'assist_odds': 'Odds', 'xa_per90': 'xA90'}), hide_index=True, use_container_width=True, height=300)

def render_team(gm, am, gf, af, pp, tags, teams, lineups):
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1: 
        # Use a selectbox or a better filtered search
        tm = st.selectbox("Team", teams, index=teams.index("Inter") if "Inter" in teams else 0)
    with c2: 
        xg = st.number_input("xG", 0.5, 4.5, 1.85, 0.05)
    with c3: 
        gen = st.button("Generate", type="primary")
    
    if gen:
        # The fuzzy_find_team inside get_team_odds will now handle 'Inter' correctly
        df, matched_name = get_team_odds(tm, xg, gm, am, gf, af, pp, tags, min_min=100, lineups=lineups)
        if df is None: st.error("Team not found"); return
        df = df[df['pos'] != 'GK']
        st.markdown(f'<p style="color:#808080;font-size:0.8rem;">{m} | xG: {xg}</p>', unsafe_allow_html=True)
        rows = [{'Player': r['name'][:18], 'Pos': r['pos'], 'st/app': f"{int(r['starts'])}{'*' if r['started_last'] else ''}/{int(r['apps'])}", 'Goal': r['goal_odds'], 'Assist': r['assist_odds'], 'xG90': r['xg_per90'], 'xA90': r['xa_per90'], 'Tags': format_tags(r['tags'], 3)} for _, r in df.nsmallest(20, 'goal_odds').iterrows()]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True, height=500)

def render_player(gm, am, gf, af, pp, tags, lineups):
    # Sort players for the selection box
    ap = sorted(pp['pf_name'].dropna().unique().tolist())
    
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1: 
        ps = st.selectbox("Select Player", ap, label_visibility="collapsed")
    with c2: 
        bxg = st.number_input("Team xG Baseline", 0.5, 4.0, 2.0, 0.1, label_visibility="collapsed")
    with c3: 
        an = st.button("Generate Deep Dive", type="primary", use_container_width=True)

    if ps:
        # 1. RETRIEVE PLAYER DATA
        pd_ = pp[pp['pf_name'] == ps]
        if len(pd_) == 0: return
        p = pd_.iloc[0]
        pid = int(p.get('player_id', 0))
        tm = p.get('pf_team', '?')
        pos = p.get('pf_position', '?')
        p_tags = tags.get(pid, [])
        
        # 2. HEADER SECTION
        st.markdown(f"### 🔍 {ps.upper()}")
        tag_html = " ".join([f'<span style="background-color: #1e3a5f; color: #e0e0e0; padding: 2px 8px; border-radius: 10px; font-size: 0.7rem; margin-right: 5px;">{t}</span>' for t in p_tags])
        st.markdown(tag_html, unsafe_allow_html=True)
        st.markdown(f'<p style="color:#808080; font-size:0.9rem;">{tm} | {pos}</p>', unsafe_allow_html=True)

        # 3. MODEL PREDICTIONS (Odds)
        df, _ = get_team_odds(tm, bxg, gm, am, gf, af, pp, tags, min_min=0, lineups=lineups)
        if df is not None:
            po = df[df['player_id'] == pid]
            if len(po) > 0:
                o = po.iloc[0]
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Goal Odds", f"{o['goal_odds']:.2f}", help="Scaled to Team xG")
                m2.metric("Goal %", f"{o['goal_prob']:.1f}%")
                m3.metric("Assist Odds", f"{o['assist_odds']:.2f}")
                m4.metric("Assist %", f"{o['assist_prob']:.1f}%")

        # 4. INTELLIGENCE REPORT (Bull/Bear/Strategy)
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        # Logic for Strategy
        xg90 = p.get('sm_xg_per90', 0) or 0
        xa90 = p.get('sm_xa_per90', 0) or 0
        shots90 = p.get('sm_shots_per90', 0) or 0
        goals = p.get('goals', 0) or 0
        xg_total = p.get('expected_goals_xg', 0) or 0
        xg_diff = goals - xg_total

        reasons_pos, reasons_neg = [], []
        if xg90 >= 0.4: reasons_pos.append(f"Elite Goal Threat ({xg90:.2f} xG/90)")
        if xa90 >= 0.25: reasons_pos.append(f"Elite Playmaker ({xa90:.2f} xA/90)")
        if 'PK' in p_tags: reasons_pos.append("Primary Penalty Taker")
        if xg_diff <= -2.0: reasons_pos.append(f"Positive Regression Candidate (Underperforming xG by {abs(xg_diff):.1f})")
        
        if xg90 < 0.1: reasons_neg.append("Low Goal Threat")
        if xg_diff >= 3.0: reasons_neg.append(f"Finishing overperformance (Unsustainable +{xg_diff:.1f})")
        if 'LATE_SUB' in p_tags: reasons_neg.append("High rotation/substitute risk")

        # Strategy Definition
        if 'PK' in p_tags and xg90 > 0.3: strategy = "💎 PRIME GOALSCORER (Volume + PKs)"
        elif xa90 > 0.20 and xg90 < 0.15: strategy = "🎯 ASSIST SPECIALIST (Fade goals, target assists)"
        elif shots90 > 3.0: strategy = "🔫 VOLUME SHOOTER (Look for Shots/SOT props)"
        elif xg_diff <= -2.5: strategy = "🍀 DUE FOR A GOAL (Positive Regression)"
        else: strategy = "📊 STANDARD USAGE (Wait for lineup confirmation)"

        st.markdown(f"**💡 STRATEGY:** {strategy}")
        col_bull, col_bear = st.columns(2)
        with col_bull:
            st.markdown('<p style="color:#4CAF50; font-size:0.8rem; font-weight:bold;">✅ BULL CASE</p>', unsafe_allow_html=True)
            for r in reasons_pos: st.markdown(f"- <span style='font-size:0.8rem;'>{r}</span>", unsafe_allow_html=True)
        with col_bear:
            st.markdown('<p style="color:#f44336; font-size:0.8rem; font-weight:bold;">⚠️ BEAR CASE</p>', unsafe_allow_html=True)
            for r in reasons_neg: st.markdown(f"- <span style='font-size:0.8rem;'>{r}</span>", unsafe_allow_html=True)

        # 5. CORE PERFORMANCE METRICS
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown('<p style="color:#606060;font-size:0.75rem;">SHOOTING</p>', unsafe_allow_html=True)
            st.markdown(f"**Shots/90:** {shots90:.2f}<br>**xG/90:** {xg90:.2f}<br>**Box Touches/90:** {(p.get('sm_touches_in_box_per90',0)):.1f}", unsafe_allow_html=True)
        with c2:
            st.markdown('<p style="color:#606060;font-size:0.75rem;">CREATION</p>', unsafe_allow_html=True)
            st.markdown(f"**xA/90:** {xa90:.2f}<br>**Chances Created:** {int(p.get('chances_created',0))}<br>**Conversion:** {p.get('sm_assist_conversion',0)*100:.1f}%", unsafe_allow_html=True)
        with c3:
            st.markdown('<p style="color:#606060;font-size:0.75rem;">CONTEXT</p>', unsafe_allow_html=True)
            st.markdown(f"**In Box %:** {p.get('sp_inside_box_pct',0)*100:.0f}%<br>**Header %:** {p.get('sp_header_pct',0)*100:.0f}%<br>**Accuracy:** {p.get('sm_shot_accuracy',0)*100:.0f}%", unsafe_allow_html=True)

        # 6. MATCH LOG (Last 10)
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown('<p style="color:#606060;font-size:0.75rem;">RECENT MATCH LOG (Form Data)</p>', unsafe_allow_html=True)
        
        if lineups is not None and not lineups.empty:
            p_log = lineups[lineups['player_id'] == pid].copy()
            if not p_log.empty:
                # Add a "Start" icon for visibility
                p_log['St'] = p_log['is_starter'].apply(lambda x: '✅' if x else '🔄')
                # Sort by match_id or date if available
                log_disp = p_log.tail(10).sort_values('match_id', ascending=False)[['match_id', 'St', 'minutes_played', 'goals_in_match', 'assists_in_match', 'rating']]
                log_disp.columns = ['Match', 'St', 'Min', 'G', 'A', 'Rating']
                st.dataframe(log_disp, hide_index=True, use_container_width=True)
            else:
                st.info("No recent match logs found for this player.")




# =============================================================================
# UPDATED LINEUP INTEL SECTION
# =============================================================================

def render_lineups(lineups, tags_map):
    if lineups is None or lineups.empty:
        st.markdown('<p style="color:#606060;">Lineup data not loaded.</p>', unsafe_allow_html=True)
        return

    # Filter out Frauen-Bundesliga if present as per your notebook logic
    lineups = lineups[lineups['competition'] != 'Frauen-Bundesliga']
    
    lt = sorted(lineups['team'].dropna().unique().tolist())
    
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1: 
        tm = st.selectbox("Select Team", lt, label_visibility="collapsed", key="lt_select")
    with c2: 
        nr = st.number_input("Last N Matches", 3, 15, 10, 1, label_visibility="collapsed")
    with c3: 
        btn = st.button("Get Deep Intel", type="primary", use_container_width=True)

    if btn:
        # Match team name
        matched = [t for t in lineups['team'].unique() if tm.lower() in t.lower()]
        if not matched:
            st.error("Team not found")
            return
        
        tn = matched[0]
        td = lineups[lineups['team'] == tn].copy()
        
        # Get recent matches
        rm_ids = td['match_id'].unique()[-nr:]
        rec = td[td['match_id'].isin(rm_ids)]
        
        st.markdown(f"### 🏟️ {tn.upper()} INTEL (Last {len(rm_ids)} Games)")

        # --- ROW 1: FORMATIONS & TACTICAL ---
        col_form, col_tact = st.columns([1, 1])
        
        with col_form:
            st.markdown('<p style="color:#606060;font-size:0.75rem;font-weight:bold;">FORMATIONS USED</p>', unsafe_allow_html=True)
            starters_only = rec[rec['is_starter'] == True]
            forms = starters_only.groupby(['match_id', 'formation']).size().reset_index().groupby('formation').size().reset_index(name='Games')
            forms['%'] = (forms['Games'] / len(rm_ids) * 100).astype(int)
            st.dataframe(forms.sort_values('Games', ascending=False), hide_index=True, use_container_width=True)

        with col_tact:
            st.markdown('<p style="color:#606060;font-size:0.75rem;font-weight:bold;">TACTICAL NOTES</p>', unsafe_allow_html=True)
            
            # Extract tactical info from tags and match data
            active_pids = rec['player_id'].unique()
            pks, corners, hooks, weapons = [], [], [], []
            
            for pid in active_pids:
                p_tags = tags_map.get(int(pid), [])
                p_name = rec[rec['player_id'] == pid]['player_name'].iloc[0]
                
                if 'PK' in p_tags: pks.append(p_name)
                if 'FK_TAKER' in p_tags or 'CROSSER' in p_tags: corners.append(p_name)
                
                # Hook Risk logic (Starts often but gets subbed early)
                p_starts = rec[(rec['player_id'] == pid) & (rec['is_starter'] == True)]
                if len(p_starts) >= 3:
                    avg_min = p_starts['minutes_played'].mean()
                    if avg_min < 75: hooks.append(f"{p_name} ({int(avg_min)}')")
                
                # Bench Weapon logic
                p_subs = rec[(rec['player_id'] == pid) & (rec['is_starter'] == False)]
                if p_subs['goals_in_match'].sum() > 0:
                    weapons.append(f"{p_name} ({int(p_subs['goals_in_match'].sum())}G)")

            st.markdown(f"**🎯 Penalties:** {', '.join(pks) if pks else 'Unknown'}")
            st.markdown(f"**🚩 Set Pieces:** {', '.join(corners[:3]) if corners else 'Unknown'}")
            if hooks: st.markdown(f"**📉 Hook Risk:** {', '.join(hooks[:3])}")
            if weapons: st.markdown(f"**🚀 Bench Weapons:** {', '.join(weapons[:3])}")

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        # --- ROW 2: SQUAD DEPTH (The Notebook Table) ---
        st.markdown('<p style="color:#606060;font-size:0.75rem;font-weight:bold;">SQUAD DEPTH & POSITION ROLES</p>', unsafe_allow_html=True)
        
        stats = rec.groupby(['player_id', 'player_name', 'position_name']).agg({
            'is_starter': 'sum',
            'minutes_played': 'mean',
            'rating': 'mean',
            'goals_in_match': 'sum',
            'assists_in_match': 'sum'
        }).reset_index()

        depth_rows = []
        for _, row in stats.iterrows():
            start_pct = (row['is_starter'] / len(rm_ids)) * 100
            
            # Status Emojis
            if start_pct >= 90: status = "🔒"
            elif start_pct >= 50: status = "✅"
            else: status = "⚠️"
            
            # Formatting tags for the "Intel" column
            p_tags = tags_map.get(int(row['player_id']), [])
            intel_tags = [t for t in p_tags if t not in ['RIGHT_FOOT', 'LEFT_FOOT', 'BOTH_FEET']]
            
            depth_rows.append({
                'St': status,
                'Name': row['player_name'],
                'Pos': row['position_name'],
                'Start%': f"{int(start_pct)}%",
                'AvgMin': f"{int(row['minutes_played'])}'",
                'G/A': f"{int(row['goals_in_match'])}/{int(row['assists_in_match'])}",
                'Rating': round(row['rating'], 2),
                'Intel': " ".join([f"[{t}]" for t in intel_tags[:2]])
            })
        
        depth_df = pd.DataFrame(depth_rows).sort_values(['Pos', 'Start%'], ascending=[True, False])
        st.dataframe(depth_df, hide_index=True, use_container_width=True, height=400)

        # --- ROW 3: PREDICTED XI ---
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown('<p style="color:#606060;font-size:0.75rem;font-weight:bold;">PREDICTED STARTING XI (Confidence Based)</p>', unsafe_allow_html=True)
        
        xi_cols = st.columns(4)
        positions = [('Goalkeeper', 'GK'), ('Defender', 'DEF'), ('Midfielder', 'MID'), ('Forward', 'FWD')]
        
        for i, (full_pos, short_pos) in enumerate(positions):
            with xi_cols[i]:
                pos_rec = starters_only[starters_only['position_name'] == full_pos]
                if not pos_rec.empty:
                    # Determine how many slots this team usually plays for this position
                    slots = int(pos_rec.groupby('match_id').size().median())
                    candidates = pos_rec.groupby('player_name').size().reset_index(name='starts').sort_values('starts', ascending=False)
                    
                    st.markdown(f"**{short_pos}**")
                    for _, cand in candidates.head(slots).iterrows():
                        conf = int((cand['starts'] / len(rm_ids)) * 100)
                        bar = "🟩" if conf >= 80 else "🟧" if conf >= 50 else "⬜"
                        st.markdown(f"{bar} {cand['player_name']} <br><small>{conf}% confidence</small>", unsafe_allow_html=True)

def main():
    try: 
        gm, am, gf, af, pp, meta, tags = load_models()
    except Exception as e: 
        st.error(f"Failed to load: {e}")
        return
        
    lineups = load_lineups()
    teams = sorted([t for t in pp['pf_team'].dropna().unique() if isinstance(t, str)])
    
    c1, c2 = st.columns([3, 1])
    with c1: 
        st.markdown('<p class="main-header">Goalscorer Odds</p>', unsafe_allow_html=True)
    with c2: 
        st.markdown(f'<p style="color:#505050;font-size:0.75rem;text-align:right;margin-top:0.5rem;">AUC: {meta["goal_model"]["test_auc"]:.3f} / {meta["assist_model"]["test_auc"]:.3f}</p>', unsafe_allow_html=True)
    
    t1, t2, t3, t4, t5 = st.tabs(["Slate", "H2H", "Team", "Player", "Lineups"])
    
    with t1: 
        render_slate(gm, am, gf, af, pp, tags, lineups)
    with t2: 
        render_h2h(gm, am, gf, af, pp, tags, teams, lineups)
    with t3: 
        render_team(gm, am, gf, af, pp, tags, teams, lineups)
    with t4: 
        render_player(gm, am, gf, af, pp, tags, lineups)
    with t5: 
        render_lineups(lineups, tags)

if __name__ == "__main__": 
    main()
