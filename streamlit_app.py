# =============================================================================
# GOALSCORER ODDS - PROFESSIONAL EDITION (FIXED SEARCH)
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path

st.set_page_config(page_title="Goalscorer Odds", page_icon="⚽", layout="wide", initial_sidebar_state="collapsed")

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
    """Refined search logic to prioritize Inter (Milan) and exact matches."""
    if not isinstance(q, str): return None
    ql = q.lower().strip()
    
    # 1. Exact Match (Prioritize this above all)
    for t in teams:
        if str(t).lower().strip() == ql: return t
            
    # 2. Specialized Case for "Inter"
    if ql == "inter":
        for t in teams:
            if str(t).lower() in ["inter", "internazionale", "inter milan"]: return t

    # 3. Starts With (Shortest string wins - helps distinguish Inter from Inter Miami)
    starts = [t for t in teams if str(t).lower().startswith(ql)]
    if starts: return min(starts, key=len)
        
    # 4. Contains
    contains = [t for t in teams if ql in str(t).lower()]
    if contains: return min(contains, key=len)

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
    if not matched: return None, f"Team '{team_name}' not found"
    
    # STRICT FILTER: Only take players who belong to the MATCHED team
    tp = pp[pp['pf_team'] == matched].copy()
    if 'sm_total_minutes' in tp.columns: tp = tp[tp['sm_total_minutes'] >= min_min]
    if len(tp) == 0: return None, f"No players for {matched}"
    
    ls = {}
    if lineups is not None and not lineups.empty:
        # Match using the formal 'matched' name, not the query
        tl = lineups[lineups['team'] == matched]
        if tl.empty: # Fallback substring for lineups
             tl = lineups[lineups['team'].str.contains(matched, case=False, na=False)]
             
        if not tl.empty:
            rm = tl['match_id'].unique()[-10:]
            rec = tl[tl['match_id'].isin(rm)]
            for pid in tp['player_id'].unique():
                pr = rec[rec['player_id'] == pid]
                if len(pr) > 0:
                    lm = pr.sort_values('match_id').iloc[-1]
                    ls[pid] = {'starts': int(pr['is_starter'].sum()), 'apps': len(pr), 'started_last': bool(lm['is_starter'])}
    
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
            
            # Combine starts/apps for column parity
            star = "*" if l['started_last'] else ""
            results.append({
                'player_id': pid, 'name': p.get('pf_name', '?'), 'position': pos, 'pos': get_pos_abbrev(pos),
                'goal_prob': round(gp*100, 2), 'goal_odds': round(go, 2), 'assist_prob': round(ap*100, 2), 'assist_odds': round(ao, 2),
                'xg_per90': round(p.get('sm_xg_per90', 0) or 0, 2), 'xa_per90': round(p.get('sm_xa_per90', 0) or 0, 2),
                'st/app': f"{l['starts']}{star}/{l['apps']}", 'started_last': l['started_last'], 'tags': tags.get(int(pid), [])})
        except: continue
        
    if not results: return None, f"No predictions for {matched}"
    df = pd.DataFrame(results)
    df = apply_market_scaling(df, xg)
    
    # Calculate value gap inside team context
    df['rank_model'] = df['goal_odds'].rank()
    pr = {'striker': 1, 'centerforward': 1, 'leftwinger': 2, 'rightwinger': 2, 'centerattackingmidfielder': 3}
    df['rank_pos'] = df['position'].apply(lambda x: pr.get(str(x).lower().strip(), 8))
    df['value_gap'] = df['rank_pos'] - df['rank_model']
    
    return df.sort_values('goal_odds'), matched

# =============================================================================
# TABS
# =============================================================================
def render_slate(gm, am, gf, af, pp, tags, lineups):
    default = "Bologna, Inter, 1.55, 1.75"
    c1, c2 = st.columns([3, 1])
    with c1: slate = st.text_area("", value=default, height=100, label_visibility="collapsed", placeholder="Home, Away, H_xG, A_xG")
    with c2:
        st.markdown('<p style="color:#606060;font-size:0.75rem;">Filename</p>', unsafe_allow_html=True)
        fname = st.text_input("", value="slate_export", label_visibility="collapsed")
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns([1, 1.5, 1, 1, 1])
    with c1: mkt = st.radio("", ["Goal", "Assist"], horizontal=True, label_visibility="collapsed")
    with c2: maxo = st.slider("Max Odds", 1.5, 50.0, 15.0, 0.5, label_visibility="collapsed")
    with c3: pk = st.checkbox("PK Only")
    with c4: vf = st.checkbox("Value ≥2")
    with c5: scan = st.button("Scan Slate", type="primary", use_container_width=True)
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    if scan:
        all_res = []
        for line in slate.strip().split('\n'):
            pts = [x.strip() for x in line.split(',')]
            if len(pts) < 4: continue
            try: ht, at, hx, ax = pts[0], pts[1], float(pts[2]), float(pts[3])
            except: continue
            for tm, xg in [(ht, hx), (at, ax)]:
                df, m = get_team_odds(tm, xg, gm, am, gf, af, pp, tags, lineups=lineups)
                if df is not None:
                    df['matchup'] = f"{ht} vs {at}"; df['team'] = m; df['team_xg'] = xg
                    all_res.append(df)
        if all_res:
            comb = pd.concat(all_res, ignore_index=True)
            oc = 'goal_odds' if mkt == "Goal" else 'assist_odds'
            filt = comb[(comb[oc] <= maxo) & (comb['pos'] != 'GK')].copy()
            if pk: filt = filt[filt['tags'].apply(lambda x: 'PK' in x if x else False)]
            if vf: filt = filt[filt['value_gap'] >= 2]
            filt = filt.sort_values(oc)
            st.markdown(f'<p style="color:#808080;font-size:0.8rem;">{len(filt)} plays found | Market: {mkt}</p>', unsafe_allow_html=True)
            
            # Format Tags for display
            filt['Tags_Disp'] = filt['tags'].apply(lambda x: format_tags(x, 3))
            
            display_cols = ['matchup', 'name', 'pos', 'team_xg', 'st/app', 'xg_per90', 'xa_per90', oc, 'Tags_Disp']
            st.dataframe(
                filt[display_cols].rename(columns={'matchup': 'Match', 'name': 'Player', 'team_xg': 'txG', oc: 'Odds', 'Tags_Disp': 'Tags'}), 
                hide_index=True, use_container_width=True, height=500
            )
            st.download_button("Export CSV", filt.to_csv(index=False), f"{fname}.csv", "text/csv")
        else: st.markdown('<p style="color:#606060;">No matches found.</p>', unsafe_allow_html=True)

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

def render_team(gm, am, gf, af, pp, tags, teams, lineups):
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1: tm = st.selectbox("Team", teams, label_visibility="collapsed")
    with c2: xg = st.number_input("xG", 0.5, 4.5, 1.85, 0.05, label_visibility="collapsed")
    with c3: gen = st.button("Generate", type="primary", key="tg")
    if gen:
        df, m = get_team_odds(tm, xg, gm, am, gf, af, pp, tags, min_min=100, lineups=lineups)
        if df is None: st.error("Team not found"); return
        df = df[df['pos'] != 'GK']
        st.markdown(f'<p style="color:#808080;font-size:0.8rem;">{m} | xG: {xg}</p>', unsafe_allow_html=True)
        rows = [{'Player': r['name'], 'Pos': r['pos'], 'st/app': r['st/app'], 'Goal': r['goal_odds'], 'Assist': r['assist_odds'], 'xG90': r['xg_per90'], 'xA90': r['xa_per90'], 'Tags': format_tags(r['tags'], 3)} for _, r in df.nsmallest(20, 'goal_odds').iterrows()]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True, height=500)

def render_player(gm, am, gf, af, pp, tags, lineups):
    ap = sorted(pp['pf_name'].dropna().unique().tolist())
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1: ps = st.selectbox("Select Player", ap, label_visibility="collapsed")
    with c2: bxg = st.number_input("Team xG", 0.5, 4.0, 2.0, 0.1, label_visibility="collapsed")
    with c3: an = st.button("Deep Dive", type="primary")
    if an:
        pd_ = pp[pp['pf_name'] == ps]
        if len(pd_) == 0: st.error("Not found"); return
        p = pd_.iloc[0]; tm, pos, pid = p.get('pf_team', '?'), p.get('pf_position', '?'), p.get('player_id', 0)
        p_tags = tags.get(int(pid), [])
        st.markdown(f"### 🔍 {ps.upper()}")
        tag_html = " ".join([f'<span style="background-color: #1e3a5f; color: #e0e0e0; padding: 2px 8px; border-radius: 10px; font-size: 0.7rem; margin-right: 5px;">{t}</span>' for t in p_tags])
        st.markdown(tag_html, unsafe_allow_html=True)
        
        df, _ = get_team_odds(tm, bxg, gm, am, gf, af, pp, tags, min_min=0, lineups=lineups)
        if df is not None:
            po = df[df['player_id'] == pid]
            if len(po) > 0:
                o = po.iloc[0]
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Goal Odds", f"{o['goal_odds']:.2f}")
                m2.metric("Goal %", f"{o['goal_prob']:.1f}%")
                m3.metric("Assist Odds", f"{o['assist_odds']:.2f}")
                m4.metric("Assist %", f"{o['assist_prob']:.1f}%")
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        # Strategy Logic
        xg90 = p.get('sm_xg_per90', 0) or 0
        xa90 = p.get('sm_xa_per90', 0) or 0
        if 'PK' in p_tags and xg90 > 0.3: strat = "💎 PRIME GOALSCORER (Volume + PKs)"
        elif xa90 > 0.20: strat = "🎯 ASSIST SPECIALIST"
        else: strat = "📊 STANDARD USAGE"
        st.markdown(f"**💡 STRATEGY:** {strat}")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown('<p style="color:#606060;font-size:0.75rem;">SHOOTING</p>', unsafe_allow_html=True)
            st.markdown(f'<p style="color:#a0a0a0;font-size:0.8rem;">xG/90: {xg90:.2f}<br>Shots: {int(p.get("sp_total_shots",0) or 0)}</p>', unsafe_allow_html=True)
        with c2:
            st.markdown('<p style="color:#606060;font-size:0.75rem;">CREATION</p>', unsafe_allow_html=True)
            st.markdown(f'<p style="color:#a0a0a0;font-size:0.8rem;">xA/90: {xa90:.2f}<br>Assists: {int(p.get("assists",0) or 0)}</p>', unsafe_allow_html=True)
        with c3:
            st.markdown('<p style="color:#606060;font-size:0.75rem;">RECENT</p>', unsafe_allow_html=True)
            if lineups is not None:
                p_log = lineups[lineups['player_id'] == pid].tail(5)
                if not p_log.empty:
                    st.dataframe(p_log[['match_id', 'minutes_played', 'rating']].rename(columns={'match_id':'Match', 'minutes_played':'Min'}), hide_index=True)

def render_lineups(lineups, tags_map):
    if lineups is None or lineups.empty: st.markdown('<p style="color:#606060;">Lineup data not loaded.</p>', unsafe_allow_html=True); return
    lt = sorted(lineups['team'].dropna().unique().tolist())
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1: tm = st.selectbox("Select Team", lt, label_visibility="collapsed", key="lt_select")
    with c2: nr = st.number_input("Last N", 3, 15, 10, 1, label_visibility="collapsed")
    with c3: btn = st.button("Deep Intel", type="primary", use_container_width=True)
    if btn:
        matched = [t for t in lineups['team'].unique() if tm.lower() in t.lower()]
        if not matched: return
        tn = matched[0]; td = lineups[lineups['team'] == tn]
        rm = td['match_id'].unique()[-nr:]; rec = td[td['match_id'].isin(rm)]
        st.markdown(f"### 🏟️ {tn.upper()}")
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown('<p style="color:#606060;font-size:0.75rem;">FORMATIONS</p>', unsafe_allow_html=True)
            starters = rec[rec['is_starter'] == True]
            forms = starters.groupby(['match_id', 'formation']).size().reset_index().groupby('formation').size().reset_index(name='Games')
            st.dataframe(forms.sort_values('Games', ascending=False), hide_index=True, use_container_width=True)
        with c2:
            st.markdown('<p style="color:#606060;font-size:0.75rem;">TACTICAL NOTES</p>', unsafe_allow_html=True)
            active_pids = rec['player_id'].unique()
            pks = [rec[rec['player_id'] == pid]['player_name'].iloc[0] for pid in active_pids if 'PK' in tags_map.get(int(pid), [])]
            if pks: st.markdown(f"**🎯 Penalties:** {', '.join(pks)}")

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        stats = rec.groupby(['player_id', 'player_name', 'position_name']).agg({'is_starter': 'sum', 'minutes_played': 'mean', 'rating': 'mean'}).reset_index()
        stats['Start%'] = (stats['is_starter'] / len(rm) * 100).astype(int)
        st.dataframe(stats[['player_name', 'position_name', 'Start%', 'minutes_played', 'rating']].sort_values('Start%', ascending=False), hide_index=True, use_container_width=True)

def main():
    try: gm, am, gf, af, pp, meta, tags = load_models()
    except Exception as e: st.error(f"Failed to load: {e}"); return
    lineups = load_lineups()
    teams = sorted([t for t in pp['pf_team'].dropna().unique() if isinstance(t, str)])
    st.markdown('<p class="main-header">Goalscorer Odds</p>', unsafe_allow_html=True)
    t1, t2, t3, t4, t5 = st.tabs(["Slate", "H2H", "Team", "Player", "Lineups"])
    with t1: render_slate(gm, am, gf, af, pp, tags, lineups)
    with t2: render_h2h(gm, am, gf, af, pp, tags, teams, lineups)
    with t3: render_team(gm, am, gf, af, pp, tags, teams, lineups)
    with t4: render_player(gm, am, gf, af, pp, tags, lineups)
    with t5: render_lineups(lineups, tags)

if __name__ == "__main__": main()
