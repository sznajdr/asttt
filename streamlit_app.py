import pandas as pd
import numpy as np
import streamlit as st
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONSTANTS
# =============================================================================

POS_MAP = {0: 'GK', 1: 'GK', 11: 'GK', 21: 'RB', 22: 'RB', 31: 'CB', 32: 'CB', 33: 'CB', 34: 'CB', 35: 'CB', 36: 'CB',
           41: 'LB', 42: 'LB', 51: 'DM', 52: 'DM', 53: 'DM', 61: 'RM', 62: 'CM', 63: 'CM', 64: 'CM', 65: 'CM', 66: 'LM',
           71: 'RW', 72: 'CAM', 73: 'CAM', 74: 'CAM', 75: 'CAM', 76: 'LW', 81: 'RW', 82: 'CF', 83: 'ST', 84: 'ST', 
           85: 'CF', 86: 'LW', 91: 'ST', 92: 'ST', 93: 'ST', 94: 'ST'}

TEXT_POS_MAP = {
    'centerdefensivemidfielder': 'DM', 'centerattackingmidfielder': 'CAM', 'centermidfielder': 'CM',
    'leftmidfielder': 'LM', 'rightmidfielder': 'RM', 'leftwinger': 'LW', 'rightwinger': 'RW',
    'leftwingback': 'LWB', 'rightwingback': 'RWB', 'leftback': 'LB', 'rightback': 'RB',
    'centerback': 'CB', 'striker': 'ST', 'keeper_long': 'GK', 'keeper': 'GK',
    'attacking': 'CAM', 'defensive': 'DM', 'winger': 'RW', 'forward': 'ST',
    'midfield': 'CM', 'defender': 'CB', 'goalkeeper': 'GK'
}

GOAL_BASELINES = {'ST': 0.45, 'CF': 0.40, 'RW': 0.25, 'LW': 0.25, 'CAM': 0.20, 'RM': 0.12, 'LM': 0.12, 'CM': 0.08,
                  'DM': 0.04, 'RWB': 0.03, 'LWB': 0.03, 'RB': 0.02, 'LB': 0.02, 'CB': 0.03, 'GK': 0.001}
ASSIST_BASELINES = {'ST': 0.15, 'CF': 0.18, 'RW': 0.22, 'LW': 0.22, 'CAM': 0.28, 'RM': 0.18, 'LM': 0.18, 'CM': 0.12,
                    'DM': 0.06, 'RWB': 0.10, 'LWB': 0.10, 'RB': 0.08, 'LB': 0.08, 'CB': 0.02, 'GK': 0.001}
POS_SORT = {'ST':1,'CF':2,'RW':3,'LW':4,'CAM':5,'RM':6,'LM':7,'CM':8,'DM':9,'RWB':10,'LWB':11,'RB':12,'LB':13,'CB':14,'GK':15}

# =============================================================================
# DATA LOADING
# =============================================================================

STATS_URL = "https://raw.githubusercontent.com/sznajdr/fb1/refs/heads/main/fotmob_multi_player_season_stats.csv"
FEATURES_URL = "https://raw.githubusercontent.com/sznajdr/fb1/refs/heads/main/player_features.csv"

def _parse_market_value(val):
    if pd.isna(val): return 0
    s = str(val).replace('€', '').replace(',', '').strip()
    try:
        if 'M' in s: return float(s.replace('M', '')) * 1_000_000
        elif 'K' in s: return float(s.replace('K', '')) * 1_000
        else: return float(s)
    except: return 0

def _normalize_pos(row):
    if 'position_id' in row and pd.notna(row['position_id']):
        pid = int(row['position_id'])
        if pid in POS_MAP: return POS_MAP[pid]
    if 'primary_position_key' in row and pd.notna(row['primary_position_key']):
        raw = str(row['primary_position_key']).lower().replace('-', '').replace(' ', '').replace('_', '')
        for k, v in TEXT_POS_MAP.items():
            if k == raw: return v
        for k, v in TEXT_POS_MAP.items():
            if k in raw: return v
    return 'CM'

@st.cache_data
def load_data():
    try:
        df_f = pd.read_csv(FEATURES_URL)
    except Exception as e:
        st.error(f"Failed to read player_features.csv: {e}")
        return pd.DataFrame()

    try:
        df_s = pd.read_csv(STATS_URL)
    except Exception as e:
        st.error(f"Failed to read fotmob_multi_player_season_stats.csv: {e}")
        return pd.DataFrame()

    # Pre-process Stats
    if 'total_minutes' in df_s.columns:
        df_s['total_minutes'] = df_s['total_minutes'].astype(str).str.replace(',', '', regex=False)
        df_s['total_minutes'] = pd.to_numeric(df_s['total_minutes'], errors='coerce').fillna(0)
    
    # Dedupe based on minutes
    df_s = df_s.sort_values('total_minutes', ascending=False).drop_duplicates('player_id')
    
    # Merge
    cols_s = ['player_id', 'player_name', 'team', 'total_minutes', 'total_goals', 'total_assists', 'position']
    if 'position_id' in df_s.columns: cols_s.append('position_id')
    cols_s = [c for c in cols_s if c in df_s.columns]
    
    df = pd.merge(df_s[cols_s], df_f, on='player_id', how='inner')

    if df.empty:
        st.error("Merge produced empty result.")
        return df

    # Process Data
    try:
        df['pos'] = df.apply(_normalize_pos, axis=1)
        df['pos_sort'] = df['pos'].map(lambda x: POS_SORT.get(x, 99))
        
        num_cols = ['shooting_xg_per90', 'passing_xa_per90', 'form_5_goal_rate', 'form_5_assist_rate', 'trait_goals', 'trait_chances_created']
        for c in num_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
            else:
                df[c] = 0.0
        
        df['market_value'] = df.get('market_value', pd.Series(0)).apply(_parse_market_value)
        
        df = df[(df['total_minutes'] >= 1) & (df['pos'] != 'GK')].copy()
        
        return df
    except Exception as e:
        st.error(f"Error during processing: {e}")
        return pd.DataFrame()

# =============================================================================
# PREDICTION
# =============================================================================

def predict_odds(df, team, team_xg=1.5):
    sub = df[df['team'] == team].copy()
    if sub.empty: return pd.DataFrame()
    xg_mod = team_xg / 1.22
    team_avg_value = sub['market_value'].replace(0, np.nan).mean() or 1
    results = []
    for _, p in sub.iterrows():
        pos = p['pos']
        base_g = GOAL_BASELINES.get(pos, 0.10)
        xg_p90 = p.get('shooting_xg_per90', 0)
        xg_pct = p.get('shooting_goals_percentile', 50) / 100
        form_g = p.get('form_5_goal_rate', 0)
        trait_g = p.get('trait_goals', 0) / 100 if p.get('trait_goals', 0) > 1 else p.get('trait_goals', 0)
        exp_g = (base_g * 0.3 + xg_p90 * 0.35 + form_g * 0.20 + trait_g * 0.15) * xg_mod
        if xg_pct > 0.8: exp_g *= 1.15
        elif xg_pct > 0.6: exp_g *= 1.05
        base_a = ASSIST_BASELINES.get(pos, 0.10)
        xa_p90 = p.get('passing_xa_per90', 0)
        xa_pct = p.get('passing_xa_percentile', 50) / 100
        cc_p90 = p.get('passing_chances_created_per90', 0)
        form_a = p.get('form_5_assist_rate', 0)
        trait_cc = p.get('trait_chances_created', 0) / 100 if p.get('trait_chances_created', 0) > 1 else p.get('trait_chances_created', 0)
        exp_a = (base_a * 0.25 + xa_p90 * 0.30 + (cc_p90 / 3) * 0.20 + form_a * 0.15 + trait_cc * 0.10) * (xg_mod * 0.82)
        if xa_pct > 0.8 or cc_p90 > 0.8: exp_a *= 1.15
        elif xa_pct > 0.6 or cc_p90 > 0.6: exp_a *= 1.05
        value_ratio = (p.get('market_value', 0) or 1) / team_avg_value
        value_boost = 1.12 if value_ratio >= 2 else 1.06 if value_ratio >= 1.5 else 1.0 if value_ratio >= 0.7 else 0.95
        exp_g *= value_boost
        exp_a *= value_boost
        prob_g = 1 - np.exp(-exp_g)
        prob_a = 1 - np.exp(-exp_a)
        odds_g = np.clip((1 / max(prob_g, 0.01)) * 1.05, 1.30, 50.0)
        odds_a = np.clip((1 / max(prob_a, 0.01)) * 1.05, 1.40, 50.0)
        if pos in ['CB','RB','LB']: odds_g = max(odds_g, 8.0)
        if pos == 'DM': odds_g = max(odds_g, 6.0)
        if pos == 'CB': odds_a = max(odds_a, 12.0)
        total_threat = xg_p90 + xa_p90
        scorer_ratio = xg_p90 / total_threat if total_threat > 0 else 0.5
        results.append({
            'Player': p.get('player_name', 'Unknown'),
            'Pos': pos,
            '⚽ ATG': round(odds_g, 2),
            '🎯 AST': round(odds_a, 2),
            'xG': round(xg_p90, 2),
            'xA': round(xa_p90, 2),
            'Type': round(scorer_ratio, 2),
            'Mins': int(p.get('total_minutes', 0)),
            'pos_sort': POS_SORT.get(pos, 99),
            'goal_odds': round(odds_g, 2),
            'assist_odds': round(odds_a, 2)
        })
    return pd.DataFrame(results).sort_values('pos_sort')

# =============================================================================
# STREAMLIT UI
# =============================================================================

st.set_page_config(page_title="Player Odds Predictor", layout="wide")

# Dark theme CSS
st.markdown("""
<style>
    .stApp {
        background-color: #1e1e1e;
    }
    .header-box {
        background: #252525;
        padding: 12px;
        margin-bottom: 12px;
        border-radius: 6px;
        font-weight: bold;
        color: #3794ff;
    }
    /* Style the HTML table */
    table {
        width: 100%;
        border-collapse: collapse;
        font-family: 'Segoe UI', sans-serif;
        font-size: 14px;
        background: #1e1e1e;
        color: #d4d4d4;
    }
    table th {
        background: #333;
        padding: 10px 12px;
        text-align: left !important;
        border-bottom: 2px solid #555;
        font-weight: 600;
    }
    table td {
        padding: 8px 12px;
        text-align: left !important;
        border-bottom: 1px solid #3a3a3a;
    }
    table tr:hover {
        background: #2d2d2d;
    }
    /* Hide the index column */
    table th:first-child, table td:first-child {
        display: none;
    }
    
    /* Mobile responsive */
    @media screen and (max-width: 768px) {
        table {
            font-size: 11px;
        }
        table th, table td {
            padding: 6px 4px;
            white-space: nowrap;
        }
        .header-box {
            font-size: 12px;
            padding: 8px;
        }
        /* Make table scrollable horizontally */
        .table-container {
            overflow-x: auto;
            -webkit-overflow-scrolling: touch;
        }
    }
    
    @media screen and (max-width: 480px) {
        table {
            font-size: 10px;
        }
        table th, table td {
            padding: 5px 3px;
        }
    }
</style>
""", unsafe_allow_html=True)

# Load data
with st.spinner("Loading data..."):
    df = load_data()

if df.empty:
    st.error("No data loaded. Check the CSV URLs.")
    st.stop()

st.success(f"Loaded {len(df)} players")

# Controls
col1, col2, col3 = st.columns([2, 3, 2])

with col1:
    teams = sorted(df['team'].unique())
    selected_team = st.selectbox("Team", teams)

with col2:
    team_xg = st.slider("Team xG", min_value=0.5, max_value=4.0, value=1.5, step=0.1)

with col3:
    sort_options = {'Goal Odds': 'goal_odds', 'Assist Odds': 'assist_odds', 'Position': 'pos_sort'}
    sort_by = st.selectbox("Sort by", list(sort_options.keys()))

# Get predictions
data = predict_odds(df, selected_team, team_xg)

if data.empty:
    st.warning("No players found for this team")
    st.stop()

# Sort data
data = data.sort_values(sort_options[sort_by])

# Header
st.markdown(f'<div class="header-box">{selected_team} | Team xG: {team_xg} | {len(data)} players</div>', unsafe_allow_html=True)

# Prepare display dataframe
display_df = data[['Player', 'Pos', '⚽ ATG', '🎯 AST', 'xG', 'xA', 'Type', 'Mins']].reset_index(drop=True)

# Position colors from original
POS_COLORS = {
    'ST': '#a94442', 'CF': '#a94442', 
    'RW': '#c97a4a', 'LW': '#c97a4a',
    'CAM': '#5a8a5a', 
    'RM': '#5a9a8a', 'LM': '#5a9a8a',
    'CM': '#5a7a9a', 
    'DM': '#7a6a9a',
    'RWB': '#6a7a8a', 'LWB': '#6a7a8a',
    'RB': '#6a7a8a', 'LB': '#6a7a8a',
    'CB': '#5a6a7a', 
    'GK': '#9a8a5a'
}

# Color functions
def color_goal_odds(val):
    if val < 4:
        return 'color: #4ec9b0; font-weight: bold'  # hot - green
    elif val < 8:
        return 'color: #dcdcaa'  # warm - yellow
    else:
        return 'color: #888'  # cold - gray

def color_assist_odds(val):
    if val < 4:
        return 'color: #4ec9b0; font-weight: bold'  # hot - green
    elif val < 8:
        return 'color: #dcdcaa'  # warm - yellow
    else:
        return 'color: #888'  # cold - gray

def color_type(val):
    if val >= 0.6:
        return 'color: #dc3545'  # scorer - red
    elif val <= 0.4:
        return 'color: #17a2b8'  # creator - blue
    else:
        return 'color: #6c757d'  # balanced - gray

def color_position(val):
    bg_color = POS_COLORS.get(val, '#666')
    return f'background-color: {bg_color}; color: white; font-weight: bold; padding: 2px 6px; border-radius: 4px'

# Style the dataframe with left alignment and colors
styled_df = display_df.style.applymap(
    color_goal_odds, subset=['⚽ ATG']
).applymap(
    color_assist_odds, subset=['🎯 AST']
).applymap(
    color_type, subset=['Type']
).applymap(
    color_position, subset=['Pos']
).set_properties(**{
    'text-align': 'left'
}).set_table_styles([
    {'selector': 'th', 'props': [('text-align', 'left')]},
    {'selector': 'td', 'props': [('text-align', 'left')]}
]).format({
    '⚽ ATG': '{:.2f}',
    '🎯 AST': '{:.2f}',
    'xG': '{:.2f}',
    'xA': '{:.2f}',
    'Type': '{:.2f}',
    'Mins': '{:d}'
})

# Display with st.write (respects styling better)
st.markdown(f'<div class="table-container">{styled_df.to_html()}</div>', unsafe_allow_html=True)
