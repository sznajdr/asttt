import pandas as pd
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import os
import requests

# Try to import XGBoost (graceful fallback)
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

# =============================================================================
# DATA CONFIG
# =============================================================================

# 1. USE YOUR NEW FILE FOR FILTERING
POSITIONS_URL = "https://raw.githubusercontent.com/sznajdr/asttt/refs/heads/main/fotmobsmart_positions.csv"

# 2. EXISTING FILES FOR FEATURES
STATS_URL = "https://raw.githubusercontent.com/sznajdr/fb1/refs/heads/main/fotmob_multi_player_season_stats.csv"
FEATURES_URL = "https://raw.githubusercontent.com/sznajdr/fb1/refs/heads/main/player_features.csv"
LINEUPS_URL = "https://raw.githubusercontent.com/sznajdr/fb1/refs/heads/main/fotmob_multi_lineups.csv"

# ... [Keep your existing POS_MAP, TEXT_POS_MAP, BASELINES, etc. here] ...

POS_SORT = {'ST':1,'CF':2,'RW':3,'LW':4,'CAM':5,'RM':6,'LM':7,'CM':8,'DM':9,'RWB':10,'LWB':11,'RB':12,'LB':13,'CB':14,'GK':15}

POS_ENCODING = {'ST': 0, 'CF': 1, 'RW': 2, 'LW': 3, 'CAM': 4, 'RM': 5, 'LM': 6, 
                'CM': 7, 'DM': 8, 'RWB': 9, 'LWB': 10, 'RB': 11, 'LB': 12, 'CB': 13}

# Feature Lists
GOAL_FEATURES = [
    'shooting_xg_per90', 'shooting_shots_per90', 'shooting_shots_on_target_per90',
    'shotmap_conversion_rate', 'shotmap_overperformance', 'shotmap_inside_box_xg',
    'shotmap_avg_xg_per_shot', 'possession_touches_in_opposition_box_per90',
    'possession_touches_per90', 'shotmap_header_xg', 'defending_aerials_won_pct',
    'shotmap_penalty_goals', 'shotmap_freekick_goals', 'career_goals_per_appearance',
    'form_5_goal_rate', 'form_10_goal_rate', 'trait_goals', 'trait_shot_attempts', 'pos_encoded'
]

ASSIST_FEATURES = [
    'passing_xa_per90', 'passing_chances_created_per90', 'passing_successful_crosses_per90',
    'passing_cross_accuracy', 'passing_accurate_passes_per90', 'passing_pass_accuracy',
    'possession_dribbles_per90', 'possession_dribbles_success_rate', 'possession_fouls_won_per90',
    'passing_accurate_long_balls_per90', 'career_assists_per_appearance',
    'form_5_assist_rate', 'form_10_assist_rate', 'trait_chances_created', 'pos_encoded'
]

# ... [Keep _parse_market_value, _normalize_pos, classify_player_type] ...

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
        # Add explicit mapping for your constants if needed
        if pid in {11, 0, 1}: return 'GK' 
        # ... (use your POS_MAP here)
        
    # Fallback to string matching
    if 'primary_position_key' in row and pd.notna(row['primary_position_key']):
        raw = str(row['primary_position_key']).lower().replace('-', '').replace(' ', '').replace('_', '')
        if 'keeper' in raw: return 'GK'
        if 'rightback' in raw: return 'RB'
        if 'leftback' in raw: return 'LB'
        if 'centerback' in raw: return 'CB'
        if 'striker' in raw or 'forward' in raw: return 'ST'
        if 'winger' in raw: return 'RW' if 'right' in raw else 'LW'
        if 'midfield' in raw:
            if 'attacking' in raw: return 'CAM'
            if 'defensive' in raw: return 'DM'
            return 'CM'
            
    return 'CM'

def classify_player_type(row):
    xg = row.get('shooting_xg_per90', 0) or 0
    xa = row.get('passing_xa_per90', 0) or 0
    box = row.get('possession_touches_in_opposition_box_per90', 0) or 0
    aerial = (row.get('defending_aerials_won_pct', 0) or 0) / 100
    dribble = row.get('possession_dribbles_per90', 0) or 0
    cross = row.get('passing_successful_crosses_per90', 0) or 0
    header_xg = row.get('shotmap_header_xg', 0) or 0
    
    if xg > 0.30 and xa < 0.12: return "Poacher"
    elif xa > 0.18 and xg < 0.12: return "Playmaker"
    elif xg > 0.18 and xa > 0.15: return "Complete"
    elif aerial > 0.55 and (header_xg > 0.15 or box > 2.5): return "Target Man"
    elif cross > 1.2 and xa > 0.10: return "Wide Creator"
    elif dribble > 2.5: return "Dribbler"
    elif box > 3.5 and xg > 0.12: return "Box Crasher"
    else: return "Utility"

# =============================================================================
# NEW DATA LOADER FOR POSITIONS/ACTIVE PLAYERS
# =============================================================================
@st.cache_data
def load_active_players_map():
    """
    Loads the match-level data to determine who is ACTUALLY playing for which team
    in the current season (2024/2025).
    Returns a dictionary: { 'TeamName': [list_of_active_player_ids] }
    """
    try:
        df = pd.read_csv(POSITIONS_URL)
        
        # Filter for the latest season available in the file
        # We assume the file contains 2024/2025 data based on your snippet
        current_season = df['season'].max() 
        df_active = df[df['season'] == current_season].copy()
        
        # Create a map of Team -> Set of Player IDs
        active_map = {}
        
        # Group by team and collect unique player IDs
        for team, group in df_active.groupby('team'):
            active_map[team] = set(group['player_id'].unique())
            
        return active_map
    except Exception as e:
        st.error(f"Failed to load positions map: {e}")
        return {}

@st.cache_data
def load_main_data():
    # Load features and aggregated stats
    df_f = pd.read_csv(FEATURES_URL)
    df_s = pd.read_csv(STATS_URL)
    
    # Merge
    if 'total_minutes' in df_s.columns:
        df_s['total_minutes'] = pd.to_numeric(df_s['total_minutes'].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
    
    # Filter for decent minutes
    df_s = df_s[df_s['total_minutes'] > 0]
    
    # Drop duplicates (keep player with most minutes if dupe)
    df_s = df_s.sort_values('total_minutes', ascending=False).drop_duplicates('player_id')
    
    cols_s = ['player_id', 'player_name', 'team', 'total_minutes', 'total_goals', 'total_assists', 'primary_position_key']
    cols_s = [c for c in cols_s if c in df_s.columns]
    
    df = pd.merge(df_s[cols_s], df_f, on='player_id', how='inner')
    
    # Preprocessing
    df['pos'] = df.apply(_normalize_pos, axis=1)
    df['pos_encoded'] = df['pos'].map(POS_ENCODING).fillna(7)
    df['market_value'] = df.get('market_value', 0).apply(_parse_market_value)
    
    # Fill numeric
    num_cols = GOAL_FEATURES + ASSIST_FEATURES + CLUSTER_FEATURES
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
    
    df['player_type'] = df.apply(classify_player_type, axis=1)
    
    return df

@st.cache_resource
def load_xgb_models():
    if not XGB_AVAILABLE: return None, None
    current_dir = os.path.dirname(os.path.abspath(__file__))
    g_path = os.path.join(current_dir, 'goal_model.json')
    a_path = os.path.join(current_dir, 'assist_model.json')
    
    if os.path.exists(g_path) and os.path.exists(a_path):
        gm = xgb.XGBRegressor(); gm.load_model(g_path)
        am = xgb.XGBRegressor(); am.load_model(a_path)
        return gm, am
    return None, None

# =============================================================================
# PREDICTION LOGIC (With Active Player Filter)
# =============================================================================

def predict_heuristic(player_row, team_xg_mod=1.0, team_avg_value=1):
    # [KEEP YOUR EXISTING HEURISTIC LOGIC HERE]
    # For brevity, using a simplified version, paste your full logic back in
    
    pos = player_row['pos']
    
    # Basic Baselines (paste your dictionaries back)
    GOAL_BASELINES = {'ST': 0.40, 'RW': 0.25, 'LW': 0.25, 'CAM': 0.20, 'CM': 0.08, 'CB': 0.04, 'RB': 0.02, 'LB': 0.02}
    base_g = GOAL_BASELINES.get(pos, 0.05)
    
    xg = player_row.get('shooting_xg_per90', 0)
    xa = player_row.get('passing_xa_per90', 0)
    
    # Simple weight
    exp_g = (base_g * 0.3 + xg * 0.7) * team_xg_mod
    exp_a = (0.1 + xa * 0.8) * team_xg_mod * 0.85
    
    return exp_g, exp_a

def prepare_xgb_features(row, features):
    data = []
    for f in features:
        data.append(row.get(f, 0))
    return np.array(data).reshape(1, -1)

def get_predictions(df, active_map, team, team_xg, use_xgb):
    # 1. FILTER: Only keep players currently in the active map for this team
    if team not in active_map:
        st.warning(f"No active roster data found for {team}. Showing all historical players.")
        sub = df[df['team'] == team].copy()
    else:
        active_ids = active_map[team]
        sub = df[(df['team'] == team) & (df['player_id'].isin(active_ids))].copy()
    
    if sub.empty: return pd.DataFrame()
    
    # Load models
    gm, am = load_xgb_models()
    models_ok = (use_xgb and gm is not None)
    
    xg_mod = team_xg / 1.3  # Normalize around avg team goals
    team_val = sub['market_value'].mean() or 1
    
    res = []
    
    for _, row in sub.iterrows():
        # Heuristic
        h_g, h_a = predict_heuristic(row, xg_mod, team_val)
        
        # XGB
        if models_ok:
            try:
                x_g_in = prepare_xgb_features(row, GOAL_FEATURES)
                x_a_in = prepare_xgb_features(row, ASSIST_FEATURES)
                x_g = gm.predict(x_g_in)[0] * xg_mod
                x_a = am.predict(x_a_in)[0] * xg_mod
                
                # Ensemble
                fin_g = (x_g * 0.65) + (h_g * 0.35)
                fin_a = (x_a * 0.65) + (h_a * 0.35)
            except:
                fin_g, fin_a = h_g, h_a
        else:
            fin_g, fin_a = h_g, h_a
            
        # Convert to Odds
        prob_g = 1 - np.exp(-fin_g)
        prob_a = 1 - np.exp(-fin_a)
        
        odds_g = 1 / prob_g if prob_g > 0.01 else 101.0
        odds_a = 1 / prob_a if prob_a > 0.01 else 101.0
        
        # Floors/Ceilings logic here...
        odds_g = max(1.4, min(odds_g, 101.0))
        odds_a = max(1.5, min(odds_a, 101.0))
        
        total_threat = row['shooting_xg_per90'] + row['passing_xa_per90']
        ratio = row['shooting_xg_per90'] / total_threat if total_threat > 0 else 0
        
        res.append({
            'Player': row['player_name'],
            'Pos': row['pos'],
            'ATG': odds_g,
            'AST': odds_a,
            'xG': row['shooting_xg_per90'],
            'xA': row['passing_xa_per90'],
            'Type': ratio,
            'PType': row['player_type'],
            'min': int(row['total_minutes']),
            'pos_sort': POS_SORT.get(row['pos'], 99)
        })
        
    return pd.DataFrame(res).sort_values('pos_sort')

# =============================================================================
# APP LAYOUT
# =============================================================================

# Load Data
df_main = load_main_data()
active_rosters = load_active_players_map()

# Sidebar / Controls
col1, col2, col3 = st.columns([3, 2, 2])

with col1:
    # Only show teams that exist in our active roster map if possible, else all
    available_teams = sorted(list(active_rosters.keys())) if active_rosters else sorted(df_main['team'].unique())
    team = st.selectbox("Select Team", available_teams)

with col2:
    team_xg = st.slider("Implied Team Total", 0.5, 3.5, 1.35, 0.05)

with col3:
    use_xgb = st.checkbox("Use XGBoost", value=True)

if team:
    preds = get_predictions(df_main, active_rosters, team, team_xg, use_xgb)
    
    if not preds.empty:
        # Format for display
        st.dataframe(
            preds.style.format({
                'ATG': '{:.2f}', 'AST': '{:.2f}', 'xG': '{:.2f}', 'xA': '{:.2f}', 'Type': '{:.2f}'
            }).background_gradient(subset=['ATG'], cmap='RdYlGn_r', vmin=2.0, vmax=15.0),
            use_container_width=True,
            height=600,
            hide_index=True
        )
    else:
        st.info("No active players found for this team (check CSV mapping).")
