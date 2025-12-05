"""
Goalscorer Prediction App - XGBoost + Clustering Version
=========================================================
Features:
- XGBoost-trained goal/assist prediction models
- Player clustering (Poacher, Playmaker, Box Crasher, etc.)
- Hybrid prediction: XGBoost + heuristic ensemble
- All quick wins from enhanced version
"""
import pandas as pd
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import pickle
import warnings
import requests
from io import BytesIO

warnings.filterwarnings('ignore')

# Try to import XGBoost (graceful fallback if not available)
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

# =============================================================================
# CLASS DEFINITIONS (REQUIRED FOR PICKLE LOADING)
# =============================================================================

class ModelConfig:
    """
    Configuration container to satisfy pickle requirements.
    """
    def __init__(self):
        self.goal_features = []
        self.assist_features = []
        self.scaler = None
        self.goal_model = None
        self.assist_model = None

class GoalAssistModel:
    """
    Reconstructs the class structure expected by the pickle file.
    Wraps the underlying model to provide predict_proba functionality.
    """
    def __init__(self, model=None, features=None, scaler=None):
        self.model = model
        self.features = features
        self.scaler = scaler

    def predict_proba(self, X):
        # Delegates prediction to the internal XGBoost/Sklearn model
        if hasattr(self, 'model') and self.model is not None:
            # Apply scaler if it exists within this wrapper
            if hasattr(self, 'scaler') and self.scaler is not None:
                try:
                    X = self.scaler.transform(X)
                except:
                    pass # Fallback to unscaled if shapes mismatch
            return self.model.predict_proba(X)
        return np.zeros((X.shape[0], 2)) # Fallback

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

GOAL_BASELINES = {'ST': 0.41, 'CF': 0.40, 'RW': 0.28, 'LW': 0.28, 'CAM': 0.23, 'RM': 0.12, 'LM': 0.12, 'CM': 0.08,
                  'DM': 0.04, 'RWB': 0.03, 'LWB': 0.03, 'RB': 0.02, 'LB': 0.02, 'CB': 0.03, 'GK': 0.001}
ASSIST_BASELINES = {'ST': 0.19, 'CF': 0.19, 'RW': 0.21, 'LW': 0.21, 'CAM': 0.22, 'RM': 0.16, 'LM': 0.16, 'CM': 0.14,
                    'DM': 0.06, 'RWB': 0.09, 'LWB': 0.09, 'RB': 0.08, 'LB': 0.08, 'CB': 0.02, 'GK': 0.001}
POS_SORT = {'ST':1,'CF':2,'RW':3,'LW':4,'CAM':5,'RM':6,'LM':7,'CM':8,'DM':9,'RWB':10,'LWB':11,'RB':12,'LB':13,'CB':14,'GK':15}

POS_ENCODING = {'ST': 0, 'CF': 1, 'RW': 2, 'LW': 3, 'CAM': 4, 'RM': 5, 'LM': 6, 
                'CM': 7, 'DM': 8, 'RWB': 9, 'LWB': 10, 'RB': 11, 'LB': 12, 'CB': 13}

# =============================================================================
# DATA LOADING
# =============================================================================

STATS_URL = "https://raw.githubusercontent.com/sznajdr/fb1/refs/heads/main/fotmob_multi_player_season_stats.csv"
FEATURES_URL = "https://raw.githubusercontent.com/sznajdr/fb1/refs/heads/main/player_features.csv"
LINEUPS_URL = "https://raw.githubusercontent.com/sznajdr/fb1/refs/heads/main/fotmob_multi_lineups.csv"
MODEL_URL = "https://raw.githubusercontent.com/sznajdr/asttt/refs/heads/main/football_model.pkl"

TACTIC_POS_COLORS = {
    'GK': '#e2b714', 'DEF': '#3794ff', 'MID': '#4ec9b0', 'FWD': '#e056fd', 'UNK': '#666'
}

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

def classify_player_type(row):
    """Classify player into offensive archetype based on stats"""
    xg = row.get('shooting_xg_per90', 0) or 0
    xa = row.get('passing_xa_per90', 0) or 0
    box = row.get('possession_touches_in_opposition_box_per90', 0) or 0
    aerial = (row.get('defending_aerials_won_pct', 0) or 0) / 100
    dribble = row.get('possession_dribbles_per90', 0) or 0
    cross = row.get('passing_successful_crosses_per90', 0) or 0
    header_xg = row.get('shotmap_header_xg', 0) or 0
    
    # Classification logic
    if xg > 0.30 and xa < 0.12:
        return "Poacher"
    elif xa > 0.18 and xg < 0.12:
        return "Playmaker"
    elif xg > 0.18 and xa > 0.15:
        return "Complete"
    elif aerial > 0.55 and (header_xg > 0.15 or box > 2.5):
        return "Target Man"
    elif cross > 1.2 and xa > 0.10:
        return "Wide Creator"
    elif dribble > 2.5:
        return "Dribbler"
    elif box > 3.5 and xg > 0.12:
        return "Box Crasher"
    else:
        return "Utility"

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

    if 'total_minutes' in df_s.columns:
        df_s['total_minutes'] = df_s['total_minutes'].astype(str).str.replace(',', '', regex=False)
        df_s['total_minutes'] = pd.to_numeric(df_s['total_minutes'], errors='coerce').fillna(0)
    
    df_s = df_s.sort_values('total_minutes', ascending=False).drop_duplicates('player_id')
    
    cols_s = ['player_id', 'player_name', 'team', 'total_minutes', 'total_goals', 'total_assists', 'position']
    if 'position_id' in df_s.columns: cols_s.append('position_id')
    cols_s = [c for c in cols_s if c in df_s.columns]
    
    df = pd.merge(df_s[cols_s], df_f, on='player_id', how='inner')

    if df.empty:
        st.error("Merge produced empty result.")
        return df

    try:
        df['pos'] = df.apply(_normalize_pos, axis=1)
        df['pos_sort'] = df['pos'].map(lambda x: POS_SORT.get(x, 99))
        df['pos_encoded'] = df['pos'].map(POS_ENCODING).fillna(7)
        
        # All numeric columns needed
        num_cols = [
            'shooting_xg_per90', 'passing_xa_per90', 'form_5_goal_rate', 'form_5_assist_rate', 
            'trait_goals', 'trait_chances_created', 'shotmap_overperformance', 'shotmap_conversion_rate',
            'possession_touches_in_opposition_box_per90', 'form_5_minutes', 'form_10_minutes',
            'career_goals_per_appearance', 'career_assists_per_appearance',
            'shotmap_freekick_goals', 'shotmap_penalty_goals', 'shotmap_header_goals', 'shotmap_header_xg',
            'defending_aerials_won_pct', 'passing_successful_crosses_per90', 'passing_cross_accuracy',
            'possession_dribbles_per90', 'possession_fouls_won_per90',
            'shotmap_inside_box_shots', 'shotmap_total_shots', 'shotmap_inside_box_xg', 'shotmap_avg_xg_per_shot',
            'shooting_goals_percentile', 'passing_xa_percentile', 'passing_chances_created_per90',
            'shooting_shots_per90', 'shooting_shots_on_target_per90', 'form_10_goal_rate', 'form_10_assist_rate',
            'trait_shot_attempts', 'passing_accurate_passes_per90', 'passing_pass_accuracy',
            'possession_dribbles_success_rate', 'passing_accurate_long_balls_per90', 'possession_touches_per90'
        ]
        for c in num_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
            else:
                df[c] = 0.0
        
        df['market_value'] = df.get('market_value', pd.Series(0)).apply(_parse_market_value)
        
        # Classify player types
        df['player_type'] = df.apply(classify_player_type, axis=1)
        
        df = df[(df['total_minutes'] >= 1) & (df['pos'] != 'GK')].copy()
        
        return df
    except Exception as e:
        st.error(f"Error during processing: {e}")
        return pd.DataFrame()

@st.cache_data
def load_lineups_data():
    try:
        df_l = pd.read_csv(LINEUPS_URL)
        cols_to_num = ['rating', 'minutes_played', 'goals_in_match', 'assists_in_match']
        for c in cols_to_num:
            if c in df_l.columns:
                df_l[c] = pd.to_numeric(df_l[c], errors='coerce').fillna(0)
        return df_l
    except Exception as e:
        return pd.DataFrame()

# =============================================================================
# NEW MODEL LOADING (football_model.pkl)
# =============================================================================

@st.cache_resource
def load_football_model():
    """Load the pre-trained football_model.pkl from GitHub"""
    try:
        response = requests.get(MODEL_URL, timeout=30)
        response.raise_for_status()
        model_data = pickle.load(BytesIO(response.content))
        return model_data
    except Exception as e:
        st.warning(f"Could not load model from GitHub: {e}")
        return None

# =============================================================================
# PREDICTION FUNCTIONS
# =============================================================================

def safe_get(row, col, default=0):
    val = row.get(col, default)
    if pd.isna(val):
        return default
    return float(val)

def get_model_attr(container, key, default=None):
    """Safely get attribute from dict or object"""
    if isinstance(container, dict):
        return container.get(key, default)
    else:
        return getattr(container, key, default)

def predict_with_pkl_model(player_row, model_data, team_xg=1.5):
    """
    Use the football_model.pkl to predict goal/assist probabilities.
    Returns (goal_prob, assist_prob) or (None, None) if model unavailable.
    """
    if model_data is None:
        return None, None
    
    try:
        # Robust retrieval handling both Dict and Object (ModelConfig) structure
        goal_model = get_model_attr(model_data, 'goal_model')
        assist_model = get_model_attr(model_data, 'assist_model')
        
        if goal_model is None or assist_model is None:
            return None, None

        # Features might be in the container or inside the goal_model itself
        goal_features = get_model_attr(model_data, 'goal_features', [])
        if not goal_features and hasattr(goal_model, 'features'):
            goal_features = goal_model.features

        assist_features = get_model_attr(model_data, 'assist_features', [])
        if not assist_features and hasattr(assist_model, 'features'):
            assist_features = assist_model.features
            
        scaler = get_model_attr(model_data, 'scaler')
        
        # Prepare goal features
        if not goal_features: return None, None
        goal_vals = []
        for f in goal_features:
            goal_vals.append(safe_get(player_row, f, 0))
        X_goal = np.array(goal_vals).reshape(1, -1)
        
        # Prepare assist features
        if not assist_features: return None, None
        assist_vals = []
        for f in assist_features:
            assist_vals.append(safe_get(player_row, f, 0))
        X_assist = np.array(assist_vals).reshape(1, -1)
        
        # Scale if scaler exists at top level
        # Note: If GoalAssistModel has internal scaler, it handles it in predict_proba
        if scaler is not None:
            try:
                X_goal = scaler.transform(X_goal)
                X_assist = scaler.transform(X_assist)
            except:
                pass # Squelch scaling errors (e.g. feature mismatch), try raw
        
        # Predict probabilities
        goal_prob = goal_model.predict_proba(X_goal)[0, 1]
        assist_prob = assist_model.predict_proba(X_assist)[0, 1]
        
        # Apply team xG modifier
        xg_modifier = team_xg / 1.5
        goal_prob = goal_prob * xg_modifier
        assist_prob = assist_prob * xg_modifier * 0.9
        
        # Clip to valid range
        goal_prob = np.clip(goal_prob, 0.001, 0.95)
        assist_prob = np.clip(assist_prob, 0.001, 0.90)
        
        return goal_prob, assist_prob
        
    except Exception as e:
        # Uncomment for debugging
        # st.toast(f"XGB Error: {str(e)}") 
        return None, None

def predict_heuristic(player_row, team_xg_mod=1.0, team_avg_value=1):
    """Enhanced heuristic prediction (quick wins version)"""
    pos = player_row['pos']
    total_minutes = safe_get(player_row, 'total_minutes', 0)
    
    # Goal prediction
    base_g = GOAL_BASELINES.get(pos, 0.10)
    xg_p90 = safe_get(player_row, 'shooting_xg_per90', 0)
    
    # Clinical finisher bonus
    overperformance = safe_get(player_row, 'shotmap_overperformance', 0)
    clinical_bonus = 1.15 if overperformance > 0.5 else 1.08 if overperformance > 0.2 else 0.92 if overperformance < -0.3 else 1.0
    
    # Box presence
    box_touches_p90 = safe_get(player_row, 'possession_touches_in_opposition_box_per90', 0)
    box_bonus = 1.12 if box_touches_p90 > 5 else 1.06 if box_touches_p90 > 3 else 1.02 if box_touches_p90 > 1.5 else 1.0
    
    # Form confidence
    form_5_minutes = safe_get(player_row, 'form_5_minutes', 0)
    form_confidence = min(form_5_minutes / 300, 1.0)
    form_g = safe_get(player_row, 'form_5_goal_rate', 0) * form_confidence
    
    # Career prior
    career_g = safe_get(player_row, 'career_goals_per_appearance', 0)
    minutes_weight = min(total_minutes / 900, 1.0)
    blended_g = (xg_p90 * minutes_weight) + (career_g * (1 - minutes_weight))
    
    # Set piece & header bonuses
    set_piece_bonus = 1.10 if safe_get(player_row, 'shotmap_freekick_goals', 0) >= 1 or safe_get(player_row, 'shotmap_penalty_goals', 0) >= 2 else 1.0
    header_goals = safe_get(player_row, 'shotmap_header_goals', 0)
    header_bonus = 1.08 if header_goals >= 2 else 1.04 if header_goals >= 1 else 1.0
    
    trait_g = safe_get(player_row, 'trait_goals', 0)
    trait_g = trait_g / 100 if trait_g > 1 else trait_g
    
    exp_g = (base_g * 0.18 + blended_g * 0.32 + form_g * 0.22 + trait_g * 0.13 + (box_touches_p90 / 20) * 0.15
            ) * team_xg_mod * clinical_bonus * box_bonus * set_piece_bonus * header_bonus
    
    # Assist prediction
    base_a = ASSIST_BASELINES.get(pos, 0.10)
    xa_p90 = safe_get(player_row, 'passing_xa_per90', 0)
    cc_p90 = safe_get(player_row, 'passing_chances_created_per90', 0)
    form_a = safe_get(player_row, 'form_5_assist_rate', 0) * form_confidence
    career_a = safe_get(player_row, 'career_assists_per_appearance', 0)
    blended_a = (xa_p90 * minutes_weight) + (career_a * (1 - minutes_weight))
    
    crosses_p90 = safe_get(player_row, 'passing_successful_crosses_per90', 0)
    dribbles_p90 = safe_get(player_row, 'possession_dribbles_per90', 0)
    
    cross_bonus = 1.12 if pos in ['RW', 'LW', 'RM', 'LM', 'RWB', 'LWB', 'RB', 'LB'] and crosses_p90 > 1.5 else 1.0
    dribble_bonus = 1.08 if dribbles_p90 > 3 else 1.04 if dribbles_p90 > 1.5 else 1.0
    
    # High xA bonus - reward creative players
    xa_bonus = 1.0
    if xa_p90 > 0.25:
        xa_bonus = 1.25  # Elite creator
    elif xa_p90 > 0.15:
        xa_bonus = 1.15  # Very good
    elif xa_p90 > 0.10:
        xa_bonus = 1.08  # Good
    
    trait_cc = safe_get(player_row, 'trait_chances_created', 0)
    trait_cc = trait_cc / 100 if trait_cc > 1 else trait_cc
    
    # Boosted xA weight (was 0.28, now 0.35)
    exp_a = (base_a * 0.15 + blended_a * 0.35 + (cc_p90 / 3) * 0.20 + form_a * 0.16 + trait_cc * 0.14
            ) * (team_xg_mod * 0.85) * cross_bonus * dribble_bonus * xa_bonus
    
    # Market value adjustment
    mv = safe_get(player_row, 'market_value', 0) or 1
    value_ratio = mv / team_avg_value
    value_boost = 1.12 if value_ratio >= 2 else 1.06 if value_ratio >= 1.5 else 0.95 if value_ratio < 0.7 else 1.0
    
    return exp_g * value_boost, exp_a * value_boost

def predict_odds(df, team, team_xg=1.5, use_xgb=True):
    """
    Combined prediction using PKL Model + Heuristic ensemble
    """
    sub = df[df['team'] == team].copy()
    if sub.empty:
        return pd.DataFrame()
    
    xg_mod = team_xg / 1.22
    team_avg_value = sub['market_value'].replace(0, np.nan).mean() or 1
    
    # Load model
    model_data = None
    models_active = False
    
    if use_xgb:
        model_data = load_football_model()
        if model_data is not None:
            models_active = True
        else:
            st.toast("⚠️ XGB selected but model not loaded. Using Heuristic.", icon="⚠️")
    
    results = []
    
    for _, p in sub.iterrows():
        pos = p['pos']
        total_minutes = safe_get(p, 'total_minutes', 0)
        
        # 1. Get heuristic prediction (returns expected goals/assists per match)
        heur_g, heur_a = predict_heuristic(p, xg_mod, team_avg_value)
        
        # 2. Get PKL model prediction (returns probabilities)
        pkl_prob_g, pkl_prob_a = None, None
        if models_active:
            pkl_prob_g, pkl_prob_a = predict_with_pkl_model(p, model_data, team_xg)
        
        # 3. Ensemble Logic
        if models_active and pkl_prob_g is not None and pkl_prob_a is not None:
            # Convert heuristic to probability for blending
            heur_prob_g = 1 - np.exp(-heur_g)
            heur_prob_a = 1 - np.exp(-heur_a)
            
            # BLEND: 70% PKL Model, 30% Heuristic
            prob_g = pkl_prob_g * 0.70 + heur_prob_g * 0.30
            prob_a = pkl_prob_a * 0.70 + heur_prob_a * 0.30
        else:
            # Heuristic only
            prob_g = 1 - np.exp(-heur_g)
            prob_a = 1 - np.exp(-heur_a)
        
        # Convert to odds
        odds_g = np.clip((1 / max(prob_g, 0.01)) * 1.05, 1.30, 50.0)
        odds_a = np.clip((1 / max(prob_a, 0.01)) * 1.05, 1.40, 50.0)
        
        xg_p90 = safe_get(p, 'shooting_xg_per90', 0)
        xa_p90 = safe_get(p, 'passing_xa_per90', 0)
        
        # --- FLOORS AND CEILINGS LOGIC ---
        
        # Ceilings
        if pos in ['ST', 'CF']: odds_g = min(odds_g, 15.0); odds_a = min(odds_a, 10.0)
        elif pos in ['RW', 'LW', 'CAM']: odds_g = min(odds_g, 20.0); odds_a = min(odds_a, 8.0)
        
        # Floors
        floor_scale = max(0.4, 1.22 / team_xg)
        if pos == 'CB': odds_g = max(odds_g, 10.0 * floor_scale)
        elif pos in ['RB', 'LB']: odds_g = max(odds_g, 8.0 * floor_scale)
        elif pos == 'DM': odds_g = max(odds_g, 6.0 * floor_scale)
        
        # Sanity Floors
        if pos not in ['ST', 'CF', 'RW', 'LW', 'CAM']:
            if xg_p90 < 0.05: odds_g = max(odds_g, 10.0 * floor_scale)
        if pos not in ['ST', 'CF', 'RW', 'LW', 'CAM', 'RM', 'LM']:
            if xa_p90 < 0.03: odds_a = max(odds_a, 15.0 * floor_scale)

        # -----------------------------------------------------------

        results.append({
            'Player': p.get('player_name', 'Unknown'),
            'Pos': pos,
            'ATG': round(odds_g, 2),
            'AST': round(odds_a, 2),
            'pos_sort': POS_SORT.get(pos, 99),
        })
    
    return pd.DataFrame(results).sort_values('pos_sort')

# =============================================================================
# STREAMLIT UI
# =============================================================================

st.set_page_config(page_title=" ", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #1e1e1e; }
    .header-box { background: #252525; padding: 12px; margin-bottom: 12px; border-radius: 6px; font-weight: bold; color: #3794ff; }
    table { width: 100%; border-collapse: collapse; font-family: 'Segoe UI', sans-serif; font-size: 14px; background: #1e1e1e; color: #d4d4d4; }
    table th { background: #333; padding: 10px 12px; text-align: left !important; border-bottom: 2px solid #555; font-weight: 600; }
    table td { padding: 8px 12px; text-align: left !important; border-bottom: 1px solid #3a3a3a; }
    table tr:hover { background: #2d2d2d; }
    table th:first-child, table td:first-child { display: none; }
    @media screen and (max-width: 768px) {
        table { font-size: 11px; }
        table th, table td { padding: 6px 4px; white-space: nowrap; }
        .header-box { font-size: 12px; padding: 8px; }
        .table-container { overflow-x: auto; -webkit-overflow-scrolling: touch; }
    }
    @media screen and (max-width: 480px) {
        table { font-size: 10px; }
        table th, table td { padding: 5px 3px; }
    }
    .player-type { padding: 2px 6px; border-radius: 3px; font-size: 10px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

with st.spinner(""):
    df = load_data()

if df.empty:
    st.error("No data loaded. Check the CSV URLs.")
    st.stop()

## Controls
col1, col2, col3, col4 = st.columns([2, 2, 2, 1])

with col1:
    teams = sorted(df['team'].unique())
    selected_team = st.selectbox("Team", teams, label_visibility="collapsed")

with col2:
    team_xg = st.slider("Team xG", min_value=0.5, max_value=4.0, value=1.15, step=0.05, label_visibility="collapsed")

with col3:
    sort_options = {'ATG': 'ATG', 'AST': 'AST'}
    sort_by = st.selectbox("Sort", list(sort_options.keys()), label_visibility="collapsed")

with col4:
    use_xgb = st.checkbox("xgb", value=False)

# Get predictions
data = predict_odds(df, selected_team, team_xg, use_xgb=use_xgb)

if data.empty:
    st.warning("No players found for this team")
    st.stop()

data = data.sort_values(sort_options[sort_by])

# Prepare display - ONLY ATG and AST columns
display_df = data[['Player', 'Pos', 'ATG', 'AST']].reset_index(drop=True)

POS_COLORS = {
    'ST': '#8F0000', 'CF': '#8F0000', 'RW': '#8D4E28', 'LW': '#8D4E28',
    'CAM': '#5a8a5a', 'RM': '#237023', 'LM': '#237023', 'CM': '#0B5E0B', 
    'DM': '#167D63', 'RWB': '#6a7a8a', 'LWB': '#6a7a8a',
    'RB': '#07166D', 'LB': '#07166D', 'CB': '#14316F', 'GK': '#9a8a5a'
}

def color_goal_odds(val):
    if val < 4: return 'color: #4ec9b0; font-weight: bold'
    elif val < 8: return 'color: #dcdcaa'
    else: return 'color: #888'

def color_assist_odds(val):
    if val < 4: return 'color: #4ec9b0; font-weight: bold'
    elif val < 8: return 'color: #dcdcaa'
    else: return 'color: #888'

def color_position(val):
    bg_color = POS_COLORS.get(val, '#666')
    return f'background-color: {bg_color}; color: white; font-weight: bold; padding: 2px 6px; border-radius: 4px'

styled_df = display_df.style.applymap(
    color_goal_odds, subset=['ATG']
).applymap(
    color_assist_odds, subset=['AST']
).applymap(
    color_position, subset=['Pos']
).set_properties(**{
    'text-align': 'left'
}).set_table_styles([
    {'selector': 'th', 'props': [('text-align', 'left')]},
    {'selector': 'td', 'props': [('text-align', 'left')]}
]).format({
    'ATG': '{:.2f}',
    'AST': '{:.2f}',
})

st.markdown(f'<div class="table-container">{styled_df.to_html()}</div>', unsafe_allow_html=True)

# =============================================================================
# TACTICAL PROFILE SECTION
# =============================================================================

def get_simple_pos(pos_name):
    p = str(pos_name).lower()
    if 'goalkeeper' in p or 'keeper' in p: return 'GK'
    if 'defender' in p or 'back' in p: return 'DEF'
    if 'midfield' in p or 'winger' in p: return 'MID'
    if 'forward' in p or 'striker' in p: return 'FWD'
    return 'MID'

def generate_player_list(df_source, is_aggregate=False):
    if is_aggregate:
        return df_source.groupby('player_name').agg({
            'is_starter': 'sum', 'minutes_played': 'sum', 'rating': 'mean',
            'goals_in_match': 'sum', 'assists_in_match': 'sum', 'position_name': 'first'
        }).reset_index().rename(columns={'is_starter': 'starts'})
    return df_source

def generate_tactical_profile(df_lineups, team):
    team_df = df_lineups[df_lineups['team'] == team].copy()
    if team_df.empty: return ""
    
    agg_df = generate_player_list(team_df, is_aggregate=True)
    agg_df['pos_simple'] = agg_df['position_name'].apply(get_simple_pos)
    
    total_matches = team_df['match_id'].nunique()
    pos_counts = agg_df['pos_simple'].value_counts()
    
    cards = []
    for pos_key in ['FWD', 'MID', 'DEF', 'GK']:
        count = pos_counts.get(pos_key, 0)
        pos_players = agg_df[agg_df['pos_simple'] == pos_key].nlargest(3, 'starts')
        top_names = ", ".join(pos_players['player_name'].tolist()[:3]) or "-"
        color = TACTIC_POS_COLORS.get(pos_key, '#666')
        cards.append(f"<div style='flex:1; min-width:100px; background:#2a2a2a; padding:8px; border-radius:4px; border-left:3px solid {color};'><div style='font-size:10px; color:#888;'>{pos_key}</div><div style='font-size:16px; font-weight:bold; color:{color};'>{count}</div><div style='font-size:9px; color:#aaa; margin-top:2px;'>{top_names}</div></div>")
    cards_html = "".join(cards)
    
    core_stats = agg_df.nlargest(11, 'starts')
    
    core_rows = ""
    for _, row in core_stats.iterrows():
        g_val = f"<span style='color:#4ec9b0; font-weight:bold;'>{int(row['goals_in_match'])}</span>" if row['goals_in_match'] > 0 else "<span style='color:#444;'>-</span>"
        a_val = f"<span style='color:#4ec9b0; font-weight:bold;'>{int(row['assists_in_match'])}</span>" if row['assists_in_match'] > 0 else "<span style='color:#444;'>-</span>"
        p_color = TACTIC_POS_COLORS.get(row['pos_simple'], '#888')
        pos_badge = f"<span style='color:{p_color}; font-weight:bold;'>{row['pos_simple']}</span>"
        core_rows += f"<tr style='border-bottom: 1px solid #333; height: 22px;'><td style='color:#ddd; font-weight:500;'>{row['player_name']}</td><td style='text-align:center; font-size:10px;'>{pos_badge}</td><td style='text-align:right;'>{int(row['starts'])}</td><td style='text-align:right; color:#888;'>{int(row['minutes_played'])}'</td><td style='text-align:right; color:#aaa;'>{row['rating']:.1f}</td><td style='text-align:right;'>{g_val}</td><td style='text-align:right;'>{a_val}</td></tr>"
    
    return f"<div class='dash-container'><div class='header-title'>{team}</div><div class='flex-row'>{cards_html}<div style='flex: 0 0 350px; background: #222; padding: 10px; border-radius: 4px; border:1px solid #333;'><div style='color: #3794ff; font-weight: bold; font-size: 12px; margin-bottom: 8px;'></div><table class='core-table'><tr><th>Player</th><th>Pos</th><th>St</th><th>Min</th><th>Rat</th><th>G</th><th>A</th></tr>{core_rows}</table></div></div></div>"

df_lineups = load_lineups_data()
if not df_lineups.empty and selected_team in df_lineups['team'].values:
    st.markdown("---")
    tactical_html = generate_tactical_profile(df_lineups, selected_team)
    full_html = f"<html><head><style>body {{ margin: 0; padding: 0; background: #1e1e1e; }}.dash-container {{ font-family: 'Segoe UI', sans-serif; background: #1e1e1e; padding: 15px; color: #d4d4d4; overflow-x: auto; }}.header-title {{ color: #3794ff; font-weight: bold; font-size: 14px; margin-bottom: 10px; }}.flex-row {{ display: flex; gap: 10px; width: 100%; flex-wrap: wrap; }}.core-table {{ width: 100%; border-collapse: collapse; font-size: 11px; }}.core-table th {{ text-align: right; color: #666; font-weight: normal; padding-bottom: 4px; border-bottom: 1px solid #444; }}.core-table th:first-child {{ text-align: left; }}.core-table th:nth-child(2) {{ text-align: center; }}.core-table td {{ padding: 4px 2px; }}@media screen and (max-width: 768px) {{ .flex-row {{ flex-direction: column; }} .flex-row > div {{ flex: 1 1 100% !important; min-width: 100% !important; }} }}</style></head><body>{tactical_html}</body></html>"
    components.html(full_html, height=450, scrolling=True)

# =============================================================================
# SUSPENSIONS, INJURIES & RISK TABLE
# =============================================================================

FOOTBALL_DATA_URL = "https://raw.githubusercontent.com/sznajdr/asttt/refs/heads/main/football_data_complete_2025-12-03.csv"

@st.cache_data
def load_football_status_data():
    try:
        df = pd.read_csv(FOOTBALL_DATA_URL)
        cols_to_keep = ['competition', 'club', 'player', 'position', 'reason', 'matches_missed', 'age', 'injury', 'yellow_cards', 'injured_since', 'injured_until', 'data_type', 'since', 'until']
        return df[[c for c in cols_to_keep if c in df.columns]]
    except: return pd.DataFrame()

st.markdown("---")
df_status = load_football_status_data()

if not df_status.empty:
    filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)
    with filter_col1:
        competitions = sorted(df_status['competition'].dropna().unique().tolist())
        selected_competitions = st.multiselect(" ", competitions, key="filter_competition")
    with filter_col2:
        positions = sorted(df_status['position'].dropna().unique().tolist())
        default_positions = [p for p in ['Centre-Forward', 'Attacking Midfield', 'Right Winger', 'Left Winger', 'Central Midfield', 'Mittelstürmer', 'Linksaußen', 'Offensives Mittelfeld', 'Rechtsaußen', 'Second Striker'] if p in positions]
        selected_positions = st.multiselect(" ", positions, default=default_positions, key="filter_position")
    with filter_col3:
        reasons = sorted(df_status['reason'].dropna().unique().tolist())
        selected_reasons = st.multiselect(" ", reasons, key="filter_reason")
    with filter_col4:
        data_types = sorted(df_status['data_type'].dropna().unique().tolist())
        default_types = ['suspensions'] if 'suspensions' in data_types else []
        selected_types = st.multiselect(" ", data_types, default=default_types, key="filter_type")
    
    df_filtered = df_status.copy()
    if selected_competitions: df_filtered = df_filtered[df_filtered['competition'].isin(selected_competitions)]
    if selected_positions: df_filtered = df_filtered[df_filtered['position'].isin(selected_positions)]
    if selected_reasons: df_filtered = df_filtered[df_filtered['reason'].isin(selected_reasons)]
    if selected_types: df_filtered = df_filtered[df_filtered['data_type'].isin(selected_types)]
    
    def color_data_type(val):
        if val == 'suspensions': return 'color: #dc3545; font-weight: bold'
        elif val == 'risk_of_suspension': return 'color: #ffc107; font-weight: bold'
        elif val == 'injuries': return 'color: #17a2b8; font-weight: bold'
        return 'color: #888'
    
    styled_status_df = df_filtered.reset_index(drop=True).style.applymap(color_data_type, subset=['data_type']).set_properties(**{'text-align': 'left'}).set_table_styles([{'selector': 'th', 'props': [('text-align', 'left')]}, {'selector': 'td', 'props': [('text-align', 'left')]}])
    st.markdown(f'<div class="table-container">{styled_status_df.to_html()}</div>', unsafe_allow_html=True)

# =============================================================================
# LIVE INJURY SCRAPER
# =============================================================================
import requests
from bs4 import BeautifulSoup

INJURY_LEAGUES = {
    'Bundesliga (GER)': 'https://www.soccerdonna.de/de/bundesliga/verletzt/wettbewerb_BL1.html',
    'Premiére Ligue (DEN)': 'https://www.soccerdonna.de/de/premire-ligue/verletzt/wettbewerb_DAN1.html',
    'WSL (ENG)': 'https://www.soccerdonna.de/de/womens-super-league/verletzt/wettbewerb_ENG1.html',
    'Serie A (ITA)': 'https://www.soccerdonna.de/de/serie-a-women/verletzt/wettbewerb_IT1.html',
    'Primera División (ESP)': 'https://www.soccerdonna.de/de/primera-division-femenina/verletzt/wettbewerb_ESP1.html',
}

@st.cache_data(ttl=3600)
def fetch_injury_data(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        all_data = []
        def parse_table(tid):
            table = soup.find('table', id=tid)
            if not table: return []
            rows = table.find_all('tr', class_=['hell', 'dunkel'])
            extracted = []
            for row in rows:
                cols = row.find_all('td', recursive=False)
                if len(cols) < 6: continue
                p_name = 'N/A'
                nested = cols[0].find('table')
                if nested and nested.find('a', class_='fb s10'): p_name = nested.find('a', class_='fb s10').text.strip()
                club = cols[1].find('a')['title'].strip() if cols[1].find('a') else 'N/A'
                injury = cols[4].text.strip()
                if 'unbekannte Verletzung' in injury: injury = 'Erdbeerwoche'
                since = cols[5].text.strip()
                pos = 'N/A'
                if nested:
                    trs = nested.find_all('tr')
                    if len(trs) > 1: pos = trs[1].text.strip()
                extracted.append({'club': club, 'name': p_name, 'pos': pos, 'injury': injury, 'since': since})
            return extracted
        all_data.extend(parse_table('reha'))
        all_data.extend(parse_table('verletzt'))
        return pd.DataFrame(all_data).sort_values(by='club')
    except: return pd.DataFrame(columns=['club', 'name', 'pos', 'injury', 'since'])

st.markdown("---")
injury_col1, injury_col2 = st.columns(2)
with injury_col1:
    selected_injury_league = st.selectbox("League", list(INJURY_LEAGUES.keys()), key="injury_league", label_visibility="collapsed")

df_injuries = fetch_injury_data(INJURY_LEAGUES[selected_injury_league])

with injury_col2:
    if not df_injuries.empty:
        teams = ['All'] + sorted(df_injuries['club'].dropna().unique().tolist())
        default_team_idx = teams.index('1. FC Köln') if '1. FC Köln' in teams else 0
        selected_injury_team = st.selectbox("Team", teams, index=default_team_idx, key="injury_team", label_visibility="collapsed")
    else:
        selected_injury_team = 'All'

if not df_injuries.empty:
    if selected_injury_team != 'All': df_injuries = df_injuries[df_injuries['club'] == selected_injury_team]
    styled_injuries = df_injuries.reset_index(drop=True).style.set_properties(**{'text-align': 'left'}).set_table_styles([{'selector': 'th', 'props': [('text-align', 'left')]}, {'selector': 'td', 'props': [('text-align', 'left')]}])
    st.markdown(f'<div class="table-container">{styled_injuries.to_html()}</div>', unsafe_allow_html=True)
else:
    st.info("No injury data available for this league.")
