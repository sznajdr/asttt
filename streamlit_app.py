import pandas as pd
import numpy as np
import streamlit as st
import pickle
import requests
from io import BytesIO
import warnings
import sys

# Suppress warnings
warnings.filterwarnings('ignore')

# Page Config
st.set_page_config(page_title="Goalscorer Odds", layout="wide")

# =============================================================================
# 1. CLASS DEFINITIONS (CRITICAL FOR PICKLE LOADING)
# =============================================================================

# Define the classes exactly as they likely appear in the pickle file
class ModelConfig:
    def __init__(self):
        self.goal_features = []
        self.assist_features = []
        self.scaler = None
        self.goal_model = None
        self.assist_model = None

class GoalAssistModel:
    def __init__(self, model=None, features=None, scaler=None):
        self.model = model
        self.features = features
        self.scaler = scaler

    def predict_proba(self, X):
        if hasattr(self, 'model') and self.model is not None:
            # Handle internal scaling if present
            if hasattr(self, 'scaler') and self.scaler is not None:
                try:
                    X = self.scaler.transform(X)
                except:
                    pass
            return self.model.predict_proba(X)
        return np.zeros((X.shape[0], 2))

# Helper to map the pickle's expectation of "__main__" to this script
class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "__main__":
            return getattr(sys.modules[__name__], name)
        return super().find_class(module, name)

# =============================================================================
# 2. CONSTANTS & MAPPINGS
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
POS_ENCODING = {'ST': 0, 'CF': 1, 'RW': 2, 'LW': 3, 'CAM': 4, 'RM': 5, 'LM': 6, 'CM': 7, 'DM': 8, 'RWB': 9, 'LWB': 10, 'RB': 11, 'LB': 12, 'CB': 13}

POS_COLORS = {
    'ST': '#8F0000', 'CF': '#8F0000', 'RW': '#8D4E28', 'LW': '#8D4E28',
    'CAM': '#5a8a5a', 'RM': '#237023', 'LM': '#237023', 'CM': '#0B5E0B', 
    'DM': '#167D63', 'RWB': '#6a7a8a', 'LWB': '#6a7a8a',
    'RB': '#07166D', 'LB': '#07166D', 'CB': '#14316F', 'GK': '#9a8a5a'
}

# =============================================================================
# 3. DATA LOADING
# =============================================================================

STATS_URL = "https://raw.githubusercontent.com/sznajdr/fb1/refs/heads/main/fotmob_multi_player_season_stats.csv"
FEATURES_URL = "https://raw.githubusercontent.com/sznajdr/fb1/refs/heads/main/player_features.csv"
MODEL_URL = "https://raw.githubusercontent.com/sznajdr/asttt/refs/heads/main/football_model.pkl"

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

def _parse_market_value(val):
    if pd.isna(val): return 0
    s = str(val).replace('€', '').replace(',', '').strip()
    try:
        if 'M' in s: return float(s.replace('M', '')) * 1_000_000
        elif 'K' in s: return float(s.replace('K', '')) * 1_000
        else: return float(s)
    except: return 0

@st.cache_data
def load_data():
    try:
        df_f = pd.read_csv(FEATURES_URL)
        df_s = pd.read_csv(STATS_URL)
        
        if 'total_minutes' in df_s.columns:
            df_s['total_minutes'] = df_s['total_minutes'].astype(str).str.replace(',', '', regex=False)
            df_s['total_minutes'] = pd.to_numeric(df_s['total_minutes'], errors='coerce').fillna(0)
        
        df_s = df_s.sort_values('total_minutes', ascending=False).drop_duplicates('player_id')
        cols_s = ['player_id', 'player_name', 'team', 'total_minutes', 'total_goals', 'total_assists', 'position', 'position_id', 'primary_position_key']
        cols_s = [c for c in cols_s if c in df_s.columns]
        
        df = pd.merge(df_s[cols_s], df_f, on='player_id', how='inner')
        
        df['pos'] = df.apply(_normalize_pos, axis=1)
        df['pos_sort'] = df['pos'].map(lambda x: POS_SORT.get(x, 99))
        df['market_value'] = df.get('market_value', pd.Series(0)).apply(_parse_market_value)
        
        # Ensure numeric types for essential features
        numeric_cols = [
            'shooting_xg_per90', 'passing_xa_per90', 'shotmap_overperformance',
            'possession_touches_in_opposition_box_per90', 'career_goals_per_appearance',
            'career_assists_per_appearance', 'form_5_goal_rate', 'form_5_assist_rate'
        ]
        for c in numeric_cols:
            if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
            else: df[c] = 0.0

        return df[(df['total_minutes'] >= 1) & (df['pos'] != 'GK')].copy()
    except Exception as e:
        st.error(f"CSV Load Error: {e}")
        return pd.DataFrame()

@st.cache_resource
def load_football_model():
    """Robust model loading with CustomUnpickler"""
    try:
        response = requests.get(MODEL_URL, timeout=30)
        response.raise_for_status()
        
        # Use CustomUnpickler to map __main__ classes to current script classes
        model_data = CustomUnpickler(BytesIO(response.content)).load()
        return model_data, "Success"
    except ImportError:
        return None, "XGBoost library not installed."
    except Exception as e:
        return None, f"Error: {str(e)}"

# =============================================================================
# 4. PREDICTION LOGIC
# =============================================================================

def safe_get(row, col, default=0):
    val = row.get(col, default)
    if pd.isna(val): return default
    return float(val)

def get_model_attr(container, key, default=None):
    """Safely extracts attributes from either a Dictionary or an Object."""
    if isinstance(container, dict):
        return container.get(key, default)
    else:
        return getattr(container, key, default)

def predict_with_pkl_model(player_row, model_data, team_xg=1.5):
    """XGBoost Prediction"""
    if model_data is None: return None, None
    
    try:
        goal_model = get_model_attr(model_data, 'goal_model')
        assist_model = get_model_attr(model_data, 'assist_model')
        scaler = get_model_attr(model_data, 'scaler')
        
        # Feature extraction
        goal_features = get_model_attr(model_data, 'goal_features', [])
        if not goal_features and hasattr(goal_model, 'features'): goal_features = goal_model.features
            
        assist_features = get_model_attr(model_data, 'assist_features', [])
        if not assist_features and hasattr(assist_model, 'features'): assist_features = assist_model.features

        if not goal_model or not assist_model or not goal_features:
            return None, None

        # Build feature arrays
        X_goal = np.array([safe_get(player_row, f, 0) for f in goal_features]).reshape(1, -1)
        X_assist = np.array([safe_get(player_row, f, 0) for f in assist_features]).reshape(1, -1)
        
        # Apply top-level scaler if exists
        if scaler:
            try:
                X_goal = scaler.transform(X_goal)
                X_assist = scaler.transform(X_assist)
            except: pass

        # Predict
        goal_prob = goal_model.predict_proba(X_goal)[0, 1]
        assist_prob = assist_model.predict_proba(X_assist)[0, 1]
        
        # Modifiers
        xg_mod = team_xg / 1.5
        return (
            np.clip(goal_prob * xg_mod, 0.001, 0.95),
            np.clip(assist_prob * xg_mod * 0.9, 0.001, 0.90)
        )
    except Exception:
        return None, None

def predict_heuristic(player_row, team_xg_mod=1.0, team_avg_value=1):
    """Heuristic Prediction (Fallback)"""
    pos = player_row['pos']
    tm = safe_get(player_row, 'total_minutes', 0)
    
    # Goal Stats
    xg = safe_get(player_row, 'shooting_xg_per90', 0)
    career_g = safe_get(player_row, 'career_goals_per_appearance', 0)
    
    # Assist Stats
    xa = safe_get(player_row, 'passing_xa_per90', 0)
    career_a = safe_get(player_row, 'career_assists_per_appearance', 0)
    
    # Blending
    w = min(tm / 900, 1.0)
    g_exp = (xg * w + career_g * (1-w)) * team_xg_mod * 1.1 
    a_exp = (xa * w + career_a * (1-w)) * team_xg_mod * 0.9

    # Positional Baselines
    g_base = GOAL_BASELINES.get(pos, 0.1)
    a_base = ASSIST_BASELINES.get(pos, 0.1)
    
    return (g_exp * 0.7 + g_base * 0.3), (a_exp * 0.7 + a_base * 0.3)

def generate_odds(df, team, team_xg=1.5, use_xgb=True, model_data=None):
    sub = df[df['team'] == team].copy()
    if sub.empty: return pd.DataFrame()
    
    xg_mod = team_xg / 1.22
    avg_val = sub['market_value'].replace(0, np.nan).mean() or 1
    
    results = []
    
    for _, p in sub.iterrows():
        # Heuristic
        h_g, h_a = predict_heuristic(p, xg_mod, avg_val)
        h_pg = 1 - np.exp(-h_g)
        h_pa = 1 - np.exp(-h_a)
        
        # XGB
        x_pg, x_pa = None, None
        if use_xgb and model_data:
            x_pg, x_pa = predict_with_pkl_model(p, model_data, team_xg)
            
        # Ensemble (70% XGB if available)
        if x_pg is not None:
            final_pg = x_pg * 0.7 + h_pg * 0.3
            final_pa = x_pa * 0.7 + h_pa * 0.3
        else:
            final_pg = h_pg
            final_pa = h_pa
            
        # Convert to Odds
        odds_g = np.clip((1 / max(final_pg, 0.01)) * 1.05, 1.30, 50.0)
        odds_a = np.clip((1 / max(final_pa, 0.01)) * 1.05, 1.40, 50.0)
        
        results.append({
            'Player': p.get('player_name', 'Unknown'),
            'Pos': p['pos'],
            'ATG': round(odds_g, 2),
            'AST': round(odds_a, 2),
            'pos_sort': p['pos_sort']
        })
        
    return pd.DataFrame(results).sort_values('pos_sort')

# =============================================================================
# 5. UI IMPLEMENTATION
# =============================================================================

# Load Data
df = load_data()

if df.empty:
    st.error("Data could not be loaded.")
    st.stop()

# Controls
c1, c2, c3, c4 = st.columns([2, 2, 2, 1])
with c1:
    team = st.selectbox("Team", sorted(df['team'].unique()), label_visibility="collapsed")
with c2:
    txg = st.slider("Team xG", 0.5, 4.0, 1.5, 0.1, label_visibility="collapsed")
with c3:
    sort = st.selectbox("Sort", ["ATG", "AST"], label_visibility="collapsed")
with c4:
    use_xgb = st.checkbox("XGB", value=True)

# Load Model
model_obj, status_msg = load_football_model()

# Feedback to User
if use_xgb:
    if "Success" in status_msg:
        st.caption(f"✅ Model Active | Status: {status_msg}")
    else:
        st.caption(f"⚠️ Model Failed (Using Heuristic) | Status: {status_msg}")

# Predict
odds_df = generate_odds(df, team, txg, use_xgb, model_obj)

# Display
if not odds_df.empty:
    odds_df = odds_df.sort_values(sort)
    
    # Styling
    def style_odds(v):
        color = '#4ec9b0' if v < 4.0 else '#dcdcaa' if v < 8.0 else '#888'
        weight = 'bold' if v < 4.0 else 'normal'
        return f'color: {color}; font-weight: {weight}'
        
    def style_pos(v):
        return f'background-color: {POS_COLORS.get(v, "#333")}; color: white; border-radius: 4px; padding: 2px 5px; font-weight: bold;'

    st.dataframe(
        odds_df[['Player', 'Pos', 'ATG', 'AST']].style
        .applymap(style_odds, subset=['ATG', 'AST'])
        .applymap(style_pos, subset=['Pos'])
        .format({'ATG': '{:.2f}', 'AST': '{:.2f}'}),
        use_container_width=True,
        hide_index=True
    )
else:
    st.info("No players found.")
