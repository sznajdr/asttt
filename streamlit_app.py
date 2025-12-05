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
# 1. HARDCODED FEATURE LIST (FALLBACK SAFETY NET)
# =============================================================================
FALLBACK_FEATURES = [
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
    'possession_dribbles_success_rate', 'passing_accurate_long_balls_per90', 'possession_touches_per90',
    'market_value'
]

# =============================================================================
# 2. CLASS DEFINITIONS & UNPICKLER
# =============================================================================

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
            return self.model.predict_proba(X)
        return np.zeros((X.shape[0], 2))

class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "__main__":
            return getattr(sys.modules[__name__], name)
        return super().find_class(module, name)

# =============================================================================
# 3. CONSTANTS
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

# =============================================================================
# 4. DATA & MODEL LOADING
# =============================================================================

STATS_URL = "https://raw.githubusercontent.com/sznajdr/fb1/refs/heads/main/fotmob_multi_player_season_stats.csv"
FEATURES_URL = "https://raw.githubusercontent.com/sznajdr/fb1/refs/heads/main/player_features.csv"
MODEL_URL = "https://raw.githubusercontent.com/sznajdr/fb1/refs/heads/main/football_model.pkl"

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
        df_s = pd.read_csv(STATS_URL)
        
        if 'total_minutes' in df_s.columns:
            df_s['total_minutes'] = pd.to_numeric(df_s['total_minutes'].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
        
        df_s = df_s.sort_values('total_minutes', ascending=False).drop_duplicates('player_id')
        cols = [c for c in ['player_id', 'player_name', 'team', 'total_minutes', 'position', 'position_id', 'primary_position_key', 'market_value'] if c in df_s.columns]
        
        df = pd.merge(df_s[cols], df_f, on='player_id', how='inner')
        df['pos'] = df.apply(_normalize_pos, axis=1)
        df['pos_sort'] = df['pos'].map(lambda x: POS_SORT.get(x, 99))
        
        # Market Value
        def parse_mv(v):
            if pd.isna(v): return 0
            s = str(v).replace('€','').replace(',','').strip()
            if 'M' in s: return float(s.replace('M',''))*1000000
            if 'K' in s: return float(s.replace('K',''))*1000
            try: return float(s)
            except: return 0
        df['market_value'] = df['market_value'].apply(parse_mv)
        
        # Ensure all fallback features exist in DF (fill with 0 if missing)
        for f in FALLBACK_FEATURES:
            if f not in df.columns:
                df[f] = 0.0
            else:
                df[f] = pd.to_numeric(df[f], errors='coerce').fillna(0)

        return df[(df['total_minutes'] >= 1) & (df['pos'] != 'GK')].copy()
    except Exception as e:
        st.error(f"Data Load Error: {e}")
        return pd.DataFrame()

@st.cache_resource
def load_football_model():
    try:
        response = requests.get(MODEL_URL, timeout=30)
        response.raise_for_status()
        model_data = CustomUnpickler(BytesIO(response.content)).load()
        return model_data, "Success"
    except Exception as e:
        return None, str(e)

# =============================================================================
# 5. PREDICTION LOGIC
# =============================================================================

def safe_get(row, col):
    val = row.get(col, 0)
    return float(val) if pd.notna(val) else 0.0

def predict_xgb(player_row, model_data, team_xg, debug):
    """Attempt XGB prediction, return None if fails"""
    try:
        # 1. Unpack Model Data
        if isinstance(model_data, dict):
            goal_model = model_data.get('goal_model')
            assist_model = model_data.get('assist_model')
            scaler = model_data.get('scaler')
            goal_feats = model_data.get('goal_features', [])
            assist_feats = model_data.get('assist_features', [])
        else:
            goal_model = getattr(model_data, 'goal_model', None)
            assist_model = getattr(model_data, 'assist_model', None)
            scaler = getattr(model_data, 'scaler', None)
            goal_feats = getattr(model_data, 'goal_features', [])
            assist_feats = getattr(model_data, 'assist_features', [])

        # 2. Check Integrity
        if not goal_model or not assist_model:
            if debug: st.error("Models missing in pickle object.")
            return None, None
            
        # 3. Aggressive Feature Finding (Robust Conversion to List)
        if hasattr(goal_model, 'feature_names_in_'): 
            goal_feats = list(goal_model.feature_names_in_)
        elif hasattr(goal_model, 'get_booster'):
            try: goal_feats = list(goal_model.get_booster().feature_names)
            except: pass
        
        if hasattr(assist_model, 'feature_names_in_'): 
            assist_feats = list(assist_model.feature_names_in_)
        elif hasattr(assist_model, 'get_booster'):
            try: assist_feats = list(assist_model.get_booster().feature_names)
            except: pass

        # 4. FINAL FALLBACK: If list is empty, use Hardcoded
        if goal_feats is None or len(goal_feats) == 0:
            if debug: st.warning("Using Hardcoded Fallback Features")
            goal_feats = FALLBACK_FEATURES
        
        if assist_feats is None or len(assist_feats) == 0:
            assist_feats = FALLBACK_FEATURES

        # 5. Build Feature Vector
        valid_goal_feats = [f for f in goal_feats if f in player_row.index]
        X_g = np.array([safe_get(player_row, f) for f in valid_goal_feats]).reshape(1, -1)
        
        valid_assist_feats = [f for f in assist_feats if f in player_row.index]
        X_a = np.array([safe_get(player_row, f) for f in valid_assist_feats]).reshape(1, -1)
        
        # 6. Scale (if Scaler exists)
        if scaler:
            try:
                X_g = scaler.transform(X_g)
                X_a = scaler.transform(X_a)
            except:
                pass # If scaling fails, run raw
        
        # 7. Predict
        prob_g = goal_model.predict_proba(X_g)[0, 1]
        prob_a = assist_model.predict_proba(X_a)[0, 1]
        
        # 8. Apply modifiers
        mod = team_xg / 1.5
        return (prob_g * mod, prob_a * mod * 0.9)

    except Exception as e:
        if debug: st.error(f"XGB Logic Error for {player_row.get('player_name', '?')}: {e}")
        return None, None

def predict_heuristic(player_row, team_xg_mod, avg_val):
    pos = player_row['pos']
    tm = safe_get(player_row, 'total_minutes')
    
    xg = safe_get(player_row, 'shooting_xg_per90')
    xa = safe_get(player_row, 'passing_xa_per90')
    cg = safe_get(player_row, 'career_goals_per_appearance')
    ca = safe_get(player_row, 'career_assists_per_appearance')
    
    w = min(tm/900, 1.0)
    g_exp = (xg*w + cg*(1-w)) * team_xg_mod
    a_exp = (xa*w + ca*(1-w)) * team_xg_mod
    
    # Simple Positional Baseline blend
    g_exp = (g_exp * 0.8) + (GOAL_BASELINES.get(pos, 0.1) * 0.2)
    a_exp = (a_exp * 0.8) + (ASSIST_BASELINES.get(pos, 0.1) * 0.2)
    
    return g_exp, a_exp

# =============================================================================
# 6. MAIN APP
# =============================================================================

df = load_data()
model_obj, status_msg = load_football_model()

# --- HEADER CONTROLS ---
c1, c2, c3, c4, c5 = st.columns([2, 2, 2, 1, 1])
with c1: team = st.selectbox("Team", sorted(df['team'].unique()), label_visibility="collapsed")
with c2: txg = st.slider("Team xG", 0.5, 4.0, 1.5, 0.1, label_visibility="collapsed")
with c3: sort = st.selectbox("Sort", ["ATG", "AST"], label_visibility="collapsed")
with c4: use_xgb = st.checkbox("XGB", value=True)
with c5: debug_mode = st.checkbox("Debug", value=False) # Enable to see why XGB fails

# --- STATUS INDICATOR ---
if use_xgb:
    if "Success" in status_msg:
        st.caption(f"✅ Model Loaded")
    else:
        st.caption(f"❌ Model Load Failed: {status_msg}")
        if debug_mode and model_obj:
            st.write("Object Dir:", dir(model_obj))

# --- PROCESS ---
sub = df[df['team'] == team].copy()
if sub.empty: st.stop()

xg_mod = txg / 1.22
avg_val = sub['market_value'].mean()

results = []
for _, p in sub.iterrows():
    # 1. Heuristic
    h_g, h_a = predict_heuristic(p, xg_mod, avg_val)
    h_pg = 1 - np.exp(-h_g)
    h_pa = 1 - np.exp(-h_a)
    
    # 2. XGB
    x_pg, x_pa = None, None
    method = "Heuristic"
    
    if use_xgb and "Success" in status_msg:
        x_pg, x_pa = predict_xgb(p, model_obj, txg, debug_mode)
    
    if x_pg is not None:
        # Ensemble: 70% XGB, 30% Heuristic
        final_pg = (x_pg * 0.7) + (h_pg * 0.3)
        final_pa = (x_pa * 0.7) + (h_pa * 0.3)
        method = "XGB+H"
    else:
        final_pg = h_pg
        final_pa = h_pa
    
    # 3. Odds
    odds_g = np.clip((1 / max(final_pg, 0.001)) * 1.05, 1.01, 50.0)
    odds_a = np.clip((1 / max(final_pa, 0.001)) * 1.05, 1.01, 50.0)
    
    results.append({
        'Player': p.get('player_name', 'Unknown'),
        'Pos': p['pos'],
        'ATG': odds_g,
        'AST': odds_a,
        'Method': method,
        'sort_idx': p['pos_sort']
    })

# --- DISPLAY ---
res_df = pd.DataFrame(results).sort_values('sort_idx')

if sort == "ATG": res_df = res_df.sort_values('ATG')
else: res_df = res_df.sort_values('AST')

def color_odds(v):
    if v < 4.0: return 'color: #4ec9b0; font-weight: bold'
    if v < 8.0: return 'color: #dcdcaa'
    return 'color: #888'

st.dataframe(
    res_df[['Player', 'Pos', 'ATG', 'AST', 'Method']].style
    .applymap(color_odds, subset=['ATG', 'AST'])
    .format({'ATG': '{:.2f}', 'AST': '{:.2f}'}),
    use_container_width=True,
    hide_index=True
)
