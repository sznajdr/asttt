# =============================================================================
# GOALSCORER ODDS PREDICTION - STREAMLIT APP
# =============================================================================
# 
# Deploy to Streamlit Cloud:
# 1. Push this file + model_artifacts/ to your GitHub repo
# 2. Go to streamlit.io/cloud
# 3. Connect your repo and deploy
#
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
from pathlib import Path

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="Goalscorer Odds",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CONSTANTS
# =============================================================================
POS_GOAL_WEIGHT = {
    'striker': 1.00, 'centerforward': 1.00, 'centerattackingmidfielder': 0.73,
    'leftwinger': 0.68, 'rightwinger': 0.67, 'right_wing_back': 0.47,
    'left_wing_back': 0.09, 'centermidfielder': 0.33, 'centerdefensivemidfielder': 0.26,
    'rightmidfielder': 0.23, 'leftmidfielder': 0.22, 'leftback': 0.21,
    'rightback': 0.20, 'centerback': 0.21, 'keeper': 0.00, 'keeper_long': 0.00,
}

POS_ASSIST_WEIGHT = {
    'leftwinger': 1.00, 'right_wing_back': 0.65, 'rightwinger': 0.93,
    'centerattackingmidfielder': 0.85, 'left_wing_back': 0.65, 'centermidfielder': 0.7,
    'rightback': 0.5, 'leftback': 0.5, 'centerdefensivemidfielder': 0.52,
    'leftmidfielder': 0.55, 'striker': 0.81, 'centerforward': 0.81,
    'rightmidfielder': 0.55, 'centerback': 0.24, 'keeper': 0.02, 'keeper_long': 0.02,
}

POS_ABBREV = {
    'striker': 'ST', 'centerforward': 'CF', 'leftwinger': 'LW', 'rightwinger': 'RW',
    'centerattackingmidfielder': 'CAM', 'leftmidfielder': 'LM', 'rightmidfielder': 'RM',
    'centermidfielder': 'CM', 'centerdefensivemidfielder': 'CDM', 'leftback': 'LB',
    'rightback': 'RB', 'left_wing_back': 'LWB', 'right_wing_back': 'RWB',
    'centerback': 'CB', 'keeper': 'GK', 'keeper_long': 'GK', 'goalkeeper': 'GK'
}

# Position bounds for odds
POS_BOUNDS = {
    'goal': {
        'striker': (1.5, 50), 'centerforward': (1.5, 50), 'leftwinger': (2, 80),
        'rightwinger': (2, 80), 'centerattackingmidfielder': (2.5, 100),
        'centermidfielder': (5, 150), 'centerdefensivemidfielder': (8, 200),
        'leftback': (10, 200), 'rightback': (10, 200), 'centerback': (10, 200),
        'left_wing_back': (6, 150), 'right_wing_back': (6, 150),
        'keeper': (100, 500), 'keeper_long': (100, 500),
    },
    'assist': {
        'striker': (3, 80), 'centerforward': (3, 80), 'leftwinger': (2, 50),
        'rightwinger': (2, 50), 'centerattackingmidfielder': (2, 50),
        'centermidfielder': (3, 80), 'centerdefensivemidfielder': (5, 150),
        'leftback': (5, 150), 'rightback': (5, 150), 'centerback': (15, 300),
        'left_wing_back': (3, 100), 'right_wing_back': (3, 100),
        'keeper': (150, 500), 'keeper_long': (150, 500),
    }
}


# =============================================================================
# MODEL WRAPPER
# =============================================================================
class CalibratedModel:
    """Wrapper that combines raw model + isotonic calibrator"""
    def __init__(self, model, calibrator):
        self.model = model
        self.calibrator = calibrator
    
    def predict_proba(self, X):
        raw = self.model.predict_proba(X)[:, 1]
        calibrated = self.calibrator.predict(raw)
        return np.column_stack([1 - calibrated, calibrated])


# =============================================================================
# LOAD MODELS (CACHED)
# =============================================================================
@st.cache_resource
def load_models():
    """Load models from model_artifacts folder"""
    artifact_dir = Path("model_artifacts")
    
    # Load goal model
    goal_bundle = joblib.load(artifact_dir / "goal_model_bundle.pkl")
    goal_model = CalibratedModel(goal_bundle['model'], goal_bundle['calibrator'])
    goal_features = goal_bundle['features']
    
    # Load assist model
    assist_bundle = joblib.load(artifact_dir / "assist_model_bundle.pkl")
    assist_model = CalibratedModel(assist_bundle['model'], assist_bundle['calibrator'])
    assist_features = assist_bundle['features']
    
    # Load player profile
    player_profile = pd.read_pickle(artifact_dir / "player_profile.pkl")
    
    # Load metadata
    with open(artifact_dir / "model_metadata.json", 'r') as f:
        metadata = json.load(f)
    
    return goal_model, assist_model, goal_features, assist_features, player_profile, metadata


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def get_pos_abbrev(position):
    return POS_ABBREV.get(str(position).lower().strip(), 'MF')


def prob_to_odds(prob):
    return 1 / prob if prob > 0 else 500.0


def apply_position_bounds(odds, position, market='goal'):
    """Apply position-specific odds bounds"""
    pos_lower = str(position).lower().strip()
    bounds = POS_BOUNDS.get(market, {}).get(pos_lower, (1.5, 200))
    return max(bounds[0], min(bounds[1], odds))


def fuzzy_find_team(query, team_list):
    """Find team by name (exact, substring, or fuzzy)"""
    if not isinstance(query, str):
        return None
    
    query_lower = query.lower().strip()
    
    # Exact match
    for team in team_list:
        if pd.notna(team) and isinstance(team, str):
            if team.lower().strip() == query_lower:
                return team
    
    # Substring match
    for team in team_list:
        if pd.notna(team) and isinstance(team, str):
            if query_lower in team.lower():
                return team
    
    # Reverse substring
    for team in team_list:
        if pd.notna(team) and isinstance(team, str):
            if team.lower() in query_lower:
                return team
    
    return None


# =============================================================================
# MARKET SCALING (from original notebook)
# =============================================================================
def apply_market_scaling(df, market_xg):
    """Scale probabilities to match expected team goals"""
    if df is None or len(df) == 0:
        return df
    
    df = df.copy()
    
    # Goal scaling
    g_probs = df['goal_prob'] / 100
    g_lambdas = -np.log(1 - g_probs.clip(upper=0.99))
    implied_goals = g_lambdas.sum()
    
    if implied_goals > 0 and market_xg > implied_goals:
        xg_weights = (df['xg_per90'].fillna(0) + 0.05)
        xg_weights = xg_weights / xg_weights.sum()
        base_factor = market_xg / implied_goals
        weight_boost = 1 + (xg_weights * len(df) - 1) * 0.2
        new_lambdas = g_lambdas * base_factor * weight_boost
        df['goal_prob'] = (1 - np.exp(-new_lambdas)) * 100
    
    # Assist scaling
    a_probs = df['assist_prob'] / 100
    a_lambdas = -np.log(1 - a_probs.clip(upper=0.99))
    implied_assists = a_lambdas.sum()
    target_assists = market_xg * 0.88
    
    if implied_assists > 0 and target_assists > implied_assists:
        xa_weights = (df['xa_per90'].fillna(0) + 0.02)
        xa_weights = xa_weights / xa_weights.sum()
        base_factor = target_assists / implied_assists
        weight_boost = 1 + (xa_weights * len(df) - 1) * 0.3
        new_lambdas = a_lambdas * base_factor * weight_boost
        df['assist_prob'] = (1 - np.exp(-new_lambdas)) * 100
    
    # Recalculate odds
    df['goal_odds'] = df.apply(
        lambda x: apply_position_bounds(prob_to_odds(x['goal_prob']/100), x['position'], 'goal'), axis=1)
    df['assist_odds'] = df.apply(
        lambda x: apply_position_bounds(prob_to_odds(x['assist_prob']/100), x['position'], 'assist'), axis=1)
    
    return df


# =============================================================================
# MAIN INFERENCE FUNCTION
# =============================================================================
def get_team_odds(team_name, market_xg, goal_model, assist_model, 
                  goal_features, assist_features, player_profile, min_minutes=50):
    """Generate goal/assist odds for all players on a team"""
    
    # Find team
    teams = player_profile['pf_team'].dropna().unique()
    matched_team = fuzzy_find_team(team_name, teams)
    
    if matched_team is None:
        return None, f"Team '{team_name}' not found"
    
    # Get players
    team_players = player_profile[player_profile['pf_team'] == matched_team].copy()
    if 'sm_total_minutes' in team_players.columns:
        team_players = team_players[team_players['sm_total_minutes'] >= min_minutes]
    
    if len(team_players) == 0:
        return None, f"No players found for {matched_team}"
    
    results = []
    
    for _, player in team_players.iterrows():
        position = player.get('pf_position', 'midfielder')
        
        if str(position) in ['0', 'nan', '', 'None', '0.0']:
            continue
        
        # Build feature vectors
        goal_row = {f: player.get(f, 0) for f in goal_features}
        goal_row['match_is_starter'] = 1
        goal_row['match_pos_goal_weight'] = POS_GOAL_WEIGHT.get(str(position).lower(), 0.3)
        goal_row['pos_goal_weight'] = POS_GOAL_WEIGHT.get(str(position).lower(), 0.3)
        
        assist_row = {f: player.get(f, 0) for f in assist_features}
        assist_row['match_is_starter'] = 1
        assist_row['match_pos_assist_weight'] = POS_ASSIST_WEIGHT.get(str(position).lower(), 0.3)
        assist_row['pos_assist_weight'] = POS_ASSIST_WEIGHT.get(str(position).lower(), 0.3)
        
        try:
            X_goal = pd.DataFrame([goal_row])[goal_features].fillna(0)
            X_assist = pd.DataFrame([assist_row])[assist_features].fillna(0)
            
            goal_prob = goal_model.predict_proba(X_goal)[0, 1]
            assist_prob = assist_model.predict_proba(X_assist)[0, 1]
            
            goal_odds = apply_position_bounds(prob_to_odds(goal_prob), position, 'goal')
            assist_odds = apply_position_bounds(prob_to_odds(assist_prob), position, 'assist')
            
            results.append({
                'name': player.get('pf_name', 'Unknown'),
                'position': position,
                'pos': get_pos_abbrev(position),
                'goal_prob': round(goal_prob * 100, 1),
                'goal_odds': round(goal_odds, 2),
                'assist_prob': round(assist_prob * 100, 1),
                'assist_odds': round(assist_odds, 2),
                'xg_per90': round(player.get('sm_xg_per90', 0) or 0, 2),
                'xa_per90': round(player.get('sm_xa_per90', 0) or 0, 2),
            })
        except Exception as e:
            continue
    
    if not results:
        return None, f"Could not generate predictions for {matched_team}"
    
    df = pd.DataFrame(results)
    df = apply_market_scaling(df, market_xg)
    
    return df.sort_values('goal_odds'), matched_team


# =============================================================================
# STREAMLIT APP
# =============================================================================
def main():
    # Load models
    try:
        goal_model, assist_model, goal_features, assist_features, player_profile, metadata = load_models()
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        st.info("Make sure model_artifacts/ folder is in the same directory as app.py")
        return
    
    # Header
    st.title("⚽ Goalscorer & Assist Odds")
    st.markdown("*Predict anytime goalscorer and assist odds for football matches*")
    
    # Sidebar - Team Selection
    st.sidebar.header("Match Setup")
    
    # Get unique teams
    teams = sorted([t for t in player_profile['pf_team'].dropna().unique() if isinstance(t, str)])
    
    # Team selection with search
    home_team = st.sidebar.selectbox("Home Team", teams, index=0)
    away_team = st.sidebar.selectbox("Away Team", teams, index=min(1, len(teams)-1))
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Expected Goals (xG)")
    
    home_xg = st.sidebar.slider("Home xG", 0.5, 4.0, 1.8, 0.1)
    away_xg = st.sidebar.slider("Away xG", 0.5, 4.0, 1.4, 0.1)
    
    # Generate predictions button
    if st.sidebar.button("🎯 Generate Odds", type="primary", use_container_width=True):
        st.session_state['generate'] = True
    
    # Model info in sidebar
    st.sidebar.markdown("---")
    st.sidebar.caption(f"**Model Info**")
    st.sidebar.caption(f"Goal AUC: {metadata['goal_model']['test_auc']:.3f}")
    st.sidebar.caption(f"Assist AUC: {metadata['assist_model']['test_auc']:.3f}")
    st.sidebar.caption(f"Trained: {metadata['data_info']['train_date_range']}")
    
    # Main content
    if st.session_state.get('generate', False) or True:  # Always show on load
        
        col1, col2 = st.columns(2)
        
        # Home team
        with col1:
            home_df, home_matched = get_team_odds(
                home_team, home_xg, goal_model, assist_model,
                goal_features, assist_features, player_profile
            )
            
            if home_df is not None:
                st.subheader(f"🏠 {home_matched}")
                st.caption(f"Expected Goals: {home_xg}")
                
                # Goalscorers
                st.markdown("**⚽ Anytime Goalscorer**")
                goal_df = home_df.nsmallest(12, 'goal_odds')[['name', 'pos', 'goal_odds']].copy()
                goal_df.columns = ['Player', 'Pos', 'Odds']
                st.dataframe(goal_df, hide_index=True, use_container_width=True)
                
                # Assists
                st.markdown("**🎯 Anytime Assist**")
                assist_df = home_df.nsmallest(12, 'assist_odds')[['name', 'pos', 'assist_odds']].copy()
                assist_df.columns = ['Player', 'Pos', 'Odds']
                st.dataframe(assist_df, hide_index=True, use_container_width=True)
            else:
                st.error(home_matched)  # Error message
        
        # Away team
        with col2:
            away_df, away_matched = get_team_odds(
                away_team, away_xg, goal_model, assist_model,
                goal_features, assist_features, player_profile
            )
            
            if away_df is not None:
                st.subheader(f"✈️ {away_matched}")
                st.caption(f"Expected Goals: {away_xg}")
                
                # Goalscorers
                st.markdown("**⚽ Anytime Goalscorer**")
                goal_df = away_df.nsmallest(12, 'goal_odds')[['name', 'pos', 'goal_odds']].copy()
                goal_df.columns = ['Player', 'Pos', 'Odds']
                st.dataframe(goal_df, hide_index=True, use_container_width=True)
                
                # Assists
                st.markdown("**🎯 Anytime Assist**")
                assist_df = away_df.nsmallest(12, 'assist_odds')[['name', 'pos', 'assist_odds']].copy()
                assist_df.columns = ['Player', 'Pos', 'Odds']
                st.dataframe(assist_df, hide_index=True, use_container_width=True)
            else:
                st.error(away_matched)
        
        # Full data expander
        st.markdown("---")
        with st.expander("📊 View Full Data"):
            tab1, tab2 = st.tabs([f"{home_matched}", f"{away_matched}"])
            
            with tab1:
                if home_df is not None:
                    st.dataframe(home_df, hide_index=True, use_container_width=True)
            
            with tab2:
                if away_df is not None:
                    st.dataframe(away_df, hide_index=True, use_container_width=True)


if __name__ == "__main__":
    main()
