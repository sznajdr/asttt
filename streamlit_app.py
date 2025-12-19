# =============================================================================
# GOALSCORER ODDS PREDICTION - STREAMLIT APP
# =============================================================================
# 
# Replicates the Colab notebook dashboard with:
# - Match H2H Preview
# - Team Projections  
# - Player Deep Dive
# - Slate Scanner
#
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
from pathlib import Path
from datetime import datetime

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="⚽ Goalscorer Odds",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        background-color: #262730;
        border-radius: 4px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4;
    }
    .metric-card {
        background-color: #262730;
        padding: 15px;
        border-radius: 8px;
        margin: 5px 0;
    }
    .player-header {
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .tag {
        background-color: #1f77b4;
        padding: 2px 8px;
        border-radius: 4px;
        margin-right: 4px;
        font-size: 12px;
    }
    .value-play {
        background-color: #2d4a3e;
        padding: 2px 6px;
        border-radius: 3px;
    }
</style>
""", unsafe_allow_html=True)

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
    
    # Substring match (query in team)
    for team in team_list:
        if pd.notna(team) and isinstance(team, str):
            if query_lower in team.lower():
                return team
    
    # Reverse substring (team in query)
    for team in team_list:
        if pd.notna(team) and isinstance(team, str):
            if team.lower() in query_lower:
                return team
    
    return None


# =============================================================================
# MARKET SCALING
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
        goal_row = {}
        for f in goal_features:
            if f in player.index:
                goal_row[f] = player.get(f, 0)
            else:
                goal_row[f] = 0
        goal_row['match_is_starter'] = 1
        goal_row['match_pos_goal_weight'] = POS_GOAL_WEIGHT.get(str(position).lower(), 0.3)
        goal_row['pos_goal_weight'] = POS_GOAL_WEIGHT.get(str(position).lower(), 0.3)
        
        assist_row = {}
        for f in assist_features:
            if f in player.index:
                assist_row[f] = player.get(f, 0)
            else:
                assist_row[f] = 0
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
                'player_id': player.get('player_id', 0),
                'name': player.get('pf_name', 'Unknown'),
                'position': position,
                'pos': get_pos_abbrev(position),
                'goal_prob': round(goal_prob * 100, 1),
                'goal_odds': round(goal_odds, 2),
                'assist_prob': round(assist_prob * 100, 1),
                'assist_odds': round(assist_odds, 2),
                'xg_per90': round(player.get('sm_xg_per90', 0) or 0, 2),
                'xa_per90': round(player.get('sm_xa_per90', 0) or 0, 2),
                'shots_per90': round(player.get('sm_shots_per90', 0) or 0, 2),
                'minutes': int(player.get('sm_total_minutes', 0) or 0),
            })
        except Exception as e:
            continue
    
    if not results:
        return None, f"Could not generate predictions for {matched_team}"
    
    df = pd.DataFrame(results)
    df = apply_market_scaling(df, market_xg)
    
    return df.sort_values('goal_odds'), matched_team


# =============================================================================
# TAB 1: HEAD-TO-HEAD MATCH PREVIEW
# =============================================================================
def render_h2h_tab(goal_model, assist_model, goal_features, assist_features, player_profile, teams):
    st.header("⚔️ Head-to-Head Match Preview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        home_team = st.selectbox("🏠 Home Team", teams, key="h2h_home")
        home_xg = st.slider("Home xG", 0.5, 4.0, 1.85, 0.05, key="h2h_home_xg")
    
    with col2:
        away_team = st.selectbox("✈️ Away Team", teams, index=min(1, len(teams)-1), key="h2h_away")
        away_xg = st.slider("Away xG", 0.5, 4.0, 1.45, 0.05, key="h2h_away_xg")
    
    if st.button("🎯 Generate Odds", type="primary", key="h2h_btn"):
        
        home_df, home_matched = get_team_odds(
            home_team, home_xg, goal_model, assist_model,
            goal_features, assist_features, player_profile
        )
        away_df, away_matched = get_team_odds(
            away_team, away_xg, goal_model, assist_model,
            goal_features, assist_features, player_profile
        )
        
        if home_df is None or away_df is None:
            st.error("Could not load one or both teams")
            return
        
        st.markdown(f"### {home_matched} ({home_xg}) vs {away_matched} ({away_xg})")
        
        # Goalscorers
        st.markdown("#### ⚽ Anytime Goalscorer")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**{home_matched}**")
            home_goal = home_df.nsmallest(12, 'goal_odds')[['name', 'pos', 'goal_odds', 'xg_per90']].copy()
            home_goal.columns = ['Player', 'Pos', 'Odds', 'xG/90']
            st.dataframe(home_goal, hide_index=True, use_container_width=True)
        
        with col2:
            st.markdown(f"**{away_matched}**")
            away_goal = away_df.nsmallest(12, 'goal_odds')[['name', 'pos', 'goal_odds', 'xg_per90']].copy()
            away_goal.columns = ['Player', 'Pos', 'Odds', 'xG/90']
            st.dataframe(away_goal, hide_index=True, use_container_width=True)
        
        # Assists
        st.markdown("#### 🎯 Anytime Assist")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**{home_matched}**")
            home_assist = home_df.nsmallest(12, 'assist_odds')[['name', 'pos', 'assist_odds', 'xa_per90']].copy()
            home_assist.columns = ['Player', 'Pos', 'Odds', 'xA/90']
            st.dataframe(home_assist, hide_index=True, use_container_width=True)
        
        with col2:
            st.markdown(f"**{away_matched}**")
            away_assist = away_df.nsmallest(12, 'assist_odds')[['name', 'pos', 'assist_odds', 'xa_per90']].copy()
            away_assist.columns = ['Player', 'Pos', 'Odds', 'xA/90']
            st.dataframe(away_assist, hide_index=True, use_container_width=True)


# =============================================================================
# TAB 2: TEAM PROJECTIONS
# =============================================================================
def render_team_proj_tab(goal_model, assist_model, goal_features, assist_features, player_profile, teams):
    st.header("📊 Team Projections")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        team = st.selectbox("Select Team", teams, key="proj_team")
    with col2:
        team_xg = st.slider("Team xG", 0.5, 4.5, 1.85, 0.05, key="proj_xg")
    
    if st.button("📈 Show Projections", type="primary", key="proj_btn"):
        df, matched = get_team_odds(
            team, team_xg, goal_model, assist_model,
            goal_features, assist_features, player_profile, min_minutes=100
        )
        
        if df is None:
            st.error(f"Team '{team}' not found")
            return
        
        st.markdown(f"### {matched} | Team xG: {team_xg}")
        
        # Filter out goalkeepers
        df = df[df['pos'] != 'GK']
        
        # Create display dataframe
        display_df = df.nsmallest(20, 'goal_odds')[
            ['name', 'pos', 'goal_odds', 'goal_prob', 'assist_odds', 'assist_prob', 'xg_per90', 'xa_per90', 'minutes']
        ].copy()
        
        display_df.columns = ['Player', 'Pos', 'Goal Odds', 'Goal %', 'Assist Odds', 'Assist %', 'xG/90', 'xA/90', 'Minutes']
        
        # Style the dataframe
        def highlight_short_odds(val):
            if isinstance(val, (int, float)):
                if val < 4:
                    return 'background-color: #2d4a3e'
                elif val < 6:
                    return 'background-color: #3d4a3e'
            return ''
        
        st.dataframe(
            display_df.style.applymap(highlight_short_odds, subset=['Goal Odds', 'Assist Odds']),
            hide_index=True,
            use_container_width=True
        )
        
        # Summary stats
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Implied Goals", f"{(df['goal_prob']/100).apply(lambda x: -np.log(1-min(x, 0.99))).sum():.2f}")
        with col2:
            st.metric("Top Scorer", f"{df.iloc[0]['name']} ({df.iloc[0]['goal_odds']:.2f})")
        with col3:
            st.metric("Top Assister", f"{df.nsmallest(1, 'assist_odds').iloc[0]['name']} ({df.nsmallest(1, 'assist_odds').iloc[0]['assist_odds']:.2f})")


# =============================================================================
# TAB 3: PLAYER DEEP DIVE
# =============================================================================
def render_player_tab(goal_model, assist_model, goal_features, assist_features, player_profile):
    st.header("🔍 Player Deep Dive")
    
    # Get all player names for search
    all_players = sorted(player_profile['pf_name'].dropna().unique().tolist())
    
    col1, col2 = st.columns([3, 1])
    with col1:
        player_search = st.selectbox(
            "Search Player",
            all_players,
            index=0,
            key="player_search"
        )
    with col2:
        baseline_xg = st.slider("Team xG", 0.5, 4.0, 2.0, 0.1, key="player_xg")
    
    if st.button("🔎 Analyze Player", type="primary", key="player_btn"):
        # Find player
        player_data = player_profile[player_profile['pf_name'] == player_search]
        
        if len(player_data) == 0:
            st.error(f"Player '{player_search}' not found")
            return
        
        player = player_data.iloc[0]
        team = player.get('pf_team', 'Unknown')
        position = player.get('pf_position', 'Unknown')
        
        # Header
        st.markdown(f"## {player['pf_name']}")
        st.markdown(f"**{team}** | {position} | {get_pos_abbrev(position)}")
        
        # Get odds
        team_df, _ = get_team_odds(
            team, baseline_xg, goal_model, assist_model,
            goal_features, assist_features, player_profile, min_minutes=0
        )
        
        if team_df is not None:
            player_odds = team_df[team_df['player_id'] == player['player_id']]
            
            if len(player_odds) > 0:
                po = player_odds.iloc[0]
                
                # Odds display
                st.markdown("### 🎰 Model Odds")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("⚽ Goal Odds", f"{po['goal_odds']:.2f}")
                with col2:
                    st.metric("Goal Prob", f"{po['goal_prob']:.1f}%")
                with col3:
                    st.metric("🎯 Assist Odds", f"{po['assist_odds']:.2f}")
                with col4:
                    st.metric("Assist Prob", f"{po['assist_prob']:.1f}%")
                
                # Rank in team
                goal_rank = (team_df['goal_odds'] < po['goal_odds']).sum() + 1
                assist_rank = (team_df['assist_odds'] < po['assist_odds']).sum() + 1
                
                st.markdown(f"**Team Rank:** #{goal_rank} for goals | #{assist_rank} for assists")
        
        # Performance Profile
        st.markdown("### 📈 Performance Profile")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Shooting**")
            shots = player.get('sp_total_shots', 0) or 0
            goals = player.get('goals', 0) or 0
            xg = player.get('expected_goals_xg', 0) or 0
            
            st.write(f"Shots: {int(shots)}")
            st.write(f"Goals: {int(goals)}")
            st.write(f"xG: {xg:.2f}")
            st.write(f"xG/Shot: {xg/shots:.2f}" if shots > 0 else "xG/Shot: -")
        
        with col2:
            st.markdown("**Creation**")
            chances = player.get('chances_created', 0) or 0
            assists = player.get('assists', 0) or 0
            xa = player.get('expected_assists_xa', 0) or 0
            
            st.write(f"Chances: {int(chances)}")
            st.write(f"Assists: {int(assists)}")
            st.write(f"xA: {xa:.2f}")
        
        with col3:
            st.markdown("**Per 90**")
            mins = player.get('sm_total_minutes', 0) or player.get('minutes_played', 0) or 1
            
            st.write(f"Minutes: {int(mins)}")
            st.write(f"xG/90: {player.get('sm_xg_per90', 0):.2f}")
            st.write(f"xA/90: {player.get('sm_xa_per90', 0):.2f}")
            st.write(f"Shots/90: {player.get('sm_shots_per90', 0):.2f}")
        
        # Intelligence Report
        st.markdown("### 🧠 Intelligence Report")
        
        xg_per90 = player.get('sm_xg_per90', 0) or 0
        xa_per90 = player.get('sm_xa_per90', 0) or 0
        shots_per90 = player.get('sm_shots_per90', 0) or 0
        
        bull_case = []
        bear_case = []
        
        if xg_per90 >= 0.4:
            bull_case.append(f"Elite Goal Threat ({xg_per90:.2f} xG/90)")
        elif xg_per90 < 0.1:
            bear_case.append("Low Goal Threat")
        
        if xa_per90 >= 0.25:
            bull_case.append(f"Elite Playmaker ({xa_per90:.2f} xA/90)")
        
        if shots_per90 >= 3.0:
            bull_case.append(f"High Shot Volume ({shots_per90:.1f}/90)")
        
        goals = player.get('goals', 0) or 0
        xg = player.get('expected_goals_xg', 0) or 0
        xg_diff = goals - xg
        
        if xg_diff <= -2.0:
            bull_case.append(f"Underperforming xG by {abs(xg_diff):.1f} (Due for regression)")
        if xg_diff >= 3.0:
            bear_case.append(f"Overperforming xG by {xg_diff:.1f} (Unsustainable)")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**✅ Bull Case**")
            if bull_case:
                for item in bull_case:
                    st.write(f"• {item}")
            else:
                st.write("No strong positives")
        
        with col2:
            st.markdown("**⚠️ Bear Case**")
            if bear_case:
                for item in bear_case:
                    st.write(f"• {item}")
            else:
                st.write("No major concerns")


# =============================================================================
# TAB 4: SLATE SCANNER
# =============================================================================
def render_slate_tab(goal_model, assist_model, goal_features, assist_features, player_profile):
    st.header("📋 Slate Scanner")
    
    st.markdown("Enter matches (one per line): `Home, Away, Home_xG, Away_xG`")
    
    default_slate = """Liverpool, Chelsea, 2.15, 1.10
Bournemouth, Arsenal, 0.95, 1.95
Wolves, Man City, 0.65, 2.85"""
    
    slate_input = st.text_area("Match Slate", value=default_slate, height=120, key="slate_input")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        market = st.radio("Market", ["Goal", "Assist"], horizontal=True, key="slate_market")
    with col2:
        max_odds = st.slider("Max Odds", 2.0, 50.0, 10.0, 0.5, key="slate_max_odds")
    with col3:
        min_xg90 = st.slider("Min xG/90" if market == "Goal" else "Min xA/90", 0.0, 0.5, 0.0, 0.05, key="slate_min_xg")
    
    if st.button("🔍 Scan Slate", type="primary", key="slate_btn"):
        all_results = []
        
        # Parse slate
        for line in slate_input.strip().split('\n'):
            parts = [p.strip() for p in line.split(',')]
            if len(parts) != 4:
                continue
            
            try:
                home_team, away_team = parts[0], parts[1]
                home_xg, away_xg = float(parts[2]), float(parts[3])
            except:
                continue
            
            # Get odds for both teams
            for team, xg in [(home_team, home_xg), (away_team, away_xg)]:
                df, matched = get_team_odds(
                    team, xg, goal_model, assist_model,
                    goal_features, assist_features, player_profile
                )
                
                if df is not None:
                    df['matchup'] = f"{home_team} vs {away_team}"
                    df['team'] = matched
                    df['team_xg'] = xg
                    all_results.append(df)
        
        if not all_results:
            st.error("No valid matches found")
            return
        
        # Combine all results
        combined = pd.concat(all_results, ignore_index=True)
        
        # Filter
        odds_col = 'goal_odds' if market == "Goal" else 'assist_odds'
        xg_col = 'xg_per90' if market == "Goal" else 'xa_per90'
        
        filtered = combined[
            (combined[odds_col] <= max_odds) &
            (combined[xg_col] >= min_xg90) &
            (combined['pos'] != 'GK')
        ].sort_values(odds_col)
        
        st.markdown(f"### Found {len(filtered)} plays")
        
        # Display
        display_cols = ['matchup', 'name', 'pos', 'team_xg', odds_col, xg_col]
        display_df = filtered[display_cols].head(50).copy()
        display_df.columns = ['Match', 'Player', 'Pos', 'Team xG', 'Odds', 'xG/90' if market == "Goal" else 'xA/90']
        
        st.dataframe(display_df, hide_index=True, use_container_width=True)
        
        # Download button
        csv = filtered.to_csv(index=False)
        st.download_button(
            "📥 Download CSV",
            csv,
            "slate_results.csv",
            "text/csv",
            key="download_slate"
        )


# =============================================================================
# MAIN APP
# =============================================================================
def main():
    # Load models
    try:
        goal_model, assist_model, goal_features, assist_features, player_profile, metadata = load_models()
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        st.info("Make sure model_artifacts/ folder exists with all required files.")
        st.code("""
Required files:
- model_artifacts/goal_model_bundle.pkl
- model_artifacts/assist_model_bundle.pkl  
- model_artifacts/player_profile.pkl
- model_artifacts/model_metadata.json
        """)
        return
    
    # Get teams
    teams = sorted([t for t in player_profile['pf_team'].dropna().unique() if isinstance(t, str)])
    
    # Title
    st.title("⚽ Goalscorer & Assist Odds")
    
    # Model info
    with st.expander("ℹ️ Model Info"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Goal Model AUC", f"{metadata['goal_model']['test_auc']:.3f}")
        with col2:
            st.metric("Assist Model AUC", f"{metadata['assist_model']['test_auc']:.3f}")
        with col3:
            st.metric("Players", f"{len(player_profile):,}")
        st.caption(f"Trained on: {metadata['data_info']['train_date_range']}")
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["⚔️ H2H Match", "📊 Team Proj", "🔍 Player", "📋 Slate"])
    
    with tab1:
        render_h2h_tab(goal_model, assist_model, goal_features, assist_features, player_profile, teams)
    
    with tab2:
        render_team_proj_tab(goal_model, assist_model, goal_features, assist_features, player_profile, teams)
    
    with tab3:
        render_player_tab(goal_model, assist_model, goal_features, assist_features, player_profile)
    
    with tab4:
        render_slate_tab(goal_model, assist_model, goal_features, assist_features, player_profile)


if __name__ == "__main__":
    main()
