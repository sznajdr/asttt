# =============================================================================
# GOALSCORER ODDS PREDICTION - STREAMLIT APP (FULL VERSION)
# =============================================================================
# 
# Features:
# - H2H Match Preview
# - Team Projections  
# - Player Deep Dive (with tags)
# - Slate Scanner
# - Lineup Intelligence (formations, predicted XI)
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

# Custom CSS
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        background-color: #262730;
        border-radius: 4px;
    }
    .stTabs [aria-selected="true"] { background-color: #1f77b4; }
    .tag { 
        background-color: #1f77b4; 
        padding: 2px 8px; 
        border-radius: 4px; 
        margin: 2px;
        font-size: 11px;
        display: inline-block;
    }
    .tag-pk { background-color: #4a3d2d; }
    .tag-hot { background-color: #2d4a3e; }
    .tag-cold { background-color: #4a2d2d; }
    .locked { color: #4CAF50; }
    .rotation { color: #FFC107; }
    .unlikely { color: #f44336; }
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

# Tag styling
TAG_COLORS = {
    'PK': '#4a3d2d', 'FK_TAKER': '#4a3d2d', 'CORNER_TARGET': '#4a3d2d',
    'FORM_HOT': '#2d4a3e', 'ATG_UNDERP': '#2d4a3e', '100P': '#2d4a3e',
    'FORM_COLD': '#4a2d2d', 'ATG_OVERP': '#4a2d2d',
    'HIGH_VOL': '#3d3d4a', 'BOX': '#3d3d4a', 'POACHER': '#3d3d4a',
    'PLAYMAKER': '#3d4a3d', 'CROSSER': '#3d4a3d',
    'STARTER': '#2d3d4a', '90M': '#2d3d4a',
    'LATE_SUB': '#4a3d4a', 'SUPER_SUB': '#4a3d4a',
}


# =============================================================================
# MODEL WRAPPER
# =============================================================================
class CalibratedModel:
    def __init__(self, model, calibrator):
        self.model = model
        self.calibrator = calibrator
    
    def predict_proba(self, X):
        raw = self.model.predict_proba(X)[:, 1]
        calibrated = self.calibrator.predict(raw)
        return np.column_stack([1 - calibrated, calibrated])


# =============================================================================
# LOAD DATA (CACHED)
# =============================================================================
@st.cache_resource
def load_models():
    """Load models from model_artifacts folder"""
    artifact_dir = Path("model_artifacts")
    
    goal_bundle = joblib.load(artifact_dir / "goal_model_bundle.pkl")
    goal_model = CalibratedModel(goal_bundle['model'], goal_bundle['calibrator'])
    goal_features = goal_bundle['features']
    
    assist_bundle = joblib.load(artifact_dir / "assist_model_bundle.pkl")
    assist_model = CalibratedModel(assist_bundle['model'], assist_bundle['calibrator'])
    assist_features = assist_bundle['features']
    
    player_profile = pd.read_pickle(artifact_dir / "player_profile.pkl")
    
    with open(artifact_dir / "model_metadata.json", 'r') as f:
        metadata = json.load(f)
    
    # Load player tags if exists
    tags_path = artifact_dir / "player_tags.json"
    if tags_path.exists():
        with open(tags_path, 'r') as f:
            player_tags = {int(k): v for k, v in json.load(f).items()}
    else:
        player_tags = {}
    
    return goal_model, assist_model, goal_features, assist_features, player_profile, metadata, player_tags


@st.cache_data
def load_lineups():
    """Load lineups data"""
    lineups_path = Path("model_artifacts/recent_lineups.csv")
    if lineups_path.exists():
        lineups = pd.read_csv(lineups_path)
        # Rename if needed
        if 'team_name' in lineups.columns and 'team' not in lineups.columns:
            lineups = lineups.rename(columns={'team_name': 'team'})
        return lineups
    return None


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def get_pos_abbrev(position):
    return POS_ABBREV.get(str(position).lower().strip(), 'MF')


def prob_to_odds(prob):
    return 1 / prob if prob > 0 else 500.0


def apply_position_bounds(odds, position, market='goal'):
    pos_lower = str(position).lower().strip()
    bounds = POS_BOUNDS.get(market, {}).get(pos_lower, (1.5, 200))
    return max(bounds[0], min(bounds[1], odds))


def fuzzy_find_team(query, team_list):
    if not isinstance(query, str):
        return None
    query_lower = query.lower().strip()
    
    for team in team_list:
        if pd.notna(team) and isinstance(team, str):
            if team.lower().strip() == query_lower:
                return team
    
    for team in team_list:
        if pd.notna(team) and isinstance(team, str):
            if query_lower in team.lower():
                return team
    
    for team in team_list:
        if pd.notna(team) and isinstance(team, str):
            if team.lower() in query_lower:
                return team
    
    return None


def render_tags(tags, max_tags=5):
    """Render tags as colored badges"""
    if not tags:
        return ""
    html = ""
    for tag in tags[:max_tags]:
        color = TAG_COLORS.get(tag, '#3d3d3d')
        html += f'<span class="tag" style="background-color: {color};">{tag}</span> '
    return html


def get_player_tags(player_id, player_tags):
    """Get tags for a player"""
    return player_tags.get(int(player_id), [])


# =============================================================================
# MARKET SCALING
# =============================================================================
def apply_market_scaling(df, market_xg):
    if df is None or len(df) == 0:
        return df
    
    df = df.copy()
    
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
    
    df['goal_odds'] = df.apply(
        lambda x: apply_position_bounds(prob_to_odds(x['goal_prob']/100), x['position'], 'goal'), axis=1)
    df['assist_odds'] = df.apply(
        lambda x: apply_position_bounds(prob_to_odds(x['assist_prob']/100), x['position'], 'assist'), axis=1)
    
    return df


# =============================================================================
# MAIN INFERENCE FUNCTION
# =============================================================================
def get_team_odds(team_name, market_xg, goal_model, assist_model, 
                  goal_features, assist_features, player_profile, player_tags, min_minutes=50):
    
    teams = player_profile['pf_team'].dropna().unique()
    matched_team = fuzzy_find_team(team_name, teams)
    
    if matched_team is None:
        return None, f"Team '{team_name}' not found"
    
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
        
        goal_row = {}
        for f in goal_features:
            goal_row[f] = player.get(f, 0) if f in player.index else 0
        goal_row['match_is_starter'] = 1
        goal_row['match_pos_goal_weight'] = POS_GOAL_WEIGHT.get(str(position).lower(), 0.3)
        goal_row['pos_goal_weight'] = POS_GOAL_WEIGHT.get(str(position).lower(), 0.3)
        
        assist_row = {}
        for f in assist_features:
            assist_row[f] = player.get(f, 0) if f in player.index else 0
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
            
            player_id = player.get('player_id', 0)
            tags = get_player_tags(player_id, player_tags)
            
            results.append({
                'player_id': player_id,
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
                'tags': tags,
            })
        except Exception:
            continue
    
    if not results:
        return None, f"Could not generate predictions for {matched_team}"
    
    df = pd.DataFrame(results)
    df = apply_market_scaling(df, market_xg)
    
    return df.sort_values('goal_odds'), matched_team


# =============================================================================
# LINEUP INTELLIGENCE FUNCTIONS
# =============================================================================
def get_lineup_team_intel(lineups_df, team_name, n_recent=10):
    """Get team intel from lineup data"""
    if lineups_df is None or lineups_df.empty:
        return None, None, None, None
    
    teams = lineups_df['team'].unique()
    matched = [t for t in teams if team_name.lower() in t.lower()]
    if not matched:
        return None, None, None, None
    
    team = matched[0]
    team_df = lineups_df[lineups_df['team'] == team].copy()
    
    matches = team_df['match_id'].unique()
    recent_matches = matches[-n_recent:] if len(matches) > n_recent else matches
    recent_df = team_df[team_df['match_id'].isin(recent_matches)]
    
    # Formations
    starters = recent_df[recent_df['is_starter'] == True]
    formations = starters.groupby(['match_id', 'formation']).size().reset_index()
    formations = formations.groupby('formation').size().reset_index(name='matches')
    formations['pct'] = (formations['matches'] / len(recent_matches) * 100).round(0)
    formations = formations.sort_values('matches', ascending=False)
    
    # Player stats
    player_stats = recent_df.groupby(['player_id', 'player_name', 'position_name']).agg({
        'is_starter': 'sum',
        'minutes_played': 'mean',
        'rating': 'mean',
        'goals_in_match': 'sum',
        'assists_in_match': 'sum'
    }).reset_index()
    
    player_stats.columns = ['player_id', 'name', 'pos', 'starts', 'avg_mins', 'rating', 'goals', 'assists']
    player_stats['start_pct'] = (player_stats['starts'] / len(recent_matches) * 100).round(0)
    player_stats = player_stats.sort_values(['pos', 'starts'], ascending=[True, False])
    
    return team, formations, player_stats, len(recent_matches)


def get_predicted_xi(lineups_df, team_name, n_recent=5):
    """Generate predicted starting XI"""
    if lineups_df is None or lineups_df.empty:
        return None, None, None
    
    teams = lineups_df['team'].unique()
    matched = [t for t in teams if team_name.lower() in t.lower()]
    if not matched:
        return None, None, None
    
    team = matched[0]
    team_df = lineups_df[lineups_df['team'] == team].copy()
    recent_matches = team_df['match_id'].unique()[-n_recent:]
    recent_df = team_df[team_df['match_id'].isin(recent_matches)]
    
    starters = recent_df[recent_df['is_starter'] == True]
    formation = starters['formation'].mode().iloc[0] if not starters.empty else "Unknown"
    
    xi = []
    rotation_risks = {'Goalkeeper': [], 'Defender': [], 'Midfielder': [], 'Forward': []}
    
    for pos in ['Goalkeeper', 'Defender', 'Midfielder', 'Forward']:
        pos_starters = starters[starters['position_name'] == pos]
        players = pos_starters.groupby('player_name').size().reset_index(name='starts')
        players = players.sort_values('starts', ascending=False)
        
        slots = int(pos_starters.groupby('match_id').size().median()) if not pos_starters.empty else 1
        
        for idx, (_, p) in enumerate(players.iterrows()):
            conf = int(p['starts'] / n_recent * 100)
            
            if idx < slots:
                status = "🔒" if conf >= 80 else "✅" if conf >= 50 else "⚠️"
                xi.append({
                    'pos': pos[:3].upper(),
                    'name': p['player_name'],
                    'conf': conf,
                    'status': status
                })
            elif conf >= 20:
                rotation_risks[pos].append(f"{p['player_name']} ({conf}%)")
    
    return formation, xi, rotation_risks


# =============================================================================
# TAB 1: HEAD-TO-HEAD MATCH PREVIEW
# =============================================================================
def render_h2h_tab(goal_model, assist_model, goal_features, assist_features, player_profile, player_tags, teams):
    st.header("⚔️ Head-to-Head Match Preview")
    
    col1, col2 = st.columns(2)
    with col1:
        home_team = st.selectbox("🏠 Home Team", teams, key="h2h_home")
        home_xg = st.slider("Home xG", 0.5, 4.0, 1.85, 0.05, key="h2h_home_xg")
    with col2:
        away_team = st.selectbox("✈️ Away Team", teams, index=min(1, len(teams)-1), key="h2h_away")
        away_xg = st.slider("Away xG", 0.5, 4.0, 1.45, 0.05, key="h2h_away_xg")
    
    if st.button("🎯 Generate Odds", type="primary", key="h2h_btn"):
        home_df, home_matched = get_team_odds(home_team, home_xg, goal_model, assist_model,
                                               goal_features, assist_features, player_profile, player_tags)
        away_df, away_matched = get_team_odds(away_team, away_xg, goal_model, assist_model,
                                               goal_features, assist_features, player_profile, player_tags)
        
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
def render_team_proj_tab(goal_model, assist_model, goal_features, assist_features, player_profile, player_tags, teams):
    st.header("📊 Team Projections")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        team = st.selectbox("Select Team", teams, key="proj_team")
    with col2:
        team_xg = st.slider("Team xG", 0.5, 4.5, 1.85, 0.05, key="proj_xg")
    
    if st.button("📈 Show Projections", type="primary", key="proj_btn"):
        df, matched = get_team_odds(team, team_xg, goal_model, assist_model,
                                     goal_features, assist_features, player_profile, player_tags, min_minutes=100)
        
        if df is None:
            st.error(f"Team '{team}' not found")
            return
        
        st.markdown(f"### {matched} | Team xG: {team_xg}")
        
        df = df[df['pos'] != 'GK']
        
        # Add tags column for display
        df['tags_display'] = df['tags'].apply(lambda x: ', '.join(x[:3]) if x else '')
        
        display_df = df.nsmallest(20, 'goal_odds')[
            ['name', 'pos', 'goal_odds', 'assist_odds', 'xg_per90', 'xa_per90', 'tags_display']
        ].copy()
        display_df.columns = ['Player', 'Pos', 'Goal', 'Assist', 'xG/90', 'xA/90', 'Tags']
        
        st.dataframe(display_df, hide_index=True, use_container_width=True)
        
        # Summary
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            implied = (df['goal_prob']/100).apply(lambda x: -np.log(1-min(x, 0.99))).sum()
            st.metric("Implied Goals", f"{implied:.2f}")
        with col2:
            st.metric("Top Scorer", f"{df.iloc[0]['name']} ({df.iloc[0]['goal_odds']:.2f})")
        with col3:
            top_assist = df.nsmallest(1, 'assist_odds').iloc[0]
            st.metric("Top Assister", f"{top_assist['name']} ({top_assist['assist_odds']:.2f})")


# =============================================================================
# TAB 3: PLAYER DEEP DIVE
# =============================================================================
def render_player_tab(goal_model, assist_model, goal_features, assist_features, player_profile, player_tags):
    st.header("🔍 Player Deep Dive")
    
    all_players = sorted(player_profile['pf_name'].dropna().unique().tolist())
    
    col1, col2 = st.columns([3, 1])
    with col1:
        player_search = st.selectbox("Search Player", all_players, key="player_search")
    with col2:
        baseline_xg = st.slider("Team xG", 0.5, 4.0, 2.0, 0.1, key="player_xg")
    
    if st.button("🔎 Analyze Player", type="primary", key="player_btn"):
        player_data = player_profile[player_profile['pf_name'] == player_search]
        
        if len(player_data) == 0:
            st.error(f"Player '{player_search}' not found")
            return
        
        player = player_data.iloc[0]
        team = player.get('pf_team', 'Unknown')
        position = player.get('pf_position', 'Unknown')
        player_id = player.get('player_id', 0)
        tags = get_player_tags(player_id, player_tags)
        
        # Header with tags
        st.markdown(f"## {player['pf_name']}")
        st.markdown(f"**{team}** | {position} | {get_pos_abbrev(position)}")
        
        if tags:
            st.markdown(render_tags(tags), unsafe_allow_html=True)
        
        # Get odds
        team_df, _ = get_team_odds(team, baseline_xg, goal_model, assist_model,
                                    goal_features, assist_features, player_profile, player_tags, min_minutes=0)
        
        if team_df is not None:
            player_odds = team_df[team_df['player_id'] == player_id]
            
            if len(player_odds) > 0:
                po = player_odds.iloc[0]
                
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
                
                goal_rank = (team_df['goal_odds'] < po['goal_odds']).sum() + 1
                assist_rank = (team_df['assist_odds'] < po['assist_odds']).sum() + 1
                st.markdown(f"**Team Rank:** #{goal_rank} for goals | #{assist_rank} for assists")
        
        # Performance Profile
        st.markdown("### 📈 Performance Profile")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Shooting**")
            st.write(f"Shots: {int(player.get('sp_total_shots', 0) or 0)}")
            st.write(f"Goals: {int(player.get('goals', 0) or 0)}")
            st.write(f"xG: {(player.get('expected_goals_xg', 0) or 0):.2f}")
        
        with col2:
            st.markdown("**Creation**")
            st.write(f"Chances: {int(player.get('chances_created', 0) or 0)}")
            st.write(f"Assists: {int(player.get('assists', 0) or 0)}")
            st.write(f"xA: {(player.get('expected_assists_xa', 0) or 0):.2f}")
        
        with col3:
            st.markdown("**Per 90**")
            st.write(f"xG/90: {(player.get('sm_xg_per90', 0) or 0):.2f}")
            st.write(f"xA/90: {(player.get('sm_xa_per90', 0) or 0):.2f}")
            st.write(f"Shots/90: {(player.get('sm_shots_per90', 0) or 0):.2f}")
        
        # Intelligence Report
        st.markdown("### 🧠 Intelligence Report")
        
        xg_per90 = player.get('sm_xg_per90', 0) or 0
        xa_per90 = player.get('sm_xa_per90', 0) or 0
        shots_per90 = player.get('sm_shots_per90', 0) or 0
        
        bull_case, bear_case = [], []
        
        if xg_per90 >= 0.4: bull_case.append(f"Elite Goal Threat ({xg_per90:.2f} xG/90)")
        elif xg_per90 < 0.1: bear_case.append("Low Goal Threat")
        
        if xa_per90 >= 0.25: bull_case.append(f"Elite Playmaker ({xa_per90:.2f} xA/90)")
        if shots_per90 >= 3.0: bull_case.append(f"High Shot Volume ({shots_per90:.1f}/90)")
        
        if 'PK' in tags: bull_case.append("Penalty Taker")
        if 'FORM_HOT' in tags: bull_case.append("Hot Form")
        if 'ATG_UNDERP' in tags: bull_case.append("Underperforming xG (Due)")
        
        if 'FORM_COLD' in tags: bear_case.append("Cold Form")
        if 'ATG_OVERP' in tags: bear_case.append("Overperforming xG (Regression risk)")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**✅ Bull Case**")
            for item in bull_case if bull_case else ["No strong positives"]:
                st.write(f"• {item}")
        with col2:
            st.markdown("**⚠️ Bear Case**")
            for item in bear_case if bear_case else ["No major concerns"]:
                st.write(f"• {item}")


# =============================================================================
# TAB 4: SLATE SCANNER
# =============================================================================
def render_slate_tab(goal_model, assist_model, goal_features, assist_features, player_profile, player_tags):
    st.header("📋 Slate Scanner")
    
    default_slate = """Liverpool, Chelsea, 2.15, 1.10
Bournemouth, Arsenal, 0.95, 1.95
Wolves, Man City, 0.65, 2.85"""
    
    slate_input = st.text_area("Match Slate (Home, Away, H_xG, A_xG)", value=default_slate, height=120)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        market = st.radio("Market", ["Goal", "Assist"], horizontal=True)
    with col2:
        max_odds = st.slider("Max Odds", 2.0, 50.0, 10.0, 0.5)
    with col3:
        pk_only = st.checkbox("PK Takers Only")
    with col4:
        min_xg = st.slider("Min xG/90" if market == "Goal" else "Min xA/90", 0.0, 0.5, 0.0, 0.05)
    
    if st.button("🔍 Scan Slate", type="primary"):
        all_results = []
        
        for line in slate_input.strip().split('\n'):
            parts = [p.strip() for p in line.split(',')]
            if len(parts) != 4:
                continue
            
            try:
                home_team, away_team = parts[0], parts[1]
                home_xg, away_xg = float(parts[2]), float(parts[3])
            except:
                continue
            
            for team, xg in [(home_team, home_xg), (away_team, away_xg)]:
                df, matched = get_team_odds(team, xg, goal_model, assist_model,
                                             goal_features, assist_features, player_profile, player_tags)
                if df is not None:
                    df['matchup'] = f"{home_team} vs {away_team}"
                    df['team'] = matched
                    df['team_xg'] = xg
                    all_results.append(df)
        
        if not all_results:
            st.error("No valid matches found")
            return
        
        combined = pd.concat(all_results, ignore_index=True)
        
        odds_col = 'goal_odds' if market == "Goal" else 'assist_odds'
        xg_col = 'xg_per90' if market == "Goal" else 'xa_per90'
        
        filtered = combined[
            (combined[odds_col] <= max_odds) &
            (combined[xg_col] >= min_xg) &
            (combined['pos'] != 'GK')
        ]
        
        if pk_only:
            filtered = filtered[filtered['tags'].apply(lambda x: 'PK' in x if x else False)]
        
        filtered = filtered.sort_values(odds_col)
        
        st.markdown(f"### Found {len(filtered)} plays")
        
        filtered['tags_display'] = filtered['tags'].apply(lambda x: ', '.join(x[:3]) if x else '')
        display_df = filtered[['matchup', 'name', 'pos', 'team_xg', odds_col, xg_col, 'tags_display']].head(50).copy()
        display_df.columns = ['Match', 'Player', 'Pos', 'xG', 'Odds', 'xG/90' if market == "Goal" else 'xA/90', 'Tags']
        
        st.dataframe(display_df, hide_index=True, use_container_width=True)
        
        csv = filtered.to_csv(index=False)
        st.download_button("📥 Download CSV", csv, "slate_results.csv", "text/csv")


# =============================================================================
# TAB 5: LINEUP INTELLIGENCE
# =============================================================================
def render_lineup_tab(lineups):
    st.header("🏟️ Lineup Intelligence")
    
    if lineups is None or lineups.empty:
        st.warning("Lineup data not loaded. Upload `recent_lineups.csv` to `model_artifacts/`")
        return
    
    lineup_teams = sorted(lineups['team'].dropna().unique().tolist())
    
    col1, col2 = st.columns([2, 1])
    with col1:
        team = st.selectbox("Select Team", lineup_teams, key="lineup_team")
    with col2:
        n_recent = st.slider("Last N Games", 3, 15, 5)
    
    if st.button("🏟️ Get Intel", type="primary"):
        matched_team, formations, player_stats, n_matches = get_lineup_team_intel(lineups, team, n_recent)
        
        if matched_team is None:
            st.error(f"Team '{team}' not found")
            return
        
        st.markdown(f"### {matched_team} - Last {n_matches} Matches")
        
        # Formations
        st.markdown("#### 📐 Formations")
        if formations is not None and not formations.empty:
            formations_display = formations.rename(columns={'formation': 'Formation', 'matches': 'Games', 'pct': '%'})
            st.dataframe(formations_display, hide_index=True, use_container_width=True)
        
        # Predicted XI
        st.markdown("#### 🔮 Predicted Starting XI")
        formation, xi, rotation_risks = get_predicted_xi(lineups, team, n_recent)
        
        if xi:
            st.markdown(f"**Formation: {formation}**")
            
            xi_df = pd.DataFrame(xi)
            xi_df['confidence'] = xi_df.apply(lambda x: f"{x['status']} {x['conf']}%", axis=1)
            xi_display = xi_df[['pos', 'name', 'confidence']].rename(columns={'pos': 'Pos', 'name': 'Player', 'confidence': 'Conf'})
            st.dataframe(xi_display, hide_index=True, use_container_width=True)
            
            # Rotation risks
            has_risks = any(rotation_risks.values())
            if has_risks:
                st.markdown("#### 🔄 Rotation Risks")
                for pos, players in rotation_risks.items():
                    if players:
                        st.write(f"**{pos}:** {', '.join(players)}")
        
        # Squad Depth
        st.markdown("#### 👥 Squad Depth")
        if player_stats is not None and not player_stats.empty:
            for pos in ['Goalkeeper', 'Defender', 'Midfielder', 'Forward']:
                pos_df = player_stats[player_stats['pos'] == pos].sort_values('starts', ascending=False)
                if not pos_df.empty:
                    with st.expander(f"{pos}s"):
                        display = pos_df[['name', 'starts', 'start_pct', 'avg_mins', 'goals', 'assists', 'rating']].copy()
                        display.columns = ['Player', 'Starts', 'Start%', 'Avg Mins', 'G', 'A', 'Rating']
                        display['Avg Mins'] = display['Avg Mins'].round(0).astype(int)
                        display['Rating'] = display['Rating'].round(2)
                        st.dataframe(display, hide_index=True, use_container_width=True)


# =============================================================================
# MAIN APP
# =============================================================================
def main():
    try:
        goal_model, assist_model, goal_features, assist_features, player_profile, metadata, player_tags = load_models()
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        st.info("Required files in model_artifacts/: goal_model_bundle.pkl, assist_model_bundle.pkl, player_profile.pkl, model_metadata.json")
        return
    
    lineups = load_lineups()
    teams = sorted([t for t in player_profile['pf_team'].dropna().unique() if isinstance(t, str)])
    
    st.title("⚽ Goalscorer & Assist Odds")
    
    with st.expander("ℹ️ Model Info"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Goal AUC", f"{metadata['goal_model']['test_auc']:.3f}")
        with col2:
            st.metric("Assist AUC", f"{metadata['assist_model']['test_auc']:.3f}")
        with col3:
            st.metric("Players", f"{len(player_profile):,}")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["⚔️ H2H", "📊 Team", "🔍 Player", "📋 Slate", "🏟️ Lineups"])
    
    with tab1:
        render_h2h_tab(goal_model, assist_model, goal_features, assist_features, player_profile, player_tags, teams)
    with tab2:
        render_team_proj_tab(goal_model, assist_model, goal_features, assist_features, player_profile, player_tags, teams)
    with tab3:
        render_player_tab(goal_model, assist_model, goal_features, assist_features, player_profile, player_tags)
    with tab4:
        render_slate_tab(goal_model, assist_model, goal_features, assist_features, player_profile, player_tags)
    with tab5:
        render_lineup_tab(lineups)


if __name__ == "__main__":
    main()
