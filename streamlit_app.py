import pandas as pd
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
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

GOAL_BASELINES = {'ST': 0.41, 'CF': 0.40, 'RW': 0.28, 'LW': 0.28, 'CAM': 0.23, 'RM': 0.12, 'LM': 0.12, 'CM': 0.08,
                  'DM': 0.04, 'RWB': 0.03, 'LWB': 0.03, 'RB': 0.02, 'LB': 0.02, 'CB': 0.03, 'GK': 0.001}
ASSIST_BASELINES = {'ST': 0.19, 'CF': 0.19, 'RW': 0.21, 'LW': 0.21, 'CAM': 0.22, 'RM': 0.16, 'LM': 0.16, 'CM': 0.14,
                    'DM': 0.06, 'RWB': 0.09, 'LWB': 0.09, 'RB': 0.08, 'LB': 0.08, 'CB': 0.02, 'GK': 0.001}
POS_SORT = {'ST':1,'CF':2,'RW':3,'LW':4,'CAM':5,'RM':6,'LM':7,'CM':8,'DM':9,'RWB':10,'LWB':11,'RB':12,'LB':13,'CB':14,'GK':15}

# =============================================================================
# DATA LOADING
# =============================================================================

STATS_URL = "https://raw.githubusercontent.com/sznajdr/fb1/refs/heads/main/fotmob_multi_player_season_stats.csv"
FEATURES_URL = "https://raw.githubusercontent.com/sznajdr/fb1/refs/heads/main/player_features.csv"
LINEUPS_URL = "https://raw.githubusercontent.com/sznajdr/fb1/refs/heads/main/fotmob_multi_lineups.csv"

# Tactical profile position colors
TACTIC_POS_COLORS = {
    'GK': '#e2b714',  # Gold/Yellow
    'DEF': '#3794ff', # Blue
    'MID': '#4ec9b0', # Teal/Green
    'FWD': '#e056fd', # Purple/Pink
    'UNK': '#666'
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
            'ATG': round(odds_g, 2),
            'AST': round(odds_a, 2),
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

st.set_page_config(page_title=" ", layout="wide")

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
with st.spinner(""):
    df = load_data()

if df.empty:
    st.error("No data loaded. Check the CSV URLs.")
    st.stop()

# Controls
col1, col2, col3 = st.columns([2, 3, 2])

with col1:
    teams = sorted(df['team'].unique())
    selected_team = st.selectbox(" ", teams, label_visibility="collapsed")

with col2:
    team_xg = st.slider(" ", min_value=0.5, max_value=4.0, value=1.23, step=0.05, label_visibility="collapsed")

with col3:
    sort_options = {'ATG': 'goal_odds', 'AST': 'assist_odds'}
    sort_by = st.selectbox("  ", list(sort_options.keys()), label_visibility="collapsed")

# Get predictions
data = predict_odds(df, selected_team, team_xg)

if data.empty:
    st.warning("No players found for this team")
    st.stop()

# Sort data
data = data.sort_values(sort_options[sort_by])


# Prepare display dataframe
display_df = data[['Player', 'Pos', 'ATG', 'AST', 'xG', 'xA', 'Type', 'Mins']].reset_index(drop=True)
display_df = display_df.rename(columns={'Type': '+/-', 'Mins': 'min'})

# Position colors from original
POS_COLORS = {
    'ST': '#8F0000', 'CF': '#8F0000', 
    'RW': '#8D4E28', 'LW': '#8D4E28',
    'CAM': '#5a8a5a', 
    'RM': '#237023', 'LM': '#237023',
    'CM': '#0B5E0B', 
    'DM': '#167D63',
    'RWB': '#6a7a8a', 'LWB': '#6a7a8a',
    'RB': '#07166D', 'LB': '#07166D',
    'CB': '#14316F', 
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
    color_goal_odds, subset=['ATG']
).applymap(
    color_assist_odds, subset=['AST']
).applymap(
    color_type, subset=['+/-']
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
    'xG': '{:.2f}',
    'xA': '{:.2f}',
    '+/-': '{:.2f}',
    'min': '{:d}'
})

# Display with st.write (respects styling better)
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
    html_out = ""
    
    if is_aggregate:
        p_stats = df_source.groupby(['player_name', 'last_name', 'pos_simple']).agg({
            'match_id': 'nunique',
            'minutes_played': 'median',
            'rating': 'mean',
            'goals_in_match': 'sum',
            'assists_in_match': 'sum'
        }).reset_index().sort_values(['match_id', 'rating'], ascending=[False, False])
    else:
        p_stats = df_source.sort_values('pos_simple', key=lambda x: x.map({'GK':0, 'DEF':1, 'MID':2, 'FWD':3}))

    for pos_group in ['GK', 'DEF', 'MID', 'FWD']:
        if is_aggregate:
            limit = 1 if pos_group == 'GK' else 5
            group = p_stats[p_stats['pos_simple'] == pos_group].head(limit)
        else:
            group = p_stats[p_stats['pos_simple'] == pos_group]

        if group.empty: continue
        
        p_list = []
        for _, p in group.iterrows():
            stats_parts = []
            
            if is_aggregate:
                stats_parts.append(f"{p['match_id']}x")
            
            stats_parts.append(f"{int(p['minutes_played'])}'")
            stats_parts.append(f"{p['rating']:.1f}")

            if p['goals_in_match'] > 0: 
                stats_parts.append(f"<span style='color:#4ec9b0; font-weight:bold;'>{int(p['goals_in_match'])}G</span>")
            if p['assists_in_match'] > 0: 
                stats_parts.append(f"<span style='color:#4ec9b0; font-weight:bold;'>{int(p['assists_in_match'])}A</span>")
            
            stats_str = " <span style='color:#555;'>|</span> ".join(stats_parts)
            name_display = p['last_name']
            
            p_list.append(f"<div style='margin-bottom:2px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;'>{name_display} <span style='font-size:10px; color:#888;'>({stats_str})</span></div>")
        
        pos_color = TACTIC_POS_COLORS.get(pos_group, '#888')
        html_out += f"""
        <div style="margin-bottom: 6px;">
            <div style="color: {pos_color}; font-weight:bold; font-size: 10px; border-bottom: 1px solid #333; margin-bottom:2px;">{pos_group}</div>
            { "".join(p_list) }
        </div>
        """
    return html_out

def generate_tactical_profile(df_l, team):
    df = df_l[df_l['team'] == team].copy()
    if df.empty:
        return ""
    
    df['pos_simple'] = df['position_name'].apply(get_simple_pos)
    
    last_match_id = df['match_id'].max()
    form_counts = df.groupby('formation')['match_id'].nunique().sort_values(ascending=False)
    top_formations = form_counts.head(2).index.tolist()
    
    # Core table
    core_df = df[(df['is_starter'] == True) & (df['pos_simple'] != 'GK')]
    core_stats = core_df.groupby('player_name').agg({
        'match_id': 'nunique',
        'minutes_played': 'median',
        'rating': 'mean',
        'goals_in_match': 'sum',
        'assists_in_match': 'sum',
        'pos_simple': lambda x: x.mode()[0] if not x.mode().empty else 'UNK'
    }).reset_index()
    core_stats = core_stats.rename(columns={'match_id': 'starts'})
    core_stats = core_stats.sort_values(['starts', 'goals_in_match', 'rating'], ascending=[False, False, False]).head(10)
    
    cards_html = ""
    
    # Last match card
    df_last = df[(df['match_id'] == last_match_id) & (df['is_starter'] == True)]
    if not df_last.empty:
        last_form = df_last['formation'].iloc[0]
        last_rat = df_last['rating'].mean()
        player_html = generate_player_list(df_last, is_aggregate=False)
        
        cards_html += f"""
        <div style="flex: 1; background: #252525; border-radius: 4px; padding: 10px; min-width: 180px; border:1px solid #333;">
            <div style="border-bottom: 1px solid #444; padding-bottom: 5px; margin-bottom: 8px; display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span style="font-weight: bold; color: #fff; font-size: 13px;"> </span>
                    <div style="font-size:10px; color:#888;">{last_form}</div>
                </div>
                <div style="background: #333; color: #aaa; padding: 2px 5px; border-radius: 3px; font-size: 11px; font-weight:bold;">
                    {last_rat:.1f} Rat
                </div>
            </div>
            <div style="font-size: 11px; line-height: 1.3; color: #ccc;">{player_html}</div>
        </div>
        """
    
    # Formation cards
    for form in top_formations:
        df_form = df[(df['formation'] == form) & (df['is_starter'] == True)]
        count_used = form_counts[form]
        avg_rat = df_form.groupby('match_id')['rating'].mean().mean()
        player_html = generate_player_list(df_form, is_aggregate=True)
        
        cards_html += f"""
        <div style="flex: 1; background: #252525; border-radius: 4px; padding: 10px; min-width: 180px; border:1px solid #333;">
            <div style="border-bottom: 1px solid #444; padding-bottom: 5px; margin-bottom: 8px; display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span style="font-weight: bold; color: #fff; font-size: 13px;">{form}</span>
                    <div style="font-size:10px; color:#888;">Used {count_used} times</div>
                </div>
                <div style="background: #333; color: #4ec9b0; padding: 2px 5px; border-radius: 3px; font-size: 11px; font-weight:bold;">
                    {avg_rat:.1f} Rat
                </div>
            </div>
            <div style="font-size: 11px; line-height: 1.3; color: #ccc;">{player_html}</div>
        </div>
        """
    
    # Core table rows
    core_rows = ""
    for _, row in core_stats.iterrows():
        g_val = f"<span style='color:#4ec9b0; font-weight:bold;'>{int(row['goals_in_match'])}</span>" if row['goals_in_match'] > 0 else "<span style='color:#444;'>-</span>"
        a_val = f"<span style='color:#4ec9b0; font-weight:bold;'>{int(row['assists_in_match'])}</span>" if row['assists_in_match'] > 0 else "<span style='color:#444;'>-</span>"
        p_color = TACTIC_POS_COLORS.get(row['pos_simple'], '#888')
        pos_badge = f"<span style='color:{p_color}; font-weight:bold;'>{row['pos_simple']}</span>"
        
        core_rows += f"""
        <tr style="border-bottom: 1px solid #333; height: 22px;">
            <td style="color:#ddd; font-weight:500;">{row['player_name']}</td>
            <td style="text-align:center; font-size:10px;">{pos_badge}</td>
            <td style="text-align:right;">{int(row['starts'])}</td>
            <td style="text-align:right; color:#888;">{int(row['minutes_played'])}'</td>
            <td style="text-align:right; color:#aaa;">{row['rating']:.1f}</td>
            <td style="text-align:right;">{g_val}</td>
            <td style="text-align:right;">{a_val}</td>
        </tr>
        """
    
    html = f"""
    <div class="dash-container">
        <div class="header-title">{team}  </div>
        <div class="flex-row">
            {cards_html}
            <div style="flex: 0 0 350px; background: #222; padding: 10px; border-radius: 4px; border:1px solid #333;">
                <div style="color: #3794ff; font-weight: bold; font-size: 12px; margin-bottom: 8px; display:flex; justify-content:space-between;">
                    <span> </span>
                    <span style="font-size:10px; color:#666;"> </span>
                </div>
                <table class="core-table">
                    <tr><th>Player</th><th>Pos</th><th>St</th><th>Min</th><th>Rat</th><th>G</th><th>A</th></tr>
                    {core_rows}
                </table>
            </div>
        </div>
    </div>
    """
    return html

# Load lineups and display tactical profile
df_lineups = load_lineups_data()

if not df_lineups.empty and selected_team in df_lineups['team'].values:
    st.markdown("---")
    
    tactical_html = generate_tactical_profile(df_lineups, selected_team)
    
    # Wrap in full HTML with styles
    full_html = f"""
    <html>
    <head>
    <style>
        body {{ margin: 0; padding: 0; background: #1e1e1e; }}
        .dash-container {{ font-family: 'Segoe UI', sans-serif; background: #1e1e1e; padding: 15px; color: #d4d4d4; overflow-x: auto; }}
        .header-title {{ color: #3794ff; font-weight: bold; font-size: 14px; margin-bottom: 10px; }}
        .flex-row {{ display: flex; gap: 10px; width: 100%; flex-wrap: wrap; }}
        .core-table {{ width: 100%; border-collapse: collapse; font-size: 11px; }}
        .core-table th {{ text-align: right; color: #666; font-weight: normal; padding-bottom: 4px; border-bottom: 1px solid #444; }}
        .core-table th:first-child {{ text-align: left; }}
        .core-table th:nth-child(2) {{ text-align: center; }}
        .core-table td {{ padding: 4px 2px; }}
        
        @media screen and (max-width: 768px) {{
            .flex-row {{ flex-direction: column; }}
            .flex-row > div {{ flex: 1 1 100% !important; min-width: 100% !important; }}
        }}
    </style>
    </head>
    <body>
    {tactical_html}
    </body>
    </html>
    """
    
    components.html(full_html, height=450, scrolling=True)

# =============================================================================
# SUSPENSIONS, INJURIES & RISK TABLE
# =============================================================================

FOOTBALL_DATA_URL = "https://raw.githubusercontent.com/sznajdr/asttt/refs/heads/main/football_data_complete_2025-12-03.csv"

@st.cache_data
def load_football_status_data():
    try:
        df = pd.read_csv(FOOTBALL_DATA_URL)
        # Keep only the columns we want
        cols_to_keep = ['competition', 'club', 'player', 'position', 'reason', 'matches_missed', 
                        'age', 'injury', 'yellow_cards', 'injured_since', 'injured_until', 
                        'data_type', 'since', 'until']
        existing_cols = [c for c in cols_to_keep if c in df.columns]
        return df[existing_cols]
    except Exception as e:
        st.error(f"Failed to load football status data: {e}")
        return pd.DataFrame()

st.markdown("---")

# Load the data
df_status = load_football_status_data()

if not df_status.empty:
    # Filter controls
    filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)
    
    with filter_col1:
        competitions = sorted(df_status['competition'].dropna().unique().tolist())
        selected_competitions = st.multiselect(" ", competitions, key="filter_competition")
    
    with filter_col2:
        positions = sorted(df_status['position'].dropna().unique().tolist())
        default_positions = [p for p in ['Centre-Forward', 'Attacking Midfield', 'Right Winger', 'Left Winger', 'Zentrales Mittelfeld', 'Central Midfield', 'Hängende Spitze', 'Mittelstürmer', 'Linksaußen', 'Offensives Mittelfeld', 'Rechtsaußen', 'Second Striker'] if p in positions]
        selected_positions = st.multiselect(" ", positions, default=default_positions, key="filter_position")
    
    with filter_col3:
        reasons = sorted(df_status['reason'].dropna().unique().tolist())
        selected_reasons = st.multiselect(" ", reasons, key="filter_reason")
    
    with filter_col4:
        data_types = sorted(df_status['data_type'].dropna().unique().tolist())
        default_types = ['suspensions'] if 'suspensions' in data_types else []
        selected_types = st.multiselect(" ", data_types, default=default_types, key="filter_type")
    
    # Apply filters
    df_filtered = df_status.copy()
    
    if selected_competitions:
        df_filtered = df_filtered[df_filtered['competition'].isin(selected_competitions)]
    if selected_positions:
        df_filtered = df_filtered[df_filtered['position'].isin(selected_positions)]
    if selected_reasons:
        df_filtered = df_filtered[df_filtered['reason'].isin(selected_reasons)]
    if selected_types:
        df_filtered = df_filtered[df_filtered['data_type'].isin(selected_types)]
    
    # Style for the status table
    def color_data_type(val):
        if val == 'suspensions':
            return 'color: #dc3545; font-weight: bold'  # red
        elif val == 'risk_of_suspension':
            return 'color: #ffc107; font-weight: bold'  # yellow
        elif val == 'injuries':
            return 'color: #17a2b8; font-weight: bold'  # blue
        return 'color: #888'
    
    # Prepare display
    display_status_df = df_filtered.reset_index(drop=True)
    
    # Style the dataframe
    styled_status_df = display_status_df.style.applymap(
        color_data_type, subset=['data_type']
    ).set_properties(**{
        'text-align': 'left'
    }).set_table_styles([
        {'selector': 'th', 'props': [('text-align', 'left')]},
        {'selector': 'td', 'props': [('text-align', 'left')]}
    ])
    
    # Display table
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
                if nested and nested.find('a', class_='fb s10'):
                    p_name = nested.find('a', class_='fb s10').text.strip()

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

    except Exception:
        return pd.DataFrame(columns=['club', 'name', 'pos', 'injury', 'since'])

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
    if selected_injury_team != 'All':
        df_injuries = df_injuries[df_injuries['club'] == selected_injury_team]
    
    styled_injuries = df_injuries.reset_index(drop=True).style.set_properties(**{
        'text-align': 'left'
    }).set_table_styles([
        {'selector': 'th', 'props': [('text-align', 'left')]},
        {'selector': 'td', 'props': [('text-align', 'left')]}
    ])
    
    st.markdown(f'<div class="table-container">{styled_injuries.to_html()}</div>', unsafe_allow_html=True)
else:
    st.info("No injury data available for this league.")
