import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import re

# Set page configuration
st.set_page_config(page_title="VJBL U14 Boys Analytics Hub", layout="wide")

# --- DATA PROCESSING FUNCTIONS ---

def extract_club(team_name):
    """Extracts the Club name from the team string."""
    match = re.split(r'\s+U14\s+Boys|\s+Boys', team_name, flags=re.IGNORECASE)
    return match[0].strip()

@st.cache_data(ttl=300)
def load_and_clean_data(file_path):
    try:
        df = pd.read_csv(file_path)
        df.columns = [c.strip() for c in df.columns]
        
        def parse_vjbl_date(date_str):
            try:
                clean_date = date_str.split(',')[-1].strip()
                for fmt in ('%d %b %Y', '%d %b %y'):
                    try: return pd.to_datetime(clean_date, format=fmt)
                    except: continue
                return pd.to_datetime(clean_date, errors='coerce')
            except: return pd.NaT

        df['date_obj'] = df['game_datetime_raw'].apply(parse_vjbl_date)
        df['Display Date'] = df['date_obj'].dt.strftime('%d-%b')
        df['home_score'] = pd.to_numeric(df['home_score'], errors='coerce')
        df['away_score'] = pd.to_numeric(df['away_score'], errors='coerce')
        df = df.dropna(subset=['home_score', 'away_score'])
        return df
    except Exception as e:
        st.error(f"Error processing CSV: {e}")
        return None

def calculate_ratings(df):
    teams = sorted(list(set(df['home_team'].tolist() + df['away_team'].tolist())))
    n = len(teams)
    team_to_idx = {team: i for i, team in enumerate(teams)}
    A, b = np.zeros((n, n)), np.zeros(n)
    
    for i, team in enumerate(teams):
        team_games = df[(df['home_team'] == team) | (df['away_team'] == team)]
        A[i, i] = len(team_games)
        for _, game in team_games.iterrows():
            if game['home_team'] == team:
                b[i] += (game['home_score'] - game['away_score'])
                A[i, team_to_idx[game['away_team']]] -= 1
            else:
                b[i] += (game['away_score'] - game['home_score'])
                A[i, team_to_idx[game['home_team']]] -= 1
    A[-1, :] = 1
    b[-1] = 0
    try:
        ratings = np.linalg.lstsq(A, b, rcond=None)[0]
    except:
        ratings = np.zeros(n)
        
    results = pd.DataFrame({'Team': teams, 'SRS': ratings})
    results['Club'] = results['Team'].apply(extract_club)
    # Extract team number for sorting within club (e.g., '1' from 'Team 1')
    results['TeamNum'] = results['Team'].str.extract(r'(\d+)$').fillna(0).astype(int)
    results = results.sort_values('SRS', ascending=False).reset_index(drop=True)
    results.insert(0, 'Ranking', range(1, len(results) + 1))
    return results

# --- APP LAYOUT ---

st.title("🏀 VJBL U14 Boys: Advanced Analytics Hub")

CSV_FILE = 'u14_boys_all_results.csv'
df_raw = load_and_clean_data(CSV_FILE)

if df_raw is not None:
    rankings = calculate_ratings(df_raw)
    clubs = sorted(rankings['Club'].unique())
    
    st.sidebar.header("Navigation")
    app_mode = st.sidebar.radio("View Mode", 
                                ["Club Overview", "Club Battle (New)", "Power Rankings", "Team Deep Dive", "Matchup Predictor"])

    # --- MODE 1: CLUB OVERVIEW ---
    if app_mode == "Club Overview":
        st.header("Club Strength Index (CSI)")
        club_stats = rankings.groupby('Club').agg(
            CSI=('SRS', 'mean'),
            Teams=('Team', 'count')
        ).sort_values('CSI', ascending=False).reset_index()

        c1, c2 = st.columns([1, 2])
        with c1:
            st.dataframe(club_stats, hide_index=True, use_container_width=True)
        with c2:
            fig = px.box(rankings, x="Club", y="SRS", color="Club", title="Program Depth Distribution")
            st.plotly_chart(fig, use_container_width=True)

    # --- MODE 2: CLUB BATTLE (NEW) ---
    elif app_mode == "Club Battle (New)":
        st.header("The Club Battle: Head-to-Head Depth")
        st.write("Compare two clubs across all their corresponding team ranks (1s vs 1s, 2s vs 2s, etc.)")
        
        col1, col2 = st.columns(2)
        club_a = col1.selectbox("Club A", clubs, index=0)
        club_b = col2.selectbox("Club B", clubs, index=1)
        
        # Filter and sort by team number
        df_a = rankings[rankings['Club'] == club_a].sort_values('TeamNum')
        df_b = rankings[rankings['Club'] == club_b].sort_values('TeamNum')
        
        # Merge on Team Number
        battle_df = pd.merge(df_a, df_b, on='TeamNum', suffixes=('_A', '_B'))
        battle_df['Winner'] = np.where(battle_df['SRS_A'] > battle_df['SRS_B'], club_a, club_b)
        battle_df['Margin'] = (battle_df['SRS_A'] - battle_df['SRS_B']).abs()

        if not battle_df.empty:
            wins_a = (battle_df['Winner'] == club_a).sum()
            wins_b = (battle_df['Winner'] == club_b).sum()
            
            st.divider()
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric(f"{club_a} Wins", wins_a)
            mc2.metric(f"{club_b} Wins", wins_b)
            mc3.metric("Projected Winner", club_a if wins_a > wins_b else club_b if wins_b > wins_a else "Split")

            # Comparison Chart
            plot_df = battle_df.melt(id_vars=['TeamNum'], value_vars=['SRS_A', 'SRS_B'], 
                                     var_name='ClubGroup', value_name='SRS')
            plot_df['ClubName'] = plot_df['ClubGroup'].map({'SRS_A': club_a, 'SRS_B': club_b})
            
            fig_battle = px.bar(plot_df, x='TeamNum', y='SRS', color='ClubName', barmode='group',
                                title=f"{club_a} vs {club_b} Power Comparison",
                                labels={'TeamNum': 'Team Rank (1s, 2s, 3s...)', 'SRS': 'Power Rating'})
            st.plotly_chart(fig_battle, use_container_width=True)
        else:
            st.warning("No matching team ranks found (e.g., one club might have 4 teams while the other only has 1).")

    # --- REMAINING MODES (Power Rankings, Deep Dive, Predictor) ---
    elif app_mode == "Power Rankings":
        st.header("Individual Team Power Rankings")
        st.dataframe(rankings[['Ranking', 'Team', 'Club', 'SRS']], use_container_width=True, hide_index=True)

    elif app_mode == "Team Deep Dive":
        st.header("Team Performance Search")
        c1, c2 = st.columns(2)
        sel_club = c1.selectbox("Select Club", clubs)
        sel_team = c2.selectbox("Select Team", rankings[rankings['Club'] == sel_club]['Team'])
        
        t_data = rankings[rankings['Team'] == sel_team].iloc[0]
        st.metric("Global Rank", f"#{t_data['Ranking']}", delta=f"SRS: {t_data['SRS']:.2f}")
        
        # (Simplified History Table)
        hist = df_raw[(df_raw['home_team'] == sel_team) | (df_raw['away_team'] == sel_team)].copy()
        st.dataframe(hist[['Display Date', 'home_team', 'home_score', 'away_score', 'away_team']], hide_index=True)

    elif app_mode == "Matchup Predictor":
        st.header("Score Margin Predictor")
        pc1, pc2 = st.columns(2)
        t_a = pc1.selectbox("Team A", rankings['Team'].tolist())
        t_b = pc2.selectbox("Team B", rankings['Team'].tolist())
        diff = rankings[rankings['Team'] == t_a]['SRS'].values[0] - rankings[rankings['Team'] == t_b]['SRS'].values[0]
        st.success(f"Projected Margin: {abs(diff):.1f} for {'Team A' if diff > 0 else 'Team B'}")

else:
    st.error("CSV file not found.")