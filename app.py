import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Set page configuration
st.set_page_config(page_title="VJBL U14 Boys Analytics Hub", layout="wide")

# --- DATA PROCESSING FUNCTIONS ---

@st.cache_data
def load_and_clean_data(file_path):
    """Loads and prepares the VJBL dataset with specific date and phase logic."""
    try:
        df = pd.read_csv(file_path)
        # Clean column names
        df.columns = [c.strip() for c in df.columns]
        
        # 1. Date Extraction (From "06:40 PM, Friday, 12 Dec 2025")
        # Split by comma and take the 3rd part (the date)
        date_raw = df['game_datetime_raw'].str.split(',').str[2].str.strip()
        df['date_obj'] = pd.to_datetime(date_raw, format='%d %b %Y', errors='coerce')
        # Create display version: DD-MMM (e.g., 12-Dec)
        df['Display Date'] = df['date_obj'].dt.strftime('%d-%b')
        
        # 2. Phase Normalization
        def clean_phase(phase_str):
            p = str(phase_str).lower()
            if "grading 1" in p:
                if "pool" in p: return "Grading 1 Pool"
                if "crossover" in p: return "Grading 1 Crossover"
            elif "grading 2" in p:
                if "pool" in p: return "Grading 2 Pool"
            return "Other"
        
        df['Phase Category'] = df['phase'].apply(clean_phase)
        
        # 3. Numeric Score Conversion
        df['home_score'] = pd.to_numeric(df['home_score'], errors='coerce')
        df['away_score'] = pd.to_numeric(df['away_score'], errors='coerce')
        
        # Drop rows with missing crucial data
        df = df.dropna(subset=['home_score', 'away_score', 'date_obj'])
        
        # Add Margin for SRS calculation
        df['margin'] = df['home_score'] - df['away_score']
        
        return df
    except Exception as e:
        st.error(f"Error processing CSV: {e}")
        return None

def calculate_srs(df):
    """Calculates Simple Rating System (SRS) and assigns Rank."""
    teams = sorted(list(set(df['home_team'].tolist() + df['away_team'].tolist())))
    n = len(teams)
    team_to_idx = {team: i for i, team in enumerate(teams)}
    
    A = np.zeros((n, n))
    b = np.zeros(n)
    
    for i, team in enumerate(teams):
        team_games = df[(df['home_team'] == team) | (df['away_team'] == team)]
        A[i, i] = len(team_games)
        
        total_margin = 0
        for _, game in team_games.iterrows():
            if game['home_team'] == team:
                opp = game['away_team']
                total_margin += (game['home_score'] - game['away_score'])
            else:
                opp = game['home_team']
                total_margin += (game['away_score'] - game['home_score'])
            
            opp_idx = team_to_idx[opp]
            A[i, opp_idx] -= 1
        b[i] = total_margin
    
    # Solve linear system with a sum-to-zero constraint
    A[-1, :] = 1
    b[-1] = 0
    
    try:
        ratings = np.linalg.lstsq(A, b, rcond=None)[0]
    except:
        ratings = np.zeros(n)
        
    results = pd.DataFrame({'Team': teams, 'SRS': ratings})
    results = results.sort_values('SRS', ascending=False).reset_index(drop=True)
    
    # Create the Ranking column (1, 2, 3...)
    results.insert(0, 'Ranking', range(1, len(results) + 1))
    
    return results

# --- APP LAYOUT ---

st.title("🏀 VJBL U14 Boys: Advanced Analytics Engine")

# Load the local file
CSV_FILE = 'u14_boys_all_results.csv'
df_raw = load_and_clean_data(CSV_FILE)

if df_raw is not None:
    # Pre-calculate global rankings
    rankings = calculate_srs(df_raw)
    
    # Sidebar Navigation
    st.sidebar.header("Navigation")
    app_mode = st.sidebar.radio("View", ["Power Rankings", "Team Deep Dive", "Matchup Predictor"])

    # Define targets for highlighting
    target_geelong = "Geelong United U14 Boys 4"
    target_casey = "Casey U14 Boys 6"

    # --- MODE 1: POWER RANKINGS ---
    if app_mode == "Power Rankings":
        st.header("VJBL Power Rankings (SRS)")
        st.markdown("Rankings are based on score margins adjusted for strength of schedule.")
        
        # Display table and hide the dataframe's internal index
        st.dataframe(rankings, use_container_width=True, hide_index=True)
        
        # Simple viz for top 15
        fig = px.bar(rankings.head(15), x='Team', y='SRS', 
                     title="Top 15 Teams by Performance Rating",
                     labels={'SRS': 'Power Score (SRS)'})
        st.plotly_chart(fig, use_container_width=True)

    # --- MODE 2: TEAM DEEP DIVE ---
    elif app_mode == "Team Deep Dive":
        st.header("Individual Team Analysis")
        
        selected_team = st.selectbox("Select Team", rankings['Team'].tolist(), 
                                     index=rankings['Team'].tolist().index(target_geelong) if target_geelong in rankings['Team'].tolist() else 0)
        
        # Filter for the team and SORT BY DATE (Oldest First)
        team_df = df_raw[(df_raw['home_team'] == selected_team) | (df_raw['away_team'] == selected_team)].copy()
        team_df = team_df.sort_values('date_obj', ascending=True)
        
        # Metrics
        team_rank = rankings[rankings['Team'] == selected_team]['Ranking'].values[0]
        team_srs = rankings[rankings['Team'] == selected_team]['SRS'].values[0]
        
        c1, c2 = st.columns(2)
        c1.metric("Overall Ranking", f"#{team_rank}")
        c2.metric("Power Score (SRS)", round(team_srs, 2))
        
        st.subheader("Recent Game Results")
        
        # Formatting for display table
        def format_log(row):
            if row['home_team'] == selected_team:
                opp = row['away_team']
                res = "W" if row['home_score'] > row['away_score'] else "L"
                score_str = f"{int(row['home_score'])} - {int(row['away_score'])}"
            else:
                opp = row['home_team']
                res = "W" if row['away_score'] > row['home_score'] else "L"
                score_str = f"{int(row['away_score'])} - {int(row['home_score'])}"
            
            return pd.Series([
                row['Display Date'], 
                row['Phase Category'], 
                row['round_label'], 
                opp, 
                score_str, 
                res
            ])

        if not team_df.empty:
            log_display = team_df.apply(format_log, axis=1)
            log_display.columns = ['Date', 'Phase', 'Round', 'Opponent', 'Score', 'Result']
            # Display sorted table and hide index
            st.dataframe(log_display, use_container_width=True, hide_index=True)
        else:
            st.warning("No game data found for this team.")

    # --- MODE 3: MATCHUP PREDICTOR ---
    elif app_mode == "Matchup Predictor":
        st.header("Head-to-Head Predictor")
        st.info("Uses the difference in SRS ratings to estimate a spread.")
        
        c1, c2 = st.columns(2)
        t_a = c1.selectbox("Team 1", rankings['Team'].tolist(), index=0)
        t_b = c2.selectbox("Team 2", rankings['Team'].tolist(), index=1)
        
        srs_a = rankings[rankings['Team'] == t_a]['SRS'].values[0]
        srs_b = rankings[rankings['Team'] == t_b]['SRS'].values[0]
        spread = srs_a - srs_b
        
        st.divider()
        if spread > 0:
            st.success(f"**{t_a}** is projected to win by **{abs(spread):.1f} points**.")
        elif spread < 0:
            st.error(f"**{t_b}** is projected to win by **{abs(spread):.1f} points**.")
        else:
            st.write("Matchup is projected to be a Draw.")

else:
    st.error("Missing 'u14_boys_all_results.csv'. Ensure the file is in the same folder as this script.")