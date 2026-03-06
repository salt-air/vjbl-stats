import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import re

# Set page configuration
st.set_page_config(page_title="VJBL U14 Boys Analytics Hub", layout="wide", page_icon="🏀")

# --- STATE MANAGEMENT ---
if 'page' not in st.session_state:
    st.session_state.page = "Power Rankings" 
if 'selected_team' not in st.session_state:
    st.session_state.selected_team = None

# --- DATA PROCESSING FUNCTIONS ---

def extract_club(team_name):
    """Extracts the Club name from the team string."""
    match = re.split(r'\s+U14\s+Boys|\s+Boys', str(team_name), flags=re.IGNORECASE)
    return match[0].strip()

@st.cache_data(ttl=300)
def load_and_clean_data(file_path):
    try:
        df = pd.read_csv(file_path)
        df.columns = [c.strip() for c in df.columns]
        
        def parse_vjbl_date(date_str):
            if pd.isna(date_str): return pd.NaT
            try:
                match = re.search(r'(\d{1,2}\s+[A-Za-z]+\s+\d{2,4})', str(date_str))
                if match: return pd.to_datetime(match.group(1))
                return pd.to_datetime(date_str, errors='coerce')
            except: return pd.NaT

        df['Date'] = df['game_datetime_raw'].apply(parse_vjbl_date)
        df['home_score'] = pd.to_numeric(df['home_score'], errors='coerce')
        df['away_score'] = pd.to_numeric(df['away_score'], errors='coerce')
        df = df.dropna(subset=['home_score', 'away_score'])
        return df
    except Exception as e:
        st.error(f"Error processing CSV: {e}")
        return None

@st.cache_data(ttl=300)
def calculate_ratings(df):
    """Calculates SRS Power Rankings using ALL games (including Grading) for accuracy."""
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
    results['TeamNum'] = results['Team'].str.extract(r'(\d+)$').fillna(0).astype(int)
    results = results.sort_values('SRS', ascending=False).reset_index(drop=True)
    results.insert(0, 'Ranking', range(1, len(results) + 1))
    return results

@st.cache_data(ttl=300)
def generate_ladders(df_season, rankings):
    """Calculates official ladders grouping strictly by League/Phase."""
    records = []
    
    # Group by Phase first so teams don't get cross-league points
    for phase in df_season['phase'].unique():
        phase_games = df_season[df_season['phase'] == phase]
        teams = set(phase_games['home_team'].tolist() + phase_games['away_team'].tolist())
        
        for team in teams:
            team_games = phase_games[(phase_games['home_team'] == team) | (phase_games['away_team'] == team)]
            w, l, d, pf, pa = 0, 0, 0, 0, 0
            
            for _, game in team_games.iterrows():
                is_home = game['home_team'] == team
                tm_pts = game['home_score'] if is_home else game['away_score']
                opp_pts = game['away_score'] if is_home else game['home_score']
                
                pf += tm_pts
                pa += opp_pts
                if tm_pts > opp_pts: w += 1
                elif tm_pts < opp_pts: l += 1
                else: d += 1
                    
            pts = (w * 3) + (d * 2) + (l * 1) 
            pct = (pf / pa * 100) if pa > 0 else 0
            
            pr = rankings[rankings['Team'] == team]['Ranking'].values
            pr_val = pr[0] if len(pr) > 0 else "-"
            
            records.append({
                "League": phase, "Team": team, "Pld": w+l+d, "W": w, "L": l, "D": d, 
                "PF": pf, "PA": pa, "%": round(pct, 2), "Pts": pts, "Power Rank": pr_val
            })
            
    return pd.DataFrame(records)

def style_game_result(row, team):
    is_home = row['home_team'] == team
    tm_pts = row['home_score'] if is_home else row['away_score']
    opp_pts = row['away_score'] if is_home else row['home_score']
    
    if tm_pts > opp_pts: return "🟢 W"
    elif tm_pts < opp_pts: return "🔴 L"
    else: return "⚪ D"

# --- APP NAVIGATION ---

st.title("🏀 VJBL U14 Boys: Advanced Analytics Hub")

CSV_FILE = 'u14_boys_all_results.csv' 
df_raw = load_and_clean_data(CSV_FILE)

if df_raw is not None and not df_raw.empty:
    # 1. Calculate Power Rankings using ALL games (including grading) for maximum mathematical accuracy
    rankings = calculate_ratings(df_raw)
    
    # 2. Filter out grading games for official season ladders
    df_season = df_raw[~df_raw['phase'].str.contains('Grading', case=False, na=False)].copy()
    ladders = generate_ladders(df_season, rankings)
    
    clubs = sorted(rankings['Club'].unique())
    
    # Sidebar Routing
    st.sidebar.header("Navigation")
    
    def set_page():
        st.session_state.page = st.session_state.nav_radio
        
    st.sidebar.radio(
        "View Mode", 
        ["Power Rankings", "League Ladders (New)", "Club Overview", "Club Battle", "Team Deep Dive", "Matchup Predictor"],
        key="nav_radio",
        index=["Power Rankings", "League Ladders (New)", "Club Overview", "Club Battle", "Team Deep Dive", "Matchup Predictor"].index(st.session_state.page),
        on_change=set_page
    )

    app_mode = st.session_state.page

    # --- MODE 1: POWER RANKINGS ---
    if app_mode == "Power Rankings":
        st.header("Individual Team Power Rankings")
        st.markdown("*Click a row below to instantly jump to that team's Deep Dive profile.*")
        
        event = st.dataframe(
            rankings[['Ranking', 'Team', 'Club', 'SRS']], 
            use_container_width=True, 
            hide_index=True,
            selection_mode="single-row",
            on_select="rerun"
        )
        
        if event and len(event.selection.rows) > 0:
            selected_idx = event.selection.rows[0]
            selected_team_name = rankings.iloc[selected_idx]['Team']
            st.session_state.selected_team = selected_team_name
            st.session_state.page = "Team Deep Dive"
            st.rerun()

    # --- MODE 2: LEAGUE LADDERS (NEW) ---
    elif app_mode == "League Ladders (New)":
        st.header("Official League Ladders vs Power Rankings")
        st.write("Compare actual standings (Points & %) against the predictive Power Rank.")
        
        if ladders.empty:
            st.warning("No official season games found yet. Only Grading games exist in the dataset.")
        else:
            leagues = sorted(ladders['League'].unique())
            selected_league = st.selectbox("Select League/Phase", leagues)
            
            if selected_league:
                league_ladder = ladders[ladders['League'] == selected_league].copy()
                league_ladder = league_ladder.sort_values(by=['Pts', '%'], ascending=[False, False]).reset_index(drop=True)
                league_ladder.insert(0, 'Pos', range(1, len(league_ladder) + 1))
                
                st.dataframe(
                    league_ladder[['Pos', 'Team', 'Pld', 'W', 'L', 'D', 'PF', 'PA', '%', 'Pts', 'Power Rank']], 
                    hide_index=True, use_container_width=True
                )

    # --- MODE 3: CLUB OVERVIEW ---
    elif app_mode == "Club Overview":
        st.header("Club Strength Index (CSI) - Top 10")
        
        club_stats = rankings.groupby('Club').agg(
            CSI=('SRS', 'mean'),
            Teams=('Team', 'count')
        ).sort_values('CSI', ascending=False).reset_index().head(10)

        c1, c2 = st.columns([1, 2])
        with c1:
            st.dataframe(club_stats, hide_index=True, use_container_width=True)
        with c2:
            top_10_clubs = club_stats['Club'].tolist()
            filtered_rankings = rankings[rankings['Club'].isin(top_10_clubs)]
            
            fig = px.box(filtered_rankings, x="Club", y="SRS", color="Club", 
                         title="Top 10 Program Depth Distribution",
                         category_orders={"Club": top_10_clubs})
            st.plotly_chart(fig, use_container_width=True)

    # --- MODE 4: CLUB BATTLE ---
    elif app_mode == "Club Battle":
        st.header("The Club Battle: Head-to-Head Depth")
        col1, col2 = st.columns(2)
        club_a = col1.selectbox("Club A", clubs, index=0)
        club_b = col2.selectbox("Club B", clubs, index=1 if len(clubs) > 1 else 0)
        
        df_a = rankings[rankings['Club'] == club_a].sort_values('TeamNum')
        df_b = rankings[rankings['Club'] == club_b].sort_values('TeamNum')
        battle_df = pd.merge(df_a, df_b, on='TeamNum', suffixes=('_A', '_B'))
        
        if not battle_df.empty:
            battle_df['Winner'] = np.where(battle_df['SRS_A'] > battle_df['SRS_B'], club_a, club_b)
            wins_a = (battle_df['Winner'] == club_a).sum()
            wins_b = (battle_df['Winner'] == club_b).sum()
            
            st.divider()
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric(f"{club_a} Wins", wins_a)
            mc2.metric(f"{club_b} Wins", wins_b)
            mc3.metric("Projected Winner", club_a if wins_a > wins_b else club_b if wins_b > wins_a else "Split")

            plot_df = battle_df.melt(id_vars=['TeamNum'], value_vars=['SRS_A', 'SRS_B'], var_name='ClubGroup', value_name='SRS')
            plot_df['ClubName'] = plot_df['ClubGroup'].map({'SRS_A': club_a, 'SRS_B': club_b})
            
            fig_battle = px.bar(plot_df, x='TeamNum', y='SRS', color='ClubName', barmode='group')
            st.plotly_chart(fig_battle, use_container_width=True)
        else:
            st.warning("No matching team ranks found.")

    # --- MODE 5: TEAM DEEP DIVE ---
    elif app_mode == "Team Deep Dive":
        st.header("Team Performance Search")
        
        default_team = st.session_state.selected_team if st.session_state.selected_team else rankings['Team'].iloc[0]
        default_club = extract_club(default_team)
        
        c1, c2 = st.columns(2)
        sel_club = c1.selectbox("Select Club", clubs, index=clubs.index(default_club) if default_club in clubs else 0)
        
        team_options = rankings[rankings['Club'] == sel_club]['Team'].tolist()
        team_idx = team_options.index(default_team) if default_team in team_options else 0
        sel_team = c2.selectbox("Select Team", team_options, index=team_idx)
        
        t_data = rankings[rankings['Team'] == sel_team].iloc[0]
        st.metric("Global Power Rank", f"#{t_data['Ranking']}", delta=f"SRS: {t_data['SRS']:.2f}")
        
        hist = df_raw[(df_raw['home_team'] == sel_team) | (df_raw['away_team'] == sel_team)].copy()
        
        if not hist.empty:
            hist['Result'] = hist.apply(lambda row: style_game_result(row, sel_team), axis=1)
            hist['Opponent'] = np.where(hist['home_team'] == sel_team, hist['away_team'], hist['home_team'])
            hist['Tm Pts'] = np.where(hist['home_team'] == sel_team, hist['home_score'], hist['away_score'])
            hist['Opp Pts'] = np.where(hist['home_team'] == sel_team, hist['away_score'], hist['home_score'])
            
            hist = hist.sort_values('Date', ascending=False)
            
            st.dataframe(
                hist[['Date', 'Result', 'Tm Pts', 'Opp Pts', 'Opponent', 'phase']], 
                hide_index=True, 
                use_container_width=True,
                column_config={
                    "Date": st.column_config.DateColumn("Match Date", format="DD MMM YYYY")
                }
            )

    # --- MODE 6: MATCHUP PREDICTOR ---
    elif app_mode == "Matchup Predictor":
        st.header("Advanced Matchup Predictor")
        st.write("Calculates predicted winner, expected margin, and algorithmic confidence.")
        
        pc1, pc2 = st.columns(2)
        t_a = pc1.selectbox("Home Team", rankings['Team'].tolist(), index=0)
        t_b = pc2.selectbox("Away Team", rankings['Team'].tolist(), index=1)
        
        if t_a != t_b:
            srs_a = rankings[rankings['Team'] == t_a]['SRS'].values[0]
            srs_b = rankings[rankings['Team'] == t_b]['SRS'].values[0]
            
            diff = srs_a - srs_b
            predicted_winner = t_a if diff > 0 else t_b
            margin = abs(diff)
            
            base_confidence = 50.0
            confidence_boost = min(margin * 2.5, 48.0) 
            confidence = base_confidence + confidence_boost
            
            st.divider()
            
            if margin < 1.0:
                st.warning(f"🔥 **TOSSUP MATCHUP!** Expected to be decided by 1 point or less.")
            else:
                st.success(f"🏆 **{predicted_winner}** is projected to win by **{margin:.1f} points**.")
                
            st.progress(int(confidence) / 100, text=f"Algorithm Confidence Score: {confidence:.1f}%")

else:
    st.error("No valid game data found. Please ensure your CSV is populated and named correctly.")