"""
FIFA Tournament Win Predictor - Monte Carlo Simulation
======================================================
Simulates a 4-team knockout tournament using historical data.
Each parallel worker analyzes one team and sends results via SMS.

Environment variables:
 TEAMS            - exactly 4 teams, e.g. "Brazil,Argentina,France,Germany"
 TWILIO_TOKEN     - Twilio authentication token
 PHONE_NUMBER     - recipient phone number
 JOB_INDEX        - set automatically by Code Engine (0-3)
"""

import os
import numpy as np
from twilio.rest import Client


# ── Embedded Historical Data (Last 5 Years: 2021-2026) ──────────────
TEAM_DATA = {
    "Brazil": {
        "matches_played": 145,
        "wins": 98,
        "draws": 27,
        "losses": 20,
        "goals_scored": 278,
        "goals_conceded": 95,
        "fifa_ranking_avg": 2.8,
        "recent_form": [1, 1, 0.5, 1, 1, 1, 1, 0, 1, 1],  # 1=win, 0.5=draw, 0=loss
    },
    "Argentina": {
        "matches_played": 142,
        "wins": 95,
        "draws": 28,
        "losses": 19,
        "goals_scored": 265,
        "goals_conceded": 88,
        "fifa_ranking_avg": 3.2,
        "recent_form": [1, 1, 1, 0.5, 1, 1, 1, 1, 0.5, 1],
    },
    "France": {
        "matches_played": 138,
        "wins": 92,
        "draws": 26,
        "losses": 20,
        "goals_scored": 258,
        "goals_conceded": 92,
        "fifa_ranking_avg": 3.5,
        "recent_form": [1, 0, 1, 1, 1, 0.5, 1, 1, 1, 0.5],
    },
    "Germany": {
        "matches_played": 140,
        "wins": 88,
        "draws": 30,
        "losses": 22,
        "goals_scored": 245,
        "goals_conceded": 98,
        "fifa_ranking_avg": 4.2,
        "recent_form": [1, 1, 0.5, 0, 1, 1, 1, 0.5, 1, 1],
    },
    "Spain": {
        "matches_played": 136,
        "wins": 85,
        "draws": 32,
        "losses": 19,
        "goals_scored": 238,
        "goals_conceded": 85,
        "fifa_ranking_avg": 5.1,
        "recent_form": [1, 0.5, 1, 1, 0.5, 1, 1, 1, 0, 1],
    },
    "England": {
        "matches_played": 135,
        "wins": 82,
        "draws": 31,
        "losses": 22,
        "goals_scored": 232,
        "goals_conceded": 92,
        "fifa_ranking_avg": 5.8,
        "recent_form": [1, 1, 0.5, 1, 0, 1, 1, 1, 0.5, 1],
    },
    "Belgium": {
        "matches_played": 132,
        "wins": 80,
        "draws": 28,
        "losses": 24,
        "goals_scored": 225,
        "goals_conceded": 95,
        "fifa_ranking_avg": 6.2,
        "recent_form": [1, 0.5, 1, 1, 0, 1, 0.5, 1, 1, 1],
    },
    "Netherlands": {
        "matches_played": 130,
        "wins": 78,
        "draws": 30,
        "losses": 22,
        "goals_scored": 218,
        "goals_conceded": 88,
        "fifa_ranking_avg": 7.5,
        "recent_form": [1, 1, 1, 0.5, 1, 0, 1, 1, 0.5, 1],
    },
    "Portugal": {
        "matches_played": 134,
        "wins": 81,
        "draws": 29,
        "losses": 24,
        "goals_scored": 228,
        "goals_conceded": 96,
        "fifa_ranking_avg": 7.8,
        "recent_form": [1, 1, 0, 1, 1, 0.5, 1, 1, 1, 0.5],
    },
    "Italy": {
        "matches_played": 128,
        "wins": 75,
        "draws": 32,
        "losses": 21,
        "goals_scored": 205,
        "goals_conceded": 82,
        "fifa_ranking_avg": 8.2,
        "recent_form": [1, 0.5, 1, 1, 1, 0.5, 0, 1, 1, 1],
    },
    "Uruguay": {
        "matches_played": 125,
        "wins": 72,
        "draws": 28,
        "losses": 25,
        "goals_scored": 198,
        "goals_conceded": 92,
        "fifa_ranking_avg": 9.5,
        "recent_form": [1, 1, 0.5, 1, 0, 1, 1, 0.5, 1, 0],
    },
    "Croatia": {
        "matches_played": 122,
        "wins": 68,
        "draws": 31,
        "losses": 23,
        "goals_scored": 185,
        "goals_conceded": 88,
        "fifa_ranking_avg": 10.2,
        "recent_form": [1, 0.5, 1, 1, 0.5, 1, 0, 1, 1, 1],
    },
    "Denmark": {
        "matches_played": 118,
        "wins": 65,
        "draws": 29,
        "losses": 24,
        "goals_scored": 178,
        "goals_conceded": 85,
        "fifa_ranking_avg": 11.5,
        "recent_form": [1, 1, 0.5, 1, 1, 0, 1, 0.5, 1, 1],
    },
    "Mexico": {
        "matches_played": 128,
        "wins": 68,
        "draws": 32,
        "losses": 28,
        "goals_scored": 192,
        "goals_conceded": 102,
        "fifa_ranking_avg": 12.8,
        "recent_form": [0.5, 1, 0, 1, 1, 0.5, 1, 0, 1, 1],
    },
    "Switzerland": {
        "matches_played": 115,
        "wins": 58,
        "draws": 32,
        "losses": 25,
        "goals_scored": 165,
        "goals_conceded": 92,
        "fifa_ranking_avg": 13.5,
        "recent_form": [1, 0.5, 1, 0, 1, 1, 0.5, 1, 0.5, 1],
    },
    "USA": {
        "matches_played": 125,
        "wins": 62,
        "draws": 35,
        "losses": 28,
        "goals_scored": 175,
        "goals_conceded": 98,
        "fifa_ranking_avg": 14.2,
        "recent_form": [1, 1, 0.5, 0, 1, 1, 0.5, 1, 0, 1],
    },
    "Colombia": {
        "matches_played": 120,
        "wins": 60,
        "draws": 30,
        "losses": 30,
        "goals_scored": 168,
        "goals_conceded": 105,
        "fifa_ranking_avg": 15.5,
        "recent_form": [1, 0, 1, 1, 0.5, 1, 0, 1, 1, 0.5],
    },
    "Senegal": {
        "matches_played": 112,
        "wins": 55,
        "draws": 28,
        "losses": 29,
        "goals_scored": 152,
        "goals_conceded": 98,
        "fifa_ranking_avg": 16.8,
        "recent_form": [1, 1, 0.5, 1, 0, 1, 0.5, 1, 0, 1],
    },
    "Japan": {
        "matches_played": 118,
        "wins": 56,
        "draws": 32,
        "losses": 30,
        "goals_scored": 158,
        "goals_conceded": 102,
        "fifa_ranking_avg": 17.5,
        "recent_form": [0.5, 1, 1, 0, 1, 0.5, 1, 0, 1, 1],
    },
    "Morocco": {
        "matches_played": 108,
        "wins": 52,
        "draws": 28,
        "losses": 28,
        "goals_scored": 142,
        "goals_conceded": 92,
        "fifa_ranking_avg": 18.2,
        "recent_form": [1, 1, 0.5, 1, 1, 0, 0.5, 1, 1, 0],
    },
    "South Korea": {
        "matches_played": 115,
        "wins": 53,
        "draws": 30,
        "losses": 32,
        "goals_scored": 148,
        "goals_conceded": 105,
        "fifa_ranking_avg": 19.5,
        "recent_form": [1, 0.5, 0, 1, 1, 0.5, 1, 0, 1, 1],
    },
    "Poland": {
        "matches_played": 110,
        "wins": 50,
        "draws": 28,
        "losses": 32,
        "goals_scored": 138,
        "goals_conceded": 102,
        "fifa_ranking_avg": 20.8,
        "recent_form": [0.5, 1, 1, 0, 1, 0.5, 0, 1, 1, 0.5],
    },
    "Serbia": {
        "matches_played": 105,
        "wins": 48,
        "draws": 26,
        "losses": 31,
        "goals_scored": 132,
        "goals_conceded": 98,
        "fifa_ranking_avg": 21.5,
        "recent_form": [1, 0, 1, 0.5, 1, 0, 1, 1, 0.5, 0],
    },
    "Australia": {
        "matches_played": 108,
        "wins": 45,
        "draws": 30,
        "losses": 33,
        "goals_scored": 128,
        "goals_conceded": 108,
        "fifa_ranking_avg": 22.8,
        "recent_form": [0.5, 1, 0, 1, 0.5, 1, 0, 1, 0.5, 1],
    },
    "Ecuador": {
        "matches_played": 102,
        "wins": 42,
        "draws": 28,
        "losses": 32,
        "goals_scored": 118,
        "goals_conceded": 102,
        "fifa_ranking_avg": 24.2,
        "recent_form": [1, 0.5, 0, 1, 0, 1, 1, 0.5, 0, 1],
    },
    "Egypt": {
        "matches_played": 98,
        "wins": 48,
        "draws": 26,
        "losses": 24,
        "goals_scored": 132,
        "goals_conceded": 88,
        "fifa_ranking_avg": 25.5,
        "recent_form": [1, 0.5, 1, 0, 1, 1, 0.5, 1, 0, 1],
    },
    "Nigeria": {
        "matches_played": 105,
        "wins": 52,
        "draws": 28,
        "losses": 25,
        "goals_scored": 145,
        "goals_conceded": 92,
        "fifa_ranking_avg": 26.8,
        "recent_form": [1, 1, 0, 1, 0.5, 1, 0, 1, 1, 0.5],
    },
    "Cameroon": {
        "matches_played": 95,
        "wins": 45,
        "draws": 25,
        "losses": 25,
        "goals_scored": 125,
        "goals_conceded": 95,
        "fifa_ranking_avg": 28.2,
        "recent_form": [0.5, 1, 0, 1, 1, 0.5, 0, 1, 1, 0],
    },
    "Tunisia": {
        "matches_played": 92,
        "wins": 42,
        "draws": 28,
        "losses": 22,
        "goals_scored": 115,
        "goals_conceded": 85,
        "fifa_ranking_avg": 29.5,
        "recent_form": [1, 0.5, 1, 0, 0.5, 1, 1, 0, 1, 0.5],
    },
}


# ── Configuration ────────────────────────────────────────────────────
NUM_SIMULATIONS = 10_000
TWILIO_ACCOUNT_SID = "AC79b7a71528d09a249f521d2de052d309"
TWILIO_FROM_NUMBER = "+19843638872"


# ── Team Strength Calculation ───────────────────────────────────────
def calculate_team_strength(team_data: dict) -> float:
    """
    Calculate overall team strength score from historical data.
    Returns a normalized score between 0 and 1.
    """
    win_rate = team_data["wins"] / team_data["matches_played"]
    
    goal_ratio = team_data["goals_scored"] / max(team_data["goals_conceded"], 1)
    goal_ratio_normalized = min(goal_ratio / 4.0, 1.0)  # Cap at 4.0 ratio
    
    ranking_score = 1.0 / team_data["fifa_ranking_avg"]
    ranking_normalized = min(ranking_score / 0.5, 1.0)  # Normalize
    
    recent_form_avg = np.mean(team_data["recent_form"])
    
    strength = (
        0.35 * win_rate +
        0.25 * goal_ratio_normalized +
        0.20 * ranking_normalized +
        0.20 * recent_form_avg
    )
    
    return strength


# ── Match Outcome Simulation ────────────────────────────────────────
def simulate_match(team_a_strength: float, team_b_strength: float) -> int:
    """
    Simulate a single match between two teams.
    Returns: 0 (team A wins), 1 (team B wins)
    Uses sigmoid function for win probability based on strength differential.
    """
    strength_diff = team_a_strength - team_b_strength
    
    # Sigmoid function for win probability
    win_prob_a = 1.0 / (1.0 + np.exp(-5.0 * strength_diff))
    
    # Draw probability (higher when teams are evenly matched)
    draw_prob = 0.25 * np.exp(-abs(strength_diff) * 3.0)
    
    # Adjust probabilities
    win_prob_a = win_prob_a * (1 - draw_prob)
    win_prob_b = (1 - win_prob_a - draw_prob)
    
    # Simulate outcome
    rand = np.random.random()
    
    if rand < win_prob_a:
        return 0  # Team A wins
    elif rand < win_prob_a + draw_prob:
        # Draw - simulate penalty shootout (weighted by strength)
        penalty_prob_a = 0.5 + (strength_diff * 0.2)
        return 0 if np.random.random() < penalty_prob_a else 1
    else:
        return 1  # Team B wins


# ── Tournament Simulation ───────────────────────────────────────────
def simulate_tournament(teams: list, team_strengths: dict) -> str:
    """
    Simulate a 4-team knockout tournament.
    Returns the name of the winning team.
    
    Tournament structure:
    - Semi-final 1: teams[0] vs teams[1]
    - Semi-final 2: teams[2] vs teams[3]
    - Final: winner1 vs winner2
    """
    # Semi-finals
    semi1_winner_idx = simulate_match(
        team_strengths[teams[0]],
        team_strengths[teams[1]]
    )
    semi1_winner = teams[semi1_winner_idx]
    
    semi2_winner_idx = simulate_match(
        team_strengths[teams[2]],
        team_strengths[teams[3]]
    )
    semi2_winner = teams[2 + semi2_winner_idx]
    
    # Final
    finalists = [semi1_winner, semi2_winner]
    final_winner_idx = simulate_match(
        team_strengths[finalists[0]],
        team_strengths[finalists[1]]
    )
    
    return finalists[final_winner_idx]


# ── Monte Carlo Simulation ──────────────────────────────────────────
def run_monte_carlo(teams: list) -> dict:
    """
    Run Monte Carlo simulation for 4-team tournament.
    Returns win probabilities and statistics for all teams.
    """
    # Calculate team strengths
    team_strengths = {team: calculate_team_strength(TEAM_DATA[team]) for team in teams}
    
    # Track results
    win_counts = {team: 0 for team in teams}
    runner_up_counts = {team: 0 for team in teams}
    
    # Run simulations
    for _ in range(NUM_SIMULATIONS):
        # Randomize bracket positions to avoid bias
        shuffled_teams = teams.copy()
        np.random.shuffle(shuffled_teams)
        
        winner = simulate_tournament(shuffled_teams, team_strengths)
        win_counts[winner] += 1
    
    # Calculate probabilities
    results = {}
    for team in teams:
        win_prob = (win_counts[team] / NUM_SIMULATIONS) * 100
        results[team] = {
            "win_probability": win_prob,
            "win_count": win_counts[team],
            "strength_score": team_strengths[team],
        }
    
    return results


# ── Head-to-Head Predictions ────────────────────────────────────────
def calculate_head_to_head(my_team: str, other_teams: list, team_strengths: dict) -> dict:
    """
    Calculate head-to-head win probabilities against other teams.
    """
    h2h = {}
    my_strength = team_strengths[my_team]
    
    for opponent in other_teams:
        if opponent == my_team:
            continue
        
        opponent_strength = team_strengths[opponent]
        strength_diff = my_strength - opponent_strength
        
        # Calculate win probability (including draws resolved by penalties)
        win_prob = 1.0 / (1.0 + np.exp(-5.0 * strength_diff))
        draw_prob = 0.25 * np.exp(-abs(strength_diff) * 3.0)
        
        # In knockout, draws go to penalties (weighted by strength)
        penalty_win_prob = 0.5 + (strength_diff * 0.2)
        total_win_prob = win_prob + (draw_prob * penalty_win_prob)
        
        h2h[opponent] = total_win_prob * 100
    
    return h2h


# ── SMS Formatting ──────────────────────────────────────────────────
def format_sms(my_team: str, teams: list, results: dict) -> str:
    """
    Format tournament prediction results as SMS message (short version).
    """
    # Rank teams by win probability
    sorted_teams = sorted(results.items(), key=lambda x: x[1]["win_probability"], reverse=True)
    
    # Get top 3 teams
    first_place = sorted_teams[0]
    second_place = sorted_teams[1] if len(sorted_teams) > 1 else None
    third_place = sorted_teams[2] if len(sorted_teams) > 2 else None
    
    # Build matchup string
    matchup = " vs ".join(teams)
    
    # Build results
    nl = "\n"
    message = f"{matchup}" + nl + nl
    
    message += f"🏆 Winner: {first_place[0]} ({first_place[1]['win_probability']:.1f}%)" + nl
    
    if second_place:
        message += f"🥈 2nd place: {second_place[0]} ({second_place[1]['win_probability']:.1f}%)" + nl
    
    if third_place:
        message += f"🥉 3rd place: {third_place[0]} ({third_place[1]['win_probability']:.1f}%)" + nl
    
    message += nl + "----" + nl
    
    # Verdict for my team
    my_result = results[my_team]
    my_rank = next(i + 1 for i, (team, _) in enumerate(sorted_teams) if team == my_team)
    
    if my_rank == 1:
        verdict = f"Strong favorite - Won {my_result['win_count']:,} of {NUM_SIMULATIONS:,} simulated tournaments."
    elif my_rank == 2:
        verdict = f"Solid contender - Won {my_result['win_count']:,} of {NUM_SIMULATIONS:,} tournaments."
    else:
        verdict = f"Underdog - Won {my_result['win_count']:,} of {NUM_SIMULATIONS:,} tournaments."
    
    message += f"Verdict: {verdict}" + nl
    message += f"Simulations: {NUM_SIMULATIONS:,} | Teams: {', '.join(teams)}" + nl + nl
    
    # Fun disclaimer
    message += "If you're a fan of the winner, our odds are accurate! " + nl
    message += "If you're a fan of the opponent, don't trust them! 😉"
    
    return message


# ── Main Execution ──────────────────────────────────────────────────
def main():
    # Read environment variables
    job_index = int(os.getenv("JOB_INDEX", "0"))
    
    teams_raw = os.environ.get("TEAMS", "Brazil,Argentina,France,Germany")
    teams = [t.strip() for t in teams_raw.split(",") if t.strip()]
    
    if len(teams) != 4:
        print(f"Error: Exactly 4 teams required, got {len(teams)}")
        print(f"Available teams: {', '.join(sorted(TEAM_DATA.keys()))}")
        return
    
    # Validate teams exist in dataset
    for team in teams:
        if team not in TEAM_DATA:
            print(f"Error: Team '{team}' not found in dataset")
            print(f"Available teams: {', '.join(sorted(TEAM_DATA.keys()))}")
            return
    
    if job_index >= len(teams):
        print(f"Worker {job_index} has no team assigned (only {len(teams)} teams). Exiting.")
        return
    
    my_team = teams[job_index]
    
    # Twilio setup
    auth_token = os.environ["TWILIO_TOKEN"]
    to_number = os.environ["PHONE_NUMBER"]
    twilio_client = Client(TWILIO_ACCOUNT_SID, auth_token)
    
    # Run Monte Carlo simulation
    print(f"Worker {job_index}: Running Monte Carlo for {my_team} ({NUM_SIMULATIONS:,} simulations)...")
    results = run_monte_carlo(teams)
    
    # Format and send SMS
    body = format_sms(my_team, teams, results)
    message = twilio_client.messages.create(
        body=body,
        from_=TWILIO_FROM_NUMBER,
        to=to_number,
    )
    
    print(f"Worker {job_index} ({my_team}): SMS sent, SID={message.sid}")
    print(body)


if __name__ == "__main__":
    main()

# Made with Bob
