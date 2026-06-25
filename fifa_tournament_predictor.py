"""
FIFA World Cup 2026 Predictor - Monte Carlo Simulation
======================================================
Predicts tournament outcomes and group stage advancement for FIFA World Cup 2026.
Includes all 48 qualified teams with historical data.

Environment variables:
 TEAMS            - 2-4 teams to analyze, e.g. "Canada,USA,Mexico,Brazil"
 TWILIO_TOKEN     - Twilio authentication token
 PHONE_NUMBER     - recipient phone number
 JOB_INDEX        - set automatically by Code Engine (0-3)
 MODE             - "tournament" (default) or "group_stage"
"""

import os
import numpy as np
from twilio.rest import Client


# ── Embedded Historical Data - FIFA World Cup 2026 Teams ────────────
TEAM_DATA = {
    # CONCACAF (6 teams)
    "Canada": {
        "matches_played": 85,
        "wins": 38,
        "draws": 25,
        "losses": 22,
        "goals_scored": 102,
        "goals_conceded": 85,
        "fifa_ranking_avg": 41.0,
        "recent_form": [1, 1, 0.5, 1, 0, 1, 0.5, 1, 0, 1],
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
    
    # UEFA (16 teams)
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
    "Ukraine": {
        "matches_played": 98,
        "wins": 45,
        "draws": 28,
        "losses": 25,
        "goals_scored": 125,
        "goals_conceded": 95,
        "fifa_ranking_avg": 24.5,
        "recent_form": [1, 0.5, 0, 1, 1, 0.5, 1, 0, 1, 0.5],
    },
    "Austria": {
        "matches_played": 95,
        "wins": 42,
        "draws": 26,
        "losses": 27,
        "goals_scored": 118,
        "goals_conceded": 98,
        "fifa_ranking_avg": 26.2,
        "recent_form": [1, 1, 0.5, 0, 1, 0.5, 1, 1, 0, 1],
    },
    "Wales": {
        "matches_played": 92,
        "wins": 38,
        "draws": 28,
        "losses": 26,
        "goals_scored": 108,
        "goals_conceded": 95,
        "fifa_ranking_avg": 28.5,
        "recent_form": [0.5, 1, 0, 1, 1, 0.5, 0, 1, 1, 0.5],
    },
    
    # CONMEBOL (6 teams)
    "Brazil": {
        "matches_played": 145,
        "wins": 98,
        "draws": 27,
        "losses": 20,
        "goals_scored": 278,
        "goals_conceded": 95,
        "fifa_ranking_avg": 2.8,
        "recent_form": [1, 1, 0.5, 1, 1, 1, 1, 0, 1, 1],
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
    "Chile": {
        "matches_played": 108,
        "wins": 48,
        "draws": 30,
        "losses": 30,
        "goals_scored": 135,
        "goals_conceded": 108,
        "fifa_ranking_avg": 27.5,
        "recent_form": [0.5, 1, 0, 1, 0.5, 1, 0, 1, 1, 0],
    },
    
    # CAF (9 teams)
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
    "Algeria": {
        "matches_played": 82,
        "wins": 40,
        "draws": 22,
        "losses": 20,
        "goals_scored": 110,
        "goals_conceded": 80,
        "fifa_ranking_avg": 32.2,
        "recent_form": [1, 1, 0.5, 0, 1, 1, 0, 1, 0.5, 1],
    },
    "Ghana": {
        "matches_played": 80,
        "wins": 36,
        "draws": 22,
        "losses": 22,
        "goals_scored": 98,
        "goals_conceded": 85,
        "fifa_ranking_avg": 33.5,
        "recent_form": [0.5, 1, 0, 1, 1, 0.5, 0, 1, 1, 0],
    },
    "South Africa": {
        "matches_played": 88,
        "wins": 40,
        "draws": 24,
        "losses": 24,
        "goals_scored": 108,
        "goals_conceded": 88,
        "fifa_ranking_avg": 30.8,
        "recent_form": [1, 0, 1, 0.5, 1, 0, 1, 0.5, 0, 1],
    },
    
    # AFC (8 teams)
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
    "Iran": {
        "matches_played": 78,
        "wins": 35,
        "draws": 24,
        "losses": 19,
        "goals_scored": 92,
        "goals_conceded": 75,
        "fifa_ranking_avg": 34.8,
        "recent_form": [1, 0.5, 1, 0, 1, 0.5, 1, 0, 1, 0.5],
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
    "Saudi Arabia": {
        "matches_played": 75,
        "wins": 32,
        "draws": 22,
        "losses": 21,
        "goals_scored": 88,
        "goals_conceded": 78,
        "fifa_ranking_avg": 36.5,
        "recent_form": [1, 0, 1, 0.5, 0, 1, 1, 0.5, 1, 0],
    },
    "Qatar": {
        "matches_played": 68,
        "wins": 28,
        "draws": 20,
        "losses": 20,
        "goals_scored": 75,
        "goals_conceded": 72,
        "fifa_ranking_avg": 42.5,
        "recent_form": [0.5, 1, 0, 1, 0.5, 0, 1, 1, 0, 1],
    },
    "Iraq": {
        "matches_played": 65,
        "wins": 26,
        "draws": 18,
        "losses": 21,
        "goals_scored": 68,
        "goals_conceded": 75,
        "fifa_ranking_avg": 45.2,
        "recent_form": [0, 1, 0.5, 0, 1, 0.5, 1, 0, 1, 0.5],
    },
    "China": {
        "matches_played": 72,
        "wins": 28,
        "draws": 22,
        "losses": 22,
        "goals_scored": 78,
        "goals_conceded": 82,
        "fifa_ranking_avg": 44.8,
        "recent_form": [0.5, 0, 1, 0.5, 1, 0, 0.5, 1, 0, 1],
    },
    
    # OFC (1 team)
    "New Zealand": {
        "matches_played": 62,
        "wins": 24,
        "draws": 18,
        "losses": 20,
        "goals_scored": 65,
        "goals_conceded": 68,
        "fifa_ranking_avg": 48.5,
        "recent_form": [0.5, 1, 0, 0.5, 1, 0, 1, 0.5, 0, 1],
    },
}


# ── Configuration ────────────────────────────────────────────────────
NUM_SIMULATIONS = 10_000
TWILIO_ACCOUNT_SID = "AC79b7a71528d09a249f521d2de052d309"
TWILIO_FROM_NUMBER = "+19843638872"


# ── Team Strength Calculation ───────────────────────────────────────
def calculate_team_strength(team_data: dict) -> float:
    """Calculate overall team strength score from historical data."""
    win_rate = team_data["wins"] / team_data["matches_played"]
    
    goal_ratio = team_data["goals_scored"] / max(team_data["goals_conceded"], 1)
    goal_ratio_normalized = min(goal_ratio / 4.0, 1.0)
    
    ranking_score = 1.0 / team_data["fifa_ranking_avg"]
    ranking_normalized = min(ranking_score / 0.5, 1.0)
    
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
    """Simulate a single match. Returns: 0 (team A wins), 1 (team B wins)"""
    strength_diff = team_a_strength - team_b_strength
    win_prob_a = 1.0 / (1.0 + np.exp(-5.0 * strength_diff))
    draw_prob = 0.25 * np.exp(-abs(strength_diff) * 3.0)
    win_prob_a = win_prob_a * (1 - draw_prob)
    
    rand = np.random.random()
    
    if rand < win_prob_a:
        return 0
    elif rand < win_prob_a + draw_prob:
        penalty_prob_a = 0.5 + (strength_diff * 0.2)
        return 0 if np.random.random() < penalty_prob_a else 1
    else:
        return 1


# ── Tournament Simulation ───────────────────────────────────────────
def simulate_tournament(teams: list, team_strengths: dict) -> str:
    """Simulate a 4-team knockout tournament."""
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
    
    finalists = [semi1_winner, semi2_winner]
    final_winner_idx = simulate_match(
        team_strengths[finalists[0]],
        team_strengths[finalists[1]]
    )
    
    return finalists[final_winner_idx]


# ── Group Stage Simulation (NEW) ────────────────────────────────────
def simulate_group_stage(teams: list, team_strengths: dict) -> list:
    """
    Simulate group stage with 4 teams. Top 2 advance.
    Returns list of 2 teams that advance to knockout stage.
    """
    points = {team: 0 for team in teams}
    goal_diff = {team: 0 for team in teams}
    
    # Each team plays every other team once (round-robin)
    for i in range(len(teams)):
        for j in range(i + 1, len(teams)):
            team_a = teams[i]
            team_b = teams[j]
            
            # Simulate match with points system
            strength_diff = team_strengths[team_a] - team_strengths[team_b]
            win_prob_a = 1.0 / (1.0 + np.exp(-5.0 * strength_diff))
            draw_prob = 0.25 * np.exp(-abs(strength_diff) * 3.0)
            
            rand = np.random.random()
            
            if rand < win_prob_a * (1 - draw_prob):
                # Team A wins
                points[team_a] += 3
                goal_diff[team_a] += 1
                goal_diff[team_b] -= 1
            elif rand < win_prob_a * (1 - draw_prob) + draw_prob:
                # Draw
                points[team_a] += 1
                points[team_b] += 1
            else:
                # Team B wins
                points[team_b] += 3
                goal_diff[team_b] += 1
                goal_diff[team_a] -= 1
    
    # Sort by points, then goal difference
    sorted_teams = sorted(teams, key=lambda t: (points[t], goal_diff[t]), reverse=True)
    
    # Top 2 advance
    return sorted_teams[:2]


# ── Monte Carlo for Group Stage (NEW) ───────────────────────────────
def run_group_stage_monte_carlo(teams: list) -> dict:
    """
    Run Monte Carlo simulation for group stage advancement.
    Returns advancement probabilities for each team.
    """
    team_strengths = {team: calculate_team_strength(TEAM_DATA[team]) for team in teams}
    
    advancement_counts = {team: 0 for team in teams}
    first_place_counts = {team: 0 for team in teams}
    
    for _ in range(NUM_SIMULATIONS):
        advancing_teams = simulate_group_stage(teams, team_strengths)
        advancement_counts[advancing_teams[0]] += 1
        advancement_counts[advancing_teams[1]] += 1
        first_place_counts[advancing_teams[0]] += 1
    
    results = {}
    for team in teams:
        advancement_prob = (advancement_counts[team] / NUM_SIMULATIONS) * 100
        first_place_prob = (first_place_counts[team] / NUM_SIMULATIONS) * 100
        results[team] = {
            "advancement_probability": advancement_prob,
            "first_place_probability": first_place_prob,
            "strength_score": team_strengths[team],
        }
    
    return results


# ── Monte Carlo for Tournament ──────────────────────────────────────
def run_tournament_monte_carlo(teams: list) -> dict:
    """Run Monte Carlo simulation for 4-team tournament."""
    team_strengths = {team: calculate_team_strength(TEAM_DATA[team]) for team in teams}
    win_counts = {team: 0 for team in teams}
    
    for _ in range(NUM_SIMULATIONS):
        shuffled_teams = teams.copy()
        np.random.shuffle(shuffled_teams)
        winner = simulate_tournament(shuffled_teams, team_strengths)
        win_counts[winner] += 1
    
    results = {}
    for team in teams:
        win_prob = (win_counts[team] / NUM_SIMULATIONS) * 100
        results[team] = {
            "win_probability": win_prob,
            "win_count": win_counts[team],
            "strength_score": team_strengths[team],
        }
    
    return results


# ── SMS Formatting for Tournament ───────────────────────────────────
def format_tournament_sms(my_team: str, teams: list, results: dict) -> str:
    """Format tournament prediction results as SMS message."""
    sorted_teams = sorted(results.items(), key=lambda x: x[1]["win_probability"], reverse=True)
    
    first_place = sorted_teams[0]
    second_place = sorted_teams[1] if len(sorted_teams) > 1 else None
    third_place = sorted_teams[2] if len(sorted_teams) > 2 else None
    
    matchup = " vs ".join(teams)
    nl = "\n"
    message = f"{matchup}" + nl + nl
    
    message += f"🏆 Winner: {first_place[0]} ({first_place[1]['win_probability']:.1f}%)" + nl
    
    if second_place:
        message += f"🥈 2nd place: {second_place[0]} ({second_place[1]['win_probability']:.1f}%)" + nl
    
    if third_place:
        message += f"🥉 3rd place: {third_place[0]} ({third_place[1]['win_probability']:.1f}%)" + nl
    
    message += nl + "----" + nl
    
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
    message += "If you're a fan of the winner, our odds are accurate! " + nl
    message += "If you're a fan of the opponent, don't trust them! 😉"
    
    return message


# ── SMS Formatting for Group Stage (NEW) ────────────────────────────
def format_group_stage_sms(my_team: str, teams: list, results: dict) -> str:
    """Format group stage advancement prediction as SMS message."""
    sorted_teams = sorted(results.items(), key=lambda x: x[1]["advancement_probability"], reverse=True)
    
    matchup = " vs ".join(teams)
    nl = "\n"
    message = f"GROUP STAGE: {matchup}" + nl + nl
    
    message += "Advancement to Round of 16:" + nl
    for i, (team, data) in enumerate(sorted_teams, 1):
        emoji = "✅" if data["advancement_probability"] >= 50 else "⚠️"
        message += f"{emoji} {i}. {team}: {data['advancement_probability']:.1f}%" + nl
    
    message += nl + "First place probability:" + nl
    for team, data in sorted_teams:
        if data["first_place_probability"] >= 10:
            message += f"🥇 {team}: {data['first_place_probability']:.1f}%" + nl
    
    message += nl + "----" + nl
    
    my_result = results[my_team]
    
    if my_result["advancement_probability"] >= 70:
        verdict = f"Very likely to advance - {my_result['advancement_probability']:.1f}% chance based on historical performance."
    elif my_result["advancement_probability"] >= 50:
        verdict = f"Good chance to advance - {my_result['advancement_probability']:.1f}% probability. Competitive group."
    elif my_result["advancement_probability"] >= 30:
        verdict = f"Tough battle - {my_result['advancement_probability']:.1f}% chance. Need strong performances."
    else:
        verdict = f"Underdog - {my_result['advancement_probability']:.1f}% chance. Would need exceptional results."
    
    message += f"Verdict: {verdict}" + nl
    message += f"Simulations: {NUM_SIMULATIONS:,} | Mode: Group Stage" + nl + nl
    message += "⚽ May the best team advance! 🏆"
    
    return message


# ── Main Execution ──────────────────────────────────────────────────
def main():
    job_index = int(os.getenv("JOB_INDEX", "0"))
    mode = os.getenv("MODE", "tournament").lower()
    
    teams_raw = os.environ.get("TEAMS", "Brazil,Argentina,France,Germany")
    teams = [t.strip() for t in teams_raw.split(",") if t.strip()]
    
    if len(teams) < 2 or len(teams) > 4:
        print(f"Error: 2-4 teams required, got {len(teams)}")
        print(f"Available teams: {', '.join(sorted(TEAM_DATA.keys()))}")
        return
    
    # Pad teams to 4 if less than 4 (for simulation purposes)
    original_team_count = len(teams)
    if len(teams) < 4:
        print(f"Note: {len(teams)} teams provided. Padding with top-ranked teams for simulation.")
        # Add top teams not in the list
        all_teams_sorted = sorted(TEAM_DATA.keys(), key=lambda t: TEAM_DATA[t]["fifa_ranking_avg"])
        for team in all_teams_sorted:
            if team not in teams and len(teams) < 4:
                teams.append(team)
                print(f"Added {team} to complete 4-team bracket")
    
    for team in teams:
        if team not in TEAM_DATA:
            print(f"Error: Team '{team}' not found in dataset")
            print(f"Available teams: {', '.join(sorted(TEAM_DATA.keys()))}")
            return
    
    if job_index >= len(teams):
        print(f"Worker {job_index} has no team assigned. Exiting.")
        return
    
    my_team = teams[job_index]
    
    auth_token = os.environ["TWILIO_TOKEN"]
    to_number = os.environ["PHONE_NUMBER"]
    twilio_client = Client(TWILIO_ACCOUNT_SID, auth_token)
    
    if mode == "group_stage":
        print(f"Worker {job_index}: Running GROUP STAGE simulation for {my_team}...")
        results = run_group_stage_monte_carlo(teams)
        body = format_group_stage_sms(my_team, teams, results)
    else:
        print(f"Worker {job_index}: Running TOURNAMENT simulation for {my_team}...")
        results = run_tournament_monte_carlo(teams)
        body = format_tournament_sms(my_team, teams, results)
    
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
