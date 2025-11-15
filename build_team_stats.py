# Purpose:
#   Build team-level features that summarize historical performance (dominance, form),
#   using full post-game stats ONLY from PAST matches. These aggregates can be joined
#   to pre-game predictions (by team names) without leaking future information.
#
# Outputs (saved to --out-dir, default: artifacts):
#   - team_stats_overall.csv : overall aggregates up to --cutoff-date
#   - team_stats_lastN.csv   : rolling-window aggregates (last N games per team)
#   - team_elo.csv           : optional Elo ratings up to cutoff (if --elo)
#   - team_stats_meta.json   : metadata (cutoff, window size, columns)
#
# Usage examples:
#   python build_team_stats.py --data lec_2023-2025_games.csv --cutoff-date 2025-03-01 \
#     --window-games 10 --min-games 5 --elo --k 24 --out-dir artifacts
#
# Expected CSV columns (case-sensitive):
#   game_id, game_date, team_blue, team_red, winning_team, gold_diff, gold_diff_at14, game_time
#
# Notes:
#   - The script rewrites match data into a team-perspective table (2 rows per match),
#     so that gold_diff_* are from the viewpoint of each team (positive means good for that team).
#   - If some columns are missing, the script will warn and compute only available features.

import argparse
import json
import os
from dataclasses import dataclass
from typing import List, Dict
import numpy as np
import pandas as pd

REQUIRED_BASE = {"game_id", "game_date", "team_blue", "team_red", "winning_team"}
OPTIONAL_NUMERIC = ["gold_diff", "gold_diff_at14", "game_time"]


@dataclass
class Config:
    data: str
    out_dir: str = "data"
    cutoff_date: str | None = None  # YYYY-MM-DD
    window_games: int = 10
    min_games: int = 5
    use_elo: bool = False
    k_factor: float = 24.0

def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Build team strength aggregates from historical matches")
    p.add_argument("--data", default="data/lec_2023-2025_games.csv", required=True, help="Path to LEC matches CSV")
    p.add_argument("--out-dir", default="summer", help="Directory for outputs")
    p.add_argument("--cutoff-date", default="2025-05-25", help="YYYY-MM-DD (use matches strictly < cutoff)")
    p.add_argument("--window-games", type=int, default=10, help="Rolling window size per team")
    p.add_argument("--min-games", type=int, default=5, help="Minimum games required to keep a row")
    p.add_argument("--elo", action="store_true", help="Compute Elo ratings up to cutoff")
    p.add_argument("--k", dest="k_factor", type=float, default=24.0, help="Elo K-factor")
    a = p.parse_args()
    return Config(
        data=a.data,
        out_dir=a.out_dir,
        cutoff_date=a.cutoff_date,
        window_games=a.window_games,
        min_games=a.min_games,
        use_elo=a.elo,
        k_factor=a.k_factor,
    )


def load_matches(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED_BASE - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV missing required columns: {sorted(missing)}")
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    if df["game_date"].isna().any():
        raise ValueError("Some game_date values could not be parsed. Please fix the CSV.")
    # Ensure numeric columns exist or create NaNs
    for c in OPTIONAL_NUMERIC:
        if c not in df.columns:
            df[c] = np.nan
    # Sort chronologically for rolling & Elo
    df = df.sort_values("game_date").reset_index(drop=True)
    return df

def filter_cutoff(df: pd.DataFrame, cutoff: str | None) -> pd.DataFrame:
    if cutoff is None:
        return df
    cutoff_dt = pd.to_datetime(cutoff)
    return df[df["game_date"] < cutoff_dt].copy()

def to_team_perspective(df: pd.DataFrame) -> pd.DataFrame:
    """Duplicate each match into two rows: one for blue team, one for red team.
    Create perspective metrics so that positive values mean good for 'team'.
    """
    rows = []
    for _, r in df.iterrows():
        # Blue perspective
        blue_win = int(str(r["winning_team"]) == str(r["team_blue"]))
        rows.append({
            "game_id": r["game_id"],
            "game_date": r["game_date"],
            "team": r["team_blue"],
            "opponent": r["team_red"],
            "is_blue": 1,
            "win": blue_win,
            "gold_diff_team": r["gold_diff"] if pd.notna(r["gold_diff"]) else np.nan,
            "gold_diff_at14_team": r["gold_diff_at14"] if pd.notna(r["gold_diff_at14"]) else np.nan,
            "game_time": r["game_time"] if pd.notna(r["game_time"]) else np.nan,
        })
        # Red perspective
        red_win = 1 - blue_win
        rows.append({
            "game_id": r["game_id"],
            "game_date": r["game_date"],
            "team": r["team_red"],
            "opponent": r["team_blue"],
            "is_blue": 0,
            "win": red_win,
            # Perspective flip: if gold_diff is positive for winner, negate for the loser side
            "gold_diff_team": (-r["gold_diff"]) if pd.notna(r["gold_diff"]) else np.nan,
            "gold_diff_at14_team": (-r["gold_diff_at14"]) if pd.notna(r["gold_diff_at14"]) else np.nan,
            "game_time": r["game_time"] if pd.notna(r["game_time"]) else np.nan,
        })
    td = pd.DataFrame(rows)
    # Normalize types
    td["team"] = td["team"].astype(str)
    td["opponent"] = td["opponent"].astype(str)
    return td


def aggregates_overall(td: pd.DataFrame, min_games: int) -> pd.DataFrame:
    aggs = {
        "game_id": "count",
        "win": "mean",
        "gold_diff_team": "mean",
        "gold_diff_at14_team": "mean",
        "game_time": ["mean", "median"],
    }
    # Compute with care for missing numeric columns
    df = td.groupby("team").agg(aggs)
    df.columns = ["_".join([c for c in col if c]) for col in df.columns.to_flat_index()]
    df = df.rename(columns={
        "game_id_count": "games_played",
        "win_mean": "winrate",
        "gold_diff_team_mean": "avg_gold_diff",
        "gold_diff_at14_team_mean": "avg_gold_diff_at14",
        "game_time_mean": "avg_game_time",
        "game_time_median": "median_game_time",
    }).reset_index()

    # Remove teams with too few matches
    df = df[df["games_played"] >= min_games].reset_index(drop=True)
    return df


def aggregates_lastN(td: pd.DataFrame, window_games: int, min_games: int) -> pd.DataFrame:
    # Sort by date within team and compute rolling means over last N games (excluding current)
    td = td.sort_values(["team", "game_date"]).copy()
    def _roll(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()
        g["games_played"] = np.arange(1, len(g) + 1)
        win_r = g["win"].rolling(window_games, min_periods=1).mean()
        gd_r = g["gold_diff_team"].rolling(window_games, min_periods=1).mean()
        gd14_r = g["gold_diff_at14_team"].rolling(window_games, min_periods=1).mean()
        gt_r = g["game_time"].rolling(window_games, min_periods=1).mean()
        # Take the last row per team as the summary up to cutoff
        return pd.DataFrame({
            "team": [g["team"].iloc[-1]],
            "games_played": [g.shape[0]],
            "winrate_lastN": [win_r.iloc[-1]],
            "avg_gold_diff_lastN": [gd_r.iloc[-1]],
            "avg_gold_diff_at14_lastN": [gd14_r.iloc[-1]],
            "avg_game_time_lastN": [gt_r.iloc[-1]],
        })
    rolled = td.groupby("team", group_keys=False).apply(_roll)
    rolled = rolled.reset_index(drop=True)
    rolled = rolled[rolled["games_played"] >= min_games].reset_index(drop=True)
    return rolled


def compute_elo(df_matches: pd.DataFrame, k: float) -> pd.DataFrame:
    # Simple Elo on match outcomes only (blue vs red). Uses chronological order up to cutoff.
    teams = pd.unique(pd.concat([df_matches["team_blue"], df_matches["team_red"]]).astype(str))
    rating: Dict[str, float] = {t: 1500.0 for t in teams}
    def expected(a: float, b: float) -> float:
        return 1.0 / (1.0 + 10 ** ((b - a) / 400.0))
    for _, r in df_matches.iterrows():
        tb, tr = str(r["team_blue"]), str(r["team_red"])
        rb, rr = rating[tb], rating[tr]
        eb = expected(rb, rr)
        er = 1.0 - eb
        blue_win = int(str(r["winning_team"]) == tb)
        sb, sr = (1, 0) if blue_win else (0, 1)
        rating[tb] = rb + k * (sb - eb)
        rating[tr] = rr + k * (sr - er)
    elo_df = pd.DataFrame({"team": list(rating.keys()), "elo": list(rating.values())})
    return elo_df.sort_values("elo", ascending=False).reset_index(drop=True)


def main():
    cfg = parse_args()
    os.makedirs(cfg.out_dir, exist_ok=True)

    raw = load_matches(cfg.data)
    hist = filter_cutoff(raw, cfg.cutoff_date)

    # Team perspective table (2 rows per match)
    team_df = to_team_perspective(hist)

    overall = aggregates_overall(team_df, cfg.min_games)
    lastN = aggregates_lastN(team_df, cfg.window_games, cfg.min_games)

    overall_path = os.path.join(cfg.out_dir, "team_stats_overall.csv")
    lastN_path = os.path.join(cfg.out_dir, "team_stats_lastN.csv")
    overall.to_csv(overall_path, index=False)
    lastN.to_csv(lastN_path, index=False)

    print(f"Saved overall aggregates to: {overall_path}")
    print(f"Saved last{cfg.window_games} aggregates to: {lastN_path}")

    if cfg.use_elo:
        elo_df = compute_elo(hist, cfg.k_factor)
        elo_path = os.path.join(cfg.out_dir, "team_elo.csv")
        elo_df.to_csv(elo_path, index=False)
        print(f"Saved Elo ratings to: {elo_path}")

    meta = {
        "cutoff_date": cfg.cutoff_date,
        "window_games": cfg.window_games,
        "min_games": cfg.min_games,
        "use_elo": cfg.use_elo,
        "k_factor": cfg.k_factor,
        "base_columns_present": list(sorted(set(raw.columns) & (REQUIRED_BASE | set(OPTIONAL_NUMERIC))))
    }
    meta_path = os.path.join(cfg.out_dir, "team_stats_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"Saved metadata to: {meta_path}")


if __name__ == "__main__":
    main()
