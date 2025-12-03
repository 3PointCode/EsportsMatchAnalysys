# Purpose:
#   Train a model that can make pre-game predictions using only information available before a match.
#   It learns from historical matches but uses aggregated team profiles (overall / lastN games stats)
#
# Inputs:
#   --data                 : lec_2023-2025_games.csv
#   --team-stats-overall   : team_stats_overall.csv
#   --team-stats-lastN     : team_stats_lastN.csv (optional)
#   --team-elo             : team_elo.csv (optional)
#
# Split:
#   --cutoff-date YYYY-MM-DD  -> train: game_date < cutoff, test: game_date >= cutoff
#
# Features used (configurable):
#   - Categorical: team_blue, team_red (optional via --include-team-names)
#   - Numeric from OVERALL (required): winrate, avg_gold_diff, avg_gold_diff_14, avg/median_game_time, games_played
#   - Numeric from LASTN (optional): winrate_lastN, avg_*_lastN
#   - Numeric from Elo (optional): elo
#   - Optionally, DIFF features: (blue - red) for each numeric stat (recommended)
#
# Outputs:
#   artifacts/model.cbm
#   artifacts/metadata.json
#
# Example:
#   python train.py --data data/lec_2023-2025_games.csv --team-stats-overall winter/team_stats_overall.csv --team-stats-lastN winter/team_stats_lastN.csv
#     --cutoff-date 2025-03-02 --include-team-names --include-diffs --save-dir artifacts_cat_winter

import argparse
import json
import os
from datetime import datetime, timezone
from typing import List
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, brier_score_loss

def parse_args():
    p = argparse.ArgumentParser(description="Train CatBoost (pre-game) with team strength profiles")
    p.add_argument("--data", required=True, help="Path to matches CSV (lec_2023-2025_games.csv)")
    p.add_argument("--team-stats-overall", required=True, help="Path to team_stats_overall.csv")
    p.add_argument("--team-stats-lastN", default=None, help="Path to team_stats_lastN.csv (optional)")
    p.add_argument("--team-elo", default=None, help="Path to team_elo.csv (optional)")

    p.add_argument("--cutoff-date", default=None, help="YYYY-MM-DD time-based split (train < cutoff, test >= cutoff)")
    p.add_argument("--train-end-date", default=None, help="Last date (YYYY-MM-DD) used for train set in 3-way split")
    p.add_argument("--val-end-date", default=None, help="Last date (YYYY-MM-DD) used for validation set in 3-way split")
    p.add_argument("--include-team-names", action="store_true", help="Include team_blue/team_red as categorical features")
    p.add_argument("--include-diffs", action="store_true", help="Add (blue - red) numeric diff features")

    p.add_argument("--save-dir", default="artifacts")
    p.add_argument("--iterations", type=int, default=1000)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--learning-rate", type=float, default=0.03)
    p.add_argument("--random-seed", type=int, default=0)
    p.add_argument("--eval-metric", default="AUC")
    p.add_argument("--gpu", action="store_true")
    return p.parse_args()


def load_matches(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"game_id", "game_date", "team_blue", "team_red", "winning_team"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Matches CSV missing required columns: {sorted(missing)}")
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    if df["game_date"].isna().any():
        raise ValueError("Some game_date values could not be parsed. Fix input.")
    df = df.sort_values("game_date").reset_index(drop=True)
    return df


def load_stats_overall(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"team", "games_played", "winrate", "avg_gold_diff", "avg_gold_diff_at14", "avg_game_time", "median_game_time"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"team_stats_overall missing columns: {sorted(missing)}")
    return df


def load_stats_lastN(path: str | None) -> pd.DataFrame | None:
    if not path:
        return None
    df = pd.read_csv(path)
    required = {"team", "games_played", "winrate_lastN", "avg_gold_diff_lastN", "avg_gold_diff_at14_lastN", "avg_game_time_lastN"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"team_stats_lastN missing columns: {sorted(missing)}")
    return df


def load_elo(path: str | None) -> pd.DataFrame | None:
    if not path:
        return None
    df = pd.read_csv(path)
    required = {"team", "elo"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"team_elo missing columns: {sorted(missing)}")
    return df


def merge_team_features(matches: pd.DataFrame,
                        overall: pd.DataFrame,
                        lastN: pd.DataFrame | None,
                        elo: pd.DataFrame | None,
                        include_team_names: bool,
                        include_diffs: bool) -> tuple[pd.DataFrame, List[str], List[str]]:
    """Return X (features), list of feature names, list of categorical feature names."""
    # Base: only team names (categoricals) if requested
    feat_frames = []
    cat_names: List[str] = []
    if include_team_names:
        feat_frames.append(matches[["team_blue", "team_red"]].astype(str))
        cat_names.extend(["team_blue", "team_red"])

    # Prepare renamed copies for blue/red
    ov_b = overall.rename(columns={c: f"ov_{c}_blue" for c in overall.columns if c != "team"})
    ov_b = ov_b.rename(columns={"team": "team_blue"})
    ov_r = overall.rename(columns={c: f"ov_{c}_red" for c in overall.columns if c != "team"})
    ov_r = ov_r.rename(columns={"team": "team_red"})

    X = matches[["team_blue", "team_red"]].copy()
    X = X.merge(ov_b, on="team_blue", how="left").merge(ov_r, on="team_red", how="left")

    if lastN is not None:
        ln_b = lastN.rename(columns={c: f"ln_{c}_blue" for c in lastN.columns if c != "team"})
        ln_b = ln_b.rename(columns={"team": "team_blue"})
        ln_r = lastN.rename(columns={c: f"ln_{c}_red" for c in lastN.columns if c != "team"})
        ln_r = ln_r.rename(columns={"team": "team_red"})
        X = X.merge(ln_b, on="team_blue", how="left").merge(ln_r, on="team_red", how="left")

    if elo is not None:
        e_b = elo.rename(columns={"team": "team_blue", "elo": "elo_blue"})
        e_r = elo.rename(columns={"team": "team_red", "elo": "elo_red"})
        X = X.merge(e_b, on="team_blue", how="left").merge(e_r, on="team_red", how="left")

    # If team names are not part of features, drop them from X but keep for diff computations
    numeric_cols = [c for c in X.columns if c not in ("team_blue", "team_red")]

    # Optionally add diffs for every numeric stat (blue - red) — recommended
    if include_diffs:
        # Pair columns by suffix _blue/_red
        for c in list(numeric_cols):
            if c.endswith("_blue"):
                base = c[:-5]
                red_col = base + "_red"
                if red_col in X.columns:
                    diff_name = base + "_diff"
                    X[diff_name] = X[c] - X[red_col]

    # Final feature table
    if include_team_names:
        features = ["team_blue", "team_red"] + [c for c in X.columns if c not in ("team_blue", "team_red")]
    else:
        features = [c for c in X.columns if c not in ("team_blue", "team_red")]
        X = X[features]

    # Cat features = team names (optional)
    cat_feat_names = cat_names

    return X, features, cat_feat_names


def derive_target(matches: pd.DataFrame) -> pd.Series:
    return (matches["winning_team"].astype(str) == matches["team_blue"].astype(str)).astype(int)


def time_split(df: pd.DataFrame, cutoff: str | None):
    if not cutoff:
        # If no cutoff, use last 20% as test, preserving time order
        n = len(df)
        split = int(n * 0.8)
        return df.iloc[:split].index, df.iloc[split:].index
    cutoff_dt = pd.to_datetime(cutoff)
    train_idx = df.index[df["game_date"] < cutoff_dt]
    test_idx = df.index[df["game_date"] >= cutoff_dt]
    return train_idx, test_idx

def time_split_3way(df: pd.DataFrame,
                    train_end_date: str,
                    val_end_date: str):
    """
    Returns (train_idx, val_idx, test_idx) in chronological order:
    - train: game_date <= train_end_date
    - val  : train_end_date < game_date <= val_end_date
    - test : game_date > val_end_date
    """
    df = df.sort_values("game_date")
    train_end = pd.to_datetime(train_end_date)
    val_end = pd.to_datetime(val_end_date)

    if val_end <= train_end:
        raise ValueError("val_end_date must be strictly after train_end_date.")

    train_idx = df.index[df["game_date"] <= train_end]
    val_idx = df.index[(df["game_date"] > train_end) & (df["game_date"] <= val_end)]
    test_idx = df.index[df["game_date"] > val_end]

    if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
        raise ValueError(
            f"3-way split produced empty subset(s). "
            f"Check train_end_date={train_end_date} and val_end_date={val_end_date}."
        )

    return train_idx, val_idx, test_idx


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    matches = load_matches(args.data)
    overall = load_stats_overall(args.team_stats_overall)
    lastN = load_stats_lastN(args.team_stats_lastN)
    elo = load_elo(args.team_elo)

    if overall is None or "team" not in overall.columns:
        raise SystemExit("team_stats_overall.csv must be provided and contain 'team' column.")

    X_full, feature_names, cat_feature_names = merge_team_features(
        matches,
        overall,
        lastN,
        elo,
        include_team_names=args.include_team_names,
        include_diffs=args.include_diffs,
    )

    y = derive_target(matches)

    use_3way = args.train_end_date is not None and args.val_end_date is not None

    if use_3way:
        tr_idx, val_idx, te_idx = time_split_3way(
            matches,
            args.train_end_date,
            args.val_end_date,
        )
        X_train = X_full.loc[tr_idx].reset_index(drop=True)
        X_val = X_full.loc[val_idx].reset_index(drop=True)
        X_test = X_full.loc[te_idx].reset_index(drop=True)

        y_train = y.loc[tr_idx].values
        y_val = y.loc[val_idx].values
        y_test = y.loc[te_idx].values
    else:
        tr_idx, te_idx = time_split(matches, args.cutoff_date)
        X_train = X_full.loc[tr_idx].reset_index(drop=True)
        X_val = X_full.loc[te_idx].reset_index(drop=True)
        X_test = X_val

        y_train = y.loc[tr_idx].values
        y_val = y.loc[te_idx].values
        y_test = y_val

    # Categorical indices
    if cat_feature_names:
        cat_indices = [feature_names.index(c) for c in cat_feature_names]
    else:
        cat_indices = []

    train_pool = Pool(X_train, y_train, cat_features=cat_indices or None)
    valid_pool = Pool(X_val, y_val, cat_features=cat_indices or None)
    test_pool = Pool(X_test, y_test, cat_features=cat_indices or None)

    params = dict(
        loss_function="Logloss",
        iterations=args.iterations,
        depth=args.depth,
        learning_rate=args.learning_rate,
        eval_metric=args.eval_metric,
        verbose=False,
        random_seed=args.random_seed,
    )

    model = None
    if args.gpu:
        try:
            model = CatBoostClassifier(**params, task_type="GPU", devices="0")
        except Exception:
            print("[warn] GPU not available, falling back to CPU…")
    if model is None:
        model = CatBoostClassifier(**params)

    model.fit(train_pool, eval_set=valid_pool, verbose=False)

    # Evaluation on test data
    proba = model.predict_proba(test_pool)[:, 1]
    y_pred = (proba >= 0.5).astype(int)

    def safe_auc(y_true, y_score):
        return roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else None

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": safe_auc(y_test, proba),
        "brier": float(brier_score_loss(y_test, proba)),
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_test": int(len(X_test)),
    }

    print("=== Evaluation (pre-game, CatBoost) ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k:>10}: {v:.4f}")
        else:
            print(f"{k:>10}: {v}")

    model_path = os.path.join(args.save_dir, "model.cbm")
    meta_path = os.path.join(args.save_dir, "metadata.json")

    model.save_model(model_path)

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "framework": "catboost",
        "feature_names": feature_names,
        "cat_feature_names": cat_feature_names,
        "include_team_names": args.include_team_names,
        "include_diffs": args.include_diffs,
        "team_stats_overall": os.path.abspath(args.team_stats_overall),
        "team_stats_lastN": os.path.abspath(args.team_stats_lastN) if args.team_stats_lastN else None,
        "team_elo": os.path.abspath(args.team_elo) if args.team_elo else None,
        "cutoff_date": args.cutoff_date,
        "train_end_date": args.train_end_date,
        "val_end_date": args.val_end_date,
        "metrics": metrics,
        "params": {
            "iterations": args.iterations,
            "depth": args.depth,
            "learning_rate": args.learning_rate,
            "eval_metric": args.eval_metric,
            "random_seed": args.random_seed,
        },
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Saved model to: {model_path}")
    print(f"Saved metadata to: {meta_path}")


if __name__ == "__main__":
    main()