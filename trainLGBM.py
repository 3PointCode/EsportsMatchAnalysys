# Purpose:
#   Train a LightGBM model that mirrors the CatBoost/XGBoost pre-game pipeline:
#   it uses only pre-game information by joining team-level profiles
#   (overall/lastN/Elo) and optional team names, plus optional diff features.
#
# Example:
#   python train_lgbm.py \
#     --data datalec_2023-2025_games.csv --team-stats-overall winter/team_stats_overall.csv --team-stats-lastN winter/team_stats_lastN.csv 
#     --cutoff-date 2025-03-02 --include-team-names --include-diffs --num-boost-round 3000 --early-stopping-rounds 200
#     --save-dir artifacts_lgbm_winter
#
# Outputs:
#   artifacts_lgbm/model_lgbm.txt
#   artifacts_lgbm/metadata_lgbm.json

import argparse
import json
import os
from datetime import datetime, timezone
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, brier_score_loss
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

def parse_args():
    p = argparse.ArgumentParser(description="Train LightGBM (pre-game) with team strength profiles")
    p.add_argument("--data", required=True, help="Path to matches CSV (lec_2023-2025_games.csv)")
    p.add_argument("--team-stats-overall", required=True, help="Path to team_stats_overall.csv")
    p.add_argument("--team-stats-lastN", default=None, help="Path to team_stats_lastN.csv (optional)")
    p.add_argument("--team-elo", default=None, help="Path to team_elo.csv (optional)")

    p.add_argument("--cutoff-date", default=None, help="YYYY-MM-DD time split (train < cutoff, test >= cutoff)")
    p.add_argument("--include-team-names", action="store_true", help="Include team_blue/team_red as categorical (LabelEncoded)")
    p.add_argument("--include-diffs", action="store_true", help="Add (blue - red) numeric diffs")

    p.add_argument("--save-dir", default="artifacts_lgbm")

    # LightGBM params (sane defaults; tune later)
    p.add_argument("--num-boost-round", type=int, default=100)
    p.add_argument("--early-stopping-rounds", type=int, default=0)
    p.add_argument("--num-leaves", type=int, default=31)
    p.add_argument("--learning-rate", type=float, default=0.1)
    p.add_argument("--feature-fraction", type=float, default=1.0)
    p.add_argument("--bagging-fraction", type=float, default=1.0)
    p.add_argument("--bagging-freq", type=int, default=0)
    p.add_argument("--lambda-l1", type=float, default=0.0)
    p.add_argument("--lambda-l2", type=float, default=0.0)
    p.add_argument("--max-depth", type=int, default=-1)
    p.add_argument("--gpu", action="store_true", help="Use GPU (device_type=gpu)")

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
    return df.sort_values("game_date").reset_index(drop=True)


def load_csv(path: Optional[str]) -> Optional[pd.DataFrame]:
    if not path:
        return None
    return pd.read_csv(path)


def time_split_index(df: pd.DataFrame, cutoff: Optional[str]):
    if not cutoff:
        n = len(df)
        split = int(n * 0.8)
        return df.iloc[:split].index, df.iloc[split:].index
    cutoff_dt = pd.to_datetime(cutoff)
    tr = df.index[df["game_date"] < cutoff_dt]
    te = df.index[df["game_date"] >= cutoff_dt]
    return tr, te


def derive_target(matches: pd.DataFrame) -> pd.Series:
    return (matches["winning_team"].astype(str) == matches["team_blue"].astype(str)).astype(int)


def merge_team_features(matches: pd.DataFrame,
                        overall: pd.DataFrame,
                        lastN: Optional[pd.DataFrame],
                        elo: Optional[pd.DataFrame],
                        include_team_names: bool,
                        include_diffs: bool) -> tuple[pd.DataFrame, List[str], List[str]]:
    """Return X (features), feature_names, cat_feature_names (only team names)."""
    ov_b = overall.rename(columns={c: f"ov_{c}_blue" for c in overall.columns if c != "team"}).rename(columns={"team": "team_blue"})
    ov_r = overall.rename(columns={c: f"ov_{c}_red" for c in overall.columns if c != "team"}).rename(columns={"team": "team_red"})
    X = matches[["team_blue", "team_red"]].copy()
    X = X.merge(ov_b, on="team_blue", how="left").merge(ov_r, on="team_red", how="left")

    if lastN is not None:
        ln_b = lastN.rename(columns={c: f"ln_{c}_blue" for c in lastN.columns if c != "team"}).rename(columns={"team": "team_blue"})
        ln_r = lastN.rename(columns={c: f"ln_{c}_red" for c in lastN.columns if c != "team"}).rename(columns={"team": "team_red"})
        X = X.merge(ln_b, on="team_blue", how="left").merge(ln_r, on="team_red", how="left")

    if elo is not None:
        e_b = elo.rename(columns={"team": "team_blue", "elo": "elo_blue"})
        e_r = elo.rename(columns={"team": "team_red", "elo": "elo_red"})
        X = X.merge(e_b, on="team_blue", how="left").merge(e_r, on="team_red", how="left")

    if include_diffs:
        for col in list(X.columns):
            if col.endswith("_blue"):
                base = col[:-5]
                red_col = base + "_red"
                if red_col in X.columns and pd.api.types.is_numeric_dtype(X[col]) and pd.api.types.is_numeric_dtype(X[red_col]):
                    X[base + "_diff"] = X[col].astype(float) - X[red_col].astype(float)

    feature_names: List[str] = []
    cat_feature_names: List[str] = []
    if include_team_names:
        feature_names += ["team_blue", "team_red"]
        cat_feature_names += ["team_blue", "team_red"]
    for c in X.columns:
        if c not in ("team_blue", "team_red"):
            feature_names.append(c)

    return X[feature_names].copy(), feature_names, cat_feature_names


def label_encode_teams(X_train: pd.DataFrame, X_test: pd.DataFrame, cat_cols: List[str]):
    encoders = {}
    for c in cat_cols:
        le = LabelEncoder()
        all_vals = pd.concat([X_train[c].astype(str), X_test[c].astype(str)], axis=0).fillna("<UNK>")
        le.fit(all_vals)
        X_train[c] = le.transform(X_train[c].astype(str).fillna("<UNK>"))
        X_test[c] = le.transform(X_test[c].astype(str).fillna("<UNK>"))
        encoders[c] = {"classes_": le.classes_.tolist()}
    return X_train, X_test, encoders

def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    matches = load_matches(args.data)
    overall = load_csv(args.team_stats_overall)
    lastN = load_csv(args.team_stats_lastN)
    elo = load_csv(args.team_elo)

    if overall is None or "team" not in overall.columns:
        raise SystemExit("team_stats_overall.csv must be provided and contain 'team' column.")

    X_full, feature_names, cat_feature_names = merge_team_features(
        matches, overall, lastN, elo, include_team_names=args.include_team_names, include_diffs=args.include_diffs
    )

    y = derive_target(matches)

    tr_idx, te_idx = time_split_index(matches, args.cutoff_date)
    X_train, X_test = X_full.loc[tr_idx].reset_index(drop=True), X_full.loc[te_idx].reset_index(drop=True)
    y_train, y_test = y.loc[tr_idx].values, y.loc[te_idx].values

    encoders = {}
    if cat_feature_names:
        X_train, X_test, encoders = label_encode_teams(X_train, X_test, cat_feature_names)

    # Build datasets
    lgb_train = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
    lgb_valid = lgb.Dataset(X_test, label=y_test, reference=lgb_train, free_raw_data=False)

    params = {
        "objective": "binary",
        "metric": ["auc", "binary_logloss"],
        "num_leaves": args.num_leaves,
        "learning_rate": args.learning_rate,
        "feature_fraction": args.feature_fraction,
        "bagging_fraction": args.bagging_fraction,
        "bagging_freq": args.bagging_freq,
        "lambda_l1": args.lambda_l1,
        "lambda_l2": args.lambda_l2,
        "max_depth": args.max_depth,
        "device_type": "gpu" if args.gpu else "cpu",
        "verbose": -1,
        "seed": 42,
    }

    callbacks = [
    lgb.early_stopping(stopping_rounds=args.early_stopping_rounds, verbose=False),
    lgb.log_evaluation(period=0)
    ]

    booster = lgb.train(
        params,
        lgb_train,
        num_boost_round=args.num_boost_round,
        valid_sets=[lgb_train, lgb_valid],
        valid_names=["train", "valid"],
        callbacks=callbacks,
    )

    proba = booster.predict(X_test, num_iteration=booster.best_iteration)
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
        "n_test": int(len(X_test)),
        "best_iteration": int(booster.best_iteration or 0),
    }

    print("=== Evaluation (pre-game, LightGBM) ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k:>14}: {v:.4f}")
        else:
            print(f"{k:>14}: {v}")

    model_path = os.path.join(args.save_dir, "model_lgbm.txt")
    meta_path = os.path.join(args.save_dir, "metadata_lgbm.json")

    booster.save_model(model_path)

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "framework": "lightgbm",
        "model_path": os.path.abspath(model_path),
        "feature_names": feature_names,
        "cat_feature_names": cat_feature_names,  # label-encoded
        "label_encoders": encoders,              # mapping of classes per categorical col
        "include_team_names": args.include_team_names,
        "include_diffs": args.include_diffs,
        "team_stats_overall": os.path.abspath(args.team_stats_overall),
        "team_stats_lastN": os.path.abspath(args.team_stats_lastN) if args.team_stats_lastN else None,
        "team_elo": os.path.abspath(args.team_elo) if args.team_elo else None,
        "cutoff_date": args.cutoff_date,
        "params": params,
        "metrics": metrics,
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Saved model to: {model_path}")
    print(f"Saved metadata to: {meta_path}")

if __name__ == "__main__":
    main()
