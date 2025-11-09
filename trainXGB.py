# Purpose:
#   Train an XGBoost model that mirrors the CatBoost pre-game pipeline:
#   it uses ONLY pre-game information by joining team-level profiles
#   (overall/lastN/Elo) and optional team names, plus optional diff features.
#
# Example:
#   python train_xgb.py \
#     --data lec_2023-2025_games.csv \
#     --team-stats-overall artifacts/team_stats_overall.csv \
#     --team-stats-lastN artifacts/team_stats_lastN.csv \
#     --team-elo artifacts/team_elo.csv \
#     --cutoff-date 2025-03-01 \
#     --include-team-names --include-diffs \
#     --num-boost-round 2000 --early-stopping-rounds 100 \
#     --save-dir artifacts_xgb --gpu
#
# Outputs:
#   artifacts_xgb/model_xgb.json
#   artifacts_xgb/metadata_xgb.json

import argparse
import json
import os
from datetime import datetime, timezone
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, brier_score_loss
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

def parse_args():
    p = argparse.ArgumentParser(description="Train XGBoost (pre-game) with team strength profiles")
    p.add_argument("--data", required=True, help="Path to matches CSV (lec_2023-2025_games.csv)")
    p.add_argument("--team-stats-overall", required=True, help="Path to team_stats_overall.csv")
    p.add_argument("--team-stats-lastN", default=None, help="Path to team_stats_lastN.csv (optional)")
    p.add_argument("--team-elo", default=None, help="Path to team_elo.csv (optional)")

    p.add_argument("--cutoff-date", default=None, help="YYYY-MM-DD time split (train < cutoff, test >= cutoff)")
    p.add_argument("--include-team-names", action="store_true", help="Include team_blue/team_red as categorical (LabelEncoded)")
    p.add_argument("--include-diffs", action="store_true", help="Add (blue - red) numeric diffs")

    p.add_argument("--save-dir", default="artifacts_xgb")

    # XGBoost params (sane defaults; tune later)
    p.add_argument("--num-boost-round", type=int, default=1500)
    p.add_argument("--early-stopping-rounds", type=int, default=100)
    p.add_argument("--max-depth", type=int, default=6)
    p.add_argument("--eta", type=float, default=0.05, help="learning_rate")
    p.add_argument("--subsample", type=float, default=0.9)
    p.add_argument("--colsample-bytree", type=float, default=0.9)
    p.add_argument("--reg-lambda", type=float, default=1.0)
    p.add_argument("--reg-alpha", type=float, default=0.0)
    p.add_argument("--gpu", action="store_true", help="Use GPU (tree_method=gpu_hist)")

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
    # Re-map overall to blue/red
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

    # Optional diffs for numeric columns only
    if include_diffs:
        for col in list(X.columns):
            if col.endswith("_blue"):
                base = col[:-5]
                red_col = base + "_red"
                if red_col in X.columns and pd.api.types.is_numeric_dtype(X[col]) and pd.api.types.is_numeric_dtype(X[red_col]):
                    X[base + "_diff"] = X[col].astype(float) - X[red_col].astype(float)

    # Feature list
    feature_names: List[str] = []
    cat_feature_names: List[str] = []

    if include_team_names:
        feature_names += ["team_blue", "team_red"]
        cat_feature_names += ["team_blue", "team_red"]

    # Add all numeric features (everything except raw team names)
    for c in X.columns:
        if c not in ("team_blue", "team_red"):
            feature_names.append(c)

    return X[feature_names].copy(), feature_names, cat_feature_names


def label_encode_teams(X_train: pd.DataFrame, X_test: pd.DataFrame, cat_cols: List[str]):
    encoders = {}
    for c in cat_cols:
        le = LabelEncoder()
        # fit on union of train+test team names to avoid unknowns at eval time
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

    # Build features identical to CatBoost pipeline (team profiles + diffs)
    X_full, feature_names, cat_feature_names = merge_team_features(
        matches, overall, lastN, elo, include_team_names=args.include_team_names, include_diffs=args.include_diffs
    )

    # Target
    y = derive_target(matches)

    # Time split
    tr_idx, te_idx = time_split_index(matches, args.cutoff_date)
    X_train, X_test = X_full.loc[tr_idx].reset_index(drop=True), X_full.loc[te_idx].reset_index(drop=True)
    y_train, y_test = y.loc[tr_idx].values, y.loc[te_idx].values

    # Encode categoricals (team names) for XGBoost
    encoders = {}
    if cat_feature_names:
        X_train, X_test, encoders = label_encode_teams(X_train, X_test, cat_feature_names)

    # Convert to DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_test, label=y_test)

    # XGBoost parameters
    params = {
        "objective": "binary:logistic",
        "eval_metric": ["auc", "logloss"],  # we'll compute other metrics ourselves
        "max_depth": args.max_depth,
        "eta": args.eta,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "lambda": args.reg_lambda,
        "alpha": args.reg_alpha,
        "tree_method": "gpu_hist" if args.gpu else "hist",
        "predictor": "gpu_predictor" if args.gpu else "auto",
        "verbosity": 1,
        "seed": 42,
    }

    # Train with early stopping
    evals = [(dtrain, "train"), (dvalid, "valid")]
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=args.num_boost_round,
        evals=evals,
        early_stopping_rounds=args.early_stopping_rounds,
        verbose_eval=False,
    )

    # Evaluation
    proba = booster.predict(dvalid, iteration_range=(0, booster.best_iteration + 1))
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
        "best_iteration": int(booster.best_iteration),
    }

    print("=== Evaluation (pre-game, XGBoost) ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k:>14}: {v:.4f}")
        else:
            print(f"{k:>14}: {v}")

    # Save artifacts
    model_path = os.path.join(args.save_dir, "model_xgb.json")
    meta_path = os.path.join(args.save_dir, "metadata_xgb.json")

    booster.save_model(model_path)

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "framework": "xgboost",
        "model_path": os.path.abspath(model_path),
        "feature_names": feature_names,
        "cat_feature_names": cat_feature_names,  # encoded via LabelEncoder
        "label_encoders": encoders,              # mapping of classes per categorical col
        "include_team_names": args.include_team_names,
        "include_diffs": args.include_diffs,
        "team_stats_overall": os.path.abspath(args.team_stats_overall),
        "team_stats_lastN": os.path.abspath(args.team_stats_lastN) if args.team_stats_lastN else None,
        "team_elo": os.path.abspath(args.team_elo) if args.team_elo else None,
        "cutoff_date": args.cutoff_date,
        "params": {k: v for k, v in params.items() if k != "predictor"},
        "metrics": metrics,
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Saved model to: {model_path}")
    print(f"Saved metadata to: {meta_path}")


if __name__ == "__main__":
    main()
