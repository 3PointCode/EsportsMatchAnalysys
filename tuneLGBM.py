# Purpose:
#  - Buduje cechy tak samo jak trainLGBM.py (profile drużyn + opcjonalne diffs i nazwy drużyn)
#  - Pozwala podać WIELE dat cutoff oraz LISTY wartości hiperparametrów
#  - Dla każdej kombinacji uruchamia trening z early stopping (callbacki, LGBM v4+)
#  - Zapisuje wyniki do CSV (leaderboard) i opcjonalnie najlepszy model + metadane
#
# Usage example:
#   python tune_lgbm.py
#     --data data/lec_2023-2025_games.csv
#     --team-stats-overall data/team_stats_overall.csv
#     --team-stats-lastN data/team_stats_lastN.csv
#     --cutoff-date 2025-03-02 2025-05-25
#     --include-team-names --include-diffs
#     --num-boost-round 500 1000 1500 2000
#     --early-stopping-rounds 100 200 300
#     --num-leaves 63 127
#     --learning-rate 0.03 0.04 0.05
#     --feature-fraction 0.7 0.8 0.9
#     --bagging-fraction 0.7 0.8 0.9
#     --bagging-freq 0 1
#     --lambda-l1 0.0 0.1
#     --lambda-l2 0.0 1.0
#     --max-depth 5 8 11 13
#     --save-dir artifacts_lgbm_tune --save-best-model

import argparse
import itertools
import json
import os
from datetime import datetime, timezone
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    brier_score_loss,
)
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

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


def time_split_index(df: pd.DataFrame, cutoff: Optional[str]) -> Tuple[pd.Index, pd.Index]:
    if not cutoff:
        split = int(len(df) * 0.8)
        return df.index[:split], df.index[split:]
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


def evaluate_run(y_true: np.ndarray, proba: np.ndarray) -> dict:
    y_pred = (proba >= 0.5).astype(int)
    def safe_auc(y, s):
        return roc_auc_score(y, s) if len(np.unique(y)) > 1 else None
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": safe_auc(y_true, proba),
        "brier": float(brier_score_loss(y_true, proba)),
    }

def parse_args():
    ap = argparse.ArgumentParser(description="Grid search for LightGBM (pre-game)")
    ap.add_argument("--data", required=True)
    ap.add_argument("--team-stats-overall", required=True)
    ap.add_argument("--team-stats-lastN", default=None)
    ap.add_argument("--team-elo", default=None)
    ap.add_argument("--cutoff-date", nargs="+", default=[None], help="You can pass multiple dates (YYYY-MM-DD)")
    ap.add_argument("--include-team-names", action="store_true")
    ap.add_argument("--include-diffs", action="store_true")

    ap.add_argument("--num-boost-round", nargs="+", type=int, default=[1500])
    ap.add_argument("--early-stopping-rounds", nargs="+", type=int, default=[100])
    ap.add_argument("--num-leaves", nargs="+", type=int, default=[63])
    ap.add_argument("--learning-rate", nargs="+", type=float, default=[0.05])
    ap.add_argument("--feature-fraction", nargs="+", type=float, default=[0.9])
    ap.add_argument("--bagging-fraction", nargs="+", type=float, default=[0.9])
    ap.add_argument("--bagging-freq", nargs="+", type=int, default=[0])
    ap.add_argument("--lambda-l1", nargs="+", type=float, default=[0.0])
    ap.add_argument("--lambda-l2", nargs="+", type=float, default=[0.0])
    ap.add_argument("--max-depth", nargs="+", type=int, default=[-1])
    ap.add_argument("--gpu", action="store_true")

    ap.add_argument("--save-dir", default="artifacts_lgbm_tune")
    ap.add_argument("--save-best-model", action="store_true")
    ap.add_argument("--leaderboard", default="leaderboard_lgbm.csv")

    return ap.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    matches = load_matches(args.data)
    overall = load_csv(args.team_stats_overall)
    lastN = load_csv(args.team_stats_lastN)
    elo = load_csv(args.team_elo)

    X_full, feature_names, cat_feature_names = merge_team_features(
        matches, overall, lastN, elo,
        include_team_names=args.include_team_names,
        include_diffs=args.include_diffs,
    )
    y_full = derive_target(matches).values

    grid = list(itertools.product(
        args.cutoff_date,
        args.num_boost_round,
        args.early_stopping_rounds,
        args.num_leaves,
        args.learning_rate,
        args.feature_fraction,
        args.bagging_fraction,
        args.bagging_freq,
        args.lambda_l1,
        args.lambda_l2,
        args.max_depth,
    ))
    print(f"Grid size: {len(grid)} (cutoffs x hyperparams)")

    leaderboard_path = os.path.join(args.save_dir, args.leaderboard)
    lb_exists = os.path.exists(leaderboard_path)
    rows = []
    best_key = None
    best_metrics = None
    best_model = None
    best_encoders = None

    for (cutoff_date, num_boost_round, es_rounds, num_leaves, learning_rate,
         feature_fraction, bagging_fraction, bagging_freq, lambda_l1, lambda_l2, max_depth) in grid:

        tr_idx, te_idx = time_split_index(matches, cutoff_date)
        X_train = X_full.loc[tr_idx].reset_index(drop=True)
        X_test  = X_full.loc[te_idx].reset_index(drop=True)
        y_train, y_test = y_full[tr_idx], y_full[te_idx]

        encoders = {}
        if cat_feature_names:
            X_train, X_test, encoders = label_encode_teams(X_train, X_test, cat_feature_names)

        lgb_train = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
        lgb_valid = lgb.Dataset(X_test,  label=y_test,  reference=lgb_train, free_raw_data=False)

        params = {
            "objective": "binary",
            "metric": ["auc", "binary_logloss"],
            "num_leaves": int(num_leaves),
            "learning_rate": float(learning_rate),
            "feature_fraction": float(feature_fraction),
            "bagging_fraction": float(bagging_fraction),
            "bagging_freq": int(bagging_freq),
            "lambda_l1": float(lambda_l1),
            "lambda_l2": float(lambda_l2),
            "max_depth": int(max_depth),
            "device_type": "gpu" if args.gpu else "cpu",
            "verbose": -1,
            "seed": 42,
        }

        callbacks = [
            lgb.early_stopping(stopping_rounds=int(es_rounds), verbose=False),
            lgb.log_evaluation(period=50),
        ]

        booster = lgb.train(
            params,
            lgb_train,
            num_boost_round=int(num_boost_round),
            valid_sets=[lgb_train, lgb_valid],
            valid_names=["train", "valid"],
            callbacks=callbacks,
        )

        proba = booster.predict(X_test, num_iteration=booster.best_iteration)
        m = evaluate_run(y_test, proba)

        row = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "cutoff_date": cutoff_date,
            "num_boost_round": int(num_boost_round),
            "early_stopping_rounds": int(es_rounds),
            "num_leaves": int(num_leaves),
            "learning_rate": float(learning_rate),
            "feature_fraction": float(feature_fraction),
            "bagging_fraction": float(bagging_fraction),
            "bagging_freq": int(bagging_freq),
            "lambda_l1": float(lambda_l1),
            "lambda_l2": float(lambda_l2),
            "max_depth": int(max_depth),
            "best_iteration": int(booster.best_iteration or 0),
            **m,
        }
        rows.append(row)

        key = (m["f1"] if m["f1"] is not None else m["f1"]) or 0.0
        if best_key is None or key > best_key:
            best_key = key
            best_metrics = row
            best_model = booster
            best_encoders = encoders

        print(f"Tried: {row}")

    lb_df = pd.DataFrame(rows)
    if lb_exists:
        old = pd.read_csv(leaderboard_path)
        lb_df = pd.concat([old, lb_df], ignore_index=True)
    lb_df.to_csv(leaderboard_path, index=False)
    print(f"Saved leaderboard to: {leaderboard_path}")

    best_meta = {
        "framework": "lightgbm",
        "best_config": {k: best_metrics[k] for k in [
            "cutoff_date","num_boost_round","early_stopping_rounds","num_leaves","learning_rate",
            "feature_fraction","bagging_fraction","bagging_freq","lambda_l1","lambda_l2","max_depth","best_iteration"
        ]},
        "best_metrics": {k: best_metrics[k] for k in ["accuracy","precision","recall","f1","roc_auc","brier"]},
        "feature_names": feature_names,
        "cat_feature_names": cat_feature_names,
        "label_encoders": best_encoders,
        "include_team_names": args.include_team_names,
        "include_diffs": args.include_diffs,
        "team_stats_overall": os.path.abspath(args.team_stats_overall),
        "team_stats_lastN": os.path.abspath(args.team_stats_lastN) if args.team_stats_lastN else None,
        "team_elo": os.path.abspath(args.team_elo) if args.team_elo else None,
    }

    if args.save_best_model and best_model is not None:
        model_path = os.path.join(args.save_dir, "best_model_lgbm.txt")
        meta_path = os.path.join(args.save_dir, "best_metadata_lgbm.json")
        best_model.save_model(model_path)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(best_meta, f, ensure_ascii=False, indent=2)
        print(f"Saved best model to: {model_path} Saved best metadata to: {meta_path}")
    else:
        meta_path = os.path.join(args.save_dir, "best_metadata_lgbm.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(best_meta, f, ensure_ascii=False, indent=2)
        print(f"Saved best metadata to: {meta_path}")

if __name__ == "__main__":
    main()
