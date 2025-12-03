# Purpose:
#  - Buduje cechy tak samo jak trainXGB.py / trainLGBM.py
#    (profile drużyn + opcjonalnie diffs i nazwy drużyn)
#  - Pozwala podać wiele cutoff-date oraz listy wartości hiperparametrów
#  - Dla każdej kombinacji trenuje model XGBoost z early stopping
#  - Zapisuje leaderboard (CSV) + opcjonalnie najlepszy model i metadane
#
# Usage example:
#   python tuneXGB.py
#     --data data/lec_2023-2025_games.csv --team-stats-overall summer/team_stats_overall.csv --team-stats-lastN summer/team_stats_lastN.csv
#     --cutoff-date 2025-03-02 2025-06-09 --include-team-names --include-diffs
#     --num-boost-round 500 1000 1500 2000 --early-stopping-rounds 100 200 300 --max-depth 5 8 11 13 --eta 0.03 0.04 0.05 --subsample 0.7 0.8 0.9
#     --colsample-bytree 0.7 0.8 0.9 --reg-lambda 0.1 1.0 --reg-alpha 0.0 0.5
#     --save-dir artifacts_xgb_tune --save-best-model

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
import xgboost as xgb

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

def time_split_3way_index(df: pd.DataFrame, train_end_date: str, val_end_date: str) -> tuple[pd.Index, pd.Index, pd.Index]:
    """
    3-way split:
      train: game_date < train_end_date
      val  : train_end_date <= game_date < val_end_date
      test : game_date >= val_end_date  (tu ignorowane w tuningu)
    """
    train_dt = pd.to_datetime(train_end_date)
    val_dt = pd.to_datetime(val_end_date)
    if train_dt >= val_dt:
        raise ValueError("--train-end-date must be earlier than --val-end-date")

    train_idx = df.index[df["game_date"] < train_dt]
    val_idx = df.index[(df["game_date"] >= train_dt) & (df["game_date"] < val_dt)]
    test_idx = df.index[df["game_date"] >= val_dt]

    if len(train_idx) == 0 or len(val_idx) == 0:
        raise ValueError("3-way split produced empty train or val segment. Check dates.")
    return train_idx, val_idx, test_idx

def derive_target(matches: pd.DataFrame) -> pd.Series:
    return (matches["winning_team"].astype(str) == matches["team_blue"].astype(str)).astype(int)

def merge_team_features(matches: pd.DataFrame,
                        overall: pd.DataFrame,
                        lastN: Optional[pd.DataFrame],
                        elo: Optional[pd.DataFrame],
                        include_team_names: bool,
                        include_diffs: bool) -> tuple[pd.DataFrame, List[str], List[str]]:
    """
    Buduje te same cechy co trainLGBM/trainXGB:
    - team_blue/team_red (opcjonalnie jako kategorie)
    - profile overall + lastN + elo dla obu drużyn
    - opcjonalne różnice (blue-red) dla cech numerycznych
    """
    ov_b = overall.rename(
        columns={c: f"ov_{c}_blue" for c in overall.columns if c != "team"}
    ).rename(columns={"team": "team_blue"})
    ov_r = overall.rename(
        columns={c: f"ov_{c}_red" for c in overall.columns if c != "team"}
    ).rename(columns={"team": "team_red"})

    X = matches[["team_blue", "team_red"]].copy()
    X = X.merge(ov_b, on="team_blue", how="left").merge(ov_r, on="team_red", how="left")

    if lastN is not None:
        ln_b = lastN.rename(
            columns={c: f"ln_{c}_blue" for c in lastN.columns if c != "team"}
        ).rename(columns={"team": "team_blue"})
        ln_r = lastN.rename(
            columns={c: f"ln_{c}_red" for c in lastN.columns if c != "team"}
        ).rename(columns={"team": "team_red"})
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
                if red_col in X.columns and \
                   pd.api.types.is_numeric_dtype(X[col]) and \
                   pd.api.types.is_numeric_dtype(X[red_col]):
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
    """
    XGBoost nie obsługuje kategorycznych tak wygodnie jak CatBoost,
    więc nazwy drużyn enkodujemy LabelEncoderem.
    """
    encoders = {}
    for c in cat_cols:
        le = LabelEncoder()
        all_vals = pd.concat(
            [X_train[c].astype(str), X_test[c].astype(str)],
            axis=0
        ).fillna("<UNK>")
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
    ap = argparse.ArgumentParser(description="Grid search for XGBoost (pre-game)")
    ap.add_argument("--data", required=True)
    ap.add_argument("--team-stats-overall", required=True)
    ap.add_argument("--team-stats-lastN", default=None)
    ap.add_argument("--team-elo", default=None)
    ap.add_argument("--cutoff-date", nargs="+", default=[None], help="You can pass multiple dates (YYYY-MM-DD)")
    ap.add_argument("--train-end-date", default=None, help="If set together with --val-end-date: train < train_end_date (3-way split).")
    ap.add_argument("--val-end-date", default=None, help="If set together with --train-end-date: validation in [train_end_date, val_end_date], test >= val_end_date.")
    ap.add_argument("--include-team-names", action="store_true")
    ap.add_argument("--include-diffs", action="store_true")

    ap.add_argument("--num-boost-round", nargs="+", type=int, default=[1500])
    ap.add_argument("--early-stopping-rounds", nargs="+", type=int, default=[100])
    ap.add_argument("--max-depth", nargs="+", type=int, default=[6])
    ap.add_argument("--eta", nargs="+", type=float, default=[0.05])
    ap.add_argument("--subsample", nargs="+", type=float, default=[0.9])
    ap.add_argument("--colsample-bytree", nargs="+", type=float, default=[0.9])
    ap.add_argument("--reg-lambda", nargs="+", type=float, default=[1.0])
    ap.add_argument("--reg-alpha", nargs="+", type=float, default=[0.0])
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--save-dir", default="artifacts_xgb_tune")
    ap.add_argument("--save-best-model", action="store_true")
    ap.add_argument("--leaderboard", default="leaderboard_xgb.csv")

    return ap.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    matches = load_matches(args.data)
    overall = load_csv(args.team_stats_overall)
    lastN   = load_csv(args.team_stats_lastN)
    elo     = load_csv(args.team_elo)

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
        args.max_depth,
        args.eta,
        args.subsample,
        args.colsample_bytree,
        args.reg_lambda,
        args.reg_alpha,
    ))
    print(f"Grid size: {len(grid)} (cutoffs x hyperparams)")

    leaderboard_path = os.path.join(args.save_dir, args.leaderboard)
    lb_exists = os.path.exists(leaderboard_path)
    rows = []
    best_key = None
    best_metrics = None
    best_model = None
    best_encoders = None

    for (cutoff_date,
         num_boost_round,
         es_rounds,
         max_depth,
         eta,
         subsample,
         colsample_bytree,
         reg_lambda,
         reg_alpha) in grid:

        tr_idx, te_idx = time_split_index(matches, cutoff_date)
        X_train = X_full.loc[tr_idx].reset_index(drop=True)
        X_test  = X_full.loc[te_idx].reset_index(drop=True)
        y_train, y_test = y_full[tr_idx], y_full[te_idx]

        encoders = {}
        if cat_feature_names:
            X_train, X_test, encoders = label_encode_teams(X_train, X_test, cat_feature_names)

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dvalid = xgb.DMatrix(X_test, label=y_test)

        params = {
            "objective": "binary:logistic",
            "eval_metric": ["auc", "logloss"],
            "max_depth": int(max_depth),
            "eta": float(eta),
            "subsample": float(subsample),
            "colsample_bytree": float(colsample_bytree),
            "lambda": float(reg_lambda),
            "alpha": float(reg_alpha),
            "tree_method": "gpu_hist" if args.gpu else "hist",
            "predictor": "gpu_predictor" if args.gpu else "auto",
            "seed": 42,
        }

        evals = [(dtrain, "train"), (dvalid, "valid")]

        booster = xgb.train(
            params,
            dtrain,
            num_boost_round=int(num_boost_round),
            evals=evals,
            early_stopping_rounds=int(es_rounds),
            verbose_eval=False,
        )

        proba = booster.predict(
            dvalid,
            iteration_range=(0, booster.best_iteration + 1)
        )
        m = evaluate_run(y_test, proba)

        row = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "cutoff_date": cutoff_date,
            "num_boost_round": int(num_boost_round),
            "early_stopping_rounds": int(es_rounds),
            "max_depth": int(max_depth),
            "eta": float(eta),
            "subsample": float(subsample),
            "colsample_bytree": float(colsample_bytree),
            "reg_lambda": float(reg_lambda),
            "reg_alpha": float(reg_alpha),
            "best_iteration": int(booster.best_iteration),
            **m,
        }
        rows.append(row)

        key = (m["roc_auc"] if m["roc_auc"] is not None else m["f1"]) or 0.0
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

    lb_df = lb_df.sort_values("roc_auc", ascending=False)
    lb_df.to_csv(leaderboard_path, index=False)
    print(f"Saved leaderboard to: {leaderboard_path}")

    best_meta = {
        "framework": "xgboost",
        "best_config": {k: best_metrics[k] for k in [
            "cutoff_date",
            "num_boost_round",
            "early_stopping_rounds",
            "max_depth",
            "eta",
            "subsample",
            "colsample_bytree",
            "reg_lambda",
            "reg_alpha",
            "best_iteration",
        ]},
        "best_metrics": {k: best_metrics[k] for k in [
            "accuracy", "precision", "recall", "f1", "roc_auc", "brier"
        ]},
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
        model_path = os.path.join(args.save_dir, "best_model_xgb.json")
        meta_path = os.path.join(args.save_dir, "best_metadata_xgb.json")
        best_model.save_model(model_path)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(best_meta, f, ensure_ascii=False, indent=2)
        print(f"Saved best model to: {model_path}")
        print(f"Saved best metadata to: {meta_path}")
    else:
        meta_path = os.path.join(args.save_dir, "best_metadata_xgb.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(best_meta, f, ensure_ascii=False, indent=2)
        print(f"Saved best metadata to: {meta_path}")


if __name__ == "__main__":
    main()
