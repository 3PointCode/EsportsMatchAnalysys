import argparse
import json
import os
from datetime import datetime
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

from catboost import CatBoostClassifier, Pool
import xgboost as xgb
import lightgbm as lgb

def load_matches(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"game_id", "game_date", "team_blue", "team_red"}
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


def time_filter(df: pd.DataFrame,
                cutoff_date: Optional[str],
                start_date: Optional[str],
                end_date: Optional[str]) -> pd.DataFrame:
    out = df.copy()
    if cutoff_date:
        cutoff_dt = pd.to_datetime(cutoff_date)
        out = out[out["game_date"] >= cutoff_dt]
    if start_date:
        out = out[out["game_date"] >= pd.to_datetime(start_date)]
    if end_date:
        out = out[out["game_date"] <= pd.to_datetime(end_date)]
    return out.reset_index(drop=True)


def split_stage(df: pd.DataFrame,
                split_col: Optional[str],
                split_val: Optional[str],
                stage_col: Optional[str],
                stage_val: Optional[str]) -> pd.DataFrame:
    out = df.copy()
    if split_col and split_val and split_col in out.columns:
        out = out[out[split_col] == split_val]
    if stage_col and stage_val and stage_col in out.columns:
        out = out[out[stage_col] == stage_val]
    return out.reset_index(drop=True)

def merge_team_features(matches: pd.DataFrame,
                        overall: pd.DataFrame,
                        lastN: Optional[pd.DataFrame],
                        elo: Optional[pd.DataFrame],
                        include_team_names: bool,
                        include_diffs: bool) -> pd.DataFrame:
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
                if (
                    red_col in X.columns
                    and pd.api.types.is_numeric_dtype(X[col])
                    and pd.api.types.is_numeric_dtype(X[red_col])
                ):
                    X[base + "_diff"] = X[col].astype(float) - X[red_col].astype(float)

    return X

def encode_categoricals_for_tree_models(X: pd.DataFrame,
                                        meta: dict) -> pd.DataFrame:
    encoders = meta.get("label_encoders", {})
    cat_cols = meta.get("cat_feature_names", []) or []
    if not encoders or not cat_cols:
        return X

    X = X.copy()
    for col in cat_cols:
        if col not in X.columns:
            continue
        info = encoders.get(col)
        if not info:
            continue
        classes = info.get("classes_", [])
        mapping = {cls: i for i, cls in enumerate(classes)}
        X[col] = X[col].map(mapping).fillna(0).astype(int)
    return X


def derive_target(df: pd.DataFrame) -> Optional[np.ndarray]:
    if "winning_team" not in df.columns:
        return None
    return (df["winning_team"].astype(str) == df["team_blue"].astype(str)).astype(int).values


def safe_auc(y_true: np.ndarray, proba: np.ndarray) -> Optional[float]:
    if y_true is None:
        return None
    if len(np.unique(y_true)) <= 1:
        return None
    return float(roc_auc_score(y_true, proba))

def load_model_and_meta(model_path: str, meta_path: str):
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    framework = (meta.get("framework") or "").lower()

    if not framework:
        if model_path.endswith(".cbm"):
            framework = "catboost"
        elif model_path.endswith(".json"):
            framework = "xgboost"
        elif model_path.endswith(".txt"):
            framework = "lightgbm"

    if "cat" in framework:
        model = CatBoostClassifier()
        model.load_model(model_path)
        framework = "catboost"
    elif "xgb" in framework:
        model = xgb.Booster()
        model.load_model(model_path)
        framework = "xgboost"
    elif "lightgbm" in framework or "lgbm" in framework:
        model = lgb.Booster(model_file=model_path)
        framework = "lightgbm"
    else:
        raise ValueError(f"Unknown framework in metadata: {framework}")

    meta["framework"] = framework
    return model, meta, framework



def predict_proba_framework(model, framework: str, X: pd.DataFrame, meta: dict) -> np.ndarray:
    feature_names = meta.get("feature_names")
    if feature_names:
        X = X[feature_names]

    if framework == "catboost":
        cat_feature_names = meta.get("cat_feature_names", []) or []
        if cat_feature_names:
            cat_indices = [feature_names.index(c) for c in cat_feature_names]
        else:
            cat_indices = None
        pool = Pool(X, cat_features=cat_indices)
        proba = model.predict_proba(pool)[:, 1]

    elif framework == "xgboost":
        X_enc = encode_categoricals_for_tree_models(X, meta)
        dmat = xgb.DMatrix(X_enc)
        proba = model.predict(dmat)

    elif framework == "lightgbm":
        X_enc = encode_categoricals_for_tree_models(X, meta)
        proba = model.predict(X_enc, num_iteration=getattr(model, "best_iteration", None))

    else:
        raise ValueError(f"Unsupported framework: {framework}")

    return np.asarray(proba, dtype=float)

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate trained model on historical matches (basic, no matchups).")
    p.add_argument("--data", required=True, help="Path to matches CSV (lec_2023-2025_games.csv)")
    p.add_argument("--team-stats-overall", required=True)
    p.add_argument("--team-stats-lastN", default=None)
    p.add_argument("--team-elo", default=None)

    p.add_argument("--model", required=True, help="Path to trained model file")
    p.add_argument("--metadata", required=True, help="Path to metadata JSON")

    p.add_argument("--cutoff-date", default=None, help="Evaluate only games >= this date (YYYY-MM-DD)")
    p.add_argument("--start-date", default=None)
    p.add_argument("--end-date", default=None)

    p.add_argument("--split-col", default=None, help="Column name for split (e.g. 'split')")
    p.add_argument("--split-value", default=None, help="Value for split (e.g. 'Summer')")
    p.add_argument("--stage-col", default=None, help="Column name for stage (e.g. 'stage')")
    p.add_argument("--stage-value", default=None, help="Value for stage (e.g. 'Season')")

    p.add_argument("--output-csv", default="eval_basic_predictions.csv")

    return p.parse_args()


def main():
    args = parse_args()

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)

    matches = load_matches(args.data)
    matches = split_stage(matches, args.split_col, args.split_value,
                          args.stage_col, args.stage_value)
    matches = time_filter(matches, args.cutoff_date, args.start_date, args.end_date)

    if matches.empty:
        raise SystemExit("No matches after filtering – nothing to evaluate.")

    overall = load_csv(args.team_stats_overall)
    lastN = load_csv(args.team_stats_lastN)
    elo = load_csv(args.team_elo)

    if overall is None or "team" not in overall.columns:
        raise SystemExit("team_stats_overall must exist and contain 'team' column.")

    model, meta, framework = load_model_and_meta(args.model, args.metadata)

    include_team_names = bool(meta.get("include_team_names", True))
    include_diffs = bool(meta.get("include_diffs", True))

    X = merge_team_features(matches, overall, lastN, elo,
                            include_team_names=include_team_names,
                            include_diffs=include_diffs)

    proba = predict_proba_framework(model, framework, X, meta)
    y_true = derive_target(matches)
    y_pred = (proba >= 0.5).astype(int)

    if y_true is not None:
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "roc_auc": safe_auc(y_true, proba),
            "brier": float(brier_score_loss(y_true, proba)),
            "n_matches": int(len(y_true)),
        }

        print("=== Evaluation (basic, no matchups) ===")
        for k, v in metrics.items():
            print(f"{k:>10}: {v:.4f}" if isinstance(v, float) else f"{k:>10}: {v}")
    else:
        print("WARNING: no 'winning_team' column – metrics will not be computed.")
        metrics = None

    out = matches.copy()
    out["p_blue"] = proba
    out["pred_blue_win"] = y_pred
    if y_true is not None:
        out["true_blue_win"] = y_true
        out["correct"] = (out["pred_blue_win"] == out["true_blue_win"]).astype(int)

    out.to_csv(args.output_csv, index=False)
    print(f"Saved predictions to: {args.output_csv}")

if __name__ == "__main__":
    main()
