# Purpose:
#   Use a pre-trained model (trained with team strength profiles) to predict a match outcome
#   BEFORE the game, based only on team names. The script automatically loads the same
#   team_stats files used during training (paths stored in metadata.json), joins the
#   appropriate rows for Blue/Red, reconstructs the exact feature vector, and outputs
#   probabilities.
#
# Usage (single match):
#   python predict_basic.py --model artifacts_cat_tune/best_model_cat.cbm --metadata artifacts_cat_tune/best_metadata_cat.json --team-blue "G2 Esports" --team-red "SK Gaming"
#
# Usage (batch CSV):
#   python predict_basic.py --model artifacts/model.cbm --metadata artifacts/metadata.json \
#     --in matches_to_score.csv --out preds.csv
#   # CSV must have columns: team_blue, team_red
#
# Notes:
# - The script respects training choices saved in metadata: include_team_names / include_diffs.
# - You can override paths to team_stats files with flags if needed.

import argparse
import json
import os
import sys
from typing import List

import pandas as pd
from catboost import CatBoostClassifier, Pool

def parse_args():
    p = argparse.ArgumentParser(description="Pre-game prediction using team profiles")
    p.add_argument("--model", required=True)
    p.add_argument("--metadata", required=True)

    # Single-match mode
    p.add_argument("--team-blue")
    p.add_argument("--team-red")

    # Batch mode
    p.add_argument("--in", dest="in_csv", default=None, help="CSV with columns: team_blue, team_red")
    p.add_argument("--out", dest="out_csv", default="predictions.csv", help="Where to save batch predictions")

    # Optional overrides for stats locations (else taken from metadata)
    p.add_argument("--team-stats-overall", default="summer/team_stats_overall.csv")
    p.add_argument("--team-stats-lastN", default="summer/team_stats_lastN.csv")
    p.add_argument("--team-elo", default=None)

    return p.parse_args()

def load_metadata(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_csv_safely(path: str | None) -> pd.DataFrame | None:
    if not path:
        return None
    if not os.path.exists(path):
        raise SystemExit(f"File not found: {path}")
    return pd.read_csv(path)

def ensure_required_cols(df: pd.DataFrame, cols: List[str], name: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(f"{name} is missing columns: {missing}")

def join_team_profiles(df_pairs: pd.DataFrame,
                       overall: pd.DataFrame,
                       lastN: pd.DataFrame | None,
                       elo: pd.DataFrame | None,
                       include_team_names: bool,
                       include_diffs: bool,
                       feature_names_from_training: List[str]) -> pd.DataFrame:
    # Base join on team names
    ov_b = overall.rename(columns={c: f"ov_{c}_blue" for c in overall.columns if c != "team"}).rename(columns={"team": "team_blue"})
    ov_r = overall.rename(columns={c: f"ov_{c}_red" for c in overall.columns if c != "team"}).rename(columns={"team": "team_red"})
    X = df_pairs.merge(ov_b, on="team_blue", how="left").merge(ov_r, on="team_red", how="left")

    if lastN is not None:
        ln_b = lastN.rename(columns={c: f"ln_{c}_blue" for c in lastN.columns if c != "team"}).rename(columns={"team": "team_blue"})
        ln_r = lastN.rename(columns={c: f"ln_{c}_red" for c in lastN.columns if c != "team"}).rename(columns={"team": "team_red"})
        X = X.merge(ln_b, on="team_blue", how="left").merge(ln_r, on="team_red", how="left")

    if elo is not None:
        e_b = elo.rename(columns={"team": "team_blue", "elo": "elo_blue"})
        e_r = elo.rename(columns={"team": "team_red", "elo": "elo_red"})
        X = X.merge(e_b, on="team_blue", how="left").merge(e_r, on="team_red", how="left")

    # Add diffs exactly like in training
    if include_diffs:
        for col in list(X.columns):
            if col.endswith("_blue"):
                base = col[:-5]
                red_col = base + "_red"
                if red_col in X.columns:
                    if pd.api.types.is_numeric_dtype(X[col]) and pd.api.types.is_numeric_dtype(X[red_col]):
                        X[base + "_diff"] = X[col].astype(float) - X[red_col].astype(float)

    # If team names were not included during training, we must drop them now
    if include_team_names:
        # keep as part of features
        pass
    else:
        # ensure they won't be in features order unless explicitly in training
        pass

    # Reorder/select columns to EXACTLY match training feature order
    missing = [c for c in feature_names_from_training if c not in X.columns]
    if missing:
        raise SystemExit("Cannot build feature vector — missing columns (did you build stats with the same cutoff/columns?):" + "".join(missing))
    X = X[feature_names_from_training]

    return X

def single_or_batch_df(args) -> pd.DataFrame:
    if args.in_csv:
        df = pd.read_csv(args.in_csv)
        ensure_required_cols(df, ["team_blue", "team_red"], "--in CSV")
        return df[["team_blue", "team_red"]].astype(str)
    # single match mode
    if not (args.team_blue and args.team_red):
        raise SystemExit("Provide either --in CSV or both --team-blue and --team-red")
    return pd.DataFrame([[args.team_blue, args.team_red]], columns=["team_blue", "team_red"]) 

def main():
    args = parse_args()

    meta = load_metadata(args.metadata)
    feature_names = meta["feature_names"]
    cat_feature_names = meta.get("cat_feature_names", [])
    include_team_names = bool(meta.get("include_team_names", False))
    include_diffs = bool(meta.get("include_diffs", False))

    # Resolve stats paths (flags override metadata)
    path_overall = args.team_stats_overall or meta.get("team_stats_overall")
    path_lastN = args.team_stats_lastN or meta.get("team_stats_lastN")
    path_elo = args.team_elo or meta.get("team_elo")

    if not path_overall:
        raise SystemExit("Path to team_stats_overall.csv not provided (neither metadata nor flag).")

    overall = load_csv_safely(path_overall)
    lastN = load_csv_safely(path_lastN)
    elo = load_csv_safely(path_elo)

    if "team" not in overall.columns:
        raise SystemExit("team_stats_overall.csv must contain 'team' column")

    pairs_df = single_or_batch_df(args)

    # Build features identical to training
    X = join_team_profiles(
        pairs_df,
        overall=overall,
        lastN=lastN,
        elo=elo,
        include_team_names=include_team_names,
        include_diffs=include_diffs,
        feature_names_from_training=feature_names,
    )

    # Categorical indices (team names may or may not be present depending on training)
    cat_indices = [X.columns.get_loc(c) for c in cat_feature_names if c in X.columns]

    # Load model and predict
    model = CatBoostClassifier()
    model.load_model(args.model)

    pool = Pool(X, cat_features=cat_indices if cat_indices else None)
    proba = model.predict_proba(pool)  # [:, 1] = P(blue wins)

    # Output
    out = pairs_df.copy()
    out["proba_blue_win"] = proba[:, 1]
    out["proba_red_win"] = proba[:, 0]
    out["predicted_winner"] = out.apply(lambda r: r.team_blue if r.proba_blue_win >= 0.5 else r.team_red, axis=1)

    if args.in_csv:
        out.to_csv(args.out_csv, index=False)
        print(f"Saved predictions to: {args.out_csv}")
    else:
        row = out.iloc[0]
        print("=== Pre-game Prediction (teams + profiles) ===")
        print(f"team_blue: {row.team_blue}")
        print(f"team_red : {row.team_red}")
        print(f"proba_blue_win: {row.proba_blue_win:.4f}")
        print(f"proba_red_win : {row.proba_red_win:.4f}")
        print(f"predicted_winner: {row.predicted_winner}")

if __name__ == "__main__":
    main()