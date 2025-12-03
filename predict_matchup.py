# Purpose:
#   Predict a match before the game using the trained model (teams + team profiles)
#   and blend in champion-vs-champion matchup information from counters_data.json.
#
#   We do not change the model's feature space (it's already trained). Instead we:
#     1) compute the model's base probability P_model(Blue wins),
#     2) compute a normalized matchup score in [-1, 1] from five role matchups,
#     3) adjust the logit with a tunable weight: logit(P_final) = logit(P_model) + beta * matchup_score.
#
# Usage (single match):
#   python predict_matchups.py --model artifacts/model.cbm --metadata artifacts/metadata.json \
#     --counters /path/to/counters_data.json \
#     --team-blue "G2 Esports" --team-red "SK Gaming" \
#     --top-blue Aatrox --jgl-blue Lee Sin --mid-blue Ahri --adc-blue Jinx --sup-blue Thresh \
#     --top-red Sion --jgl-red Viego --mid-red Orianna --adc-red Kai'Sa --sup-red Nautilus
#
# Usage (batch CSV):
#   python predict_matchups.py --model artifacts/model.cbm --metadata artifacts/metadata.json \
#     --counters /path/to/counters_data.json \
#     --in matches_to_score.csv --out preds.csv
#   # CSV must have: team_blue,team_red,top_blue,jgl_blue,mid_blue,adc_blue,sup_blue,top_red,jgl_red,mid_red,adc_red,sup_red
#

import argparse
import json
import os
import math
from typing import Dict, List

import pandas as pd
from catboost import CatBoostClassifier, Pool

ROLES = ["top", "jgl", "mid", "adc", "sup"]


def parse_args():
    p = argparse.ArgumentParser(description="Pre-game prediction with champion matchup blending")
    p.add_argument("--model", required=True)
    p.add_argument("--metadata", required=True)
    p.add_argument("--counters", required=True, help="Path to counters_data.json")

    # Single match
    p.add_argument("--team-blue", nargs="+")
    p.add_argument("--team-red", nargs="+")
    p.add_argument("--top-blue", nargs="+")
    p.add_argument("--jgl-blue", nargs="+")
    p.add_argument("--mid-blue", nargs="+")
    p.add_argument("--adc-blue", nargs="+")
    p.add_argument("--sup-blue", nargs="+")
    p.add_argument("--top-red", nargs="+")
    p.add_argument("--jgl-red", nargs="+")
    p.add_argument("--mid-red", nargs="+")
    p.add_argument("--adc-red", nargs="+")
    p.add_argument("--sup-red", nargs="+")


    # Batch mode
    p.add_argument("--in", dest="in_csv", default=None)
    p.add_argument("--out", dest="out_csv", default="predictions_matchups.csv")

    # Optional overrides for stats locations (else taken from metadata)
    p.add_argument("--team-stats-overall", default=None)
    p.add_argument("--team-stats-lastN", default=None)
    p.add_argument("--team-elo", default=None)

    # Blend strength – zgodne z evaluate_matchupsSummer
    p.add_argument(
        "--matchup-weight",
        type=float,
        default=0.2,
        help="How strongly matchups sway final probability (same semantics as in evaluate_matchupsSummer)",
    )

    return p.parse_args()


def load_json(path: str) -> dict:
    if not os.path.exists(path):
        raise SystemExit(f"File not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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

def _join_arg(val):
    """Join nargs='+' arguments into a single string, leave others as-is."""
    if isinstance(val, list):
        return " ".join(str(x) for x in val)
    return val


def single_or_batch_df(args) -> pd.DataFrame:
    if args.in_csv:
        df = pd.read_csv(args.in_csv)
        req = [
            "team_blue", "team_red",
            "top_blue", "jgl_blue", "mid_blue", "adc_blue", "sup_blue",
            "top_red", "jgl_red", "mid_red", "adc_red", "sup_red",
        ]
        ensure_required_cols(df, req, "--in CSV")
        return df[req].astype(str)

    # single match mode
    raw_fields = [
        args.team_blue, args.team_red,
        args.top_blue, args.jgl_blue, args.mid_blue, args.adc_blue, args.sup_blue,
        args.top_red, args.jgl_red, args.mid_red, args.adc_red, args.sup_red,
    ]
    # join lists such as ["Lee", "Sin"] -> "Lee Sin"
    fields = [_join_arg(v) for v in raw_fields]

    if any(v is None for v in fields):
        raise SystemExit("Provide either --in CSV or all single-match flags for teams and champions.")

    df = pd.DataFrame([fields], columns=[
        "team_blue", "team_red",
        "top_blue", "jgl_blue", "mid_blue", "adc_blue", "sup_blue",
        "top_red", "jgl_red", "mid_red", "adc_red", "sup_red",
    ])

    return df



def join_team_profiles(df_pairs: pd.DataFrame,
                       overall: pd.DataFrame,
                       lastN: pd.DataFrame | None,
                       elo: pd.DataFrame | None,
                       include_team_names: bool,
                       include_diffs: bool,
                       feature_names_from_training: List[str]) -> pd.DataFrame:
    ov_b = overall.rename(columns={c: f"ov_{c}_blue" for c in overall.columns if c != "team"}).rename(
        columns={"team": "team_blue"}
    )
    ov_r = overall.rename(columns={c: f"ov_{c}_red" for c in overall.columns if c != "team"}).rename(
        columns={"team": "team_red"}
    )
    X = df_pairs.merge(ov_b, on="team_blue", how="left").merge(ov_r, on="team_red", how="left")

    if lastN is not None:
        ln_b = lastN.rename(columns={c: f"ln_{c}_blue" for c in lastN.columns if c != "team"}).rename(
            columns={"team": "team_blue"}
        )
        ln_r = lastN.rename(columns={c: f"ln_{c}_red" for c in lastN.columns if c != "team"}).rename(
            columns={"team": "team_red"}
        )
        X = X.merge(ln_b, on="team_blue", how="left").merge(ln_r, on="team_red", how="left")

    if elo is not None:
        e_b = elo.rename(columns={"team": "team_blue", "elo": "elo_blue"})
        e_r = elo.rename(columns={"team": "team_red", "elo": "elo_red"})
        X = X.merge(e_b, on="team_blue", how="left").merge(e_r, on="team_red", how="left")

    if include_diffs:
        import pandas as pd  # local import
        for col in list(X.columns):
            if col.endswith("_blue"):
                base = col[:-5]
                red_col = base + "_red"
                if red_col in X.columns:
                    if pd.api.types.is_numeric_dtype(X[col]) and pd.api.types.is_numeric_dtype(X[red_col]):
                        X[base + "_diff"] = X[col].astype(float) - X[red_col].astype(float)

    # Respect feature order used in training
    missing = [c for c in feature_names_from_training if c not in X.columns]
    if missing:
        raise SystemExit("Cannot build feature vector — missing columns: " + ", ".join(missing))
    X = X[feature_names_from_training]
    return X


def compute_matchup_blue_prob(row: pd.Series, counters: Dict) -> float:
    """
    Policz średni winrate Blue z matchupów (0..1),
    tak jak w evaluate_matchupsSummer.py.
    """
    probs = []
    for role in ROLES:
        role_key = "jungle" if role == "jgl" else role
        role_data = counters.get(role_key)
        if not role_data:
            continue

        col_b = f"{role}_blue"
        col_r = f"{role}_red"
        if col_b not in row or col_r not in row:
            continue

        champ_b = str(row[col_b])
        champ_r = str(row[col_r])

        win_table = role_data.get(champ_b)
        if not win_table or not isinstance(win_table, dict):
            continue

        p_raw = win_table.get(champ_r)
        if p_raw is None:
            continue

        p = float(p_raw)
        if p > 1.0:
            p /= 100.0

        if 0.0 <= p <= 1.0:
            probs.append(p)

    if not probs:
        return 0.5
    return float(sum(probs) / len(probs))


def blend_probability_linear(p_model: float, p_matchup: float, w: float) -> float:
    """
    Ten sam wzór co w evaluate_matchupsSummer:
    p_final = p_model + w * (p_matchup - 0.5), z clipem.
    """
    p_final = p_model + w * (p_matchup - 0.5)
    p_final = max(0.01, min(0.99, p_final))
    return p_final


def main():
    args = parse_args()

    counters = load_json(args.counters)
    meta = load_metadata(args.metadata)

    feature_names = meta["feature_names"]
    cat_feature_names = meta.get("cat_feature_names", [])
    include_team_names = bool(meta.get("include_team_names", False))
    include_diffs = bool(meta.get("include_diffs", False))

    # Resolve stats paths
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

    df = single_or_batch_df(args)

    # Build model feature matrix (identical to training)
    X = join_team_profiles(
        df[["team_blue", "team_red"]].copy(),
        overall=overall,
        lastN=lastN,
        elo=elo,
        include_team_names=include_team_names,
        include_diffs=include_diffs,
        feature_names_from_training=feature_names,
    )

    # Categorical indices
    cat_indices = [X.columns.get_loc(c) for c in cat_feature_names if c in X.columns]

    # Load model and get base probabilities
    model = CatBoostClassifier()
    model.load_model(args.model)
    pool = Pool(X, cat_features=cat_indices if cat_indices else None)
    proba = model.predict_proba(pool)  # [:, 1] = P(blue wins)

    # Compute matchup-based probability and final blend
    matchup_probs = []
    p_final = []

    for i, row in df.iterrows():
        p_match = compute_matchup_blue_prob(row, counters)
        matchup_probs.append(p_match)
        p_final.append(blend_probability_linear(float(proba[i, 1]), p_match, args.matchup_weight))

    out = df.copy()
    out["proba_blue_win_model"] = proba[:, 1]
    out["proba_blue_win_matchups"] = matchup_probs
    out["proba_blue_win_final"] = p_final
    out["proba_red_win_final"] = 1.0 - out["proba_blue_win_final"]
    out["predicted_winner"] = out.apply(
        lambda r: r.team_blue if r.proba_blue_win_final >= 0.5 else r.team_red, axis=1
    )

    if args.in_csv:
        out.to_csv(args.out_csv, index=False)
        print(f"Saved predictions to: {args.out_csv}")
    else:
        r = out.iloc[0]
        print("=== Pre-game Prediction (teams + profiles + matchups) ===")
        print(f"team_blue: {r.team_blue}")
        print(f"team_red : {r.team_red}")
        for role in ROLES:
            print(f"{role:>6}: {r[f'{role}_blue']} vs {r[f'{role}_red']}")
        print(f"model_proba_blue_win    : {r.proba_blue_win_model:.4f}")
        print(f"matchup_proba_blue_win  : {r.proba_blue_win_matchups:.4f}")
        print(f"FINAL proba_blue_win    : {r.proba_blue_win_final:.4f}")
        print(f"FINAL proba_red_win     : {r.proba_red_win_final:.4f}")
        print(f"predicted_winner        : {r.predicted_winner}")


if __name__ == "__main__":
    main()
