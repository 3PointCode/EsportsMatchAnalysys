"""
Purpose:
REST API service for predicting outcomes of League of Legends esports matches.
The prediction uses:
1) Team-level statistics (team strength, last-N form, ELO, etc.)
2) Optionally — champion draft matchups

This service supports 3 ML models:
- CatBoost
- XGBoost
- LightGBM

Trained models and metadata are loaded dynamically from the below folders:
- artifacts_cat_tune
- artifacts_xgb_tune
- artifacts_lgbm_tune

The API also uses:
- summer/team_stats_overall.csv     (team-level aggregated features)
- summer/team_stats_lastN.csv       (last-N performance, optional)
- data_matchups/counters_data.json  (champion vs champion winrates)

The API intelligently decides prediction mode:
- If no champions are picked -> basic mode (team-only prediction)
- If full 5v5 draft is present -> matchup mode (team-level + matchup data)

Usage example:

Start the API:
    uvicorn api:app --reload --host 0.0.0.0 --port 8000

Basic prediction (teams only):
    curl -X POST http://localhost:8000/api/predict \
         -H "Content-Type: application/json" \
         -d '{"model":"CatBoost","blue_team":"G2","red_team":"Fnatic"}'

Matchup prediction (teams + full draft):
    curl -X POST http://localhost:8000/api/predict \
         -H "Content-Type: application/json" \
         -d '{
               "model":"CatBoost",
               "blue_team":"G2",
               "red_team":"Fnatic",
               "blue_picks": {"TOP":"Aatrox","JUNGLE":"Maokai","MID":"Ahri","ADC":"Jinx","SUPPORT":"Thresh"},
               "red_picks":  {"TOP":"Sion","JUNGLE":"Viego","MID":"Orianna","ADC":"Kai'Sa","SUPPORT":"Nautilus"}
             }'

API response format:
    {
      "blue_win_prob": 0.73,
      "red_win_prob": 0.27
    }
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Literal, Optional, Any
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from catboost import CatBoostClassifier, Pool
import xgboost as xgb
import lightgbm as lgb

logger = logging.getLogger(__name__)

ModelName = Literal["CatBoost", "XGBoost", "LightGBM"]

BASE_DIR = Path(__file__).resolve().parent

# Path mapping for all tuned models
MODEL_CONFIG: Dict[ModelName, Dict[str, Path]] = {
    "CatBoost": {
        "model": BASE_DIR / "artifacts_cat_tune" / "best_model_cat.cbm",
        "meta": BASE_DIR / "artifacts_cat_tune" / "best_metadata_cat.json",
    },
    "XGBoost": {
        "model": BASE_DIR / "artifacts_xgb_tune" / "best_model_xgb.json",
        "meta": BASE_DIR / "artifacts_xgb_tune" / "best_metadata_xgb.json",
    },
    "LightGBM": {
        "model": BASE_DIR / "artifacts_lgbm_tune" / "best_model_lgbm.txt",
        "meta": BASE_DIR / "artifacts_lgbm_tune" / "best_metadata_lgbm.json",
    },
}

# Mapping frontend -> internal role names used in team_stats + counters
FRONT_TO_INTERNAL_ROLE = {
    "TOP": ("top", "top_blue", "top_red"),
    "JUNGLE": ("jungle", "jgl_blue", "jgl_red"),
    "MID": ("mid", "mid_blue", "mid_red"),
    "ADC": ("adc", "adc_blue", "adc_red"),
    "SUPPORT": ("sup", "sup_blue", "sup_red"),
}

# Default weight for champion-vs-champion matchup contribution
DEFAULT_MATCHUP_WEIGHT = 0.22

# Encodes categorical features into integer IDs for XGBoost and LightGBM
# CatBoost does not use this as it handles categorical features internally
def encode_categoricals_for_tree_models(X: pd.DataFrame, meta: dict) -> pd.DataFrame:
    encoders = meta.get("label_encoders", {}) or {}
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

# Combines team identifiers with aggregated team statistics
# Produces: ov_*_* : base stats, ln_*_* : last-N stats (optional), elo_* : ELO ratings (optional), *_diff : blue minus red feature differences (optional)
def merge_team_features(
    matches: pd.DataFrame,
    overall: pd.DataFrame,
    lastN: Optional[pd.DataFrame],
    elo: Optional[pd.DataFrame],
    include_team_names: bool,
    include_diffs: bool,
) -> pd.DataFrame:
    # Merge team-wide aggregated stats
    ov_b = overall.rename(
        columns={c: f"ov_{c}_blue" for c in overall.columns if c != "team"}
    ).rename(columns={"team": "team_blue"})
    ov_r = overall.rename(
        columns={c: f"ov_{c}_red" for c in overall.columns if c != "team"}
    ).rename(columns={"team": "team_red"})

    X = matches[["team_blue", "team_red"]].copy()
    X = X.merge(ov_b, on="team_blue", how="left").merge(ov_r, on="team_red", how="left")

    # Merge last-N performance
    if lastN is not None:
        ln_b = lastN.rename(
            columns={c: f"ln_{c}_blue" for c in lastN.columns if c != "team"}
        ).rename(columns={"team": "team_blue"})
        ln_r = lastN.rename(
            columns={c: f"ln_{c}_red" for c in lastN.columns if c != "team"}
        ).rename(columns={"team": "team_red"})
        X = X.merge(ln_b, on="team_blue", how="left").merge(ln_r, on="team_red", how="left")
    
    # Merge ELO ratings
    if elo is not None:
        e_b = elo.rename(columns={"team": "team_blue", "elo": "elo_blue"})
        e_r = elo.rename(columns={"team": "team_red", "elo": "elo_red"})
        X = X.merge(e_b, on="team_blue", how="left").merge(e_r, on="team_red", how="left")

    # Compute difference features
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

"""
    Loads a trained model and its metadata, the type of model is determined either from metadata or file extension, supporting: CatBoost (.cbm), XGBoost (.json) and LightGBM (.txt)

    Returns:
    - model:     model instance
    - meta:      metadata dictionary
    - framework: one of: "catboost", "xgboost", "lightgbm"
"""
def load_model_and_meta(model_path: Path, meta_path: Path):
    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    framework = (meta.get("framework") or "").lower()

    # If not explicitly provided take the info from file extension
    if not framework:
        if str(model_path).endswith(".cbm"):
            framework = "catboost"
        elif str(model_path).endswith(".json"):
            framework = "xgboost"
        elif str(model_path).endswith(".txt"):
            framework = "lightgbm"

    # Model loading
    if "cat" in framework:
        model = CatBoostClassifier()
        model.load_model(str(model_path))
        framework = "catboost"
    elif "xgb" in framework:
        model = xgb.Booster()
        model.load_model(str(model_path))
        framework = "xgboost"
    elif "lightgbm" in framework or "lgbm" in framework:
        model = lgb.Booster(model_file=str(model_path))
        framework = "lightgbm"
    else:
        raise ValueError(f"Unknown framework in metadata: {framework}")

    meta["framework"] = framework
    return model, meta, framework

# Computes probability of blue_win using the correct prediction pipeline for a given framework
def predict_proba_framework(model, framework: str, X: pd.DataFrame, meta: dict) -> np.ndarray:
    # Use only required model features
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


MATCHUP_ROLES = ["top", "jungle", "mid", "adc", "support"]

# Loads counters_data.json, containing champion vs champion matchup winrates
def load_counters(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"counters_data.json not found at {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

# Maps logical role name to actual dataframe column names
def get_role_columns(role: str) -> tuple[str, str]:
    if role == "jungle":
        tag = "jgl"
    elif role == "support":
        tag = "sup"
    else:
        tag = role
    return f"{tag}_blue", f"{tag}_red"

# Computes average p(blue_win) across all champion-vs-champion matchups, returns float value from 0-1
def compute_matchup_blue_prob(row: pd.Series, counters: dict) -> float:
    probs = []
    for role in MATCHUP_ROLES:
        role_data = counters.get(role)
        if not role_data:
            continue
        col_b, col_r = get_role_columns(role)
        if col_b not in row or col_r not in row:
            continue

        champ_b = str(row[col_b])
        champ_r = str(row[col_r])

        win_table = role_data.get(champ_b)
        if not win_table:
            continue
        p_blue_raw = win_table.get(champ_r)
        if p_blue_raw is None:
            continue

        p = float(p_blue_raw)
        if p > 1.0:
            p /= 100.0

        if p < 0.0 or p > 1.0:
            continue

        probs.append(p)

    if not probs:
        return 0.5
    return float(np.mean(probs))

# Stores preloaded model, metadata, and team statistics in memory so that repeated predictions do not reload anything from disk
class RuntimeState(BaseModel):
    model_name: ModelName
    framework: str
    model: Any
    meta: dict
    overall: Any
    lastN: Any
    elo: Any

_runtime_cache: Dict[ModelName, RuntimeState] = {}
_counters_cache: Optional[dict] = None

# eturns cached champion matchup data (counters_data.json), if not loaded - attempts to load
def _get_counters() -> Optional[dict]:
    global _counters_cache
    if _counters_cache is None:
        path = BASE_DIR / "data_matchups" / "counters_data.json"
        if path.exists():
            try:
                _counters_cache = load_counters(path)
                logger.info("Loaded counters_data.json from %s", path)
            except Exception as e:
                logger.error("Failed to load counters_data.json: %s", e)
                _counters_cache = None
        else:
            logger.warning("counters_data.json not found at %s – matchups will be ignored.", path)
            _counters_cache = None
    return _counters_cache


# Loads team statistics required for prediction, looking for the below local files with fallback to metadata paths
def _load_team_stats_from_meta(meta: dict):
    summer_dir = BASE_DIR / "summer"
    local_overall = summer_dir / "team_stats_overall.csv"
    local_lastN = summer_dir / "team_stats_lastN.csv"
    local_elo = summer_dir / "team_elo.csv"

    # Local files take priority
    if local_overall.exists():
        overall = pd.read_csv(local_overall)
        lastN = pd.read_csv(local_lastN) if local_lastN.exists() else None
        elo = pd.read_csv(local_elo) if local_elo.exists() else None

        if "team" not in overall.columns:
            raise RuntimeError("team_stats_overall.csv must contain 'team' column")

        return overall, lastN, elo  

    # Fallback to metadata references paths from training environment
    def resolve(path_val: Optional[str]) -> Optional[Path]:
        if not path_val:
            return None
        p = Path(path_val)
        if not p.is_absolute():
            p = BASE_DIR / p
        return p

    path_overall = resolve(meta.get("team_stats_overall"))
    path_lastN = resolve(meta.get("team_stats_lastN"))
    path_elo = resolve(meta.get("team_elo"))

    if not path_overall or not path_overall.exists():
        raise RuntimeError(
            f"team_stats_overall.csv not found. Metadata has: {meta.get('team_stats_overall')}"
        )

    overall = pd.read_csv(path_overall)
    lastN = pd.read_csv(path_lastN) if path_lastN and path_lastN.exists() else None
    elo = pd.read_csv(path_elo) if path_elo and path_elo.exists() else None

    if "team" not in overall.columns:
        raise RuntimeError("team_stats_overall.csv must contain 'team' column")

    return overall, lastN, elo

# Initializes or retrieves cached runtime model state for a given model type, loads once per model: model object, metadata, team statistics
def _get_runtime_state(model_name: ModelName) -> RuntimeState:
    if model_name in _runtime_cache:
        return _runtime_cache[model_name]

    cfg = MODEL_CONFIG[model_name]
    model_path = cfg["model"]
    meta_path = cfg["meta"]

    if not model_path.exists():
        raise RuntimeError(f"Model file not found for {model_name}: {model_path}")
    if not meta_path.exists():
        raise RuntimeError(f"Metadata file not found for {model_name}: {meta_path}")

    model, meta, framework = load_model_and_meta(model_path, meta_path)
    overall, lastN, elo = _load_team_stats_from_meta(meta)

    state = RuntimeState(
        model_name=model_name,
        framework=framework,
        model=model,
        meta=meta,
        overall=overall,
        lastN=lastN,
        elo=elo,
    )
    _runtime_cache[model_name] = state
    logger.info("Loaded model %s (%s)", model_name, framework)
    return state

# Prediction engine: Always compute team-level features, if full 5v5 draft exists and counters_data.json is available -> compute matchup score and blend with model prediction
# Returns float value: probability of blue_win from 0-1
def _compute_blue_win_probability(
    model_name: ModelName,
    blue_team: str,
    red_team: str,
    blue_picks: Optional[Dict[str, str]] = None,
    red_picks: Optional[Dict[str, str]] = None,
) -> float:
    state = _get_runtime_state(model_name)
    meta = state.meta

    # 1) Build match row (team-only or team+champ draft)
    row: Dict[str, Any] = {
        "team_blue": blue_team,
        "team_red": red_team,
    }

    full_draft = False
    if blue_picks and red_picks:
        required_roles = set(FRONT_TO_INTERNAL_ROLE.keys())
        if required_roles.issubset(blue_picks.keys()) and required_roles.issubset(red_picks.keys()):
            full_draft = True
            # Map picks into expected feature columns
            for front_role, (_, col_blue, col_red) in FRONT_TO_INTERNAL_ROLE.items():
                row[col_blue] = blue_picks[front_role]
                row[col_red] = red_picks[front_role]

    matches_df = pd.DataFrame([row])

    # 2) Merge team-level features
    include_team_names = bool(meta.get("include_team_names", True))
    include_diffs = bool(meta.get("include_diffs", True))

    X = merge_team_features(
        matches_df,
        state.overall,
        state.lastN,
        state.elo,
        include_team_names=include_team_names,
        include_diffs=include_diffs,
    )

    # 3) Base model prediction
    p_model = predict_proba_framework(state.model, state.framework, X, meta)[0]

    # 4) Add matchup effect (only if a full draft is present)
    counters = _get_counters()
    if full_draft and counters is not None:
        p_match = compute_matchup_blue_prob(matches_df.iloc[0], counters)
        w = float(meta.get("matchup_weight", DEFAULT_MATCHUP_WEIGHT))
        
        # Blend model prediction with matchup-based signal
        p_final = p_model + w * (p_match - 0.5)
        p_final = float(np.clip(p_final, 0.01, 0.99))
        return p_final

    # Otherwise, team-only prediction
    return float(p_model)

# Incoming request format for /api/predict
class PredictRequest(BaseModel):
    model: ModelName
    blue_team: str
    red_team: str
    blue_picks: Optional[Dict[str, str]] = None
    red_picks: Optional[Dict[str, str]] = None

# Response format returned by /api/predict
class PredictionResponse(BaseModel):
    blue_win_prob: float
    red_win_prob: float

# Initialize API app + CORS for frontend
app = FastAPI(title="Esports Match Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Main prediction endpoint, automatically selects prediction mode based on picks
@app.post("/api/predict", response_model=PredictionResponse)
def api_predict(req: PredictRequest):
    try:
        p_blue = _compute_blue_win_probability(
            model_name=req.model,
            blue_team=req.blue_team,
            red_team=req.red_team,
            blue_picks=req.blue_picks,
            red_picks=req.red_picks,
        )
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    return PredictionResponse(
        blue_win_prob=p_blue,
        red_win_prob=1.0 - p_blue,
    )

# Compatibility alias for older frontend code, forwards to /api/predict with identical behavior
@app.post("/api/predict/matchup", response_model=PredictionResponse)
def api_predict_matchup(req: PredictRequest):
    return api_predict(req)
