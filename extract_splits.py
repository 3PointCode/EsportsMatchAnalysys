import os
import re
import argparse
import pandas as pd

EXCLUDE_KEYWORDS = [
    "playoff", "play-offs", "final", "finals",
    "group", "groups", "group stage", "knockout", "bracket"
]

def detect_split_column(cols):
    for c in cols:
        cl = c.lower()
        if "split" in cl or "season" in cl or "tournament" in cl or "stage" in cl:
            return c
    raise ValueError("Nie znaleziono kolumny split/season/tournament/stage w CSV.")

def is_regular_spring_or_summer(text: str) -> bool:
    t = (text or "").lower()
    has_spring_or_summer = ("spring" in t) or ("summer" in t)
    has_excluded = any(k in t for k in EXCLUDE_KEYWORDS)
    # exclude “groups”, “playoffs”, “finals”, etc.
    return has_spring_or_summer and not has_excluded

def main():
    ap = argparse.ArgumentParser(
        description="Wyodrębnij Spring/Summer SEASON (bez Playoffs/Groups) do osobnych plików z kolumnami team_blue,team_red."
    )
    ap.add_argument("--data", default="data/lec_2023-2025_games.csv", help="Ścieżka do wejściowego CSV.")
    ap.add_argument("--out-spring", default="spring/spring_matches.csv")
    ap.add_argument("--out-summer", default="summer/summer_matches.csv")
    ap.add_argument("--year", type=int, default="2025", help="(Opcjonalnie) tylko mecze z tego roku (po game_date).")
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    print("Columns:", df.columns.tolist())

    # detect the collumn that describes the split/tournament
    split_col = detect_split_column(df.columns)
    df[split_col] = df[split_col].astype(str).str.lower()

    # (optional) year filter
    if args.year is not None:
        if "game_date" not in df.columns:
            raise ValueError("Brak kolumny 'game_date' — nie mogę zastosować filtra --year.")
        dates = pd.to_datetime(df["game_date"], errors="coerce")
        df = df[dates.dt.year == args.year]
        print(f"Zastosowano filtr roku: {args.year}. Pozostało rekordów: {len(df)}")

    # team collumn validation
    for c in ["team_blue", "team_red"]:
        if c not in df.columns:
            raise ValueError(f"Brakuje wymaganej kolumny: {c}")

    mask_regular = df[split_col].apply(is_regular_spring_or_summer)

    regular = df[mask_regular].copy()

    # spring/summer split
    spring_df = regular[regular[split_col].str.contains("spring", na=False)][["team_blue", "team_red"]].copy()
    summer_df = regular[regular[split_col].str.contains("summer", na=False)][["team_blue", "team_red"]].copy()

    os.makedirs(os.path.dirname(args.out_spring) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.out_summer) or ".", exist_ok=True)
    spring_df.to_csv(args.out_spring, index=False)
    summer_df.to_csv(args.out_summer, index=False)

    print(f"Zapisano: {args.out_spring} ({len(spring_df)} meczów)")
    print(f"Zapisano: {args.out_summer} ({len(summer_df)} meczów)")

if __name__ == "__main__":
    main()
