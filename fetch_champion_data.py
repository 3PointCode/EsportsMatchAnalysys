"""
The script is responsible for scraping the champions matchups for each role from op.gg site for Master+ ranked games.
This ensures that the data may be more reliable when it comes to the pro-play. The op.gg data set is made out of nearly 470k records.

What is this script doing:
1) For each role, open a tier page.
2) Read all champions rows to get the below:
    - slug: the URL id
    - pretty name: text that is inside the <strong> tag
3) For each champion, go to its "Counters" page for a specific role and extract:
    - opponent champion name
    - win rate percentage versus that opponent example: Diana,Volibear,53.42
4) Filter out the unnecessary data
5) Add "Others" value that is equal to 50.0 to keep JSON/CSV consistent if a champion is missing certain matchups.
6) Create a JSON and CSV file for each role. 

Example usage:
    python fetch_champion_data.py
    python fetch_champion_data.py --roles <role1> <role2> --out-dir <directory> --delay <requests_time>
    (pulls champions for top and jungle roles and places the ouput in the provided directory)
"""

import csv
import json
import re
import time
import random
import argparse
import requests
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any
from bs4 import BeautifulSoup

# Global configuration (roles scraped by default, targeted data tier, base domain, reducing the chance of being blocked)
ROLES = ["top", "jungle", "mid", "adc", "support"]
TIER_QUERY = "master_plus"
BASE = "https://op.gg"
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/124.0.0.0 Safari/537.36"),
    "Accept-Language": "en,en-US;q=0.9",
}

# Fallback champions for each role in case if the page fails to load or parse
FALLBACK_BY_ROLE = {
    "top": ["aatrox","jax","camille","kled","renekton","akali","ornn","darius","riven","shen",
            "illaoi","malphite","garen","mordekaiser","gwen","rengar","irelia","gnar","singed","nasus"],
    "jungle": ["leesin","kayn","graves","viego","diana","sejuani","vi","jarvaniv","kindred","khazix",
               "wukong","elise","evelynn","masteryi","hecarim","rengar","udyr","zac","amumu","rammus"],
    "mid": ["ahri","sylas","azir","orianna","yasuo","yone","akali","leblanc","viktor","tf",
            "annie","veigar","zed","kassadin","anivia","seraphine","syndra","naafiri","neeko","katarina"],
    "adc": ["caitlyn","ezreal","jinx","xayah","aphelios","ashe","lucian","kalista","draven","kaisa",
            "tristana","zeri","sivir","missfortune","samira","varus","vayne","kogmaw","twitch","jhin"],
    "support": ["thresh","nautilus","rakan","lulu","nami","leona","morgana","blitzcrank","braum","sona",
                "senna","renata","pyke","asupport?no","soraka","taric","alistar","yuumi","amumu","rell"],
}

# Fallback cleanup in case of typos
for k, arr in list(FALLBACK_BY_ROLE.items()):
    FALLBACK_BY_ROLE[k] = [s for s in arr if s and s.replace("'", "").replace("-", "").isalpha()]

# Simple GET with retries + backoff, if op.gg rate-limits, then retry automatically
def get(url: str) -> str:
    for attempt in range(6):
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            if r.status_code in (429, 503):
                time.sleep(2 ** attempt + random.random())
                continue
            r.raise_for_status()
            return r.text
        except Exception:
            time.sleep(2 ** attempt + random.random())  # network/site issue - backoff and try again
    raise RuntimeError(f"Failed to fetch {url}")

# This function walks the JSON and pulls objects that are useful
# Returns { opponent_name: winrate percentage } with win-rate normalized to (0.0, 100.0) 
def deep_find_counters(obj: Any, subject_name: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    def visit(x: Any):
        if isinstance(x, dict):
            name, wr = None, None
            # trying common name keys
            for k in ("name","championName","opponentName","opponent"):
                if k in x and isinstance(x[k], str):
                    name = x[k].strip(); break
            # trying common win-rate keys
            for k in ("winRate","winrate","win_rate","percentage","value"):
                if k in x and isinstance(x[k], (int,float)):
                    val = float(x[k]); wr = val*(100.0 if val<=1 else 1.0); break
            # store if the data looks like a real opponent row - different that the current value
            if name and wr is not None and name.lower()!=subject_name.lower():
                if name[0].isalpha() and len(name)<30:
                    out.setdefault(name, round(wr,2))
            for v in x.values(): visit(v)
        elif isinstance(x, list):
            for v in x: visit(v)
    visit(obj)
    return out

"""
Determine the champion's pretty name from the counters page.
We try several places that op.gg uses:
    - <h1> heading example: "Lee Sin Counters - Jungle"
    - og:title meta tag
    - <title>
If all else fails, use whatever name we passed in.
"""
def parse_subject_name(soup: BeautifulSoup, fallback: str) -> str:
    h1 = soup.find("h1")
    if h1:
        n = re.sub(r"\s*Counters.*$", "", h1.get_text(" ", strip=True)).strip()
        if n: return n
    meta = soup.find("meta", attrs={"property": "og:title"})
    if meta and meta.get("content"):
        n = re.sub(r"\s*Counters.*$", "", meta["content"]).strip()
        if n: return n
    title = soup.find("title")
    if title:
        t = re.sub(r"\s*-\s*OP\.GG.*$", "", title.get_text(" ", strip=True))
        t = re.sub(r"\s*Counters.*$", "", t).strip()
        if t: return t
    return fallback

# Parse the role tier page and extract all champions for that role.
def list_role_champions(role: str) -> List[Tuple[str, str]]:
    """
    Parse: https://op.gg/lol/champions?position=<role>&tier=master_plus
    Extract slug from /lol/champions/<slug>/..., and pretty from <strong> inside the anchor (or img alt/text).
    """
    url = f"{BASE}/lol/champions?position={role}&tier={TIER_QUERY}"
    try:
        html = get(url)
    except Exception:
        print(f"! Could not fetch {role} index; using fallback list.")
        return [(s.capitalize(), s) for s in FALLBACK_BY_ROLE.get(role, [])]

    soup = BeautifulSoup(html, "html.parser")
    pairs: Set[Tuple[str,str]] = set()

    # finding all champion row for specific role
    for a in soup.select('a[href^="/lol/champions/"]'):
        href = a.get("href","")
        if f"/{role}" not in href:
            continue
        m = re.search(r"/lol/champions/([^/?#]+)/", href)
        if not m:
            continue
        slug = m.group(1).strip()
        
        # pretty name extraction
        strong = a.find("strong")
        if strong and strong.get_text(strip=True):
            pretty = strong.get_text(strip=True)
        else:
            img = a.find("img", alt=True)
            pretty = img["alt"].strip() if img and img.get("alt") else a.get_text(" ", strip=True) or slug
        pairs.add((pretty, slug))

    # clear duplicates and sort by slug for stable order
    out, seen = [], set()
    for pretty, slug in sorted(pairs, key=lambda x: x[1]):
        if slug not in seen:
            out.append((pretty, slug)); seen.add(slug)

    if not out:
        print(f"! {role} index parsed but empty; using fallback list.")
        return [(s.capitalize(), s) for s in FALLBACK_BY_ROLE.get(role, [])]
    return out

# returns (subject_name, {opponent_name: win_rate_pct}) after trying to access embedded JSON blob and visible "Win rate" list
def parse_counters_from_page(html: str, subject_guess: str) -> Tuple[str, Dict[str,float]]:
    soup = BeautifulSoup(html, "html.parser")
    subject = parse_subject_name(soup, subject_guess)

    # trying to access Next.js JSON blob
    script = soup.find("script", id="__NEXT_DATA__")
    if script and script.string:
        try:
            blob = json.loads(script.string)
            found = deep_find_counters(blob, subject)
            if found: return subject, found
        except Exception:
            pass

    # trying to access the visible "Win rate" list
    header = soup.find(lambda t: t.name in ("h2","div","span") and t.get_text(strip=True) == "Win rate")
    if header:
        container = header.find_parent()
        for _ in range(5):
            if container and container.find_all("img"): break
            container = container.find_parent()
        if container:
            results = {}
            for img in container.select('img[alt]'):
                opp = img["alt"].strip()
                if not opp or opp.lower()==subject.lower(): continue
                row = img.find_parent()
                pct = None
                for _ in range(3):
                    if not row: break
                    m = re.search(r'(\d{1,3}(?:\.\d{1,2})?)\s*%', row.get_text(" ", strip=True))
                    if m: pct = float(m.group(1)); break
                    row = row.find_next_sibling()
                if opp and pct is not None: results[opp] = pct
            if results: return subject, results

    # loose text fallback if both previous failed
    text = soup.get_text("\n", strip=True)
    results = {}
    for name, pct in re.findall(r"([A-Za-z' .&-]{3,25})\s+(\d{1,3}(?:\.\d{1,2})?)\s*%\s+\d{1,6}\b", text):
        name = name.strip()
        if name.lower() not in (subject.lower(),"win rate","games") and name[0].isupper():
            results[name] = float(pct)
    return subject, results

def scrape_counters(role: str, slug: str, pretty: str) -> Tuple[str, Dict[str,float]]:
    url = f"{BASE}/lol/champions/{slug}/counters/{role}?tier={TIER_QUERY}"
    html = get(url)
    return parse_counters_from_page(html, pretty)

# Filtering out the rows that are not champion names. Dropping everything that contains any of the below key words.
def is_valid_opponent(name: str) -> bool:
    bad = ["kill","participation","lane","pick","ban","page","rate","win",
           "match","avg","ratio","vs","per"]
    low = name.lower()
    if not re.search(r"[a-z]", low): return False
    if len(name) > 25: return False
    return not any(tok in low for tok in bad)

# End-to-end for one role: get champion list, iterate champions, parse, cleanup, add "Others": 50.0 for consistency, create JSON/CSV
def run_role(role: str, out_dir: Path, resume: bool, delay: float):
    champs = list_role_champions(role)
    print(f"[{role}] discovered {len(champs)} champions")

    out_csv = out_dir / f"{role}_{TIER_QUERY}_counters.csv"
    out_json = out_dir / f"{role}_{TIER_QUERY}_counters.json"

    data = {role: {}}
    if resume and out_json.exists():
        with out_json.open("r", encoding="utf-8") as f:
            old = json.load(f)
            if role in old: data = old

    for pretty, slug in champs:
        if resume and pretty in data[role]:
            print(f"[{role}] skip {pretty} (resume)")
            continue
        print(f"[{role}] scraping {slug} …")
        try:
            subject, counters = scrape_counters(role, slug, pretty)

            # Data cleanup, keep only real champion names
            counters = {opp: wr for opp, wr in counters.items() if is_valid_opponent(opp)}

            # Ensure that the "Others" for unknown matchups has been added
            if not counters:
                counters = {"Others": 50.0}
            counters["Others"] = 50.0

            data[role][subject] = counters

            # Write in after each matchup so the progress won't be lost due to crashing program
            with out_json.open("w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"[{role}] ! error on {slug}: {e} -> fallback Others=50.0")
            data[role][pretty] = {"Others": 50.0}
            with out_json.open("w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

        time.sleep(delay + random.random()*0.5)

    # Creaete CSV file
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["role","tier","champion","opponent","winrate"])
        for champ, opps in sorted(data[role].items()):
            for opp, wr in sorted(opps.items()):
                w.writerow([role, TIER_QUERY, champ, opp, wr])

    print(f"[{role}] done → CSV: {out_csv.name} | JSON: {out_json.name}")

# Description of additional arguments when running the script
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape OP.GG Master+ counters for multiple roles.")
    parser.add_argument(
        "--roles",
        nargs="*",
        default=ROLES,
        help="Roles to scrape (space-separated): top jungle mid adc support."
    )
    parser.add_argument(
        "--out-dir",
        default=".",
        help="Directory to save per-role CSV/JSON files."
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing JSON (skip already scraped champions)."
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Base delay between requests (seconds). Randomized slightly per request."
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for role in args.roles:
        role = role.lower().strip()
        if role not in ROLES:
            print(f"Skipping unknown role: {role}")
            continue
        run_role(role, out_dir, resume=args.resume, delay=args.delay)

    print("\nAll requested roles completed.")
