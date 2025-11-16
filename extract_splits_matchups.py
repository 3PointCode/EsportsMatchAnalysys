import re
import csv
import time
import os
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
from urllib.parse import urljoin, unquote

import requests
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

BASE = "https://gol.gg"

MATCHLIST_URLS = [
    "https://gol.gg/tournament/tournament-matchlist/LEC%202025%20Summer%20Season/",
    "https://gol.gg/tournament/tournament-matchlist/LEC%202025%20Summer%20Playoffs/",
]

HEADERS = {"User-Agent": "LEC-Analytics/2.5 (+contact: you@example.com)"}
OUTPUT_CSV = "summer/summer_matches_with_picks.csv"


@dataclass
class GameWithPicks:
    split: str
    game_id: str
    game_date: Optional[str]
    team_blue: Optional[str]
    team_red: Optional[str]
    winning_team: Optional[str]
    top_blue: Optional[str]
    jgl_blue: Optional[str]
    mid_blue: Optional[str]
    adc_blue: Optional[str]
    sup_blue: Optional[str]
    top_red: Optional[str]
    jgl_red: Optional[str]
    mid_red: Optional[str]
    adc_red: Optional[str]
    sup_red: Optional[str]


def http_get(url, sleep=0.35, retries=3):
    for i in range(retries):
        r = requests.get(url, headers=HEADERS, timeout=30)
        if r.ok:
            if sleep:
                time.sleep(sleep)
            return r
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(1.2 * (i + 1))
            continue
        r.raise_for_status()
    raise RuntimeError(f"GET failed: {url}")


def expand_series_page_for_games(summary_url: str) -> List[str]:
    abs_url = summary_url if summary_url.startswith("http") else urljoin(BASE, summary_url)
    try:
        r = http_get(abs_url)
    except Exception:
        return []

    soup = BeautifulSoup(r.text, "lxml")
    out = []
    for a in soup.select("a[href*='/game/stats/']"):
        href = a.get("href") or ""
        m = re.search(r"/game/stats/(\d+)", href)
        if not m:
            continue
        gid = m.group(1)
        game_url = urljoin(BASE, f"/game/stats/{gid}/page-game/")
        out.append(game_url)

    out = list(dict.fromkeys(out))
    return out


def get_matchlist_with_series_expansion(matchlist_url: str):
    split_name = unquote(matchlist_url.rstrip("/").split("/tournament-matchlist/")[1])
    r = http_get(matchlist_url)
    soup = BeautifulSoup(r.text, "lxml")
    id_to_date: Dict[str, str] = {}
    per_game_urls: List[Tuple[str, str]] = []

    for a in soup.select("a[href*='/game/stats/']"):
        href = a.get("href") or ""
        m = re.search(r"/game/stats/(\d+)", href)
        if not m:
            continue
        gid = m.group(1)
        per_url = urljoin(BASE, f"/game/stats/{gid}/page-game/")
        per_game_urls.append((gid, per_url))

        tr = a.find_parent("tr")
        if tr:
            tds = tr.find_all("td")
            if tds:
                last_txt = tds[-1].get_text(" ", strip=True)
                if re.match(r"\d{4}-\d{2}-\d{2}", last_txt):
                    id_to_date[gid] = last_txt

    series_links = []
    series_links += [a.get("href") for a in soup.select("a[href*='/page-summary/']") if a.get("href")]
    series_links += [a.get("href") for a in soup.select("a[href*='/match/stats/']") if a.get("href")]

    for href in series_links:
        abs_summary = urljoin(BASE, href)
        sub_games = expand_series_page_for_games(abs_summary)
        for per_url in sub_games:
            m = re.search(r"/game/stats/(\d+)/", per_url)
            if not m:
                continue
            gid = m.group(1)
            per_game_urls.append((gid, per_url))

    seen = set()
    game_entries = []
    for gid, per_url in per_game_urls:
        if gid in seen:
            continue
        seen.add(gid)
        game_date = id_to_date.get(gid, None)
        game_entries.append((split_name, gid, game_date, per_url))

    return game_entries


def parse_game_date_from_text(text: str):
    if not text:
        return None
    m = re.search(r"\b\d{4}-\d{2}-\d{2}\b", text)
    return m.group(0) if m else None


JS_EXTRACT = """
(() => {
  function txt(el){ return (el && el.textContent || '').trim(); }

  const blueHeader = document.querySelector('.blue-line-header');
  const redHeader  = document.querySelector('.red-line-header');
  const teamBlue = txt(blueHeader?.querySelector('a[href*="/team"]')) || txt(blueHeader)?.split(' - ')[0]?.trim() || null;
  const teamRed  = txt(redHeader ?.querySelector('a[href*="/team"]')) || txt(redHeader )?.split(' - ')[0]?.trim() || null;

  // ustalenie zwycięzcy na podstawie tekstu nagłówków (np. "NAVI - WIN")
  const blueText = txt(blueHeader).toLowerCase();
  const redText  = txt(redHeader).toLowerCase();
  let winningTeam = null;
  if (blueText.includes('win')) {
    winningTeam = teamBlue;
  } else if (redText.includes('win')) {
    winningTeam = teamRed;
  }

  const teamCols = Array.from(document.querySelectorAll('div.col-12.col-sm-6'));
  const colBlue = teamCols[0] || null;
  const colRed  = teamCols[1] || null;

  function getPicksFromCol(col) {
    if (!col) return [];

    const rows = Array.from(col.querySelectorAll('div.row'));
    for (const row of rows) {
      const labelDiv = row.querySelector('div.col-2');
      if (!labelDiv) continue;
      const labelText = (labelDiv.textContent || '').trim().toLowerCase();
      if (labelText !== 'picks') continue;

      const iconsContainer = row.querySelector('div.col-10');
      if (!iconsContainer) continue;

      const champs = [];
      iconsContainer.querySelectorAll('img.champion_icon_medium').forEach(img => {
        const alt = (img.getAttribute('alt') || '').trim();
        if (!alt) return;
        champs.push(alt);
      });

      if (champs.length >= 5) {
        return champs.slice(0, 5);
      }
    }

    return [];
  }

  const picksBlue = getPicksFromCol(colBlue);
  const picksRed  = getPicksFromCol(colRed);

  const pageText = document.body ? document.body.innerText : '';

  return {
    teamBlue,
    teamRed,
    winningTeam,
    picksBlue,
    picksRed,
    pageText
  };
})();
"""


def main():
    os.makedirs(os.path.dirname(OUTPUT_CSV) or ".", exist_ok=True)

    all_games = []
    seen = set()
    for url in MATCHLIST_URLS:
        entries = get_matchlist_with_series_expansion(url)
        for split_name, gid, game_date, per_url in entries:
            if gid in seen:
                continue
            seen.add(gid)
            all_games.append((split_name, gid, game_date, per_url))
        print(f"[{unquote(url.split('/tournament-matchlist/')[1].rstrip('/'))}] found {len(entries)} games.")

    print(f"Total unique games: {len(all_games)}")

    out_rows: List[GameWithPicks] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context()
        page = ctx.new_page()

        for split_name, gid, game_date, url in all_games:
            print(f" -> {split_name} game {gid}: {url}")
            try:
                page.goto(url, wait_until="networkidle", timeout=60000)
            except Exception as e:
                print(f"[WARN] goto failed {url}: {e}")
                continue

            try:
                data = page.evaluate(JS_EXTRACT)
            except Exception as e:
                print(f"[WARN] JS extract failed {gid}: {e}")
                continue

            if not game_date:
                game_date = parse_game_date_from_text(data.get("pageText") or "")

            team_blue = data.get("teamBlue")
            team_red = data.get("teamRed")
            winning_team = data.get("winningTeam")
            picks_blue = data.get("picksBlue") or []
            picks_red = data.get("picksRed") or []

            if len(picks_blue) != 5 or len(picks_red) != 5:
                print(f"[WARN] game {gid}: picks len blue={len(picks_blue)} red={len(picks_red)}")
                continue

            row = GameWithPicks(
                split=split_name,
                game_id=gid,
                game_date=game_date,
                team_blue=team_blue,
                team_red=team_red,
                winning_team=winning_team,
                top_blue=picks_blue[0],
                jgl_blue=picks_blue[1],
                mid_blue=picks_blue[2],
                adc_blue=picks_blue[3],
                sup_blue=picks_blue[4],
                top_red=picks_red[0],
                jgl_red=picks_red[1],
                mid_red=picks_red[2],
                adc_red=picks_red[3],
                sup_red=picks_red[4],
            )
            out_rows.append(row)

        browser.close()

    if not out_rows:
        print("No games with valid picks scraped.")
        return

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(asdict(out_rows[0]).keys())
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in out_rows:
            w.writerow(asdict(r))

    print(f"Saved {len(out_rows)} rows -> {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
