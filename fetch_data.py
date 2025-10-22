import re
import csv
import time
import requests
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

BASE = "https://gol.gg"
MATCHLIST_URL = "https://gol.gg/tournament/tournament-matchlist/LEC%20Winter%202023/"
HEADERS = {"User-Agent": "LEC-Analytics/2.3 (+contact: you@example.com)"}

def http_get(url, sleep=0.4, retries=3):
    for i in range(retries):
        r = requests.get(url, headers=HEADERS, timeout=30)
        if r.ok:
            if sleep: time.sleep(sleep)
            return r
        if r.status_code in (429,500,502,503,504):
            time.sleep(1.2*(i+1)); continue
        r.raise_for_status()
    raise RuntimeError(f"GET failed: {url}")

def parse_duration_seconds(page_text):
    m = re.search(r"\b(?:(\d+):)?(\d{1,2}):(\d{2})\b", page_text)
    if not m: return None
    h = m.group(1); mm = int(m.group(2)); ss = int(m.group(3))
    total = mm*60 + ss
    if h: total += int(h)*3600
    return total

def get_matchlist():
    resp = http_get(MATCHLIST_URL)
    soup = BeautifulSoup(resp.text, "lxml")
    rows = []
    for a in soup.select("a[href*='/game/stats/']"):
        href = a.get("href") or ""
        m = re.search(r"/game/stats/(\d+)", href)
        if not m: continue
        gid = m.group(1)
        tr = a.find_parent("tr")
        game_date = None
        if tr:
            tds = tr.find_all("td")
            if tds:
                last_txt = tds[-1].get_text(" ", strip=True)
                if re.match(r"\d{4}-\d{2}-\d{2}", last_txt):
                    game_date = last_txt
        url = urljoin(BASE, href if href.startswith("/") else f"/{href}")
        rows.append((gid, game_date, url))
    seen = set()
    return [(g,d,u) for (g,d,u) in rows if not (g in seen or seen.add(g))]

JS_EXTRACT = """
(() => {
  function txt(el){ return (el && el.textContent || '').trim(); }

  function parseAllNumbersWithK(s){
    if(!s) return [];
    s = s.replace(/\\u00A0|\\u2007|\\u202F/g, ' ');
    const re = /(-?\\d+(?:[.,]\\d+)?)(\\s*[kK])?/g;
    let m, out = [];
    while ((m = re.exec(s)) !== null) {
      const num = parseFloat(m[1].replace(',', '.'));
      const hasK = !!m[2];
      const scaled = Math.round(hasK ? num * 1000 : num);
      out.push({ scaled, hasK, raw: m[0] });
    }
    return out;
  }

  function readScoreBoxGold(selector){
    const candidates = Array.from(document.querySelectorAll(selector))
      .filter(sp => sp.querySelector('img[alt*="Team Gold"]'));
    if (!candidates.length) return null;
    const sp = candidates[0];

    const nums = parseAllNumbersWithK(sp.textContent || '');
    if (!nums.length) return null;

    const withK = nums.filter(x => x.hasK);
    if (withK.length) {
      withK.sort((a,b) => b.scaled - a.scaled);
      return withK[0].scaled;
    }

    nums.sort((a,b) => b.scaled - a.scaled);
    return nums[0].scaled;
  }

  const blueHeader = document.querySelector('.blue-line-header');
  const redHeader  = document.querySelector('.red-line-header');
  const teamBlue = txt(blueHeader?.querySelector('a[href*="/team"]')) || txt(blueHeader)?.split(' - ')[0]?.trim() || null;
  const teamRed  = txt(redHeader ?.querySelector('a[href*="/team"]')) || txt(redHeader )?.split(' - ')[0]?.trim() || null;
  const blueWin = /\\bWIN\\b/i.test(txt(blueHeader));
  const redWin  = /\\bWIN\\b/i.test(txt(redHeader));

  const blueGold = readScoreBoxGold('span.score-box.blue_line');
  const redGold  = readScoreBoxGold('span.score-box.red_line');

  const canvas = document.getElementById('GoldLine');
  let labels = null;
  let datasets = [];
  const ChartObj = window.Chart;

  function pushFromChart(chart){
    if(!chart || !chart.data) return false;
    labels = chart.data.labels || null;
    if (Array.isArray(chart.data.datasets)) {
      datasets = chart.data.datasets.map(d => Array.isArray(d.data) ? d.data.slice() : []);
    }
    return true;
  }

  let found = false;
  if (ChartObj) {
    if (ChartObj.getChart) {
      try { if (pushFromChart(ChartObj.getChart(canvas))) found = true; } catch(e){}
    }
    if (!found && ChartObj.instances) {
      try {
        const inst = ChartObj.instances;
        const arr = Array.isArray(inst) ? inst : Object.values(inst || {});
        const c = arr.find(x => x && x.canvas === canvas);
        if (pushFromChart(c)) found = true;
      } catch(e){}
    }
  }

  return {
    teamBlue, teamRed, blueWin, redWin,
    blueGold, redGold,
    labels, datasets,
    pageText: document.body?.innerText || ''
  };
})();
"""

def labels_to_seconds(labels):
    if not labels: return None
    out = []
    for lab in labels:
        if lab is None: continue
        s = str(lab).strip()
        m = re.match(r"^(\d{1,2}):(\d{2})$", s)
        if m: out.append(int(m.group(1))*60 + int(m.group(2)))
        else:
            try: out.append(int(round(float(s))) * 60)
            except: return None
    return out if out else None

def pick_series_by_height(datasets):
    best = None; best_score = -1.0
    for series in datasets or []:
        if not series: continue
        vals = []
        for v in series:
            try: vals.append(float(v))
            except: vals.append(0.0)
        if not vals: continue
        score = max(abs(x) for x in vals) + 0.001*len(vals)
        if score > best_score:
            best_score = score; best = vals
    return best

def main():
    games = get_matchlist()
    print(f"Found {len(games)} games on matchlist.")

    out_rows = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context()
        page = ctx.new_page()

        for gid, game_date, url in games:
            try:
                page.goto(url, wait_until="networkidle", timeout=60000)
            except Exception as e:
                print(f"[WARN] goto failed {url}: {e}")
                continue

            data = page.evaluate(JS_EXTRACT)

            team_blue = data.get("teamBlue")
            team_red  = data.get("teamRed")
            winning_team = team_blue if data.get("blueWin") and not data.get("redWin") else (team_red if data.get("redWin") and not data.get("blueWin") else None)

            blue_gold = data.get("blueGold")
            red_gold  = data.get("redGold")
            gold_diff = int(blue_gold - red_gold) if blue_gold is not None and red_gold is not None else None

            game_time = parse_duration_seconds(data.get("pageText") or "")

            labels = data.get("labels")
            datasets = data.get("datasets") or []
            labels_seconds = labels_to_seconds(labels)

            gold_diff_at14 = None
            if datasets:
                series = pick_series_by_height(datasets)
                if series:
                    if gold_diff is not None and series[-1] != 0:
                        if (series[-1] > 0 and gold_diff < 0) or (series[-1] < 0 and gold_diff > 0):
                            series = [-x for x in series]

                    if labels_seconds:
                        target = 14*60
                        idx = min(range(len(series)), key=lambda i: abs((labels_seconds[i] if i < len(labels_seconds) else i*60) - target))
                        gold_diff_at14 = int(round(series[idx]))
                    else:
                        if len(series) > 14:
                            gold_diff_at14 = int(round(series[14]))

            if not winning_team and gold_diff is not None:
                if gold_diff > 0 and team_blue:
                    winning_team = team_blue
                elif gold_diff < 0 and team_red:
                    winning_team = team_red

            out_rows.append({
                "game_id": gid,
                "game_date": game_date,
                "team_blue": team_blue,
                "team_red": team_red,
                "winning_team": winning_team,
                "game_time": int(game_time) if isinstance(game_time, (int, float)) else (int(game_time) if game_time else None),
                "gold_diff": gold_diff,
                "gold_diff_at14": gold_diff_at14
            })

        browser.close()

    out_path = "lec_winter_2023_games.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "game_id","game_date","team_blue","team_red","winning_team","game_time","gold_diff","gold_diff_at14"
        ])
        w.writeheader()
        for row in out_rows:
            w.writerow(row)

    print(f"Saved {len(out_rows)} rows -> {out_path}")

if __name__ == "__main__":
    main()
