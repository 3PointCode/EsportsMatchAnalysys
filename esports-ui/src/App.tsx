import React, { useMemo, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Search, Plus, X, Loader2 } from "lucide-react";

const ALL_TEAMS = [
  "G2 Esports","Fnatic","Team Vitality","Movistar KOI","SK Gaming",
  "Team Heretics","Karmine Corp","NAVI",
];

const ALL_CHAMPIONS = [
  "Aatrox", "Ahri", "Akali", "Akshan", "Alistar", "Ambessa", "Amumu", "Anivia", "Annie", "Aphelios", "Ashe", "Aurelion Sol", "Aurora", "Azir", "Bard", "Bel'Veth", "Blitzcrank", "Brand", "Braum", "Briar", "Caitlyn", "Camille", "Cassiopeia", "Cho'Gath", "Corki", "Darius", "Diana", "Dr. Mundo", "Draven", "Ekko", "Elise", "Evelynn", "Ezreal", "Fiddlesticks", "Fiora", "Fizz", "Galio", "Gangplank", "Garen", "Gnar", "Gragas", "Graves", "Gwen", "Hecarim", "Heimerdinger", "Hwei", "Illaoi", "Irelia", "Ivern", "Janna", "Jarvan IV", "Jax", "Jayce", "Jhin", "Jinx", "K'Sante", "Kai'Sa", "Kalista", "Karma", "Karthus", "Kassadin", "Katarina", "Kayle", "Kayn", "Kennen", "Kha'Zix", "Kindred", "Kled", "Kog'Maw", "LeBlanc", "Lee Sin", "Leona", "Lillia", "Lissandra", "Lucian", "Lulu", "Lux", "Malphite", "Malzahar", "Maokai", "Master Yi", "Mel", "Milio", "Miss Fortune", "Mordekaiser", "Morgana", "Naafiri", "Nami", "Nasus", "Nautilus", "Neeko", "Nidalee", "Nilah", "Nocturne", "Nunu & Willump", "Olaf", "Orianna", "Ornn", "Pantheon", "Poppy", "Pyke", "Qiyana", "Quinn", "Rakan", "Rammus", "Rek'Sai", "Rell", "Renata Glasc", "Renekton", "Rengar", "Riven", "Rumble", "Ryze", "Samira", "Sejuani", "Senna", "Seraphine", "Sett", "Shaco", "Shen", "Shyvana", "Singed", "Sion", "Sivir", "Skarner", "Smolder", "Sona", "Soraka", "Swain", "Sylas", "Syndra", "Tahm Kench", "Taliyah", "Talon", "Taric", "Teemo", "Thresh", "Tristana", "Trundle", "Tryndamere", "Twisted Fate", "Twitch", "Udyr", "Urgot", "Varus", "Vayne", "Veigar", "Vel'Koz", "Vex", "Vi", "Viego", "Viktor", "Vladimir", "Volibear", "Warwick", "Wukong", "Xayah", "Xerath", "Xin Zhao", "Yasuo", "Yone", "Yorick", "Yunara", "Yuumi", "Zac", "Zed", "Zeri", "Ziggs", "Zilean", "Zoe", "Zyra"
];

const Pill = ({ children }: { children: React.ReactNode }) => (
  <span className="inline-flex items-center gap-2 rounded-2xl px-3 py-1 text-sm
                   bg-zinc-800 border border-zinc-700 text-zinc-100">
    {children}
  </span>
);

function useFuzzyFilter(
  list: string[],
  q: string,
  opts?: { limitOnSearch?: number; showAllWhenEmpty?: boolean }
) {
  const { limitOnSearch = 3, showAllWhenEmpty = true } = opts ?? {};
  return useMemo(() => {
    const s = q.trim().toLowerCase();

    // If nothing typed → return ALL alphabetically
    if (!s) {
      const all = [...list].sort((a, b) => a.localeCompare(b));
      return showAllWhenEmpty ? all : [];
    }

    // Typed → fuzzy score then take top N
    const scored = list.map((name) => {
      const a = name.toLowerCase();

      // subsequence hits
      let i = 0, j = 0, hits = 0;
      while (i < a.length && j < s.length) { if (a[i] === s[j]) { hits++; j++; } i++; }

      // bonuses
      const prefixBonus = a.startsWith(s) ? 2 : 0;
      const exactBonus  = a === s ? 5 : 0;
      const score = hits / (s.length + 0.5 * (a.length - hits)) + prefixBonus + exactBonus;

      return { name, score };
    });

    return scored
      .sort((x, y) => y.score - x.score)
      .slice(0, limitOnSearch)
      .map((x) => x.name);
  }, [list, q, limitOnSearch, showAllWhenEmpty]);
}

function TeamSearch({
  label,
  value,
  onChange,
  taken,
  side, // "blue" | "red"
}: {
  label: string;
  value?: string;
  onChange: (t: string) => void;
  taken: Set<string>;
  side: "blue" | "red";
}) {
  const [open, setOpen] = useState(false);
  const [q, setQ] = useState("");

  const filtered = useFuzzyFilter(ALL_TEAMS, q, { limitOnSearch: 5, showAllWhenEmpty: true });

  const panelTint =
    side === "blue"
      ? "border-sky-600/40 bg-sky-950/20"
      : "border-rose-600/40 bg-rose-950/20";

  return (
    <div className="relative">
      <div className={`rounded-2xl p-4 min-h-[160px] shadow-sm border ${panelTint}`}>
        {/* Top bar */}
        <div className="flex items-center justify-between mb-3">
          <h3 className="font-semibold text-zinc-100">{label}</h3>
          <button
            className="inline-flex items-center gap-2 rounded-2xl border border-zinc-700 px-3 py-1 text-sm hover:bg-zinc-700/70"
            onClick={() => setOpen(true)}
          >
            <Plus className="h-4 w-4" /> Add team
          </button>
        </div>

        {/* Centered team name */}
        <div className="grid place-items-center h-[110px]">
          {value ? (
            <div className="text-3xl md:text-4xl font-extrabold tracking-tight text-center">
              {value}
            </div>
          ) : (
            <div className="text-zinc-400 text-sm">No team selected.</div>
          )}
        </div>
      </div>

      <AnimatePresence>
        {open && (
          <motion.div
            className="absolute z-20 mt-2 w-[520px] max-w-[85vw] rounded-2xl border border-zinc-700 bg-zinc-800 shadow-xl"
            initial={{ opacity: 0, y: -8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
          >
            <div className="flex items-center gap-2 p-3 border-b border-zinc-700">
              <Search className="h-4 w-4" />
              <input
                autoFocus
                placeholder="Search teams…"
                className="w-full bg-transparent outline-none"
                value={q}
                onChange={(e) => setQ(e.target.value)}
              />
              <button className="p-1 rounded-lg hover:bg-zinc-700/70" onClick={() => setOpen(false)}>
                <X className="h-4 w-4" />
              </button>
            </div>

            <div className="max-h-72 overflow-auto p-2">
              {filtered.map((t) => {
                const isTaken = taken.has(t) && t !== value;
                return (
                  <button
                    key={t}
                    disabled={isTaken}
                    title={isTaken ? "Already picked" : ""}
                    className={
                      "w-full text-left px-3 py-2 rounded-xl focus:outline-none focus:ring-2 focus:ring-zinc-500 " +
                      (isTaken ? "opacity-40 cursor-not-allowed" : "hover:bg-zinc-700/70")
                    }
                    onClick={() => {
                      if (!isTaken) {
                        onChange(t);
                        setOpen(false);
                        setQ("");
                      }
                    }}
                  >
                    {t}
                  </button>
                );
              })}
              {filtered.length === 0 && <div className="p-4 text-sm text-zinc-500">No results.</div>}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function ChampionSlot({
  owner,
  index,
  champion,
  onPick,
  taken,
  side, // "blue" | "red"
}: {
  owner: string;
  index: number;
  champion?: string;
  onPick: (name?: string) => void;
  taken: Set<string>;
  side: "blue" | "red";
}) {
  const [open, setOpen] = useState(false);
  const [q, setQ] = useState("");

  // top-3 when typing, A→Z when empty
  const filtered = useFuzzyFilter(ALL_CHAMPIONS, q, {
    limitOnSearch: 3,
    showAllWhenEmpty: true,
  });

  const panelTint =
    side === "blue"
      ? "border-sky-600/40 bg-sky-950/20"
      : "border-rose-600/40 bg-rose-950/20";

  return (
    <div className="relative">
      {/* container */}
      <div className={`flex items-center gap-3 rounded-xl px-3 py-3 min-h-[72px] border ${panelTint}`}>
        {/* left: label + centered name */}
        <div className="flex-1">
          <div className="text-[11px] uppercase tracking-wide text-zinc-500">
            {owner} · Pick {index + 1}
          </div>
          {/* centered champion name */}
          <div className="h-8 md:h-10 grid place-items-center">
            <div
              className={
                "text-lg md:text-xl font-semibold tracking-tight " +
                (champion ? "" : "text-zinc-400")
              }
            >
              {champion ?? "Empty"}
            </div>
          </div>
        </div>

        {/* right: actions */}
        <div className="flex gap-1">
          {champion && (
            <button
              className="text-xs px-2 py-1 rounded-lg border border-zinc-700 hover:bg-zinc-700/70"
              onClick={() => onPick(undefined)}
            >
              Clear
            </button>
          )}
          <button
            className="inline-flex items-center gap-1 text-xs px-2 py-1 rounded-lg border border-zinc-700 hover:bg-zinc-700/70"
            onClick={() => setOpen(true)}
          >
            <Plus className="h-4 w-5" /> Pick
          </button>
        </div>
      </div>

      {/* picker */}
      <AnimatePresence>
        {open && (
          <motion.div
            className="absolute z-30 mt-2 w-[420px] max-w-[90vw] rounded-2xl border border-zinc-700 bg-zinc-800 shadow-xl"
            initial={{ opacity: 0, y: -8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
          >
            <div className="flex items-center gap-2 p-3 border-b border-zinc-700">
              <Search className="h-4 w-4" />
              <input
                autoFocus
                placeholder="Search champions…"
                className="w-full bg-transparent outline-none"
                value={q}
                onChange={(e) => setQ(e.target.value)}
              />
              <button
                className="p-1 rounded-lg hover:bg-zinc-700/70"
                onClick={() => setOpen(false)}
              >
                <X className="h-4 w-4" />
              </button>
            </div>

            <div className="max-h-72 overflow-auto p-2 grid grid-cols-2 gap-1">
              {filtered.map((c) => {
                const isTaken = taken.has(c) && c !== champion;
                return (
                  <button
                    key={c}
                    disabled={isTaken}
                    title={isTaken ? "Already picked" : ""}
                    className={
                      "w-full text-left px-3 py-2 rounded-xl focus:outline-none focus:ring-2 focus:ring-zinc-500 " +
                      (isTaken ? "opacity-40 cursor-not-allowed" : "hover:bg-zinc-700/70")
                    }
                    onClick={() => {
                      if (!isTaken) {
                        onPick(c);
                        setOpen(false);
                        setQ("");
                      }
                    }}
                  >
                    {c}
                  </button>
                );
              })}
              {filtered.length === 0 && (
                <div className="p-4 text-sm text-zinc-500 col-span-2">
                  No results.
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function PredictionBar({ bluePct, redPct, loading }:
  { bluePct: number; redPct: number; loading: boolean }) {
  const clamp = (v: number) => Math.max(0, Math.min(100, Math.round(v)));
  const b = clamp(bluePct), r = clamp(redPct);
  return (
    <div className="rounded-2xl border border-zinc-700 overflow-hidden">
      <div className="flex">
        <motion.div
          className="flex-1 min-w-[10%] px-4 py-6 text-center font-semibold"
          style={{ background: "#1f2a44" }}
          animate={{ width: `${b}%` }}
          transition={{ type: "spring", stiffness: 120, damping: 18 }}
        >
          <div className="text-sm">BLUE</div>
          <div className="text-2xl">{loading ? "…" : `${b}%`}</div>
        </motion.div>
        <motion.div
          className="flex-1 min-w-[10%] px-4 py-6 text-center font-semibold"
          style={{ background: "#3a2121" }}
          animate={{ width: `${r}%` }}
          transition={{ type: "spring", stiffness: 120, damping: 18 }}
        >
          <div className="text-sm">RED</div>
          <div className="text-2xl">{loading ? "…" : `${r}%`}</div>
        </motion.div>
      </div>
    </div>
  );
}

export default function App() {
  const [team1, setTeam1] = useState<string | undefined>();
  const [team2, setTeam2] = useState<string | undefined>();
  const [blueDraft, setBlueDraft] = useState<(string | undefined)[]>(Array(5).fill(undefined));
  const [redDraft, setRedDraft] = useState<(string | undefined)[]>(Array(5).fill(undefined));
  const [loading, setLoading] = useState(false);
  const [scores, setScores] = useState({ blue: 65, red: 35 });

  // Already-picked champions across both teams
  const taken = useMemo(() => {
    return new Set<string>([...blueDraft, ...redDraft].filter(Boolean) as string[]);
  }, [blueDraft, redDraft]);

  // Already-picked teams (prevents duplicates)
  const takenTeams = useMemo(() => {
    return new Set<string>([team1, team2].filter(Boolean) as string[]);
  }, [team1, team2]);

  async function analyze() {
    setLoading(true);
    try {
      const seed =
        (team1?.length ?? 0) - (team2?.length ?? 0) +
        blueDraft.filter(Boolean).length - redDraft.filter(Boolean).length;
      const bluePct = 50 + Math.max(-25, Math.min(25, seed * 3));
      const redPct = 100 - bluePct;
      await new Promise((r) => setTimeout(r, 700));
      setScores({ blue: bluePct, red: redPct });
    } finally {
      setLoading(false);
    }
  }

  const ready = Boolean(team1 && team2);

  return (
    <div className="min-h-screen bg-zinc-900 text-zinc-100">
      <div className="mx-auto max-w-[1600px] p-6 space-y-6">

        {/* Instruction at the top, full width */}
        <div className="rounded-2xl border border-dashed border-zinc-700 bg-zinc-900/40 p-6 text-center">
          <span className="text-zinc-400 text-sm">
            Draft champions using the panels on the sides. When you are ready, click Analyze.
          </span>
        </div>

        {/* Main stage – BLUE left, RED right, spacer in the middle */}
        <div className="grid grid-cols-1 xl:grid-cols-12 gap-6">

          {/* LEFT: BLUE column */}
          <div className="xl:col-span-5 space-y-6">
            <TeamSearch
              label="TEAM 1 (BLUE)"
              value={team1}
              onChange={setTeam1}
              taken={takenTeams}
              side="blue"
            />
            <div className="space-y-3">
              {Array.from({ length: 5 }).map((_, i) => (
                <ChampionSlot
                  key={i}
                  owner="BLUE"
                  index={i}
                  champion={blueDraft[i]}
                  taken={taken}
                  side="blue"
                  onPick={(c) =>
                    setBlueDraft((p) => {
                      const n = [...p];
                      n[i] = c;
                      return n;
                    })
                  }
                />
              ))}
            </div>
          </div>

          {/* MIDDLE spacer (free for future stats) */}
          <div className="xl:col-span-2" />

          {/* RIGHT: RED column */}
          <div className="xl:col-span-5 space-y-6">
            <TeamSearch
              label="TEAM 2 (RED)"
              value={team2}
              onChange={setTeam2}
              taken={takenTeams}
              side="red"
            />
            <div className="space-y-3">
              {Array.from({ length: 5 }).map((_, i) => (
                <ChampionSlot
                  key={i}
                  owner="RED"
                  index={i}
                  champion={redDraft[i]}
                  taken={taken}
                  side="red"
                  onPick={(c) =>
                    setRedDraft((p) => {
                      const n = [...p];
                      n[i] = c;
                      return n;
                    })
                  }
                />
              ))}
            </div>
          </div>
        </div>

        {/* Status + Analyze */}
        <div className="flex items-center justify-between gap-4">
          <div className="flex gap-2 flex-wrap">
            <Pill>Mode: Draft</Pill>
            <Pill>Team 1: {team1 ?? "—"}</Pill>
            <Pill>Team 2: {team2 ?? "—"}</Pill>
          </div>
          <button
            onClick={analyze}
            disabled={!ready || loading}
            className="inline-flex items-center gap-2 rounded-2xl px-5 py-2 border border-zinc-700 bg-zinc-900 text-white hover:bg-zinc-800 disabled:opacity-50"
          >
            {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Search className="h-4 w-4" />}
            Analyze
          </button>
        </div>

        {/* Prediction */}
        <div className="space-y-2">
          <div className="text-sm text-zinc-500">Prediction</div>
          <PredictionBar bluePct={scores.blue} redPct={scores.red} loading={loading} />
        </div>
      </div>
    </div>
  );
}