import React, { useMemo, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Search, Plus, X, Loader2, Megaphone } from "lucide-react";

const ALL_TEAMS = [
  "G2 Esports","Fnatic","Team Vitality","Movistar KOI","SK Gaming",
  "Team Heretics","Karmine Corp","Natus Vincere",
];

const ROLES = ["TOP", "JUNGLE", "MID", "ADC", "SUPPORT"] as const;

const ALL_CHAMPIONS = [
  "Aatrox", "Ahri", "Akali", "Akshan", "Alistar", "Ambessa", "Amumu", "Anivia", "Annie", "Aphelios", "Ashe", "Aurelion Sol", "Aurora", "Azir", "Bard", "Bel'Veth", "Blitzcrank", "Brand", "Braum", "Briar", "Caitlyn", "Camille", "Cassiopeia", "Cho'Gath", "Corki", "Darius", "Diana", "Dr. Mundo", "Draven", "Ekko", "Elise", "Evelynn", "Ezreal", "Fiddlesticks", "Fiora", "Fizz", "Galio", "Gangplank", "Garen", "Gnar", "Gragas", "Graves", "Gwen", "Hecarim", "Heimerdinger", "Hwei", "Illaoi", "Irelia", "Ivern", "Janna", "Jarvan IV", "Jax", "Jayce", "Jhin", "Jinx", "K'Sante", "Kai'Sa", "Kalista", "Karma", "Karthus", "Kassadin", "Katarina", "Kayle", "Kayn", "Kennen", "Kha'Zix", "Kindred", "Kled", "Kog'Maw", "LeBlanc", "Lee Sin", "Leona", "Lillia", "Lissandra", "Lucian", "Lulu", "Lux", "Malphite", "Malzahar", "Maokai", "Master Yi", "Mel", "Milio", "Miss Fortune", "Mordekaiser", "Morgana", "Naafiri", "Nami", "Nasus", "Nautilus", "Neeko", "Nidalee", "Nilah", "Nocturne", "Nunu & Willump", "Olaf", "Orianna", "Ornn", "Pantheon", "Poppy", "Pyke", "Qiyana", "Quinn", "Rakan", "Rammus", "Rek'Sai", "Rell", "Renata Glasc", "Renekton", "Rengar", "Riven", "Rumble", "Ryze", "Samira", "Sejuani", "Senna", "Seraphine", "Sett", "Shaco", "Shen", "Shyvana", "Singed", "Sion", "Sivir", "Skarner", "Smolder", "Sona", "Soraka", "Swain", "Sylas", "Syndra", "Tahm Kench", "Taliyah", "Talon", "Taric", "Teemo", "Thresh", "Tristana", "Trundle", "Tryndamere", "Twisted Fate", "Twitch", "Udyr", "Urgot", "Varus", "Vayne", "Veigar", "Vel'Koz", "Vex", "Vi", "Viego", "Viktor", "Vladimir", "Volibear", "Warwick", "Wukong", "Xayah", "Xerath", "Xin Zhao", "Yasuo", "Yone", "Yorick", "Yunara", "Yuumi", "Zac", "Zed", "Zeri", "Ziggs", "Zilean", "Zoe", "Zyra"
];

const Pill = ({ children }: { children: React.ReactNode }) => (
  <span className="inline-flex items-center gap-2 rounded-2xl px-3 py-1 text-sm bg-zinc-800 border border-zinc-700 text-zinc-100">
    {children}
  </span>
);

type ModelName = "CatBoost" | "XGBoost" | "LightGBM";


// Fuzzy filter: A→Z gdy puste; top-N podczas pisania
function useFuzzyFilter(
  list: string[],
  q: string,
  opts?: { limitOnSearch?: number; showAllWhenEmpty?: boolean }
) {
  const { limitOnSearch = 3, showAllWhenEmpty = true } = opts ?? {};
  return useMemo(() => {
    const s = q.trim().toLowerCase();
    if (!s) {
      const all = [...list].sort((a, b) => a.localeCompare(b));
      return showAllWhenEmpty ? all : [];
    }
    const scored = list.map((name) => {
      const a = name.toLowerCase();
      let i = 0, j = 0, hits = 0;
      while (i < a.length && j < s.length) { if (a[i] === s[j]) { hits++; j++; } i++; }
      const prefixBonus = a.startsWith(s) ? 2 : 0;
      const exactBonus  = a === s ? 5 : 0;
      const score = hits / (s.length + 0.5 * (a.length - hits)) + prefixBonus + exactBonus;
      return { name, score };
    });
    return scored.sort((x, y) => y.score - x.score).slice(0, limitOnSearch).map((x) => x.name);
  }, [list, q, limitOnSearch, showAllWhenEmpty]);
}

function TeamSearch({
  label, value, onChange, taken, side,
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
  const panelTint = side === "blue" ? "border-sky-600/40 bg-sky-950/20" : "border-rose-600/40 bg-rose-950/20";

  return (
    <div className="relative">
      <div className={`rounded-2xl p-3 min-h-[120px] shadow-sm border ${panelTint}`}>
        <div className="flex items-center justify-between mb-3">
          <h3 className="font-semibold text-zinc-100">{label}</h3>
          <button
            className="inline-flex items-center gap-2 rounded-2xl border border-zinc-700 px-3 py-1 text-sm hover:bg-zinc-700/70"
            onClick={() => setOpen(true)}
          >
            <Plus className="h-4 w-4" /> Add team
          </button>
        </div>
        {/* wyśrodkowana nazwa drużyny */}
        <div className="grid place-items-center h-[110px]">
          {value ? (
            <div className="text-3xl md:text-4xl font-extrabold tracking-tight text-center">{value}</div>
          ) : (
            <div className="text-zinc-400 text-sm">No team selected.</div>
          )}
        </div>
      </div>

      <AnimatePresence>
        {open && (
          <motion.div
            className="absolute z-20 mt-2 w-[520px] max-w-[85vw] rounded-2xl border border-zinc-700 bg-zinc-800 shadow-xl"
            initial={{ opacity: 0, y: -8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -8 }}
          >
            <div className="flex items-center gap-2 p-3 border-b border-zinc-700">
              <Search className="h-4 w-4" />
              <input autoFocus placeholder="Search teams…" className="w-full bg-transparent outline-none"
                     value={q} onChange={(e) => setQ(e.target.value)} />
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
                    className={"w-full text-left px-3 py-2 rounded-xl focus:outline-none focus:ring-2 focus:ring-zinc-500 " +
                      (isTaken ? "opacity-40 cursor-not-allowed" : "hover:bg-zinc-700/70")}
                    onClick={() => { if (!isTaken) { onChange(t); setOpen(false); setQ(""); } }}
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
  owner, role, champion, onPick, taken, side,
}: {
  owner: "BLUE" | "RED";
  role: typeof ROLES[number];
  champion?: string;
  onPick: (name?: string) => void;
  taken: Set<string>;
  side: "blue" | "red";
}) {
  const [open, setOpen] = useState(false);
  const [q, setQ] = useState("");
  const filtered = useFuzzyFilter(ALL_CHAMPIONS, q, { limitOnSearch: 3, showAllWhenEmpty: true });
  const panelTint = side === "blue" ? "border-sky-600/40 bg-sky-950/20" : "border-rose-600/40 bg-rose-950/20";

  return (
    <div className="relative">
      <div className={`flex items-center gap-3 rounded-xl px-3 py-3 min-h-[72px] border ${panelTint}`}>
        <div className="flex-1">
          <div className="text-[11px] uppercase tracking-wide text-zinc-500">
            {owner} · {role}
          </div>
          <div className="h-8 md:h-10 grid place-items-center">
            <div className={"text-lg md:text-xl font-semibold tracking-tight " + (champion ? "" : "text-zinc-400")}>
              {champion ?? "Empty"}
            </div>
          </div>
        </div>
        <div className="flex gap-1">
          {champion && (
            <button className="text-xs px-2 py-1 rounded-lg border border-zinc-700 hover:bg-zinc-700/70"
                    onClick={() => onPick(undefined)}>
              Clear
            </button>
          )}
          <button className="inline-flex items-center gap-1 text-xs px-2 py-1 rounded-lg border border-zinc-700 hover:bg-zinc-700/70"
                  onClick={() => setOpen(true)}>
            <Plus className="h-3 w-3" /> Pick
          </button>
        </div>
      </div>

      <AnimatePresence>
        {open && (
          <motion.div
            className="absolute z-30 mt-2 w-[420px] max-w-[90vw] rounded-2xl border border-zinc-700 bg-zinc-800 shadow-xl"
            initial={{ opacity: 0, y: -8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -8 }}
          >
            <div className="flex items-center gap-2 p-3 border-b border-zinc-700">
              <Search className="h-4 w-4" />
              <input autoFocus placeholder="Search champions…" className="w-full bg-transparent outline-none"
                     value={q} onChange={(e) => setQ(e.target.value)} />
              <button className="p-1 rounded-lg hover:bg-zinc-700/70" onClick={() => setOpen(false)}>
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
                    className={"w-full text-left px-3 py-2 rounded-xl focus:outline-none focus:ring-2 focus:ring-zinc-500 " +
                      (isTaken ? "opacity-40 cursor-not-allowed" : "hover:bg-zinc-700/70")}
                    onClick={() => { if (!isTaken) { onPick(c); setOpen(false); setQ(""); } }}
                  >
                    {c}
                  </button>
                );
              })}
              {filtered.length === 0 && <div className="p-4 text-sm text-zinc-500 col-span-2">No results.</div>}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function PredictionBar({ bluePct, redPct, loading }:{
  bluePct: number; redPct: number; loading: boolean
}) {
  const clamp = (v: number) => Math.max(0, Math.min(100, Math.round(v)));
  const b = clamp(bluePct), r = clamp(redPct);
  return (
    <div className="rounded-2xl border border-zinc-700 overflow-hidden">
      <div className="flex">
        <motion.div className="flex-1 min-w-[10%] px-4 py-6 text-center font-semibold"
          style={{ background: "#1f2a44" }} animate={{ width: `${b}%` }}
          transition={{ type: "spring", stiffness: 120, damping: 18 }}>
          <div className="text-sm">BLUE</div>
          <div className="text-2xl">{loading ? "…" : `${b}%`}</div>
        </motion.div>
        <motion.div className="flex-1 min-w-[10%] px-4 py-6 text-center font-semibold"
          style={{ background: "#3a2121" }} animate={{ width: `${r}%` }}
          transition={{ type: "spring", stiffness: 120, damping: 18 }}>
          <div className="text-sm">RED</div>
          <div className="text-2xl">{loading ? "…" : `${r}%`}</div>
        </motion.div>
      </div>
    </div>
  );
}

function ModelToggle({
  value,
  onChange,
}: {
  value: "CatBoost" | "XGBoost" | "LightGBM";
  onChange: (m: "CatBoost" | "XGBoost" | "LightGBM") => void;
}) {
  const buttons: ModelName[] = ["CatBoost", "XGBoost", "LightGBM"];
  return (
    <div className="inline-flex items-stretch ml-4 border border-zinc-700 rounded-xl overflow-hidden">
      {buttons.map((name, i) => {
        const active = value === name;
        return (
          <button
            key={name}
            type="button"
            onClick={() => onChange(name)}
            className={
              "px-3 py-1.5 text-sm transition-colors duration-150 focus:outline-none " +
              (i === 0 ? "rounded-l-xl " : i === buttons.length - 1 ? "rounded-r-xl " : "") +
              (active
                ? "bg-zinc-700/70 text-zinc-100 font-semibold"
                : "bg-transparent text-zinc-400 hover:bg-zinc-800/60")
            }
          >
            {name}
          </button>
        );
      })}
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
  const [model, setModel] = useState<ModelName>("CatBoost");


  // zabrane champy (obie drużyny)
  const taken = useMemo(() => new Set<string>([...blueDraft, ...redDraft].filter(Boolean) as string[]),
    [blueDraft, redDraft]);

  // zabrane teamy (blokada duplikatów)
  const takenTeams = useMemo(() => new Set<string>([team1, team2].filter(Boolean) as string[]),
    [team1, team2]);

  async function analyze() {
    setLoading(true);
    try {
      const seed = (team1?.length ?? 0) - (team2?.length ?? 0)
        + blueDraft.filter(Boolean).length - redDraft.filter(Boolean).length;
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

        {/* Baner nagłówkowy*/}
        <div className="rounded-2xl border border-zinc-700 bg-gradient-to-r from-zinc-900 to-zinc-800 p-5 shadow-lg">
          <div className="flex items-center gap-4">
            <div className="shrink-0 rounded-xl bg-zinc-700/50 p-3">
              <Megaphone className="h-6 w-6 text-zinc-200" />
            </div>
            <div className="flex-1">
              <h1 className="text-xl md:text-2xl font-extrabold tracking-tight text-zinc-100">
                Wybierz drużyny i postacie – następnie kliknij <span className="zinc-500/60">Analyze</span>
              </h1>
              <p className="text-sm text-zinc-400 mt-1">
                Wybór samych drużyn - predykcja na podstawie ich ostatnich występów. Wybór drużyn i postaci uwzględni również dane kontekstowe takie jak matchupy.
              </p>
            </div>
          </div>
        </div>

        {/* Rząd 1: Drużyny + Analyze pośrodku */}
        <div className="grid grid-cols-1 xl:grid-cols-12 gap-6 items-stretch">
          <div className="xl:col-span-5">
            <TeamSearch
              label="TEAM BLUE"
              value={team1}
              onChange={setTeam1}
              taken={takenTeams}
              side="blue"
            />
          </div>

          {/* Analyze + wybór modelu jeden pod drugim */}
          <div className="xl:col-span-2 flex flex-col items-center justify-center gap-3">
            <button
              onClick={analyze}
              disabled={!ready || loading}
              className="inline-flex items-center gap-2 rounded-2xl px-6 py-3 border border-zinc-700 bg-zinc-900 text-white hover:bg-zinc-800 disabled:opacity-50"
            >
              {loading ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Search className="h-4 w-4" />
              )}
              Analyze
            </button>

            <ModelToggle value={model} onChange={setModel} />

            <p className="text-xs text-zinc-500">
              Selected model:{" "}
              <span className="text-zinc-300 font-semibold">{model}</span>
            </p>
          </div>

          <div className="xl:col-span-5">
            <TeamSearch
              label="TEAM RED"
              value={team2}
              onChange={setTeam2}
              taken={takenTeams}
              side="red"
            />
          </div>
        </div>

        {/* Rząd 2: Sloty championów*/}
        <div className="grid grid-cols-1 xl:grid-cols-12 gap-6">
          {/* BLUE */}
          <div className="xl:col-span-5 space-y-3">
            {ROLES.map((role, i) => (
              <ChampionSlot
                key={role}
                owner="BLUE"
                role={role}
                champion={blueDraft[i]}
                taken={taken}
                side="blue"
                onPick={(c) => setBlueDraft((p) => { const n = [...p]; n[i] = c; return n; })}
              />
            ))}
          </div>

          <div className="xl:col-span-2" />

          {/* RED */}
          <div className="xl:col-span-5 space-y-3">
            {ROLES.map((role, i) => (
              <ChampionSlot
                key={role}
                owner="RED"
                role={role}
                champion={redDraft[i]}
                taken={taken}
                side="red"
                onPick={(c) => setRedDraft((p) => { const n = [...p]; n[i] = c; return n; })}
              />
            ))}
          </div>
        </div>

        {/* Status */}
        <div className="flex gap-2 flex-wrap">
          <Pill>Mode: Draft</Pill>
          <Pill>Team Blue: {team1 ?? "—"}</Pill>
          <Pill>Team Red: {team2 ?? "—"}</Pill>
          <Pill>Model: {model}</Pill>
        </div>

        {/* Predykcja */}
        <div className="space-y-2">
          <div className="text-sm text-zinc-500">Prediction</div>
          <PredictionBar bluePct={scores.blue} redPct={scores.red} loading={loading} />
        </div>
      </div>
    </div>
  );
}
