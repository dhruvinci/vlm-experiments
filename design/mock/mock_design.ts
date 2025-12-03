import React, { useMemo, useState } from "react";
import { Play, Pause, SkipBack, SkipForward, Volume2, SlidersHorizontal, Settings, ChevronRight, ChevronLeft, Info, Wand2, Flag, Share, Clock, TrendingUp, Flame, Gauge, Download } from "lucide-react";

// --- Sample data to render the mock ---
const DURATION = 16 * 60 + 10; // 16:10
const segmentsSeed = [
  {
    id: 1,
    start: 0,
    end: 90,
    label: "standing",
    phase: "neutral",
    heat: 0.2,
    action: 0.2,
    control: 0.2,
    notes: "Feeling out: hand fighting, collar ties, distance checks.",
    micro: {
      Strategy:
        "Both look to establish grips and reactions without over‑committing. Tackett shoulder‑posts to manage range.",
      Setup:
        "Collar/wrist ties. Stiff arm from Tackett as Galvão feints level changes.",
      Execution:
        "Neutral grip‑fighting; no committed shots yet.",
      Outcome:
        "Reset exchanges on the feet; patience from both."
    }
  },
  {
    id: 2,
    start: 90,
    end: 103,
    label: "scramble",
    phase: "transition",
    heat: 0.8,
    action: 0.9,
    control: 0.2,
    notes: "Level change → front headlock threat → spin out.",
    micro: {
      Strategy: "Create chaos to open pathways to top.",
      Setup: "Snap + go‑behind attempt off a collar tie break.",
      Execution: "Both change directions twice; brief exposure of the back.",
      Outcome: "Settles into open guard." 
    }
  },
  {
    id: 3,
    start: 103,
    end: 192,
    label: "guard",
    phase: "control_buildup",
    heat: 0.5,
    action: 0.4,
    control: 0.6,
    notes: "Top pins hips, bottom frames; knee‑cut threats.",
    micro: {
      Strategy: "Top looks for knee‑cut; bottom retains with shin shield.",
      Setup: "Cross‑face denied; inside tie pummeling.",
      Execution: "Series of knee‑cut → backstep → re‑pummel cycles.",
      Outcome: "Top opens path to back exposure." 
    }
  },
  {
    id: 4,
    start: 192,
    end: 248,
    label: "back_control",
    phase: "dominance",
    heat: 0.9,
    action: 0.6,
    control: 0.95,
    notes: "Seatbelt established; short choke attempts; strong ride.",
    micro: {
      Strategy: "Prioritize chest‑to‑back connection over rushed finish.",
      Setup: "Chair‑sit off knee‑cut back take.",
      Execution: "Seatbelt + far‑side hook; traps wrist, peels defenses.",
      Outcome: "Opponent escapes to side control off a scramble." 
    }
  },
  {
    id: 5,
    start: 248,
    end: 368,
    label: "side_control",
    phase: "control",
    heat: 0.6,
    action: 0.4,
    control: 0.9,
    notes: "Kesa variations; near‑side underhook shutdown; knee‑on‑belly pressure.",
    micro: {
      Strategy: "Slow cooking with chest pressure to force frames.",
      Setup: "Head‑arm grip; switches to reverse kesa for mount entry.",
      Execution: "Isolates near arm; transitions to knee‑on‑belly.",
      Outcome: "Slides knee through to mount." 
    }
  },
  {
    id: 6,
    start: 368,
    end: DURATION,
    label: "mount",
    phase: "dominance",
    heat: 0.5,
    action: 0.3,
    control: 0.85,
    notes: "High mount attempts; grapevines; time expires with control.",
    micro: {
      Strategy: "Stabilize, climb high, threaten cross‑collar/armbar.",
      Setup: "Knee slide from knee‑on‑belly.",
      Execution: "Cross‑face + underhook; grapevines to blunt bridges.",
      Outcome: "Maintains until bell." 
    }
  }
];

function secondsToMMSS(s: number) {
  const m = Math.floor(s / 60)
    .toString()
    .padStart(2, "0");
  const sec = Math.floor(s % 60)
    .toString()
    .padStart(2, "0");
  return `${m}:${sec}`;
}

function classNames(...args: any[]) {
  return args.filter(Boolean).join(" ");
}

const phaseColor: Record<string, string> = {
  neutral: "bg-zinc-500",
  transition: "bg-amber-500",
  control_buildup: "bg-indigo-500",
  control: "bg-blue-500",
  dominance: "bg-emerald-500"
};

export default function SensaiMobileMock() {
  const [selected, setSelected] = useState<number | null>(null);
  const [tab, setTab] = useState<"watch" | "segments" | "summary">("watch");
  const [isPlaying, setIsPlaying] = useState(false);

  const totals = useMemo(() => {
    const action = segmentsSeed.reduce((a, s) => a + s.action * (s.end - s.start), 0) / DURATION;
    const control = segmentsSeed.reduce((a, s) => a + s.control * (s.end - s.start), 0) / DURATION;
    const heat = segmentsSeed.reduce((a, s) => a + s.heat * (s.end - s.start), 0) / DURATION;
    return { action, control, heat };
  }, []);

  return (
    <div className="w-full min-h-screen flex items-start justify-center py-6 bg-zinc-950 text-zinc-50">
      <div className="w-[390px] rounded-2xl border border-zinc-800 shadow-2xl bg-zinc-900 overflow-hidden">
        {/* Header */}
        <div className="px-4 py-3 flex items-center gap-2 border-b border-zinc-800 bg-zinc-900/80 sticky top-0 z-10">
          <ChevronLeft className="w-5 h-5 opacity-70" />
          <div className="flex-1">
            <div className="text-[15px] font-semibold leading-tight">ADCC Trials – Galvão vs Tackett</div>
            <div className="text-[11px] opacity-60">Jan 12 · {secondsToMMSS(DURATION)}</div>
          </div>
          <Share className="w-5 h-5 opacity-70" />
          <Settings className="w-5 h-5 opacity-70" />
        </div>

        {/* Top Tabs */}
        <div className="grid grid-cols-3 text-xs">
          {[
            { id: "watch", label: "Watch" },
            { id: "segments", label: "Segments" },
            { id: "summary", label: "Summary" }
          ].map((t) => (
            <button
              key={t.id}
              onClick={() => setTab(t.id as any)}
              className={classNames(
                "py-2.5 font-medium border-b",
                tab === t.id ? "border-emerald-500 text-emerald-400" : "border-transparent text-zinc-400"
              )}
            >
              {t.label}
            </button>
          ))}
        </div>

        {tab === "watch" && <WatchTab onOpen={(id) => setSelected(id)} isPlaying={isPlaying} setIsPlaying={setIsPlaying} />}
        {tab === "segments" && <SegmentsTab onOpen={(id) => setSelected(id)} />}
        {tab === "summary" && <SummaryTab totals={totals} />}

        {/* Drawer for segment details */}
        {selected && (
          <DetailsDrawer seg={segmentsSeed.find((s) => s.id === selected)!} onClose={() => setSelected(null)} />
        )}
      </div>
    </div>
  );
}

function WatchTab({ onOpen, isPlaying, setIsPlaying }: { onOpen: (id: number) => void; isPlaying: boolean; setIsPlaying: (v: boolean) => void }) {
  return (
    <div className="p-4 space-y-4">
      {/* Video player mock */}
      <div className="relative rounded-xl overflow-hidden bg-black aspect-video border border-zinc-800">
        <div className="absolute inset-0 grid place-items-center text-zinc-400 text-xs select-none">
          <div className="text-center">
            <div className="text-[11px] opacity-60 mb-1">Preview</div>
            <div className="text-sm font-medium">16:10 Jiu‑Jitsu Bout</div>
          </div>
        </div>
        <div className="absolute bottom-2 left-2 px-2 py-0.5 rounded text-[11px] bg-zinc-900/70 border border-zinc-700">00:00 / {secondsToMMSS(DURATION)}</div>
        <div className="absolute top-2 right-2 flex items-center gap-2">
          <div className="px-2 py-0.5 rounded text-[11px] bg-zinc-900/70 border border-zinc-700 flex items-center gap-1"><Clock className="w-3 h-3"/> Live</div>
        </div>
      </div>

      {/* Controls row */}
      <div className="grid grid-cols-5 gap-2 text-[11px]">
        <IconBtn icon={<Volume2 className="w-4 h-4" />} label="Sound" />
        <PillBtn label="SD" sub="Quality" />
        <PillBtn label="1x" sub="Speed" />
        <PillBtn label="⤢" sub="Layout" />
        <IconBtn icon={<SlidersHorizontal className="w-4 h-4" />} label="More" />
      </div>

      {/* Timeline */}
      <Timeline onOpen={onOpen} />

      {/* Transport */}
      <div className="flex items-center justify-center gap-4 pt-1">
        <button className="p-3 rounded-full bg-zinc-800 border border-zinc-700" title="Back 10s">
          <SkipBack className="w-5 h-5" />
        </button>
        <button
          className="p-4 rounded-full bg-emerald-500 text-zinc-950 font-semibold shadow-lg"
          onClick={() => setIsPlaying(!isPlaying)}
        >
          {isPlaying ? <Pause className="w-6 h-6" /> : <Play className="w-6 h-6" />}
        </button>
        <button className="p-3 rounded-full bg-zinc-800 border border-zinc-700" title="Forward 10s">
          <SkipForward className="w-5 h-5" />
        </button>
      </div>

      {/* Day + actions */}
      <div className="flex items-center justify-between pt-1">
        <div className="px-2.5 py-1 text-[11px] rounded-lg bg-zinc-800 border border-zinc-700">Jan 12</div>
        <div className="flex items-center gap-2 text-[11px]">
          <button className="px-2.5 py-1 rounded-lg bg-zinc-800 border border-zinc-700">Loop</button>
          <button className="px-2.5 py-1 rounded-lg bg-zinc-800 border border-zinc-700">Pause at Boundaries</button>
        </div>
      </div>

      {/* Events list */}
      <div>
        <div className="flex items-center justify-between mb-2">
          <div className="text-sm font-semibold">Segment Events</div>
          <button className="text-[11px] opacity-70">Show less</button>
        </div>
        <div className="space-y-2">
          {segmentsSeed.map((s) => (
            <button key={s.id} onClick={() => onOpen(s.id)} className="w-full text-left bg-zinc-900/60 border border-zinc-800 rounded-xl p-3 hover:bg-zinc-900">
              <div className="flex items-center justify-between">
                <div className="text-[12px] font-medium capitalize">{secondsToMMSS(s.start)}–{secondsToMMSS(s.end)} · {s.label.replace("_", " ")}</div>
                <div className="text-[11px] flex items-center gap-1 opacity-80"><Flame className="w-3 h-3"/> {Math.round(s.heat * 100)}%</div>
              </div>
              <div className="mt-1 flex items-center gap-2 text-[11px] opacity-80">
                <span>Action {Math.round(s.action * 100)}%</span>
                <span className="opacity-50">•</span>
                <span>Control {Math.round(s.control * 100)}%</span>
              </div>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

function SegmentsTab({ onOpen }: { onOpen: (id: number) => void }) {
  const [filter, setFilter] = useState<string>("all");
  const filtered = useMemo(
    () => segmentsSeed.filter((s) => (filter === "all" ? true : s.label === filter)),
    [filter]
  );
  const tags = ["all", "standing", "scramble", "guard", "back_control", "side_control", "mount"];
  return (
    <div className="p-4 space-y-3">
      <div className="flex flex-wrap gap-2">
        {tags.map((t) => (
          <button
            key={t}
            className={classNames(
              "px-2.5 py-1 rounded-full border text-[11px]",
              filter === t ? "bg-emerald-500 text-zinc-900 border-emerald-400" : "bg-zinc-900/50 border-zinc-700"
            )}
            onClick={() => setFilter(t)}
          >
            {t.replace("_", " ")}
          </button>
        ))}
      </div>

      <div className="space-y-2">
        {filtered.map((s) => (
          <button key={s.id} onClick={() => onOpen(s.id)} className="w-full text-left bg-zinc-900/60 border border-zinc-800 rounded-xl p-3 hover:bg-zinc-900">
            <div className="flex items-center justify-between">
              <div className="text-[12px] font-semibold capitalize">{s.label.replace("_", " ")}</div>
              <div className="text-[11px] opacity-70">{secondsToMMSS(s.start)}–{secondsToMMSS(s.end)}</div>
            </div>
            <div className="mt-1 flex items-center gap-2 text-[11px]">
              <span className="px-1.5 py-0.5 rounded bg-zinc-800 border border-zinc-700">AI</span>
              <span className="px-1.5 py-0.5 rounded bg-zinc-800 border border-zinc-700">Coach</span>
              <span className="ml-auto flex items-center gap-1 opacity-90"><Flame className="w-3 h-3"/> {Math.round(s.heat * 100)}%</span>
            </div>
            <div className="mt-2 h-1.5 rounded-full overflow-hidden bg-zinc-800">
              <div className="h-full bg-emerald-500" style={{ width: `${s.control * 100}%` }} />
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}

function SummaryTab({ totals }: { totals: { action: number; control: number; heat: number } }) {
  const stats = [
    { label: "Duration", value: secondsToMMSS(DURATION) },
    { label: "Segments", value: segmentsSeed.length },
    { label: "Avg Control", value: `${Math.round(totals.control * 100)}%` },
    { label: "Avg Action", value: `${Math.round(totals.action * 100)}%` }
  ];
  return (
    <div className="p-4 space-y-4">
      {/* Key stats */}
      <div className="grid grid-cols-2 gap-2">
        {stats.map((s) => (
          <div key={s.label} className="rounded-xl bg-zinc-900/60 border border-zinc-800 p-3">
            <div className="text-[11px] opacity-70">{s.label}</div>
            <div className="text-lg font-semibold">{s.value}</div>
          </div>
        ))}
      </div>

      {/* Heat over time (simple bar row) */}
      <div className="rounded-xl bg-zinc-900/60 border border-zinc-800 p-3">
        <div className="text-sm font-semibold mb-2 flex items-center gap-2"><TrendingUp className="w-4 h-4"/> Heat over time</div>
        <div className="flex gap-1 h-14 items-end">
          {segmentsSeed.map((s) => (
            <div key={s.id} className="flex-1">
              <div
                className="w-full rounded-t bg-emerald-500/80"
                style={{ height: `${Math.max(6, Math.round(s.heat * 56))}px` }}
                title={`${s.label} ${Math.round(s.heat * 100)}%`}
              />
            </div>
          ))}
        </div>
        <div className="mt-2 text-[11px] opacity-70">Higher bars = spicier moments</div>
      </div>

      {/* Top moments */}
      <div className="rounded-xl bg-zinc-900/60 border border-zinc-800 p-3 space-y-2">
        <div className="text-sm font-semibold">Key Moments</div>
        {segmentsSeed
          .filter((s) => s.heat > 0.6)
          .map((s) => (
            <div key={s.id} className="text-[12px] flex items-center justify-between">
              <span className="capitalize">{s.label.replace("_", " ")}</span>
              <span className="opacity-70">{secondsToMMSS(s.start)}–{secondsToMMSS(s.end)}</span>
            </div>
          ))}
      </div>

      {/* Coaching notes */}
      <div className="rounded-xl bg-zinc-900/60 border border-zinc-800 p-3">
        <div className="text-sm font-semibold mb-1">Coaching Summary</div>
        <ul className="list-disc pl-5 text-[12px] space-y-1 opacity-90">
          <li>Excellent ride retention during back control; continue focusing on wrist traps before choke entries.</li>
          <li>Knee‑cut series is effective—drill backstep exit to chair‑sit chain for faster back exposure.</li>
          <li>Standing phase: earlier level‑change commitment after establishing shoulder‑post reactions.</li>
        </ul>
        <div className="mt-3 flex items-center gap-2 text-[12px]">
          <button className="px-2.5 py-1 rounded-lg bg-zinc-800 border border-zinc-700 flex items-center gap-1"><Download className="w-3.5 h-3.5"/> Export PDF</button>
          <button className="px-2.5 py-1 rounded-lg bg-zinc-800 border border-zinc-700 flex items-center gap-1"><Share className="w-3.5 h-3.5"/> Share</button>
        </div>
      </div>
    </div>
  );
}

function Timeline({ onOpen }: { onOpen: (id: number) => void }) {
  return (
    <div className="rounded-xl bg-zinc-900/60 border border-zinc-800 p-3">
      <div className="text-[11px] opacity-70 mb-2">{secondsToMMSS(0)} to {secondsToMMSS(DURATION)}</div>
      <div className="w-full h-2 rounded-full overflow-hidden bg-zinc-800 flex">
        {segmentsSeed.map((s) => {
          const width = ((s.end - s.start) / DURATION) * 100;
          return (
            <button
              key={s.id}
              onClick={() => onOpen(s.id)}
              className={classNames("h-full", phaseColor[s.phase], "hover:opacity-90")}
              style={{ width: `${width}%` }}
              title={`${s.label} ${secondsToMMSS(s.start)}–${secondsToMMSS(s.end)}`}
            />
          );
        })}
      </div>
      <div className="mt-2 flex items-center justify-between text-[11px] opacity-70">
        <span>Phase colors: neutral · transition · control build‑up · control · dominance</span>
      </div>
    </div>
  );
}

function DetailsDrawer({ seg, onClose }: { seg: typeof segmentsSeed[number]; onClose: () => void }) {
  return (
    <div className="fixed inset-0 z-40">
      <div className="absolute inset-0 bg-black/60" onClick={onClose} />
      <div className="absolute inset-x-0 bottom-0 rounded-t-2xl bg-zinc-950 border-t border-zinc-800 p-4 max-h-[75vh] overflow-auto">
        <div className="flex items-center gap-2 mb-2">
          <div className="w-10 h-1.5 rounded-full bg-zinc-700 mx-auto" />
        </div>
        <div className="flex items-center justify-between mb-1">
          <div className="text-[13px] font-semibold capitalize">{seg.label.replace("_", " ")}</div>
          <div className="text-[11px] opacity-70">{secondsToMMSS(seg.start)}–{secondsToMMSS(seg.end)}</div>
        </div>
        <div className="text-[12px] opacity-90 mb-3">{seg.notes}</div>

        <div className="grid grid-cols-3 gap-2 mb-3">
          <Metric label="Action" value={`${Math.round(seg.action * 100)}%`} icon={<Wand2 className="w-3.5 h-3.5"/>} />
          <Metric label="Control" value={`${Math.round(seg.control * 100)}%`} icon={<Gauge className="w-3.5 h-3.5"/>} />
          <Metric label="Heat" value={`${Math.round(seg.heat * 100)}%`} icon={<Flame className="w-3.5 h-3.5"/>} />
        </div>

        <div className="space-y-2">
          {Object.entries(seg.micro).map(([k, v]) => (
            <div key={k} className="rounded-xl bg-zinc-900/60 border border-zinc-800 p-3">
              <div className="text-[12px] font-semibold mb-1">{k}</div>
              <div className="text-[12px] opacity-90">{v}</div>
            </div>
          ))}
        </div>

        <div className="mt-3 flex items-center justify-between">
          <div className="flex items-center gap-2 text-[12px]">
            <button className="px-2.5 py-1 rounded-lg bg-zinc-800 border border-zinc-700 flex items-center gap-1"><Wand2 className="w-3.5 h-3.5"/> Approve</button>
            <button className="px-2.5 py-1 rounded-lg bg-zinc-800 border border-zinc-700 flex items-center gap-1"><Info className="w-3.5 h-3.5"/> Edit</button>
          </div>
          <button className="px-2.5 py-1 rounded-lg bg-red-500/90 text-zinc-950 font-medium flex items-center gap-1"><Flag className="w-3.5 h-3.5"/> Flag</button>
        </div>
      </div>
    </div>
  );
}

function IconBtn({ icon, label }: { icon: React.ReactNode; label: string }) {
  return (
    <button className="flex flex-col items-center justify-center gap-1 rounded-xl bg-zinc-900/60 border border-zinc-800 py-2">
      {icon}
      <div className="text-[11px] opacity-80">{label}</div>
    </button>
  );
}

function PillBtn({ label, sub }: { label: string; sub: string }) {
  return (
    <button className="rounded-xl bg-zinc-900/60 border border-zinc-800 py-2 text-center">
      <div className="text-sm font-semibold leading-none">{label}</div>
      <div className="text-[10px] opacity-70 mt-1">{sub}</div>
    </button>
  );
}

function Metric({ label, value, icon }: { label: string; value: string | number; icon?: React.ReactNode }) {
  return (
    <div className="rounded-xl bg-zinc-900/60 border border-zinc-800 p-3">
      <div className="text-[11px] opacity-70 flex items-center gap-1">{icon} {label}</div>
      <div className="text-base font-semibold">{value}</div>
    </div>
  );
}
