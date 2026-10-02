import { useEffect, useRef, useState } from "react";
import { CheckCircle2, Loader2, Play, XCircle } from "lucide-react";
import { api, apiError } from "../services/api";
import type { TrainStatus } from "../types";
import { ErrorBox } from "./ui";

// Coarse pipeline steps we expect the backend to report, so we can render a
// stable checklist even before each step has been reached.
const EXPECTED = [
  "Loading dataset",
  "Preprocessing",
  "Training LDA",
  "Training Bayesian Logistic Regression",
  "Training Linear SVM",
  "Training Kernel SVM",
  "Evaluating models",
  "Saving artifacts",
];

export default function TrainingPanel({ onDone }: { onDone?: () => void }) {
  const [status, setStatus] = useState<TrainStatus | null>(null);
  const [err, setErr] = useState("");
  const [starting, setStarting] = useState(false);
  const poll = useRef<number | null>(null);

  async function refresh() {
    try {
      const s = await api.trainStatus();
      setStatus(s);
      if (s.status === "completed") {
        stop();
        onDone?.();
      } else if (s.status === "failed") {
        stop();
        setErr(s.message);
      }
    } catch (e) {
      setErr(apiError(e));
      stop();
    }
  }

  function stop() {
    if (poll.current) {
      window.clearInterval(poll.current);
      poll.current = null;
    }
  }

  useEffect(() => {
    refresh();
    return stop;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function start(force: boolean) {
    setErr("");
    setStarting(true);
    try {
      await api.train(force);
      poll.current = window.setInterval(refresh, 1500);
    } catch (e) {
      setErr(apiError(e));
    } finally {
      setStarting(false);
    }
  }

  const running = status?.status === "running";
  const doneSteps = new Set(
    (status?.steps || []).filter((s) => s.status === "done").map((s) => s.name)
  );
  const activeStep = (status?.steps || [])
    .filter((s) => s.status === "running")
    .map((s) => s.name)
    .pop();

  return (
    <div className="space-y-4">
      {err && <ErrorBox message={err} />}

      <div className="flex flex-wrap items-center gap-3">
        <button
          className="btn-primary"
          disabled={running || starting}
          onClick={() => start(false)}
        >
          {running ? (
            <Loader2 className="h-4 w-4 animate-spin" />
          ) : (
            <Play className="h-4 w-4" />
          )}
          {running ? "Training…" : "Train Models"}
        </button>
        <button className="btn-ghost" disabled={running} onClick={() => start(true)}>
          Force retrain
        </button>
        {status?.status === "completed" && (
          <div className="inline-flex items-center gap-1.5 rounded-lg border border-emerald-500/30 bg-emerald-500/10 px-3 py-1.5 text-xs font-medium text-emerald-400">
            <CheckCircle2 className="h-4 w-4" />
            Training completed successfully. 4 models trained.
          </div>
        )}
      </div>

      <ol className="space-y-2">
        {EXPECTED.map((name) => {
          const done = doneSteps.has(name);
          const active = activeStep === name;
          return (
            <li key={name} className="flex items-center gap-3 text-sm">
              {done ? (
                <CheckCircle2 className="h-4 w-4 text-accent-cyan" />
              ) : active ? (
                <Loader2 className="h-4 w-4 animate-spin text-accent-blue" />
              ) : status?.status === "failed" ? (
                <XCircle className="h-4 w-4 text-amber-400" />
              ) : (
                <span className="h-4 w-4 rounded-full border border-line" />
              )}
              <span className={done || active ? "text-white" : "text-muted"}>
                {name}
              </span>
            </li>
          );
        })}
      </ol>

      {status?.message && (
        <p className="font-mono text-xs text-muted">{status.message}</p>
      )}
    </div>
  );
}
