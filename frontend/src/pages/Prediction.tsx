import { useEffect, useState } from "react";
import { Dna, Shuffle, Sparkles, Target, Zap } from "lucide-react";
import { api, apiError } from "../services/api";
import { Card, ErrorBox, Loading, SectionTitle } from "../components/ui";
import UncertaintyBar from "../components/UncertaintyBar";
import type { PredictionResult } from "../types";

const MODELS = [
  { key: "automatic", label: "Automatic (Highest Test F1)" },
  { key: "bayesian", label: "Bayesian Logistic Regression (PyMC)" },
  { key: "lda", label: "Linear Discriminant Analysis (LDA)" },
  { key: "linear_svm", label: "Linear SVM (Calibrated)" },
  { key: "kernel_svm", label: "Kernel SVM (RBF)" },
];

const PRESETS = [0, 10, 50, 250, 1254, 3000];

export default function Prediction() {
  const [mode, setMode] = useState<"sample" | "advanced">("sample");
  const [model, setModel] = useState("automatic");
  const [sampleId, setSampleId] = useState(1254);
  const [totalSamples, setTotalSamples] = useState<number | null>(null);
  const [schema, setSchema] = useState<any>(null);
  const [featureValues, setFeatureValues] = useState<Record<string, string>>({});
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [err, setErr] = useState("");
  const [busy, setBusy] = useState(false);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    (async () => {
      try {
        const s = await api.samples(1);
        setTotalSamples(s.total_test_samples);
        setSchema(await api.predictSchema());
        setReady(true);
      } catch (e) {
        setErr(apiError(e));
      }
    })();
  }, []);

  async function run() {
    setErr("");
    setBusy(true);
    setResult(null);
    try {
      if (mode === "sample") {
        setResult(await api.predict({ model, sample_id: sampleId }));
      } else {
        const features: Record<string, number> = {};
        for (const [k, v] of Object.entries(featureValues)) {
          if (v !== "") features[k] = Number(v);
        }
        setResult(await api.predict({ model, features }));
      }
    } catch (e) {
      setErr(apiError(e));
    } finally {
      setBusy(false);
    }
  }

  function pickRandom() {
    if (!totalSamples) return;
    const rnd = Math.floor(Math.random() * totalSamples);
    setSampleId(rnd);
  }

  if (!ready && !err) return <Loading label="Preparing prediction interface…" />;

  return (
    <div className="space-y-8">
      <SectionTitle
        title="Cognitive Workload Prediction"
        subtitle="Classify cognitive workload and evaluate predictive uncertainty from HCI behavioral features."
      />
      {err && <ErrorBox message={err} />}

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <Card className="space-y-5">
          <div className="flex gap-2 border-b border-line pb-4">
            <button
              className={`flex-1 rounded-lg py-2 text-sm font-medium transition-colors ${
                mode === "sample"
                  ? "bg-accent-blue/15 text-accent-blue border border-accent-blue/30"
                  : "bg-base-900/60 text-muted hover:text-white border border-line"
              }`}
              onClick={() => setMode("sample")}
            >
              Option A — Test Dataset Sample
            </button>
            <button
              className={`flex-1 rounded-lg py-2 text-sm font-medium transition-colors ${
                mode === "advanced"
                  ? "bg-accent-blue/15 text-accent-blue border border-accent-blue/30"
                  : "bg-base-900/60 text-muted hover:text-white border border-line"
              }`}
              onClick={() => setMode("advanced")}
            >
              Option B — Advanced Input
            </button>
          </div>

          <div>
            <label className="mb-1.5 block text-xs font-semibold uppercase tracking-wider text-muted">
              Select Classification Model
            </label>
            <select
              value={model}
              onChange={(e) => setModel(e.target.value)}
              className="w-full rounded-lg border border-line bg-base-900 px-3.5 py-2.5 text-sm text-white outline-none focus:border-accent-blue transition-colors"
            >
              {MODELS.map((m) => (
                <option key={m.key} value={m.key}>
                  {m.label}
                </option>
              ))}
            </select>
            {model === "automatic" && (
              <p className="mt-1.5 text-xs text-muted">
                Automatic mode selects the estimator with highest test F1-score from the empirical comparison.
              </p>
            )}
          </div>

          {mode === "sample" ? (
            <div className="space-y-4">
              <div>
                <div className="flex items-center justify-between mb-1.5">
                  <label className="text-xs font-semibold uppercase tracking-wider text-muted">
                    Test Sample ID
                  </label>
                  <span className="text-xs font-mono text-muted">
                    Available: 0 – {totalSamples != null ? (totalSamples - 1).toLocaleString() : "..."}
                  </span>
                </div>
                <div className="flex gap-2">
                  <input
                    type="number"
                    min={0}
                    max={totalSamples ? totalSamples - 1 : undefined}
                    value={sampleId}
                    onChange={(e) => setSampleId(Number(e.target.value))}
                    className="flex-1 rounded-lg border border-line bg-base-900 px-3.5 py-2.5 font-mono text-sm text-white outline-none focus:border-accent-blue transition-colors"
                    placeholder="Enter Sample ID, e.g. 1254"
                  />
                  <button
                    type="button"
                    onClick={pickRandom}
                    className="btn-ghost px-3 text-xs flex items-center gap-1.5"
                    title="Pick random held-out sample"
                  >
                    <Shuffle className="h-3.5 w-3.5" /> Random
                  </button>
                </div>
              </div>

              <div>
                <span className="text-xs text-muted block mb-2 font-medium">Quick Select Presets:</span>
                <div className="flex flex-wrap gap-2">
                  {PRESETS.map((p) => (
                    <button
                      key={p}
                      type="button"
                      onClick={() => setSampleId(p)}
                      className={`px-2.5 py-1 text-xs font-mono rounded-md border transition-all ${
                        sampleId === p
                          ? "border-accent-blue bg-accent-blue/20 text-white"
                          : "border-line bg-base-900/60 text-muted hover:text-white"
                      }`}
                    >
                      #{p}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          ) : (
            <div className="space-y-3">
              <p className="text-xs text-muted">
                {schema?.note} Displaying {schema?.numeric_features?.length ?? 0} manageable features.
              </p>
              <div className="grid grid-cols-2 gap-3 max-h-72 overflow-y-auto pr-1">
                {(schema?.numeric_features ?? []).map((f: string) => (
                  <div key={f}>
                    <label className="mb-1 block truncate font-mono text-[11px] text-muted">
                      {f}
                    </label>
                    <input
                      type="number"
                      step="any"
                      value={featureValues[f] ?? ""}
                      onChange={(e) =>
                        setFeatureValues((s) => ({ ...s, [f]: e.target.value }))
                      }
                      placeholder="median impute"
                      className="w-full rounded-lg border border-line bg-base-900 px-2 py-1.5 font-mono text-xs text-white outline-none focus:border-accent-blue"
                    />
                  </div>
                ))}
              </div>
            </div>
          )}

          <button
            className="btn-primary w-full py-3 justify-center text-sm shadow-lg shadow-accent-blue/10"
            disabled={busy}
            onClick={run}
          >
            <Sparkles className="h-4 w-4" />
            {busy ? "Analyzing HCI Patterns…" : "Analyze Cognitive Workload"}
          </button>
        </Card>

        <ResultCard result={result} />
      </div>
    </div>
  );
}

function ResultCard({ result }: { result: PredictionResult | null }) {
  if (!result)
    return (
      <Card className="flex flex-col items-center justify-center text-center p-12 text-muted space-y-3 border-dashed">
        <Target className="h-10 w-10 text-muted/40 animate-pulse" />
        <div className="text-base font-medium text-white/80">Awaiting Prediction</div>
        <p className="text-xs max-w-sm">
          Select a held-out test sample ID (or enter feature values), choose a model, and click <strong>Analyze</strong> to run inference.
        </p>
      </Card>
    );

  const isHigh =
    result.prediction_class === "1" ||
    result.prediction_label?.toLowerCase().includes("high");

  return (
    <Card className="space-y-6 relative overflow-hidden border border-line/80 shadow-xl">
      <div
        className={`absolute -right-16 -top-16 h-36 w-36 rounded-full blur-3xl opacity-20 pointer-events-none ${
          isHigh ? "bg-purple-500" : "bg-cyan-500"
        }`}
      />

      <div>
        <div className="flex items-center justify-between">
          <div className="text-xs font-semibold tracking-wider uppercase text-muted flex items-center gap-1.5">
            <Zap className="h-3.5 w-3.5 text-accent-blue" />
            Predicted Cognitive Workload
          </div>
          <span
            className={`px-2.5 py-0.5 rounded-full text-xs font-mono font-semibold uppercase tracking-wider ${
              isHigh
                ? "bg-purple-500/20 text-purple-300 border border-purple-500/30"
                : "bg-cyan-500/20 text-cyan-300 border border-cyan-500/30"
            }`}
          >
            {isHigh ? "High Workload" : "Low Workload"}
          </span>
        </div>

        <div className="mt-2 text-3xl font-extrabold tracking-tight text-white flex items-baseline gap-2">
          <span>{result.prediction_label || result.prediction_class}</span>
          <span className="text-xs font-mono text-muted">
            (Class {result.prediction_class})
          </span>
        </div>

        {result.true_label !== undefined && (
          <div className="mt-2 flex items-center gap-2 text-xs">
            <span className="text-muted">Held-out Ground Truth:</span>
            <span
              className={`font-mono px-2 py-0.5 rounded ${
                result.prediction_class === result.true_label
                  ? "bg-emerald-500/10 text-emerald-400 border border-emerald-500/30"
                  : "bg-amber-500/10 text-amber-400 border border-amber-500/30"
              }`}
            >
              {result.true_label_display || `Class ${result.true_label}`}{" "}
              {result.prediction_class === result.true_label ? "✓ Correct" : "× Misclassified"}
            </span>
          </div>
        )}
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div className="rounded-xl border border-line bg-base-900/60 p-4">
          <div className="text-xs font-medium text-muted">Winning Probability</div>
          <div className="text-3xl font-bold font-mono text-accent-cyan mt-1">
            {result.probability != null
              ? `${(result.probability * 100).toFixed(1)}%`
              : "—"}
          </div>
        </div>
        <div className="rounded-xl border border-line bg-base-900/60 p-4">
          <div className="text-xs font-medium text-muted">Prediction Uncertainty</div>
          <div className="text-3xl font-bold font-mono text-accent-purple mt-1">
            {result.uncertainty != null
              ? `${(result.uncertainty * 100).toFixed(1)}%`
              : "—"}
          </div>
        </div>
      </div>

      {result.probabilities_breakdown && (
        <div className="space-y-2.5">
          <div className="text-xs font-semibold uppercase tracking-wider text-muted flex items-center justify-between">
            <span>Class-by-Class Posterior Probabilities</span>
            <span className="text-[11px] font-mono text-muted">Multimodal Output</span>
          </div>
          <div className="space-y-2">
            {result.probabilities_breakdown.map((pb) => {
              const active = pb.class === result.prediction_class;
              return (
                <div key={pb.class} className="space-y-1">
                  <div className="flex justify-between text-xs">
                    <span
                      className={`font-medium ${
                        active ? "text-white font-semibold" : "text-muted"
                      }`}
                    >
                      {pb.label} (Class {pb.class})
                    </span>
                    <span className="font-mono text-white font-semibold">
                      {pb.percentage.toFixed(1)}%
                    </span>
                  </div>
                  <div className="h-2 w-full rounded-full bg-base-900 overflow-hidden border border-line/40">
                    <div
                      className={`h-full rounded-full transition-all duration-500 ${
                        pb.class === "1" ? "bg-accent-purple" : "bg-accent-cyan"
                      }`}
                      style={{ width: `${Math.min(100, Math.max(0, pb.percentage))}%` }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {result.uncertainty != null && (
        <div>
          <div className="mb-2 flex items-center justify-between text-xs">
            <span className="font-medium text-muted">Confidence Interpretation</span>
            <span className="font-semibold text-white">
              {result.confidence_interpretation}
            </span>
          </div>
          <UncertaintyBar value={result.uncertainty} />
        </div>
      )}

      <div className="space-y-1.5 border-t border-line pt-4 text-xs">
        <div className="flex justify-between">
          <span className="text-muted">Inference Model</span>
          <span className="font-mono text-white">{result.model}</span>
        </div>
        {result.epistemic_std != null && (
          <div className="flex justify-between">
            <span className="text-muted">Posterior Epistemic Std Dev</span>
            <span className="font-mono text-accent-blue font-semibold">
              {result.epistemic_std.toFixed(4)}
            </span>
          </div>
        )}
        <div className="flex justify-between">
          <span className="text-muted">Input Mode</span>
          <span className="font-mono text-muted capitalize">
            {result.input_mode.replace("_", " ")}
            {result.sample_id !== undefined ? ` (ID: ${result.sample_id})` : ""}
          </span>
        </div>
      </div>

      <div className="rounded-lg bg-base-900/80 border border-line/60 p-3 text-[11px] text-muted flex items-start gap-2">
        <Dna className="h-4 w-4 text-accent-blue shrink-0 mt-0.5" />
        <span>{result.selection_reason}</span>
      </div>
    </Card>
  );
}
