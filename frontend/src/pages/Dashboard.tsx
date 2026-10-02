import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Legend,
  CartesianGrid,
} from "recharts";
import {
  Activity,
  ArrowRight,
  Cpu,
  Database,
  Gauge,
  Sparkles,
} from "lucide-react";
import { api, apiError } from "../services/api";
import { Card, ErrorBox, Loading, SectionTitle, CHART_COLORS } from "../components/ui";
import type { DatasetInfo, EvaluationResponse, Health, UncertaintyResponse } from "../types";
import TrainingPanel from "../components/TrainingPanel";

export default function Dashboard() {
  const nav = useNavigate();
  const [health, setHealth] = useState<Health | null>(null);
  const [info, setInfo] = useState<DatasetInfo | null>(null);
  const [evalData, setEvalData] = useState<EvaluationResponse | null>(null);
  const [uncData, setUncData] = useState<UncertaintyResponse | null>(null);
  const [err, setErr] = useState("");
  const [loading, setLoading] = useState(true);

  async function load() {
    setLoading(true);
    setErr("");
    try {
      const h = await api.health();
      setHealth(h);
      try {
        setInfo(await api.datasetInfo());
      } catch {
        /* dataset info might be empty before first run */
      }
      if (h.models_trained) {
        const [ev, unc] = await Promise.all([
          api.evaluation(),
          api.uncertainty().catch(() => null),
        ]);
        setEvalData(ev);
        setUncData(unc);
      }
    } catch (e) {
      setErr(apiError(e));
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    load();
  }, []);

  if (loading) return <Loading label="Connecting to NeuroShield backend & loading artifacts…" />;
  if (err) return <ErrorBox message={err} />;

  const accHi = evalData?.highlights?.accuracy;
  const f1Hi = evalData?.highlights?.f1;

  return (
    <div className="space-y-8">
      <div className="flex flex-wrap items-center justify-between gap-4 border-b border-line pb-6">
        <div>
          <div className="flex items-center gap-2">
            <span className="h-2 w-2 rounded-full bg-emerald-400 animate-pulse" />
            <span className="text-xs font-mono uppercase tracking-wider text-muted">
              System Status: Active
            </span>
          </div>
          <h1 className="text-3xl font-extrabold tracking-tight text-white mt-1">
            NeuroShield Research Dashboard
          </h1>
          <p className="text-sm text-muted mt-1">
            Probabilistic Cognitive Workload &amp; Fatigue Classification from Multimodal HCI Data
          </p>
        </div>
        <div className="flex gap-2">
          <button className="btn-primary" onClick={() => nav("/prediction")}>
            <Sparkles className="h-4 w-4" /> Run Prediction
          </button>
          <button className="btn-ghost" onClick={() => nav("/training")}>
            <Cpu className="h-4 w-4" /> Training Studio
          </button>
        </div>
      </div>

      {/* 4 Core Summary Cards: Dataset Card, Model Card, Evaluation Card, Uncertainty Card */}
      <div className="grid grid-cols-1 gap-5 md:grid-cols-2 lg:grid-cols-4">
        {/* 1. Dataset Card */}
        <Card className="flex flex-col justify-between border-t-2 border-t-accent-blue hover:border-accent-blue/50 transition-all">
          <div>
            <div className="flex items-center justify-between text-muted text-xs">
              <span className="font-semibold uppercase tracking-wider">Dataset Card</span>
              <Database className="h-4 w-4 text-accent-blue" />
            </div>
            <div className="mt-3 text-2xl font-bold text-white font-mono">
              {info ? info.n_samples.toLocaleString() : "57,472"}
            </div>
            <div className="text-xs text-muted">Total observations recorded</div>
            <div className="mt-4 space-y-1.5 text-xs">
              <div className="flex justify-between border-b border-line/40 pb-1">
                <span className="text-muted">Features</span>
                <span className="font-mono text-white font-semibold">
                  {info ? info.n_features : 414} HCI/EEG
                </span>
              </div>
              <div className="flex justify-between border-b border-line/40 pb-1">
                <span className="text-muted">Target</span>
                <span className="font-mono text-accent-cyan">
                  {info ? info.target_column : "cognitive_load"}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted">Classes</span>
                <span className="font-mono text-white">2 (Low / High)</span>
              </div>
            </div>
          </div>
          <button
            onClick={() => nav("/dataset")}
            className="mt-4 flex items-center justify-between text-xs text-accent-blue hover:text-white pt-2 border-t border-line/50 transition-colors"
          >
            <span>Explore Dataset</span>
            <ArrowRight className="h-3.5 w-3.5" />
          </button>
        </Card>

        {/* 2. Model Card */}
        <Card className="flex flex-col justify-between border-t-2 border-t-accent-purple hover:border-accent-purple/50 transition-all">
          <div>
            <div className="flex items-center justify-between text-muted text-xs">
              <span className="font-semibold uppercase tracking-wider">Model Card</span>
              <Cpu className="h-4 w-4 text-accent-purple" />
            </div>
            <div className="mt-3 text-2xl font-bold text-white font-mono">4 Models</div>
            <div className="text-xs text-muted">Diverse ML architectures</div>
            <div className="mt-4 space-y-1.5 text-xs">
              <div className="flex justify-between border-b border-line/40 pb-1">
                <span className="text-muted">Linear Discriminant</span>
                <span className="font-mono text-accent-blue font-semibold">LDA</span>
              </div>
              <div className="flex justify-between border-b border-line/40 pb-1">
                <span className="text-muted">Probabilistic NUTS</span>
                <span className="font-mono text-accent-purple font-semibold">PyMC Bayes</span>
              </div>
              <div className="flex justify-between border-b border-line/40 pb-1">
                <span className="text-muted">Discriminative</span>
                <span className="font-mono text-accent-cyan font-semibold">Linear SVM</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted">Nonlinear Boundary</span>
                <span className="font-mono text-amber-400 font-semibold">Kernel SVM (RBF)</span>
              </div>
            </div>
          </div>
          <button
            onClick={() => nav("/models")}
            className="mt-4 flex items-center justify-between text-xs text-accent-purple hover:text-white pt-2 border-t border-line/50 transition-colors"
          >
            <span>Compare Models</span>
            <ArrowRight className="h-3.5 w-3.5" />
          </button>
        </Card>

        {/* 3. Evaluation Card */}
        <Card className="flex flex-col justify-between border-t-2 border-t-accent-cyan hover:border-accent-cyan/50 transition-all">
          <div>
            <div className="flex items-center justify-between text-muted text-xs">
              <span className="font-semibold uppercase tracking-wider">Evaluation Card</span>
              <Activity className="h-4 w-4 text-accent-cyan" />
            </div>
            <div className="mt-3 text-2xl font-bold text-accent-cyan font-mono">
              {accHi ? `${(accHi.value * 100).toFixed(1)}%` : "Pending"}
            </div>
            <div className="text-xs text-muted">
              {accHi ? `Highest accuracy (${accHi.model})` : "Train models to evaluate"}
            </div>
            <div className="mt-4 space-y-1.5 text-xs">
              <div className="flex justify-between border-b border-line/40 pb-1">
                <span className="text-muted">Highest F1 Score</span>
                <span className="font-mono text-white font-semibold">
                  {f1Hi ? f1Hi.value.toFixed(4) : "—"}
                </span>
              </div>
              <div className="flex justify-between border-b border-line/40 pb-1">
                <span className="text-muted">Test Split Size</span>
                <span className="font-mono text-white">14,368 rows (25%)</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted">Metric Averaging</span>
                <span className="font-mono text-accent-cyan">Weighted</span>
              </div>
            </div>
          </div>
          <button
            onClick={() => nav("/evaluation")}
            className="mt-4 flex items-center justify-between text-xs text-accent-cyan hover:text-white pt-2 border-t border-line/50 transition-colors"
          >
            <span>Detailed Metrics</span>
            <ArrowRight className="h-3.5 w-3.5" />
          </button>
        </Card>

        {/* 4. Uncertainty Card */}
        <Card className="flex flex-col justify-between border-t-2 border-t-amber-400 hover:border-amber-400/50 transition-all">
          <div>
            <div className="flex items-center justify-between text-muted text-xs">
              <span className="font-semibold uppercase tracking-wider">Uncertainty Card</span>
              <Gauge className="h-4 w-4 text-amber-400" />
            </div>
            <div className="mt-3 text-2xl font-bold text-amber-400 font-mono">
              {uncData ? `${(uncData.mean_uncertainty * 100).toFixed(1)}%` : "Pending"}
            </div>
            <div className="text-xs text-muted">Mean Bayesian posterior uncertainty</div>
            <div className="mt-4 space-y-1.5 text-xs">
              <div className="flex justify-between border-b border-line/40 pb-1">
                <span className="text-muted">Median Uncertainty</span>
                <span className="font-mono text-white">
                  {uncData ? `${(uncData.median_uncertainty * 100).toFixed(1)}%` : "—"}
                </span>
              </div>
              <div className="flex justify-between border-b border-line/40 pb-1">
                <span className="text-muted">Max Uncertainty</span>
                <span className="font-mono text-white">
                  {uncData ? `${(uncData.max_uncertainty * 100).toFixed(1)}%` : "—"}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted">Posterior Samples</span>
                <span className="font-mono text-amber-400">400 MCMC Draws</span>
              </div>
            </div>
          </div>
          <button
            onClick={() => nav("/uncertainty")}
            className="mt-4 flex items-center justify-between text-xs text-amber-400 hover:text-white pt-2 border-t border-line/50 transition-colors"
          >
            <span>Uncertainty Analysis</span>
            <ArrowRight className="h-3.5 w-3.5" />
          </button>
        </Card>
      </div>

      {!health?.models_trained && (
        <Card className="border border-accent-blue/30 bg-accent-blue/5">
          <SectionTitle
            title="Model Artifacts Not Detected"
            subtitle="Trigger the training pipeline to fit all four models and generate genuine empirical evaluation artifacts."
          />
          <TrainingPanel onDone={load} />
        </Card>
      )}

      {evalData && (
        <Card className="space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-4 border-b border-line pb-4">
            <div>
              <SectionTitle
                title="Cross-Model Performance Benchmark"
                subtitle={`Empirical test metrics measured across all 14,368 held-out observations.`}
              />
            </div>
            <div className="flex items-center gap-2">
              <button
                className="btn-ghost text-xs"
                onClick={() => nav("/evaluation")}
              >
                Full Metric Breakdown
              </button>
            </div>
          </div>

          <div className="h-80 w-full pt-2">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={evalData.bar_chart} margin={{ top: 12, right: 12, left: -10, bottom: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#243049" />
                <XAxis dataKey="model" tick={{ fill: "#8a97b1", fontSize: 12 }} interval={0} />
                <YAxis domain={[0, 1]} tick={{ fill: "#8a97b1", fontSize: 12 }} />
                <Tooltip
                  contentStyle={{
                    background: "#0f1523",
                    border: "1px solid #243049",
                    borderRadius: 8,
                  }}
                />
                <Legend wrapperStyle={{ paddingTop: 10 }} />
                <Bar dataKey="accuracy" name="Accuracy" fill={CHART_COLORS[0]} radius={[4, 4, 0, 0]} />
                <Bar dataKey="precision" name="Precision" fill={CHART_COLORS[1]} radius={[4, 4, 0, 0]} />
                <Bar dataKey="recall" name="Recall" fill={CHART_COLORS[2]} radius={[4, 4, 0, 0]} />
                <Bar dataKey="f1" name="F1 Score" fill={CHART_COLORS[3]} radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>
      )}
    </div>
  );
}
