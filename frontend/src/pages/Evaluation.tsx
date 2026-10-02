import { useEffect, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { BarChart2, Layers } from "lucide-react";
import { api, apiError } from "../services/api";
import { Card, CHART_COLORS, ErrorBox, Loading, SectionTitle } from "../components/ui";
import ConfusionMatrix from "../components/ConfusionMatrix";
import type { ComparisonResponse, EvaluationResponse } from "../types";

type MetricFilter = "all" | "accuracy" | "precision" | "recall" | "f1";

const METRIC_LABELS: Record<MetricFilter, string> = {
  all: "All Metrics",
  accuracy: "Accuracy",
  precision: "Precision (Weighted)",
  recall: "Recall (Weighted)",
  f1: "F1 Score (Weighted)",
};

const METRIC_COLORS: Record<string, string> = {
  accuracy: "#4f8cff",
  precision: "#8b7dff",
  recall: "#38d9c6",
  f1: "#f59e0b",
};

export default function Evaluation() {
  const [data, setData] = useState<EvaluationResponse | null>(null);
  const [cmp, setCmp] = useState<ComparisonResponse | null>(null);
  const [err, setErr] = useState("");
  const [loading, setLoading] = useState(true);
  const [selected, setSelected] = useState("lda");
  const [activeMetric, setActiveMetric] = useState<MetricFilter>("all");

  useEffect(() => {
    (async () => {
      try {
        const [evalRes, cmpRes] = await Promise.all([
          api.evaluation(),
          api.comparison().catch(() => null),
        ]);
        setData(evalRes);
        setCmp(cmpRes);
      } catch (e) {
        setErr(apiError(e));
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  if (loading) return <Loading label="Loading model evaluation results…" />;
  if (err) return <ErrorBox message={err} />;
  if (!data) return null;

  return (
    <div className="space-y-8">
      <SectionTitle
        title="Model Evaluation &amp; Comparison"
        subtitle={`All metrics are evaluated on the exact same held-out test split (14,368 observations). Evaluated using ${data.averaging} averaging. Results are reported objectively without unwarranted claims of real-world clinical fatigue detection.`}
      />

      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        {Object.entries(data.highlights).map(([metric, h]) => (
          <Card key={metric} className="border-t-2 border-t-accent-blue/40">
            <div className="card-title capitalize text-xs">Highest {metric}</div>
            <div className="mt-2 text-base font-semibold text-white">{h.model}</div>
            <div className="mt-1 font-mono text-2xl font-bold text-accent-cyan">
              {h.value.toFixed(4)}
            </div>
            <div className="mt-1 text-[11px] text-muted">
              Measured on held-out split
            </div>
          </Card>
        ))}
      </div>

      {cmp && (
        <Card>
          <div className="flex items-center gap-2 mb-4">
            <BarChart2 className="h-4 w-4 text-accent-blue" />
            <SectionTitle
              title="Four-Model Benchmark Table"
              subtitle={`Empirical test-set evaluation (${cmp.averaging} averaging). Higher is better.`}
            />
          </div>
          <div className="overflow-x-auto">
            <table className="min-w-full text-sm">
              <thead>
                <tr className="border-b border-line text-left text-xs uppercase tracking-wider text-muted">
                  <th className="p-3 font-semibold">Classification Model</th>
                  <th className="p-3 text-right font-semibold">Accuracy</th>
                  <th className="p-3 text-right font-semibold">Precision</th>
                  <th className="p-3 text-right font-semibold">Recall</th>
                  <th className="p-3 text-right font-semibold">F1 Score</th>
                  <th className="p-3 text-right font-semibold">ROC-AUC</th>
                  <th className="p-3 text-right font-semibold">Training Time</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-line/40">
                {cmp.models.map((m) => (
                  <tr key={m.model_key} className="hover:bg-base-700/30 transition-colors">
                    <td className="p-3 font-medium text-white flex items-center gap-2">
                      <span className="h-2 w-2 rounded-full bg-accent-blue" />
                      {m.model}
                    </td>
                    <td className="p-3 text-right font-mono font-semibold text-accent-blue">
                      {m.accuracy.toFixed(4)}
                    </td>
                    <td className="p-3 text-right font-mono text-white/90">
                      {m.precision.toFixed(4)}
                    </td>
                    <td className="p-3 text-right font-mono text-white/90">
                      {m.recall.toFixed(4)}
                    </td>
                    <td className="p-3 text-right font-mono font-semibold text-accent-cyan">
                      {m.f1.toFixed(4)}
                    </td>
                    <td className="p-3 text-right font-mono text-muted">
                      {m.roc_auc != null ? m.roc_auc.toFixed(4) : "—"}
                    </td>
                    <td className="p-3 text-right font-mono text-muted">
                      {m.training_time_seconds != null
                        ? `${m.training_time_seconds.toFixed(2)}s`
                        : "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      <Card className="space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-4 border-b border-line pb-4">
          <div>
            <SectionTitle
              title="Interactive Metric Comparison Chart"
              subtitle="Filter and compare performance across models for individual metrics."
            />
          </div>
          <div className="flex flex-wrap gap-1 bg-base-900 p-1 rounded-lg border border-line">
            {(["all", "accuracy", "precision", "recall", "f1"] as MetricFilter[]).map(
              (metric) => (
                <button
                  key={metric}
                  onClick={() => setActiveMetric(metric)}
                  className={`px-3 py-1.5 rounded-md text-xs font-medium transition-all ${
                    activeMetric === metric
                      ? "bg-accent-blue text-white shadow-sm"
                      : "text-muted hover:text-white hover:bg-base-800"
                  }`}
                >
                  {METRIC_LABELS[metric]}
                </button>
              )
            )}
          </div>
        </div>

        <div className="h-80 w-full pt-2">
          <ResponsiveContainer width="100%" height="100%">
            {activeMetric === "all" ? (
              <BarChart data={data.bar_chart} margin={{ top: 12, right: 12, left: -10, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#243049" />
                <XAxis dataKey="model" tick={{ fill: "#8a97b1", fontSize: 12 }} interval={0} />
                <YAxis domain={[0, 1]} tick={{ fill: "#8a97b1", fontSize: 12 }} />
                <Tooltip
                  contentStyle={{
                    background: "#0f1523",
                    border: "1px solid #243049",
                    borderRadius: 8,
                    boxShadow: "0 8px 24px rgba(0,0,0,0.5)",
                  }}
                />
                <Legend wrapperStyle={{ paddingTop: 10 }} />
                <Bar dataKey="accuracy" name="Accuracy" fill={CHART_COLORS[0]} radius={[4, 4, 0, 0]} />
                <Bar dataKey="precision" name="Precision" fill={CHART_COLORS[1]} radius={[4, 4, 0, 0]} />
                <Bar dataKey="recall" name="Recall" fill={CHART_COLORS[2]} radius={[4, 4, 0, 0]} />
                <Bar dataKey="f1" name="F1 Score" fill={CHART_COLORS[3]} radius={[4, 4, 0, 0]} />
              </BarChart>
            ) : (
              <BarChart data={data.bar_chart} margin={{ top: 12, right: 12, left: -10, bottom: 20 }}>
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
                <Bar
                  dataKey={activeMetric}
                  name={METRIC_LABELS[activeMetric]}
                  fill={METRIC_COLORS[activeMetric]}
                  radius={[6, 6, 0, 0]}
                >
                  {data.bar_chart.map((_entry, index) => (
                    <Cell
                      key={`cell-${index}`}
                      fill={
                        index === 0
                          ? "#4f8cff"
                          : index === 1
                          ? "#8b7dff"
                          : index === 2
                          ? "#38d9c6"
                          : "#f59e0b"
                      }
                    />
                  ))}
                </Bar>
              </BarChart>
            )}
          </ResponsiveContainer>
        </div>
      </Card>

      <Card>
        <div className="mb-6 flex flex-wrap items-center justify-between gap-4 border-b border-line pb-4">
          <div className="flex items-center gap-2">
            <Layers className="h-5 w-5 text-accent-blue" />
            <div>
              <SectionTitle
                title="Confusion Matrix Analysis"
                subtitle="Select an estimator to inspect its True vs. Predicted distribution across cognitive workload levels."
              />
            </div>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-xs text-muted font-medium">Select Model:</span>
            <select
              value={selected}
              onChange={(e) => setSelected(e.target.value)}
              className="rounded-lg border border-line bg-base-900 px-3.5 py-2 text-sm text-white font-medium outline-none focus:border-accent-blue transition-colors"
            >
              {Object.entries(data.confusion_matrices).map(([key, cm]) => (
                <option key={key} value={key}>
                  {cm.model}
                </option>
              ))}
            </select>
          </div>
        </div>

        {data.confusion_matrices[selected] && (
          <ConfusionMatrix data={data.confusion_matrices[selected]} />
        )}
      </Card>
    </div>
  );
}
