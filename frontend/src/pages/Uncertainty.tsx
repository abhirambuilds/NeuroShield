import { useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";
import { AlertCircle, Gauge, Layers, Sparkles } from "lucide-react";
import { api, apiError } from "../services/api";
import { Card, ErrorBox, Loading, SectionTitle, StatCard } from "../components/ui";
import type { UncertaintyResponse, UncertaintySample } from "../types";

type SortKey = keyof Pick<UncertaintySample, "uncertainty" | "probability" | "sample" | "prediction">;

export default function Uncertainty() {
  const [data, setData] = useState<UncertaintyResponse | null>(null);
  const [err, setErr] = useState("");
  const [loading, setLoading] = useState(true);
  const [sortKey, setSortKey] = useState<SortKey>("uncertainty");
  const [asc, setAsc] = useState(false);
  const [search, setSearch] = useState("");

  useEffect(() => {
    (async () => {
      try {
        setData(await api.uncertainty());
      } catch (e) {
        setErr(apiError(e));
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  const sorted = useMemo(() => {
    if (!data) return [];
    let rows = [...data.high_uncertainty_samples];
    if (search.trim()) {
      const q = search.toLowerCase();
      rows = rows.filter(
        (r) =>
          r.sample.toString().includes(q) ||
          r.prediction.toLowerCase().includes(q)
      );
    }
    rows.sort((a, b) => {
      const av = a[sortKey];
      const bv = b[sortKey];
      if (typeof av === "string" && typeof bv === "string") {
        return asc ? av.localeCompare(bv) : bv.localeCompare(av);
      }
      return asc ? Number(av) - Number(bv) : Number(bv) - Number(av);
    });
    return rows;
  }, [data, sortKey, asc, search]);

  // Scatter plot data pairing probability and uncertainty
  const scatterPoints = useMemo(() => {
    if (!data) return [];
    return data.high_uncertainty_samples.map((s) => ({
      sample: s.sample,
      probability: s.probability,
      uncertainty: s.uncertainty,
      prediction: s.prediction,
    }));
  }, [data]);

  if (loading) return <Loading label="Computing Bayesian posterior uncertainty distributions…" />;
  if (err) return <ErrorBox message={err} />;
  if (!data) return null;

  const uHist = toBars(data.uncertainty_histogram);
  const pHist = toBars(data.probability_histogram);

  function setSort(k: SortKey) {
    if (k === sortKey) setAsc((v) => !v);
    else {
      setSortKey(k);
      setAsc(false);
    }
  }

  return (
    <div className="space-y-8">
      <SectionTitle
        title="Bayesian Uncertainty Analysis"
        subtitle={`Empirical analysis of the Bayesian predictive distribution across all ${data.n_predictions.toLocaleString()} held-out test predictions.`}
      />

      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        <StatCard
          label="Mean Uncertainty"
          value={data.mean_uncertainty.toFixed(4)}
          hint={`Std dev: ${data.std_uncertainty.toFixed(4)}`}
          icon={<Gauge className="h-6 w-6 text-accent-purple" />}
        />
        <StatCard
          label="Median Uncertainty"
          value={data.median_uncertainty.toFixed(4)}
          hint="50th percentile"
          icon={<Sparkles className="h-6 w-6 text-accent-blue" />}
        />
        <StatCard
          label="Maximum Uncertainty"
          value={data.max_uncertainty.toFixed(4)}
          hint="Boundary: 0.50 (binary entropy max)"
          icon={<AlertCircle className="h-6 w-6 text-amber-400" />}
        />
        <StatCard
          label="Minimum Uncertainty"
          value={data.min_uncertainty.toFixed(4)}
          hint="Highest confidence prediction"
          icon={<Layers className="h-6 w-6 text-accent-cyan" />}
        />
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <Card className="space-y-2">
          <SectionTitle
            title="Uncertainty Distribution"
            subtitle="Frequency histogram of posterior predictive uncertainty values"
          />
          <Histo data={uHist} color="#8b7dff" xLabel="Uncertainty Band" />
        </Card>

        <Card className="space-y-2">
          <SectionTitle
            title="Predicted Probability Distribution"
            subtitle="Frequency histogram of winning posterior class probabilities"
          />
          <Histo data={pHist} color="#4f8cff" xLabel="Max Probability Band" />
        </Card>
      </div>

      <Card className="space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-2 border-b border-line pb-3">
          <div>
            <SectionTitle
              title="Probability vs. Uncertainty Relationship"
              subtitle="Scatter visualization of sample predictions mapping winning probability against Bayesian uncertainty."
            />
          </div>
          <div className="flex items-center gap-1.5 text-xs text-muted font-mono">
            <span className="h-2.5 w-2.5 rounded-full bg-accent-purple inline-block" />
            Theoretical constraint: Uncertainty = 1 - P(max)
          </div>
        </div>
        <div className="h-72 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart margin={{ top: 10, right: 20, bottom: 20, left: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#243049" />
              <XAxis
                type="number"
                dataKey="probability"
                name="Probability"
                domain={[0.5, 1]}
                tick={{ fill: "#8a97b1", fontSize: 11 }}
                unit=""
                label={{
                  value: "Winning Class Probability",
                  position: "insideBottom",
                  offset: -10,
                  fill: "#8a97b1",
                  fontSize: 12,
                }}
              />
              <YAxis
                type="number"
                dataKey="uncertainty"
                name="Uncertainty"
                domain={[0, 0.5]}
                tick={{ fill: "#8a97b1", fontSize: 11 }}
                label={{
                  value: "Uncertainty",
                  angle: -90,
                  position: "insideLeft",
                  fill: "#8a97b1",
                  fontSize: 12,
                }}
              />
              <ZAxis range={[60, 60]} />
              <Tooltip
                cursor={{ strokeDasharray: "3 3" }}
                contentStyle={{
                  background: "#0f1523",
                  border: "1px solid #243049",
                  borderRadius: 8,
                }}
                formatter={(val: any, name: string) => [
                  Number(val).toFixed(4),
                  name,
                ]}
              />
              <Scatter name="Test Predictions" data={scatterPoints} fill="#8b7dff" />
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      </Card>

      <Card>
        <div className="mb-4 flex flex-wrap items-center justify-between gap-4 border-b border-line pb-4">
          <div>
            <SectionTitle
              title="High-Uncertainty Predictions"
              subtitle="Held-out observations where the Bayesian posterior distribution exhibited highest ambiguity."
            />
          </div>
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search by sample ID or label..."
            className="rounded-lg border border-line bg-base-900 px-3 py-1.5 text-xs text-white outline-none focus:border-accent-blue"
          />
        </div>

        <div className="overflow-x-auto">
          <table className="min-w-full text-sm">
            <thead>
              <tr className="border-b border-line text-left text-xs uppercase tracking-wider text-muted">
                <Th onClick={() => setSort("sample")} active={sortKey === "sample"} asc={asc}>
                  Sample ID
                </Th>
                <Th onClick={() => setSort("prediction")} active={sortKey === "prediction"} asc={asc}>
                  Posterior Prediction
                </Th>
                <Th onClick={() => setSort("probability")} active={sortKey === "probability"} asc={asc} right>
                  Posterior Probability
                </Th>
                <Th onClick={() => setSort("uncertainty")} active={sortKey === "uncertainty"} asc={asc} right>
                  Uncertainty Score
                </Th>
              </tr>
            </thead>
            <tbody className="divide-y divide-line/40">
              {sorted.map((s) => (
                <tr key={s.sample} className="hover:bg-base-700/30 transition-colors">
                  <td className="p-3 font-mono text-xs text-white">#{s.sample}</td>
                  <td className="p-3 text-xs">
                    <span
                      className={`inline-block px-2 py-0.5 rounded text-[11px] font-mono font-medium ${
                        s.prediction === "1" || s.prediction.includes("High")
                          ? "bg-purple-500/15 text-purple-300 border border-purple-500/30"
                          : "bg-cyan-500/15 text-cyan-300 border border-cyan-500/30"
                      }`}
                    >
                      {s.prediction === "1"
                        ? "High Cognitive Workload"
                        : s.prediction === "0"
                        ? "Low Cognitive Workload"
                        : s.prediction}
                    </span>
                  </td>
                  <td className="p-3 text-right font-mono text-xs text-white font-semibold">
                    {(s.probability * 100).toFixed(2)}%
                  </td>
                  <td className="p-3 text-right font-mono text-xs text-accent-purple font-semibold">
                    {(s.uncertainty * 100).toFixed(2)}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}

function toBars(h: { counts: number[]; bin_edges: number[] }) {
  return h.counts.map((c, i) => ({
    bin: `${h.bin_edges[i].toFixed(2)}`,
    count: c,
  }));
}

function Histo({
  data,
  color,
  xLabel,
}: {
  data: { bin: string; count: number }[];
  color: string;
  xLabel?: string;
}) {
  return (
    <div className="h-64">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 8, right: 8, bottom: 20, left: -10 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#243049" />
          <XAxis
            dataKey="bin"
            tick={{ fill: "#8a97b1", fontSize: 10 }}
            interval={1}
            label={
              xLabel
                ? { value: xLabel, position: "insideBottom", offset: -10, fill: "#8a97b1", fontSize: 11 }
                : undefined
            }
          />
          <YAxis tick={{ fill: "#8a97b1", fontSize: 11 }} />
          <Tooltip
            contentStyle={{
              background: "#0f1523",
              border: "1px solid #243049",
              borderRadius: 8,
            }}
          />
          <Bar dataKey="count" name="Frequency" fill={color} radius={[4, 4, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

function Th({
  children,
  onClick,
  active,
  asc,
  right,
}: {
  children: ReactNode;
  onClick: () => void;
  active: boolean;
  asc: boolean;
  right?: boolean;
}) {
  return (
    <th
      onClick={onClick}
      className={`cursor-pointer select-none p-3 font-semibold ${
        right ? "text-right" : ""
      } ${active ? "text-accent-blue" : "hover:text-white"}`}
    >
      <div className={`inline-flex items-center gap-1 ${right ? "justify-end" : ""}`}>
        <span>{children}</span>
        {active && <span>{asc ? "▲" : "▼"}</span>}
      </div>
    </th>
  );
}
