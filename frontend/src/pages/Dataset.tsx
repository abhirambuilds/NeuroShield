import { useEffect, useState } from "react";
import type { ReactNode } from "react";
import {
  Cell,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
} from "recharts";
import { api, apiError } from "../services/api";
import {
  Card,
  CHART_COLORS,
  ErrorBox,
  Loading,
  SectionTitle,
  StatCard,
} from "../components/ui";
import type { DatasetInfo, DatasetPage } from "../types";

export default function Dataset() {
  const [info, setInfo] = useState<DatasetInfo | null>(null);
  const [page, setPage] = useState<DatasetPage | null>(null);
  const [pageNum, setPageNum] = useState(1);
  const [query, setQuery] = useState("");
  const [err, setErr] = useState("");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    (async () => {
      setErr("");
      try {
        setInfo(await api.datasetInfo());
      } catch (e) {
        setErr(apiError(e));
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  useEffect(() => {
    (async () => {
      try {
        setPage(await api.datasetRows(pageNum, 25));
      } catch (e) {
        setErr(apiError(e));
      }
    })();
  }, [pageNum]);

  if (loading) return <Loading label="Loading dataset…" />;
  if (err) return <ErrorBox message={err} />;
  if (!info) return <ErrorBox message="Dataset unavailable." />;

  const dist = Object.entries(info.class_distribution).map(([name, value]) => ({
    name,
    value,
  }));

  const cols = page?.columns ?? [];
  const visibleCols = query
    ? cols.filter((c) => c.toLowerCase().includes(query.toLowerCase()))
    : cols.slice(0, 12);

  return (
    <div className="space-y-8">
      <SectionTitle
        title="Dataset explorer"
        subtitle={`${info.dataset_name} — target column: ${info.target_column}`}
      />

      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        <StatCard label="Rows" value={info.n_samples.toLocaleString()} />
        <StatCard label="Columns" value={(info.n_features + 1).toLocaleString()} hint={`${info.n_features} features`} />
        <StatCard label="Missing values" value={info.missing_values.toLocaleString()} />
        <StatCard label="Duplicate rows" value={info.duplicate_rows.toLocaleString()} />
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <Card>
          <SectionTitle title="Class distribution" />
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie data={dist} dataKey="value" nameKey="name" outerRadius={90} label>
                  {dist.map((_, i) => (
                    <Cell key={i} fill={CHART_COLORS[i % CHART_COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip
                  contentStyle={{
                    background: "#0f1523",
                    border: "1px solid #243049",
                    borderRadius: 8,
                  }}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </Card>

        <Card>
          <SectionTitle title="Feature types" />
          <div className="space-y-3 text-sm">
            <Row label="Numeric features" value={info.numeric_features} />
            <Row label="Categorical features" value={info.categorical_features} />
            <Row label="Train samples" value={info.train_samples ?? "—"} />
            <Row label="Test samples" value={info.test_samples ?? "—"} />
            <Row label="Classes" value={info.classes.join(", ")} />
          </div>
        </Card>
      </div>

      <Card>
        <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
          <SectionTitle title="Sample records" subtitle="Backend-paginated (25 rows/page)." />
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Filter columns…"
            className="rounded-lg border border-line bg-base-900 px-3 py-2 text-sm outline-none focus:border-accent-blue"
          />
        </div>
        <div className="overflow-x-auto">
          <table className="min-w-full text-xs">
            <thead>
              <tr className="text-left text-muted">
                {visibleCols.map((c) => (
                  <th key={c} className="whitespace-nowrap p-2 font-mono">
                    {c}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {(page?.rows ?? []).map((row, i) => (
                <tr key={i} className="border-t border-line/60">
                  {visibleCols.map((c) => (
                    <td key={c} className="whitespace-nowrap p-2 font-mono text-muted">
                      {formatCell((row as Record<string, unknown>)[c])}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {page && (
          <div className="mt-4 flex items-center justify-between text-sm text-muted">
            <span>
              Page {page.page} / {page.total_pages} · {page.total_rows.toLocaleString()} rows
            </span>
            <div className="flex gap-2">
              <button
                className="btn-ghost"
                disabled={pageNum <= 1}
                onClick={() => setPageNum((p) => p - 1)}
              >
                Prev
              </button>
              <button
                className="btn-ghost"
                disabled={pageNum >= (page.total_pages || 1)}
                onClick={() => setPageNum((p) => p + 1)}
              >
                Next
              </button>
            </div>
          </div>
        )}
      </Card>
    </div>
  );
}

function Row({ label, value }: { label: string; value: ReactNode }) {
  return (
    <div className="flex items-center justify-between border-b border-line/50 pb-2">
      <span className="text-muted">{label}</span>
      <span className="font-mono text-white">{value}</span>
    </div>
  );
}

function formatCell(v: unknown): string {
  if (v === null || v === undefined) return "—";
  if (typeof v === "number") return Number.isInteger(v) ? String(v) : v.toFixed(4);
  return String(v);
}
