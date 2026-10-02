import { useEffect, useState } from "react";
import { api, apiError } from "../services/api";
import { Card, ErrorBox, Loading, SectionTitle } from "../components/ui";
import ConfusionMatrix from "../components/ConfusionMatrix";
import TrainingPanel from "../components/TrainingPanel";
import type { ComparisonResponse, EvaluationResponse, ModelsList } from "../types";

export default function Models() {
  const [list, setList] = useState<ModelsList | null>(null);
  const [cmp, setCmp] = useState<ComparisonResponse | null>(null);
  const [evalData, setEvalData] = useState<EvaluationResponse | null>(null);
  const [selected, setSelected] = useState<string>("lda");
  const [err, setErr] = useState("");
  const [loading, setLoading] = useState(true);

  async function load() {
    setErr("");
    setLoading(true);
    try {
      const l = await api.models();
      setList(l);
      if (l.trained) {
        setCmp(await api.comparison());
        setEvalData(await api.evaluation());
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

  if (loading) return <Loading label="Loading models…" />;

  return (
    <div className="space-y-8">
      <SectionTitle
        title="Models"
        subtitle="Train, compare, and inspect the four classifiers. Metrics are weighted-averaged over classes."
      />

      {err && <ErrorBox message={err} />}

      <Card>
        <SectionTitle title="Training" subtitle="Artifacts are cached on disk and reused on subsequent runs." />
        <TrainingPanel onDone={load} />
        {list?.bayesian_config && Object.keys(list.bayesian_config).length > 0 && (
          <p className="mt-3 font-mono text-xs text-muted">
            Bayesian sampler config: {JSON.stringify(list.bayesian_config)}
          </p>
        )}
      </Card>

      {cmp && (
        <Card>
          <SectionTitle
            title="Model comparison"
            subtitle={`Averaging: ${cmp.averaging}. Higher is better; values are experimental, not claims of real-world fatigue detection.`}
          />
          <div className="overflow-x-auto">
            <table className="min-w-full text-sm">
              <thead>
                <tr className="text-left text-muted">
                  <th className="p-2">Model</th>
                  <th className="p-2 text-right">Accuracy</th>
                  <th className="p-2 text-right">Precision</th>
                  <th className="p-2 text-right">Recall</th>
                  <th className="p-2 text-right">F1</th>
                  <th className="p-2 text-right">ROC-AUC</th>
                  <th className="p-2 text-right">Train (s)</th>
                </tr>
              </thead>
              <tbody>
                {cmp.models.map((m) => (
                  <tr key={m.model_key} className="border-t border-line/60">
                    <td className="p-2 font-medium text-white">{m.model}</td>
                    <td className="p-2 text-right font-mono">{m.accuracy.toFixed(4)}</td>
                    <td className="p-2 text-right font-mono">{m.precision.toFixed(4)}</td>
                    <td className="p-2 text-right font-mono">{m.recall.toFixed(4)}</td>
                    <td className="p-2 text-right font-mono">{m.f1.toFixed(4)}</td>
                    <td className="p-2 text-right font-mono">
                      {m.roc_auc != null ? m.roc_auc.toFixed(4) : "—"}
                    </td>
                    <td className="p-2 text-right font-mono">
                      {m.training_time_seconds != null ? m.training_time_seconds.toFixed(2) : "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {evalData && (
        <Card>
          <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
            <SectionTitle title="Confusion matrix" subtitle="Select a model to view its held-out confusion matrix." />
            <select
              value={selected}
              onChange={(e) => setSelected(e.target.value)}
              className="rounded-lg border border-line bg-base-900 px-3 py-2 text-sm outline-none focus:border-accent-blue"
            >
              {Object.entries(evalData.confusion_matrices).map(([key, cm]) => (
                <option key={key} value={key}>
                  {cm.model}
                </option>
              ))}
            </select>
          </div>
          {evalData.confusion_matrices[selected] && (
            <ConfusionMatrix data={evalData.confusion_matrices[selected]} />
          )}
        </Card>
      )}
    </div>
  );
}
