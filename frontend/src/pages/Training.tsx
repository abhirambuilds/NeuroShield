import { useEffect, useState } from "react";
import { CheckCircle2, Clock, Cpu, HardDrive, ShieldCheck } from "lucide-react";
import { api, apiError } from "../services/api";
import { Card, ErrorBox, Loading, SectionTitle, StatCard } from "../components/ui";
import TrainingPanel from "../components/TrainingPanel";
import type { ModelsList } from "../types";

export default function Training() {
  const [models, setModels] = useState<ModelsList | null>(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState("");

  async function load() {
    setLoading(true);
    setErr("");
    try {
      const data = await api.models();
      setModels(data);
    } catch (e) {
      setErr(apiError(e));
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    load();
  }, []);

  if (loading) return <Loading label="Checking model training status…" />;

  const trainedCount = models?.models.filter((m) => m.artifact_exists).length ?? 0;

  return (
    <div className="space-y-8">
      <SectionTitle
        title="Model Training"
        subtitle="Train the four cognitive workload classifiers on the multimodal HCI dataset. Trained models and evaluation matrices are persisted to disk."
      />

      {err && <ErrorBox message={err} />}

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <StatCard
          label="Trained Models"
          value={`${trainedCount} / 4`}
          hint={trainedCount === 4 ? "All models ready" : "Training required"}
          icon={<Cpu className="h-6 w-6 text-accent-blue" />}
        />
        <StatCard
          label="Artifact Cache"
          value={models?.trained ? "Active" : "Uninitialized"}
          hint="Reused across app restarts"
          icon={<HardDrive className="h-6 w-6 text-accent-purple" />}
        />
        <StatCard
          label="Last Trained"
          value={
            models?.trained_at
              ? new Date(models.trained_at * 1000).toLocaleTimeString([], {
                  hour: "2-digit",
                  minute: "2-digit",
                })
              : "Never"
          }
          hint={
            models?.trained_at
              ? new Date(models.trained_at * 1000).toLocaleDateString()
              : "Awaiting run"
          }
          icon={<Clock className="h-6 w-6 text-accent-cyan" />}
        />
        <StatCard
          label="Evaluation Status"
          value={models?.trained ? "Ready" : "Pending"}
          hint="Metrics generated on held-out split"
          icon={<ShieldCheck className="h-6 w-6 text-emerald-400" />}
        />
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        <Card className="lg:col-span-2">
          <SectionTitle
            title="Training Pipeline"
            subtitle="Click below to run the end-to-end pipeline: load Kaggle dataset, preprocess with stratified split, fit all 4 models, evaluate, and persist artifacts."
          />
          <TrainingPanel onDone={load} />
        </Card>

        <Card className="space-y-4">
          <SectionTitle
            title="Model Registry"
            subtitle="Persistence &amp; status of all 4 estimators"
          />
          <div className="space-y-3">
            {[
              {
                key: "lda",
                name: "Linear Discriminant Analysis",
                type: "Generative Linear Classifier",
                color: "border-blue-500/30 bg-blue-500/10",
              },
              {
                key: "bayesian",
                name: "Bayesian Logistic Regression",
                type: "PyMC Probabilistic NUTS MCMC",
                color: "border-purple-500/30 bg-purple-500/10",
              },
              {
                key: "linear_svm",
                name: "Linear SVM",
                type: "Calibrated Linear Boundary",
                color: "border-cyan-500/30 bg-cyan-500/10",
              },
              {
                key: "kernel_svm",
                name: "Kernel SVM (RBF)",
                type: "Nonlinear Radial Basis Function",
                color: "border-amber-500/30 bg-amber-500/10",
              },
            ].map((m) => {
              const info = models?.models.find((mod) => mod.key === m.key);
              const ready = info?.artifact_exists ?? false;
              return (
                <div
                  key={m.key}
                  className={`rounded-lg border p-3 transition-colors ${m.color}`}
                >
                  <div className="flex items-center justify-between">
                    <span className="font-semibold text-white">{m.name}</span>
                    {ready ? (
                      <span className="inline-flex items-center gap-1 text-xs text-emerald-400 font-mono">
                        <CheckCircle2 className="h-3.5 w-3.5" /> Ready
                      </span>
                    ) : (
                      <span className="text-xs text-muted font-mono">Not trained</span>
                    )}
                  </div>
                  <div className="mt-1 text-xs text-muted">{m.type}</div>
                </div>
              );
            })}
          </div>

          <div className="rounded-lg border border-line bg-base-900/60 p-3 text-xs text-muted space-y-1.5">
            <div className="font-semibold text-white flex items-center gap-1.5">
              <HardDrive className="h-3.5 w-3.5 text-accent-blue" />
              Caching Guarantee
            </div>
            <p>
              Trained models and test evaluation results are saved under{" "}
              <code className="text-accent-cyan font-mono">backend/models/</code> and{" "}
              <code className="text-accent-cyan font-mono">backend/results/</code>. Models are not retrained when restarting the application.
            </p>
          </div>
        </Card>
      </div>
    </div>
  );
}
