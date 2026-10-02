import { Card, SectionTitle } from "../components/ui";

export default function About() {
  return (
    <div className="space-y-6">
      <SectionTitle
        title="About NeuroShield"
        subtitle="Probabilistic Cognitive Workload & Fatigue Analysis"
      />

      <Card>
        <h3 className="mb-2 font-semibold text-white">Problem</h3>
        <p className="text-sm leading-relaxed text-muted">
          Human cognitive performance can vary significantly during prolonged
          interaction with digital systems. Traditional productivity systems often
          rely on task completion time, error counts, and simple productivity
          metrics. These may not fully capture the complex behavioural patterns
          associated with cognitive workload.
        </p>
      </Card>

      <Card>
        <h3 className="mb-2 font-semibold text-white">Proposed solution</h3>
        <p className="text-sm leading-relaxed text-muted">
          NeuroShield evaluates whether human–computer interaction (HCI) behavioural
          features can be used to classify cognitive workload/fatigue levels, using
          probabilistic and machine-learning methods. It reports predictions,
          calibrated probabilities, and — for the Bayesian model — predictive
          uncertainty, rather than issuing confident diagnostic claims.
        </p>
      </Card>

      <Card>
        <h3 className="mb-2 font-semibold text-white">Models compared</h3>
        <ul className="space-y-1 text-sm text-muted">
          <li>• Linear Discriminant Analysis (generative)</li>
          <li>• Bayesian Logistic Regression (probabilistic, PyMC)</li>
          <li>• Linear Support Vector Machine (discriminative)</li>
          <li>• Kernel Support Vector Machine (RBF)</li>
        </ul>
      </Card>

      <Card>
        <h3 className="mb-2 font-semibold text-white">Research questions</h3>
        <ol className="list-decimal space-y-1 pl-5 text-sm text-muted">
          <li>
            Can nonlinear relationships among behavioural features improve
            classification of cognitive workload?
          </li>
          <li>
            Can Bayesian probability estimates provide more informative,
            uncertainty-aware predictions?
          </li>
          <li>
            How do generative, discriminative, Bayesian, and kernel-based methods
            compare on this dataset?
          </li>
        </ol>
      </Card>

      <Card>
        <h3 className="mb-2 font-semibold text-white">Scientific note</h3>
        <p className="text-sm leading-relaxed text-muted">
          Reported metrics are experimental results measured on a held-out split of
          the loaded dataset. NeuroShield does not claim to accurately detect
          fatigue, and does not assert that any model is objectively best — it
          presents the measured performance and lets you compare the approaches.
        </p>
      </Card>
    </div>
  );
}
