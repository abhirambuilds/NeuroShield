// Shared TypeScript types mirroring the FastAPI responses.

export interface Health {
  status: string;
  models_trained: boolean;
  version: string;
}

export interface DatasetInfo {
  dataset_name: string;
  main_file?: string;
  n_samples: number;
  n_features: number;
  train_samples: number | null;
  test_samples: number | null;
  missing_values: number;
  classes: string[];
  class_distribution: Record<string, number>;
  numeric_features: number;
  categorical_features: number;
  duplicate_rows: number;
  target_column: string;
  source: string;
}

export interface DatasetPage {
  page: number;
  page_size: number;
  total_rows: number;
  total_pages: number;
  columns: string[];
  rows: Record<string, unknown>[];
}

export interface ModelMetrics {
  model: string;
  model_key: string;
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  roc_auc: number | null;
  training_time_seconds: number | null;
  averaging: string;
}

export interface ComparisonResponse {
  models: ModelMetrics[];
  averaging: string;
}

export interface ConfusionEntry {
  model: string;
  matrix: number[][];
  labels: string[];
}

export interface EvaluationResponse {
  averaging: string;
  metrics: Record<string, any>;
  bar_chart: Array<Record<string, number | string>>;
  confusion_matrices: Record<string, ConfusionEntry>;
  highlights: Record<string, { model: string; value: number }>;
}

export interface Histogram {
  counts: number[];
  bin_edges: number[];
}

export interface UncertaintySample {
  sample: number;
  prediction: string;
  probability: number;
  uncertainty: number;
}

export interface UncertaintyResponse {
  n_predictions: number;
  mean_uncertainty: number;
  median_uncertainty: number;
  max_uncertainty: number;
  min_uncertainty: number;
  std_uncertainty: number;
  mean_probability: number;
  std_probability: number;
  uncertainty_histogram: Histogram;
  probability_histogram: Histogram;
  high_uncertainty_samples: UncertaintySample[];
  definition: string;
  config?: Record<string, unknown>;
}

export interface PredictionResult {
  prediction_index: number;
  prediction_class: string;
  prediction_label?: string;
  probability: number | null;
  uncertainty: number | null;
  epistemic_std: number | null;
  confidence_interpretation: string;
  model_key: string;
  model: string;
  probabilities: Record<string, number> | null;
  probabilities_breakdown?: {
    class: string;
    label: string;
    probability: number;
    percentage: number;
  }[] | null;
  selection_reason: string;
  input_mode: string;
  sample_id?: number;
  true_label?: string;
  true_label_display?: string;
}

export interface TrainStatus {
  status: "idle" | "running" | "completed" | "failed";
  steps: { name: string; status: string; t: number }[];
  message: string;
  started_at: number | null;
  finished_at: number | null;
}

export interface ModelsList {
  trained: boolean;
  trained_at?: number;
  selection_metric?: string;
  bayesian_config?: Record<string, unknown>;
  class_labels?: string[];
  models: { key: string; name: string; artifact_exists: boolean }[];
}
