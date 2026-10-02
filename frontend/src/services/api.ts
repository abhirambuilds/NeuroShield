import axios, { AxiosError } from "axios";
import type {
  ComparisonResponse,
  DatasetInfo,
  DatasetPage,
  EvaluationResponse,
  Health,
  ModelsList,
  PredictionResult,
  TrainStatus,
  UncertaintyResponse,
} from "../types";

// Same-origin: Vite proxies /api to the backend in dev; in production the
// backend can serve behind the same host or via VITE_API_BASE.
const baseURL = import.meta.env.VITE_API_BASE || "";

const client = axios.create({ baseURL, timeout: 60000 });

/** Extract a clean, user-facing message from an Axios error. */
export function apiError(err: unknown): string {
  const e = err as AxiosError<{ detail?: string }>;
  if (e.response?.data?.detail) return e.response.data.detail;
  if (e.message) return e.message;
  return "Unexpected error contacting the backend.";
}

export const api = {
  health: () => client.get<Health>("/api/health").then((r) => r.data),

  datasetInfo: () =>
    client.get<DatasetInfo>("/api/dataset/info").then((r) => r.data),

  datasetRows: (page: number, pageSize: number) =>
    client
      .get<DatasetPage>("/api/dataset/rows", {
        params: { page, page_size: pageSize },
      })
      .then((r) => r.data),

  train: (force = false) =>
    client
      .post<{ status: string; detail: string }>("/api/models/train", {
        force,
        background: true,
      })
      .then((r) => r.data),

  trainStatus: () =>
    client.get<TrainStatus>("/api/models/train/status").then((r) => r.data),

  comparison: () =>
    client.get<ComparisonResponse>("/api/models/comparison").then((r) => r.data),

  models: () => client.get<ModelsList>("/api/models").then((r) => r.data),

  evaluation: () =>
    client.get<EvaluationResponse>("/api/evaluation").then((r) => r.data),

  uncertainty: () =>
    client.get<UncertaintyResponse>("/api/uncertainty").then((r) => r.data),

  predictSchema: () =>
    client.get<any>("/api/predict/schema").then((r) => r.data),

  samples: (limit = 50) =>
    client
      .get<any>("/api/predict/samples", { params: { limit } })
      .then((r) => r.data),

  predict: (body: {
    model: string;
    sample_id?: number;
    features?: Record<string, unknown>;
  }) => client.post<PredictionResult>("/api/predict", body).then((r) => r.data),
};
