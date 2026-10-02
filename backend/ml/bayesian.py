"""Bayesian multinomial logistic regression backed by PyMC.

Design goals:
  * Provide genuine *posterior* probability estimates, not just point
    predictions, so downstream code can report epistemic uncertainty.
  * Stay tractable on a high-dimensional, tens-of-thousands-of-rows dataset
    by (a) reducing dimensionality with PCA and (b) sub-sampling rows for the
    sampler. Both are configurable via environment variables.
  * At predict time we do NOT need PyMC: we keep a thinned set of posterior
    weight draws as plain numpy arrays and vectorise the softmax over them.
"""
from __future__ import annotations

import logging

import numpy as np
from sklearn.decomposition import PCA

from config import (
    BAYES_CHAINS,
    BAYES_DRAWS,
    BAYES_MAX_FEATURES,
    BAYES_MAX_ROWS,
    BAYES_TUNE,
    RANDOM_SEED,
)

logger = logging.getLogger("neuroshield.bayesian")


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


class BayesianLogisticRegression:
    """A minimal sklearn-like Bayesian classifier.

    Attributes set after ``fit``:
      classes_        : sorted class labels (0..K-1)
      pca_            : fitted PCA used to compress features
      W_samples_      : posterior weight draws, shape (S, d, K)
      b_samples_      : posterior bias draws,   shape (S, K)
      config_         : sampling configuration actually used
    """

    def __init__(
        self,
        draws: int = BAYES_DRAWS,
        tune: int = BAYES_TUNE,
        chains: int = BAYES_CHAINS,
        max_features: int = BAYES_MAX_FEATURES,
        max_rows: int = BAYES_MAX_ROWS,
        random_seed: int = RANDOM_SEED,
    ) -> None:
        self.draws = draws
        self.tune = tune
        self.chains = chains
        self.max_features = max_features
        self.max_rows = max_rows
        self.random_seed = random_seed
        self.pca_: PCA | None = None
        self.W_samples_: np.ndarray | None = None
        self.b_samples_: np.ndarray | None = None
        self.classes_: np.ndarray | None = None
        self.config_: dict = {}
        self._trace = None  # arviz InferenceData, saved separately as .nc

    # -- fitting --------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BayesianLogisticRegression":
        import pymc as pm  # imported lazily so predict-time needs no PyMC

        X = np.asarray(X, dtype=float)
        y = np.asarray(y).astype(int)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        # Dimensionality reduction to keep the sampler tractable.
        n_comp = int(min(self.max_features, X.shape[1], max(2, X.shape[0] - 1)))
        self.pca_ = PCA(n_components=n_comp, random_state=self.random_seed)
        Xr = self.pca_.fit_transform(X)

        # Row sub-sampling for the sampler (fit still summarises full signal
        # via PCA; this only limits likelihood evaluations).
        rng = np.random.default_rng(self.random_seed)
        if Xr.shape[0] > self.max_rows:
            sel = rng.choice(Xr.shape[0], size=self.max_rows, replace=False)
            Xs, ys = Xr[sel], y[sel]
        else:
            Xs, ys = Xr, y

        d = Xs.shape[1]
        logger.info(
            "Bayesian fit: %d rows x %d comps, %d classes, draws=%d tune=%d chains=%d",
            Xs.shape[0], d, n_classes, self.draws, self.tune, self.chains,
        )

        with pm.Model() as model:  # noqa: F841
            sigma = pm.HalfNormal("sigma", sigma=2.0)
            W = pm.Normal("W", mu=0.0, sigma=sigma, shape=(d, n_classes))
            b = pm.Normal("b", mu=0.0, sigma=2.0, shape=(n_classes,))
            eta = pm.math.dot(Xs, W) + b
            p = pm.math.softmax(eta, axis=1)
            pm.Categorical("y_obs", p=p, observed=ys)
            idata = pm.sample(
                draws=self.draws,
                tune=self.tune,
                chains=self.chains,
                cores=1,
                random_seed=self.random_seed,
                progressbar=False,
                target_accept=0.9,
                compute_convergence_checks=False,
            )

        self._trace = idata
        # Flatten (chain, draw) -> sample dimension.
        W_post = idata.posterior["W"].stack(sample=("chain", "draw")).values
        b_post = idata.posterior["b"].stack(sample=("chain", "draw")).values
        # Reorder to (S, d, K) and (S, K).
        self.W_samples_ = np.moveaxis(W_post, -1, 0)
        self.b_samples_ = np.moveaxis(b_post, -1, 0)

        self.config_ = {
            "draws": self.draws,
            "tune": self.tune,
            "chains": self.chains,
            "pca_components": d,
            "sampler_rows": int(Xs.shape[0]),
            "posterior_samples": int(self.W_samples_.shape[0]),
        }
        return self

    # -- prediction -----------------------------------------------------------
    def _posterior_probs(self, X: np.ndarray) -> np.ndarray:
        """Return per-posterior-sample probabilities, shape (S, n, K)."""
        assert self.pca_ is not None and self.W_samples_ is not None
        Xr = self.pca_.transform(np.asarray(X, dtype=float))
        # eta: (S, n, K) = Xr(n,d) . W(S,d,K) + b(S,K)
        eta = np.einsum("nd,sdk->snk", Xr, self.W_samples_)
        eta = eta + self.b_samples_[:, None, :]
        return _softmax(eta)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Posterior-mean class probabilities, shape (n, K)."""
        return self._posterior_probs(X).mean(axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def predict_uncertainty(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """Return predictive-probability-based uncertainty measures.

        * uncertainty      : 1 - max posterior-mean class probability
                             (bounded [0, (K-1)/K]; equals min(p,1-p) when binary).
        * epistemic_std    : std across posterior draws of the top-class prob.
        * entropy          : Shannon entropy of the mean probability vector.
        """
        post = self._posterior_probs(X)          # (S, n, K)
        mean_p = post.mean(axis=0)               # (n, K)
        top_idx = np.argmax(mean_p, axis=1)      # (n,)
        top_p = mean_p[np.arange(len(top_idx)), top_idx]
        uncertainty = 1.0 - top_p
        # Epistemic: variability of the chosen class prob across the posterior.
        top_samples = post[:, np.arange(len(top_idx)), top_idx]  # (S, n)
        epistemic_std = top_samples.std(axis=0)
        eps = 1e-12
        entropy = -np.sum(mean_p * np.log(mean_p + eps), axis=1)
        return {
            "uncertainty": uncertainty,
            "epistemic_std": epistemic_std,
            "entropy": entropy,
            "max_probability": top_p,
        }
