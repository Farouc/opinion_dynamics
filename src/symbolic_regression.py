"""Simple symbolic-regression style model search via sparse basis selection."""

from __future__ import annotations

import itertools
import os
from pathlib import Path
from typing import Any, Callable

import numpy as np


def _basis_functions() -> dict[str, Callable[[np.ndarray], np.ndarray]]:
    return {
        "1": lambda r: np.ones_like(r, dtype=float),
        "rho": lambda r: r,
        "rho2": lambda r: r**2,
        "inv_rho": lambda r: 1.0 / r,
        "inv_rho2": lambda r: 1.0 / (r**2),
        "log_inv_rho": lambda r: np.log(1.0 / r),
        "inv_rho_log_inv_rho": lambda r: np.log(1.0 / r) / r,
    }


def _human_term(name: str) -> str:
    mapping = {
        "1": "1",
        "rho": "rho",
        "rho2": "rho^2",
        "inv_rho": "1/rho",
        "inv_rho2": "1/rho^2",
        "log_inv_rho": "log(1/rho)",
        "inv_rho_log_inv_rho": "log(1/rho)/rho",
    }
    return mapping[name]


def _design_matrix(rho: np.ndarray, terms: list[str]) -> np.ndarray:
    funcs = _basis_functions()
    cols = [funcs[t](rho) for t in terms]
    return np.column_stack(cols)


def _scores(y_true: np.ndarray, y_pred: np.ndarray, n_params: int) -> dict[str, float]:
    n = y_true.size
    resid = y_true - y_pred
    rss = float(np.sum(resid**2))
    tss = float(np.sum((y_true - np.mean(y_true)) ** 2))

    eps = 1e-15
    mse = rss / max(n, 1)
    rmse = float(np.sqrt(mse))
    r2 = float(1.0 - rss / (tss + eps))

    # Information criteria for model-complexity control.
    aic = float(n * np.log(mse + eps) + 2 * n_params)
    bic = float(n * np.log(mse + eps) + n_params * np.log(max(n, 1)))

    return {
        "rss": rss,
        "rmse": rmse,
        "r2": r2,
        "aic": aic,
        "bic": bic,
    }


def fit_sparse_symbolic_regression(
    rho,
    target,
    max_terms: int = 3,
    criterion: str = "bic",
) -> dict:
    """Fit a symbolic-like model by sparse linear combination of basis terms.

    Parameters
    ----------
    rho : array-like
        Input rho values (must be > 0).
    target : array-like
        Target values (e.g., mean stopping times).
    max_terms : int
        Maximum total number of terms, including the constant term.
    criterion : {'bic', 'aic'}
        Model-selection score.
    """
    r = np.asarray(rho, dtype=float)
    y = np.asarray(target, dtype=float)

    if r.ndim != 1 or y.ndim != 1:
        raise ValueError("rho and target must be 1D arrays.")
    if r.size != y.size:
        raise ValueError("rho and target must have equal length.")
    if np.any(r <= 0.0):
        raise ValueError("rho values must be strictly positive.")
    if max_terms < 1:
        raise ValueError("max_terms must be >= 1.")
    if criterion not in {"bic", "aic"}:
        raise ValueError("criterion must be either 'bic' or 'aic'.")

    all_terms = [
        "rho",
        "rho2",
        "inv_rho",
        "inv_rho2",
        "log_inv_rho",
        "inv_rho_log_inv_rho",
    ]

    best: dict | None = None

    for k_extra in range(0, max_terms):
        for combo in itertools.combinations(all_terms, k_extra):
            terms = ["1", *combo]
            X = _design_matrix(r, terms)
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            y_pred = X @ coef

            metrics = _scores(y, y_pred, n_params=len(terms))
            score = metrics[criterion]

            candidate = {
                "terms": terms,
                "coefficients": coef.astype(float).tolist(),
                "metrics": metrics,
                "selection_score": float(score),
                "criterion": criterion,
            }

            if best is None or score < best["selection_score"]:
                best = candidate

    if best is None:
        raise RuntimeError("Model search failed to produce any candidate.")

    pieces = []
    for c, t in zip(best["coefficients"], best["terms"]):
        if abs(c) < 1e-12:
            continue
        pieces.append(f"({c:.6g})*{_human_term(t)}")

    expr = "tau(rho) = " + (" + ".join(pieces) if pieces else "0")
    expr = expr.replace("+ (-", "- (")

    best["expression"] = expr
    best["terms_human"] = [_human_term(t) for t in best["terms"]]
    return best


def predict_symbolic_model(model: dict, rho) -> np.ndarray:
    """Predict values from fitted symbolic model."""
    r = np.asarray(rho, dtype=float)
    terms = list(model["terms"])
    coef = np.asarray(model["coefficients"], dtype=float)

    X = _design_matrix(r, terms)
    return X @ coef


def _to_builtin(obj: Any) -> Any:
    """Convert numpy/pandas scalar containers to JSON-friendly Python types."""
    if isinstance(obj, dict):
        return {str(k): _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_builtin(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_to_builtin(v) for v in obj)
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _prepare_pysr_environment() -> None:
    """Configure writable Julia/PySR paths for restricted environments."""
    if "PYTHON_JULIAPKG_PROJECT" not in os.environ:
        os.environ["PYTHON_JULIAPKG_PROJECT"] = "/tmp/pyjuliapkg_project"
    if "JULIA_DEPOT_PATH" not in os.environ:
        os.environ["JULIA_DEPOT_PATH"] = "/tmp/julia_depot"
    Path(os.environ["PYTHON_JULIAPKG_PROJECT"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["JULIA_DEPOT_PATH"]).mkdir(parents=True, exist_ok=True)


def _fit_pysr_core(
    X: np.ndarray,
    y: np.ndarray,
    seed: int = 0,
    niterations: int = 400,
    population_size: int = 50,
    maxsize: int = 20,
    binary_operators: list[str] | None = None,
    unary_operators: list[str] | None = None,
    model_selection: str = "best",
) -> tuple[dict, Any, np.ndarray]:
    """Internal PySR fit routine on a 2D feature matrix."""
    _prepare_pysr_environment()

    try:
        from pysr import PySRRegressor  # type: ignore
    except Exception as exc:  # pragma: no cover - import path depends on env
        raise RuntimeError(
            "PySR is not installed. Install it in your environment first "
            "(and ensure Julia is available)."
        ) from exc

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array (n_samples, n_features).")
    if y.ndim != 1:
        raise ValueError("target y must be 1D.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have matching sample counts.")

    if binary_operators is None:
        binary_operators = ["+", "-", "*", "/"]
    if unary_operators is None:
        unary_operators = ["log", "exp"]

    params = {
        "niterations": int(niterations),
        "population_size": int(population_size),
        "maxsize": int(maxsize),
        "binary_operators": list(binary_operators),
        "unary_operators": list(unary_operators),
        "model_selection": str(model_selection),
        "elementwise_loss": "loss(x, y) = (x - y)^2",
        "deterministic": True,
        "parallelism": "serial",
        "progress": False,
        "verbosity": 0,
        "random_state": int(seed),
    }

    # Compatibility fallback for older/newer PySR versions.
    try:
        model = PySRRegressor(**params)
    except TypeError:
        params["loss"] = params.pop("elementwise_loss")
        params.pop("deterministic", None)
        params.pop("parallelism", None)
        params.pop("random_state", None)
        model = PySRRegressor(**params)

    model.fit(X, y)
    y_pred = np.asarray(model.predict(X), dtype=float).reshape(-1)
    metrics = _scores(y, y_pred, n_params=1)

    expression = "unknown"
    best_info: dict[str, Any] = {}
    equations_top: list[dict[str, Any]] = []

    try:
        best_row = model.get_best()
        if hasattr(best_row, "to_dict"):
            best_info = _to_builtin(best_row.to_dict())
        elif isinstance(best_row, dict):
            best_info = _to_builtin(best_row)
        else:
            expression = str(best_row)

        if "equation" in best_info:
            expression = str(best_info["equation"])
    except Exception:
        pass

    try:
        eq_df = model.equations_
        if hasattr(eq_df, "sort_values"):
            if "loss" in eq_df.columns:
                eq_df = eq_df.sort_values("loss", ascending=True)
            eq_df = eq_df.head(10)
            equations_top = _to_builtin(eq_df.to_dict(orient="records"))

            if expression == "unknown" and len(equations_top) > 0:
                expression = str(equations_top[0].get("equation", "unknown"))
    except Exception:
        pass

    summary = {
        "model_type": "pysr_symbolic_regression",
        "expression": expression,
        "metrics": metrics,
        "pysr_params": {
            "niterations": int(niterations),
            "population_size": int(population_size),
            "maxsize": int(maxsize),
            "binary_operators": list(binary_operators),
            "unary_operators": list(unary_operators),
            "model_selection": str(model_selection),
            "seed": int(seed),
        },
        "best_equation_info": best_info,
        "equations_top": equations_top,
    }

    return summary, model, y_pred


def fit_pysr_symbolic_regression(
    rho,
    target,
    seed: int = 0,
    niterations: int = 400,
    population_size: int = 50,
    maxsize: int = 20,
    binary_operators: list[str] | None = None,
    unary_operators: list[str] | None = None,
    model_selection: str = "best",
) -> tuple[dict, Any]:
    """Fit 1D symbolic regression tau(rho) with PySR."""
    r = np.asarray(rho, dtype=float)
    y = np.asarray(target, dtype=float)

    if r.ndim != 1 or y.ndim != 1:
        raise ValueError("rho and target must be 1D arrays.")
    if r.size != y.size:
        raise ValueError("rho and target must have equal length.")
    if np.any(r <= 0.0):
        raise ValueError("rho values must be strictly positive.")

    summary, model, _ = _fit_pysr_core(
        X=r.reshape(-1, 1),
        y=y,
        seed=seed,
        niterations=niterations,
        population_size=population_size,
        maxsize=maxsize,
        binary_operators=binary_operators,
        unary_operators=unary_operators,
        model_selection=model_selection,
    )
    summary["feature_names"] = ["rho"]
    return summary, model


def fit_pysr_multivariate_symbolic_regression(
    features,
    target,
    feature_names: list[str] | None = None,
    seed: int = 0,
    niterations: int = 400,
    population_size: int = 50,
    maxsize: int = 20,
    binary_operators: list[str] | None = None,
    unary_operators: list[str] | None = None,
    model_selection: str = "best",
) -> tuple[dict, Any]:
    """Fit multivariate symbolic regression with PySR.

    Parameters
    ----------
    features : array-like, shape (n_samples, n_features)
    target : array-like, shape (n_samples,)
    feature_names : list[str] | None
        Optional names corresponding to PySR variables x0, x1, ...
    """
    X = np.asarray(features, dtype=float)
    y = np.asarray(target, dtype=float)
    if X.ndim != 2:
        raise ValueError("features must be 2D (n_samples, n_features).")
    if y.ndim != 1:
        raise ValueError("target must be 1D.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("features and target must have matching sample counts.")

    summary, model, _ = _fit_pysr_core(
        X=X,
        y=y,
        seed=seed,
        niterations=niterations,
        population_size=population_size,
        maxsize=maxsize,
        binary_operators=binary_operators,
        unary_operators=unary_operators,
        model_selection=model_selection,
    )
    if feature_names is None:
        feature_names = [f"x{i}" for i in range(X.shape[1])]
    summary["feature_names"] = list(feature_names)
    return summary, model


def predict_pysr_model(model: Any, rho) -> np.ndarray:
    """Predict tau(rho) from a fitted PySR model."""
    r = np.asarray(rho, dtype=float)
    X = r.reshape(-1, 1)
    return np.asarray(model.predict(X), dtype=float).reshape(-1)


def predict_pysr_multivariate_model(model: Any, features) -> np.ndarray:
    """Predict from a fitted multivariate PySR model."""
    X = np.asarray(features, dtype=float)
    if X.ndim != 2:
        raise ValueError("features must be 2D (n_samples, n_features).")
    return np.asarray(model.predict(X), dtype=float).reshape(-1)
