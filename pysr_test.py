import os
from pathlib import Path

import numpy as np

# Keep Julia/PySR state in writable locations (helpful on managed systems).
os.environ.setdefault("PYTHON_JULIAPKG_PROJECT", "/tmp/pyjuliapkg_project")
os.environ.setdefault("JULIA_DEPOT_PATH", "/tmp/julia_depot")
Path(os.environ["PYTHON_JULIAPKG_PROJECT"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["JULIA_DEPOT_PATH"]).mkdir(parents=True, exist_ok=True)

from pysr import PySRRegressor

# PySR expects X to be 2D: (n_samples, n_features).
x = np.array([1, 2, 3, 4, 5, 6], dtype=float)
X = x.reshape(-1, 1)
y = 2.0 * x

model = PySRRegressor(
    niterations=60,
    population_size=25,
    maxsize=12,
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["cos", "exp", "log"],
    model_selection="best",
    progress=True,
    verbosity=1,
)

model.fit(X, y)
print("Best equation:")
print(model.get_best())
