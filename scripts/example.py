"""Example: train a toy model, save weights, then run the pipeline.

Note: Uses pickle for sklearn model serialization (standard practice).
Only load weights from trusted sources.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge

from hooke.config import DataConfig, ModelConfig, PipelineConfig
from hooke.model import HookeModel
from hooke.preprocessing import PreprocessingPipeline, StandardScaler

# 1. Train a toy sklearn model
rng = np.random.default_rng(42)
X_train = rng.standard_normal((100, 3))
y_train = X_train @ np.array([1.0, -2.0, 0.5]) + rng.standard_normal(100) * 0.1

model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

# 2. Save weights
weights_path = Path("/tmp/hooke_example_weights.pkl")
with open(weights_path, "wb") as f:
    pickle.dump(model, f)

# 3. Build config programmatically
config = PipelineConfig(
    model=ModelConfig(
        model_class="hooke.model.HookeModel",
        weights_path=weights_path,
        device="cpu",
    ),
    data=DataConfig(
        feature_names=["gene_1", "gene_2", "gene_3"],
        preprocessing_steps=[],
    ),
    name="example",
)

# 4. Load model and predict
hooke_model = HookeModel(config.model)
hooke_model.load_weights(weights_path)

scaler = StandardScaler()
scaler.fit(X_train)

preprocessor = PreprocessingPipeline(steps=[scaler])

# 5. Run prediction
X_test = rng.standard_normal((5, 3))
X_scaled = preprocessor.transform(X_test)
result = hooke_model.predict({"features": X_scaled})

print("Predictions:", result["predictions"])
print("Pipeline config:", config.model_dump())
