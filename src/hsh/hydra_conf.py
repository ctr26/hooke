"""Register Pydantic configs with Hydra's ConfigStore via hydra-zen.

No YAML files are produced or consumed.  ``builds()`` creates structured
configs that carry ``_target_`` metadata so ``zen()`` can instantiate
the original Pydantic types at runtime.
"""

from __future__ import annotations

from hydra_zen import builds, make_config

from hsh.config import EvalConfig, FinetuneConfig, InferConfig, ModelConfig, TrainConfig

ModelConf = builds(ModelConfig, populate_full_signature=True)
TrainConf = builds(TrainConfig, populate_full_signature=True)
FinetuneConf = builds(FinetuneConfig, populate_full_signature=True)
EvalConf = builds(EvalConfig, populate_full_signature=True)
InferConf = builds(InferConfig, populate_full_signature=True)

TrainTaskConf = make_config(train_config=TrainConf, model_config=ModelConf)
FinetuneTaskConf = make_config(config=FinetuneConf)
EvalTaskConf = make_config(config=EvalConf)
InferTaskConf = make_config(config=InferConf)
