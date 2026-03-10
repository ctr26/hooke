from __future__ import annotations


def test_import_hsh() -> None:
    import hsh

    assert hasattr(hsh, "__version__")


def test_import_model() -> None:
    from hsh.model import FlowMatchingMLP

    assert callable(FlowMatchingMLP)


def test_import_train() -> None:
    from hsh.train import train

    assert callable(train)


def test_import_finetune() -> None:
    from hsh.finetune import finetune

    assert callable(finetune)


def test_import_eval() -> None:
    from hsh.eval import evaluate

    assert callable(evaluate)


def test_import_infer() -> None:
    from hsh.infer import infer

    assert callable(infer)


def test_import_configs() -> None:
    from hsh.config import EvalConfig, FinetuneConfig, InferConfig, ModelConfig, TrainConfig

    assert all(callable(c) for c in [ModelConfig, TrainConfig, FinetuneConfig, EvalConfig, InferConfig])
