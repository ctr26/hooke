"""Tests for preprocessing components."""

from __future__ import annotations

import numpy as np
import pytest

from hooke.preprocessing import LabelEncoder, PreprocessingPipeline, StandardScaler


class TestStandardScaler:
    def test_fit_transform(self) -> None:
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        scaler = StandardScaler()
        result = scaler.fit_transform(X)
        np.testing.assert_allclose(result.mean(axis=0), 0.0, atol=1e-10)
        np.testing.assert_allclose(result.std(axis=0), 1.0, atol=1e-10)

    def test_transform_without_fit(self) -> None:
        scaler = StandardScaler()
        with pytest.raises(RuntimeError, match="not fitted"):
            scaler.transform(np.array([[1.0]]))

    def test_zero_variance_column(self) -> None:
        X = np.array([[1.0, 5.0], [1.0, 5.0], [1.0, 5.0]])
        scaler = StandardScaler()
        result = scaler.fit_transform(X)
        assert np.all(np.isfinite(result))


class TestLabelEncoder:
    def test_fit_transform(self) -> None:
        X = np.array(["cat", "dog", "cat", "bird"])
        encoder = LabelEncoder()
        result = encoder.fit_transform(X)
        assert result.shape == X.shape
        assert set(result.tolist()) == {0.0, 1.0, 2.0}

    def test_transform_without_fit(self) -> None:
        encoder = LabelEncoder()
        with pytest.raises(RuntimeError, match="not fitted"):
            encoder.transform(np.array(["cat"]))


class TestPreprocessingPipeline:
    def test_empty_pipeline(self) -> None:
        pipeline = PreprocessingPipeline()
        X = np.array([[1.0, 2.0]])
        result = pipeline.fit_transform(X)
        np.testing.assert_array_equal(result, X)

    def test_chained_scalers(self) -> None:
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        pipeline = PreprocessingPipeline(steps=[StandardScaler()])
        result = pipeline.fit_transform(X)
        np.testing.assert_allclose(result.mean(axis=0), 0.0, atol=1e-10)

    def test_fit_then_transform(self) -> None:
        X_train = np.array([[1.0], [2.0], [3.0]])
        X_test = np.array([[4.0], [5.0]])
        pipeline = PreprocessingPipeline(steps=[StandardScaler()])
        pipeline.fit(X_train)
        result = pipeline.transform(X_test)
        assert result.shape == X_test.shape
