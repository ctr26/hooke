from unittest.mock import Mock

import numpy as np

from vcb.data_models.dataset.anndata import AnnotatedDataMatrix
from vcb.preprocessing.steps.log1p import InverseLog1pStep, Log1pStep


def test_log1p_transform():
    """Test that Log1pStep correctly applies log1p transformation."""

    mock_ground_truth = Mock(spec=AnnotatedDataMatrix)
    mock_predictions = Mock(spec=AnnotatedDataMatrix)

    # Test data with various values including zeros
    test_data_gt = np.array([[0, 1, 2, 10, 100]])
    test_data_pred = np.array([[0.5, 5, 50, 500]])

    mock_ground_truth.X = test_data_gt.copy()
    mock_predictions.X = test_data_pred.copy()

    step = Log1pStep(transform_ground_truth=True, transform_predictions=True)

    result_gt, result_pred = step.transform(mock_ground_truth, mock_predictions)

    # Verify the same objects are returned
    assert result_gt is mock_ground_truth
    assert result_pred is mock_predictions

    # Verify log1p transformation was applied
    np.testing.assert_array_almost_equal(result_gt.X, np.log1p(test_data_gt))
    np.testing.assert_array_almost_equal(result_pred.X, np.log1p(test_data_pred))


def test_log1p_transform_flags():
    """Test that transform flags control which data gets transformed."""

    mock_gt = Mock(spec=AnnotatedDataMatrix)
    mock_pred = Mock(spec=AnnotatedDataMatrix)

    test_data = np.array([[1, 2, 3]])

    mock_gt.X = test_data.copy()
    mock_pred.X = test_data.copy()

    # Only transform ground truth
    step = Log1pStep(transform_ground_truth=True, transform_predictions=False)
    result_gt, result_pred = step.transform(mock_gt, mock_pred)

    # Ground truth should be transformed, predictions should remain unchanged
    np.testing.assert_array_almost_equal(result_gt.X, np.log1p(test_data))
    np.testing.assert_array_equal(result_pred.X, test_data)


def test_inverse_log1p_transform():
    """Test that InverseLog1pStep correctly applies inverse log1p transformation."""

    mock_ground_truth = Mock(spec=AnnotatedDataMatrix)
    mock_predictions = Mock(spec=AnnotatedDataMatrix)

    # Test data with log-transformed values
    test_data_gt = np.array([[0, 0.693, 1.099, 2.398]])  # log1p of [0, 1, 2, 10]
    test_data_pred = np.array([[0.405, 1.609, 3.912]])  # log1p of [0.5, 4, 49]

    mock_ground_truth.X = test_data_gt.copy()
    mock_predictions.X = test_data_pred.copy()

    step = InverseLog1pStep(transform_ground_truth=True, transform_predictions=True)

    result_gt, result_pred = step.transform(mock_ground_truth, mock_predictions)

    # Verify the same objects are returned
    assert result_gt is mock_ground_truth
    assert result_pred is mock_predictions

    # Verify inverse log1p transformation was applied
    expected_gt = np.exp(test_data_gt) - 1
    expected_pred = np.exp(test_data_pred) - 1

    np.testing.assert_array_almost_equal(result_gt.X, expected_gt)
    np.testing.assert_array_almost_equal(result_pred.X, expected_pred)


def test_inverse_log1p_transform_flags():
    """Test that transform flags control which data gets transformed."""

    mock_gt = Mock(spec=AnnotatedDataMatrix)
    mock_pred = Mock(spec=AnnotatedDataMatrix)

    test_data = np.array([[0.693, 1.099]])  # log1p of [1, 2]

    mock_gt.X = test_data.copy()
    mock_pred.X = test_data.copy()

    # Only transform predictions (default behavior)
    step = InverseLog1pStep(transform_ground_truth=False, transform_predictions=True)
    result_gt, result_pred = step.transform(mock_gt, mock_pred)

    # Ground truth should remain unchanged, predictions should be transformed
    np.testing.assert_array_equal(result_gt.X, test_data)
    np.testing.assert_array_almost_equal(result_pred.X, np.exp(test_data) - 1)


def test_log1p_inverse_roundtrip():
    """Test that log1p followed by inverse log1p returns original data."""

    mock_gt = Mock(spec=AnnotatedDataMatrix)
    mock_pred = Mock(spec=AnnotatedDataMatrix)

    original_data = np.array([[0, 1, 2, 10, 100]])

    mock_gt.X = original_data.copy()
    mock_pred.X = original_data.copy()

    # Apply log1p transformation
    log1p_step = Log1pStep(transform_ground_truth=True, transform_predictions=True)
    result_gt, result_pred = log1p_step.transform(mock_gt, mock_pred)

    # Apply inverse log1p transformation
    inverse_step = InverseLog1pStep(transform_ground_truth=True, transform_predictions=True)
    final_gt, final_pred = inverse_step.transform(result_gt, result_pred)

    # Should be close to original data (within numerical precision)
    np.testing.assert_array_almost_equal(final_gt.X, original_data)
    np.testing.assert_array_almost_equal(final_pred.X, original_data)
