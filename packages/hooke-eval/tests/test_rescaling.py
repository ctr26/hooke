from unittest.mock import Mock

import numpy as np
import pytest

from vcb.data_models.dataset.anndata import AnnotatedDataMatrix
from vcb.preprocessing.steps.scale_counts import ScaleCountsStep


def test_scale_counts_fit_and_transform():
    """Test that ScaleCountsStep correctly fits desired library size and transforms data."""

    mock_ground_truth = Mock(spec=AnnotatedDataMatrix)
    mock_predictions = Mock(spec=AnnotatedDataMatrix)

    # Ground truth: 3 cells with library sizes [1000, 2000, 3000] -> median = 2000
    test_data_gt = np.array(
        [
            [100, 200, 300, 400],  # sum = 1000
            [200, 400, 600, 800],  # sum = 2000
            [300, 600, 900, 1200],  # sum = 3000
        ]
    )

    # Predictions: 3 cells with library sizes [500, 1500, 2500]
    test_data_pred = np.array(
        [
            [50, 100, 150, 200],  # sum = 500
            [150, 300, 450, 600],  # sum = 1500
            [250, 500, 750, 1000],  # sum = 2500
        ]
    )

    mock_ground_truth.X = test_data_gt
    mock_predictions.X = test_data_pred

    step = ScaleCountsStep(transform_ground_truth=True, transform_predictions=True)
    assert not step.fitted

    step.fit(mock_ground_truth, mock_predictions)
    assert step.fitted
    assert step.desired_library_size == 2000

    step.transform(mock_ground_truth, mock_predictions)

    # Verify ground truth transformation
    # Original GT sums: [1000, 2000, 3000]
    # After scaling to 2000: [2000, 2000, 2000]
    expected_gt_sums = np.array([2000, 2000, 2000])
    actual_gt_sums = np.sum(mock_ground_truth.X, axis=1)
    np.testing.assert_array_almost_equal(actual_gt_sums, expected_gt_sums)

    # Verify predictions transformation
    # Original pred sums: [500, 1500, 2500]
    # After scaling to 2000: [2000, 2000, 2000]
    expected_pred_sums = np.array([2000, 2000, 2000])
    actual_pred_sums = np.sum(mock_predictions.X, axis=1)
    np.testing.assert_array_almost_equal(actual_pred_sums, expected_pred_sums)


def test_scale_counts_with_predefined_library_size():
    """Test that ScaleCountsStep works with a predefined library size."""

    mock_gt = Mock(spec=AnnotatedDataMatrix)
    mock_pred = Mock(spec=AnnotatedDataMatrix)

    test_data = np.array(
        [
            [100, 200, 300],  # sum = 600
            [200, 400, 600],  # sum = 1200
        ]
    )

    mock_gt.X = test_data.copy()
    mock_pred.X = test_data.copy()

    step = ScaleCountsStep(library_size=1000, transform_ground_truth=True, transform_predictions=True)
    assert step.fitted
    assert step.desired_library_size == 1000

    step.fit(mock_gt, mock_pred)
    assert step.desired_library_size == 1000

    step.transform(mock_gt, mock_pred)

    expected_sums = np.array([1000, 1000])
    np.testing.assert_array_almost_equal(np.sum(mock_gt.X, axis=1), expected_sums)
    np.testing.assert_array_almost_equal(np.sum(mock_pred.X, axis=1), expected_sums)


def test_scale_counts_transform_flags():
    """Test that transform flags control which data gets transformed."""

    mock_gt = Mock(spec=AnnotatedDataMatrix)
    mock_pred = Mock(spec=AnnotatedDataMatrix)

    test_data = np.array([[100, 200, 300]])  # sum = 600

    mock_gt.X = test_data.copy()
    mock_pred.X = test_data.copy()

    step = ScaleCountsStep(library_size=1000, transform_ground_truth=True, transform_predictions=False)
    step.transform(mock_gt, mock_pred)

    assert np.sum(mock_gt.X) == 1000
    assert np.sum(mock_pred.X) == 600


def test_scale_counts_validation():
    """Test validation of desired_library_size property."""

    step = ScaleCountsStep()

    # Test setting valid values
    step.desired_library_size = 1000
    assert step.desired_library_size == 1000

    # Test setting None should raise ValueError
    with pytest.raises(ValueError):
        step.desired_library_size = None

    # Test setting negative value should raise ValueError
    with pytest.raises(ValueError):
        step.desired_library_size = -100

    # Test setting zero should raise ValueError
    with pytest.raises(ValueError):
        step.desired_library_size = 0


def test_scale_counts_not_fitted_error():
    """Test that transform raises error when not fitted."""

    step = ScaleCountsStep()

    mock_gt = Mock(spec=AnnotatedDataMatrix)
    mock_pred = Mock(spec=AnnotatedDataMatrix)

    with pytest.raises(RuntimeError):
        # RuntimeError: The desired library size is not set.
        step.transform(mock_gt, mock_pred)
