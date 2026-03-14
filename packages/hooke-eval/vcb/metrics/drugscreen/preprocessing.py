import numpy as np
from loguru import logger
from sklearn.ensemble import IsolationForest


def isolation_outlier_mask(
    data: np.ndarray,
    n_estimators: int = 500,
    contamination: str | float = "auto",
    n_jobs: int = -1,
    random_state: int = 0,
):
    """Identify outliers using an isolation forest model.

    Fits an isolation forest model to the query population, adds outlier
    predictions to `data` according to the supplied name, and adds
    outlier records to the `transmogrifier_filter_column` for automatic
    filtering in subsequent estimator fitting and query operations.
    """
    outlier_detector = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        n_jobs=n_jobs,
        random_state=random_state,
    )

    outlier_predictions = outlier_detector.fit_predict(data)
    outlier_mask = outlier_predictions == -1
    logger.debug(
        f"Detected {sum(outlier_mask)} ({sum(outlier_mask) / len(data) * 100:.2f}%) "
        "outliers using the isolation forest approach"
    )
    return ~outlier_mask
