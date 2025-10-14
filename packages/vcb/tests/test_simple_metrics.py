import numpy as np

from vcb.metrics.simple import cosine, cosine_delta, mse, pearson, pearson_delta


def test_metric_mse():
    rng = np.random.default_rng(42)

    a = rng.normal(size=1000)
    b = rng.normal(size=1000)
    c = a.copy()

    assert np.isclose(mse(a, c), 0.0)
    assert mse(a, b) > 0.0


def test_metric_pearson():
    rng = np.random.default_rng(42)

    a = rng.random(size=1000)
    b = rng.random(size=1000)
    c = a * -1

    assert np.isclose(pearson(a, a + 10), 1.0)  # Same, but offset
    assert np.isclose(pearson(a, b), 0.0, atol=0.05)  # Random
    assert np.isclose(pearson(a, c), -1.0)  # Opposite

    # Adding a random base should not change anything
    base = rng.random(size=1000)
    assert np.isclose(pearson_delta(a + base, a + base + 10, base), 1.0)
    assert np.isclose(pearson_delta(a + base, b + base, base), 0.0, atol=0.05)
    assert np.isclose(pearson_delta(a + base, c + base, base), -1.0)


def test_metric_cosine():
    a = np.array([0, 0, 1])
    b = np.array([0, 0, 2])
    c = np.array([0, 1, 0])
    d = np.array([0, 0, -1])

    assert np.isclose(cosine(a, b), 1.0)  # parallel
    assert np.isclose(cosine(a, c), 0.0)  # perpendicular
    assert np.isclose(cosine(a, d), -1.0)  # opposite

    rng = np.random.default_rng(42)

    # Adding a random base should not change anything
    base = rng.random(size=3)
    assert np.isclose(cosine_delta(a + base, b + base, base), 1.0)
    assert np.isclose(cosine_delta(a + base, c + base, base), 0.0)
    assert np.isclose(cosine_delta(a + base, d + base, base), -1.0)
