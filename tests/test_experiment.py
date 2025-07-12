import os, sys, pathlib
import pytest
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from ternary_dfa_experiment import make_dataset, sweep_and_log


def test_make_dataset_shape():
    X, Y = make_dataset(freq=2, n=10, seed=0)
    assert X.shape == (1, 10)
    assert Y.shape == (1, 10)

def test_make_dataset_explicit_dataset():
    X, Y = make_dataset(freq=1, n=5, seed=1, dataset="synthetic")
    assert X.shape == (1, 5)
    assert Y.shape == (1, 5)

def test_make_dataset_limit():
    X, Y = make_dataset(freq=1, n=10, seed=0, dataset="synthetic", max_points=3)
    assert X.shape == (1, 3)
    assert Y.shape == (1, 3)

def test_make_dataset_mnist_shape():
    try:
        X, Y = make_dataset(freq=1, dataset="mnist")
    except Exception:
        pytest.skip("MNIST dataset not available")
    assert X.shape[0] == 1 and X.shape[1] == 784
    assert Y.shape == (1, 1)


@pytest.mark.parametrize("ds", ["mnist", "tinystories", "ucr:GunPoint"])
def test_make_dataset_downloadables(ds):
    try:
        X, Y = make_dataset(freq=1, dataset=ds, max_points=3)
    except Exception:
        pytest.skip(f"{ds} dataset not available")
    assert X.shape == (1, 3)
    assert Y.shape == (1, 3)


def test_sweep_returns_tables(tmp_path):
    tables = sweep_and_log(
        ['Backprop'], [1], [1], range(1), epochs=2, outdir=str(tmp_path), dataset="synthetic"
    )
    assert isinstance(tables, dict)
    assert 'Backprop' in tables
    assert tables['Backprop'].shape == (1, 1)
