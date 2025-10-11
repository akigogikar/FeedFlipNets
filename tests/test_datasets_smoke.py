"""Smoke tests for the unified dataset registry."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from feedflipnets.data.registry import get_dataset


def _first_batch(name: str, cache_dir: Path, **options):
    spec = get_dataset(name, offline=True, cache_dir=cache_dir, **options)
    batch = next(spec.loader("train", 4))
    return spec, batch


def _assert_float_batch(batch):
    assert batch.inputs.dtype == np.float32
    assert batch.targets.dtype == np.float32


def test_mnist_offline(tmp_path):
    spec, batch = _first_batch("mnist", tmp_path)
    _assert_float_batch(batch)
    assert spec.data_spec.d_in == 784
    assert spec.data_spec.d_out == 10
    assert spec.data_spec.task_type == "multiclass"
    assert sum(spec.splits.values()) > 0


def test_ucr_offline(tmp_path):
    spec, batch = _first_batch("ucr", tmp_path, ucr_name="GunPoint")
    _assert_float_batch(batch)
    assert spec.data_spec.task_type == "multiclass"
    assert spec.data_spec.extra["sequence_length"] > 0
    assert batch.inputs.shape[1] == spec.data_spec.d_in


def test_california_housing_offline(tmp_path):
    spec, batch = _first_batch("california_housing", tmp_path)
    _assert_float_batch(batch)
    assert spec.data_spec.task_type == "regression"
    assert spec.data_spec.d_out == 1
    assert spec.splits["train"] > 0


def test_20newsgroups_offline(tmp_path):
    spec, batch = _first_batch("20newsgroups", tmp_path)
    _assert_float_batch(batch)
    assert spec.data_spec.d_in == 4096
    assert spec.data_spec.d_out >= 5


def test_csv_regression_offline(tmp_path):
    fixture = Path("feedflipnets/data/_fixtures/csv_regression_fixture.csv")
    spec, batch = _first_batch(
        "csv_regression", tmp_path, csv_path=str(fixture), target_col="target"
    )
    _assert_float_batch(batch)
    assert spec.data_spec.task_type == "regression"
    assert spec.data_spec.d_out == 1


def test_csv_classification_offline(tmp_path):
    fixture = Path("feedflipnets/data/_fixtures/csv_classification_fixture.csv")
    spec, batch = _first_batch(
        "csv_classification", tmp_path, csv_path=str(fixture), target_col="target"
    )
    _assert_float_batch(batch)
    assert spec.data_spec.task_type == "multiclass"
    assert batch.targets.shape[1] == spec.data_spec.d_out


def test_deterministic_splits(tmp_path):
    spec_a = get_dataset("mnist", offline=True, cache_dir=tmp_path, seed=123)
    spec_b = get_dataset("mnist", offline=True, cache_dir=tmp_path, seed=123)
    batch_a = next(spec_a.loader("train", 8))
    batch_b = next(spec_b.loader("train", 8))
    np.testing.assert_allclose(batch_a.inputs, batch_b.inputs)
    np.testing.assert_allclose(batch_a.targets, batch_b.targets)
