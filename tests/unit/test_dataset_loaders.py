from feedflipnets.data.registry import get_dataset


def test_mnist_offline(tmp_path, monkeypatch):
    monkeypatch.setenv("FFN_DATA_OFFLINE", "1")
    spec = get_dataset("mnist", offline=True, cache_dir=tmp_path, one_hot=True)
    batch = next(spec.loader("train", 2))
    assert batch.inputs.shape[1] == spec.data_spec.d_in
    assert batch.targets.shape[1] == spec.data_spec.d_out


def test_tinystories_offline(tmp_path, monkeypatch):
    monkeypatch.setenv("FFN_DATA_OFFLINE", "1")
    spec = get_dataset("tinystories", offline=True, cache_dir=tmp_path, window=3)
    batch = next(spec.loader("train", 2))
    assert batch.inputs.shape[1] == spec.data_spec.d_in
    assert batch.targets.shape[1] == spec.data_spec.d_out


def test_ucr_offline(tmp_path, monkeypatch):
    monkeypatch.setenv("FFN_DATA_OFFLINE", "1")
    spec = get_dataset("ucr", offline=True, cache_dir=tmp_path)
    batch = next(spec.loader("train", 2))
    assert batch.inputs.shape[1] == spec.data_spec.d_in
    assert batch.targets.shape[1] == spec.data_spec.d_out
