from feedflipnets.data import registry


def test_mnist_offline(tmp_path, monkeypatch):
    monkeypatch.setenv("FEEDFLIP_DATA_OFFLINE", "1")
    spec = registry.get(
        "mnist",
        offline=True,
        cache_dir=tmp_path,
        subset="train",
        max_items=4,
        one_hot=False,
    )
    batch = next(spec.loader("train", 2))
    assert batch.inputs.shape[1] == 784
    assert batch.targets.shape[1] == 1


def test_tinystories_offline(tmp_path, monkeypatch):
    monkeypatch.setenv("FEEDFLIP_DATA_OFFLINE", "1")
    spec = registry.get("tinystories", offline=True, cache_dir=tmp_path, window=3)
    batch = next(spec.loader("train", 2))
    assert batch.inputs.shape[1] == 3


def test_ucr_offline(tmp_path, monkeypatch):
    monkeypatch.setenv("FEEDFLIP_DATA_OFFLINE", "1")
    spec = registry.get("ucr_uea", offline=True, cache_dir=tmp_path, name="demo")
    batch = next(spec.loader("train", 2))
    assert batch.inputs.ndim == 2
    assert batch.targets.shape[1] > 0
