import warnings

from feedflipnets import train


def test_train_single_shim(tmp_path, monkeypatch):
    monkeypatch.setenv("FEEDFLIP_DATA_OFFLINE", "1")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    with warnings.catch_warnings(record=True) as caught:
        losses, auc, t01 = train.train_single("Backprop", depth=1, freq=2, seed=0, epochs=2)
    assert losses
    assert auc >= 0.0
    assert t01 >= 0
    assert any("deprecated" in str(w.message).lower() for w in caught)
