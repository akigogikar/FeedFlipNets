from feedflipnets.experiments import registry


def test_config_hash_is_stable_under_key_order():
    base = {
        "model": {"strategy": "dfa", "hidden": [4]},
        "train": {"seed": 11, "lr": 0.01},
        "data": {"name": "synth_fixture", "options": {"seed": 11}},
        "offline": True,
    }
    reordered = {
        "offline": True,
        "data": {"options": {"seed": 11}, "name": "synth_fixture"},
        "train": {"lr": 0.01, "seed": 11},
        "model": {"hidden": [4], "strategy": "dfa"},
    }
    assert registry.config_hash(base) == registry.config_hash(reordered)


def test_config_hash_changes_on_seed():
    experiment = registry.get_experiment("dfa_baseline")
    config = experiment.to_pipeline_config()
    baseline = registry.config_hash(config)

    mutated = experiment.to_pipeline_config()
    mutated["train"]["seed"] = int(mutated["train"]["seed"]) + 1
    assert registry.config_hash(mutated) != baseline
