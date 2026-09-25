import pandas as pd
import pytest

from subscription_analytics.generate import generate_dataset


def test_generation_is_reproducible(tmp_path):
    first = generate_dataset(tmp_path / "first", seed=7, n_users=25, days=40)
    second = generate_dataset(tmp_path / "second", seed=7, n_users=25, days=40)

    assert set(first) == set(second)
    for name in first:
        assert first[name].read_bytes() == second[name].read_bytes()


def test_small_dataset_has_all_raw_tables(tmp_path):
    files = generate_dataset(tmp_path, seed=3, n_users=10, days=30)

    assert set(files) == {
        "users",
        "events",
        "subscriptions",
        "acquisition",
        "experiment_assignments",
    }
    assert all(path.exists() for path in files.values())
    assert pd.read_csv(files["users"]).user_id.nunique() == 10


@pytest.mark.parametrize("kwargs", [{"n_users": 0}, {"days": 29}])
def test_generation_rejects_invalid_size(tmp_path, kwargs):
    with pytest.raises(ValueError):
        generate_dataset(tmp_path, **kwargs)
