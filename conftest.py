# conftest.py file to share fixtures between multiple test files
# https://docs.pytest.org/en/latest/fixture.html#conftest-py-sharing-fixture-functions

import numpy as np
import pytest

from data import generate_binary

SEED = 0
np.random.seed(seed=SEED)


# skipping slow tests if --runslow not provided in cli
def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )

def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


N_SAMPLES_SMALL = 100
N_FEATURES_SMALL = 20


@pytest.fixture(scope="session")
def dataset():
    n_samples = N_SAMPLES_SMALL
    n_features = N_FEATURES_SMALL
    return generate_binary(n_samples, n_features)

# @pytest.fixture(scope="session")
# def X():
#     n_samples = N_SAMPLES_SMALL
#     n_features = N_FEATURES_SMALL
#     return np.random.rand(n_samples, n_features)

# @pytest.fixture(scope="session")
# def y():
#     n_samples = N_SAMPLES_SMALL
#     return np.sign(np.random.rand(n_samples, 1))

@pytest.fixture(scope="session")
def lmbd():
    return .1