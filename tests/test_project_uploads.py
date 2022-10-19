import numpy as np
import pandas as pd
import pytest

from giskard.client.project import GiskardProject

data = np.array(["g", "e", "e", "k", "s"])


@pytest.mark.parametrize('pred', [
    [[0.81, 0.32]],
    [[0.9, 0.21]],
    [[1.5, 1]],
    [[-1, 2]],
    [[0.9, -0.1]],
    [[0, -1], [0.8, 0.5]]
])
def test__validate_classification_prediction_warn(pred):
    with pytest.warns():
        GiskardProject._validate_classification_prediction(['one', 'two'],
                                                           np.array(pred))


@pytest.mark.parametrize('pred', [
    [[0.1, 0.2, 0.7]],
])
def test__validate_classification_prediction_fail(pred):
    with pytest.raises(ValueError):
        GiskardProject._validate_classification_prediction(["one", "two"], np.array(pred))


@pytest.mark.parametrize("pred", [[[0, 1]], [[0.999999999999999, 0.000000000000001]]])
def test__validate_classification_prediction_pass(pred):
    GiskardProject._validate_classification_prediction(["one", "two"], np.array(pred))


@pytest.mark.parametrize("data", [pd.Series(data)])
def test_verify_is_pandasdataframe_fail(data):
    with pytest.raises(AssertionError):
        GiskardProject._validate_is_pandasdataframe(data)


@pytest.mark.parametrize("data", [pd.DataFrame(data)])
def test_verify_is_pandasdataframe_pass(data):
    GiskardProject._validate_is_pandasdataframe(data)


def _test_prediction_function(data):
    return np.random.rand(5, 1)


prev_pred = _test_prediction_function(data)


@pytest.mark.parametrize('data,prev_prediction,prediction_function', [
    (pd.DataFrame(data), prev_pred, _test_prediction_function)
])
def test_validate_deterministic_model(data, prev_prediction, prediction_function):
    with pytest.raises(AssertionError):
        GiskardProject._validate_deterministic_model(data, prev_prediction, prediction_function)
