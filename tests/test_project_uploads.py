import numpy as np
import pytest

from giskard.model import SupportedModelTypes
from giskard.project import GiskardProject


@pytest.mark.parametrize('pred', [
    [[1, 1]],
    [[0.1, 0.2]],
    [[-1, 2]],
    [[-1, 2]],
    [[0.1, 0.2, 0.7]],
    [[0, 1], [0.8, 0.5]]
])
def test__validate_classification_prediction_fail(pred):
    with pytest.raises(ValueError):
        GiskardProject._validate_classification_prediction(['one', 'two'],
                                                           SupportedModelTypes.CLASSIFICATION.value,
                                                           np.array(pred))


@pytest.mark.parametrize('pred', [
    [[0, 1]],
    [[0.999999999999999, 0.000000000000001]]
])
def test__validate_classification_prediction_pass(pred):
    GiskardProject._validate_classification_prediction(['one', 'two'],
                                                       SupportedModelTypes.CLASSIFICATION.value,
                                                       np.array(pred))
