from enum import Enum
from typing import Callable, Iterable, Union, List

import pandas as pd


class SupportedModelTypes(Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class SupportedColumnType(Enum):
    NUMERIC = "numeric"
    CATEGORY = "category"
    TEXT = "text"


class GiskardModel:
    def __init__(self,
                 prediction_function: Callable[[pd.DataFrame], Iterable[Union[str, float, int]]],
                 model_type: str,
                 feature_names: List[str],
                 classification_labels: List[str] = None,
                 classification_threshold: float = None,
                 ) -> None:
        self.prediction_function = prediction_function
        self.model_type = model_type
        self.classification_threshold = classification_threshold
        self.feature_names = feature_names
        self.classification_labels = classification_labels
