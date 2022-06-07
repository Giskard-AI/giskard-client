import json
import logging
from typing import Callable, Dict, Iterable, List, Optional, Union

import numpy as np
import pandas as pd
import requests
from requests import Session

from giskard.io_utils import compress, pickle_dumps, save_df
from giskard.model import SupportedModelTypes, SupportedColumnType
from giskard.python_utils import get_python_requirements, get_python_version


class GiskardProject:
    def __init__(self, session: Session, project_key: str) -> None:
        self.project_key = project_key
        self.session = session

    @staticmethod
    def _serialize(prediction_function: Callable[
        [pd.DataFrame],
        Iterable[Union[str, float, int]],
    ]) -> bytes:
        compressed_pickle: bytes = compress(pickle_dumps(prediction_function))
        print(f'Compressed model size: {len(compressed_pickle)} bytes')
        return compressed_pickle

    def upload_model(
            self,
            prediction_function: Callable[[pd.DataFrame], Iterable[Union[str, float, int]]],
            model_type: str,
            feature_names: List[str],
            name: str = None,
            classification_threshold: Optional[float] = 0.5,
            classification_labels: Optional[List[str]] = None,
            validate_df: pd.DataFrame = None
    ):
        print(f"Uploading model '{name}' to project '{self.project_key}'...")

        self._validate_classification_threshold(classification_threshold)
        self._validate_model_type(model_type)
        self._validate_prediction_function(prediction_function)
        classification_labels = self._validate_classification_labels(classification_labels, model_type)

        if validate_df is not None:
            self._validate_model_execution(prediction_function, validate_df)

        model = self._serialize(prediction_function)
        requirements = get_python_requirements()
        params = {
            "name": name,
            "projectKey": self.project_key,
            "languageVersion": get_python_version(),
            "modelType": model_type,
            "threshold": classification_threshold,
            "featureNames": feature_names,
            "language": "PYTHON",
            "classificationLabels": classification_labels
        }

        files = [
            ('metadata', (None, json.dumps(params), 'application/json')),
            ('modelFile', model),
            ('requirementsFile', requirements)
        ]
        self.session.post('project/models/upload', data={}, files=files)
        print(f"Uploading model '{name}' to project '{self.project_key}': Done!")

    def upload_df(
            self,
            df: pd.DataFrame,
            feature_types: Dict[str, str],
            target: str = None,
            name: str = None,
    ) -> requests.Response:
        logging.info(f"Uploading dataset '{name}' to project '{self.project_key}'...")
        self.validate_df(df, feature_types)
        self._validate_input_types(feature_types)
        data = compress(save_df(df))
        params = {
            "projectKey": self.project_key,
            "name": name,
            "featureTypes": feature_types,
            "target": target
        }

        files = [
            ('metadata', (None, json.dumps(params), 'application/json')),
            ('file', data)
        ]

        logging.info(f"Uploading dataset '{name}' to project '{self.project_key}': Done!")
        return self.session.post("project/data/upload", data={}, files=files)

    def upload_model_and_df(
            self,
            prediction_function: Callable[[pd.DataFrame], Iterable[Union[str, float, int]]],
            prediction_task: str,
            feature_names: List[str],
            df: pd.DataFrame,
            feature_types: Dict[str, str],
            target: str,
            model_name: str = None,
            dataset_name: str = None,
            classification_threshold: Optional[float] = 0.5,
            classification_labels: Optional[List[str]] = None,
    ) -> None:
        self.upload_model(prediction_function,
                          prediction_task,
                          feature_names,
                          model_name,
                          classification_threshold,
                          classification_labels,
                          df)
        self.upload_df(
            df=df,
            name=dataset_name,
            feature_types=feature_types,
            target=target)

    @staticmethod
    def _validate_model_type(prediction_task):
        if prediction_task not in {task.value for task in SupportedModelTypes}:
            raise ValueError(
                f"Invalid prediction_task parameter: {prediction_task}. "
                + f"Please choose one of {[task.value for task in SupportedModelTypes]}."
            )

    @staticmethod
    def _validate_input_types(input_types):
        if input_types and type(input_types) is dict:
            if set(input_types.values()) > {column_type.value for column_type in SupportedColumnType}:
                raise ValueError(
                    f"Invalid input_types parameter: {input_types}. "
                    + f"Please choose types among {[column_type.value for column_type in SupportedColumnType]}."
                )
        else:
            raise ValueError(
                f"Invalid input_types parameter: {input_types}. Please specify non-empty dictionary."
            )

    @staticmethod
    def _validate_prediction_function(prediction_function):
        if not callable(prediction_function):
            raise ValueError(
                f"Invalid prediction_function parameter: {prediction_function}. Please specify Python function."
            )

    @staticmethod
    def _validate_classification_threshold(classification_threshold):
        if classification_threshold is not None and not isinstance(classification_threshold, (int, float)):
            raise ValueError(
                f"Invalid classification_threshold parameter: {classification_threshold}. Please specify valid number."
            )

    @staticmethod
    def _validate_classification_labels(classification_labels, prediction_task):
        res = None
        if prediction_task == SupportedModelTypes.CLASSIFICATION.value:
            if (
                    classification_labels is not None
                    and hasattr(classification_labels, "__iter__")
                    and not isinstance(classification_labels, (str, dict))  # type: ignore
            ):
                if len(classification_labels) > 1:
                    res: Optional[List[str]] = [str(label) for label in classification_labels]
                else:
                    raise ValueError(
                        f"Invalid classification_labels parameter: {classification_labels}. Please specify more than 1 label."
                    )
            else:
                raise ValueError(
                    f"Invalid classification_labels parameter: {classification_labels}. Please specify valid list of strings."
                )
        return res

    @staticmethod
    def _validate_model_execution(prediction_function, df: pd.DataFrame) -> None:
        prediction = prediction_function(df)
        if not isinstance(prediction, np.ndarray):
            raise ValueError("Model should return numpy array")

    @staticmethod
    def validate_df(df: pd.DataFrame, input_types) -> pd.DataFrame:
        if set(input_types.values()) < set(df.columns):
            missing_columns = set(df.columns) - set(input_types.values())
            raise ValueError(f"Missing input_types for columns: {missing_columns}")
        elif set(input_types.values()) > set(df.columns):
            missing_columns = set(input_types.values()) - set(df.columns)
            raise ValueError(
                f"Missing columns in dataframe according to input_types: {missing_columns}"
            )
        else:
            pandas_inferred_input_types = df.dtypes.to_dict()
            for column, dtype in pandas_inferred_input_types.items():
                if (
                        input_types.get(column) == SupportedColumnType.NUMERIC.value
                        and dtype == "object"
                ):
                    df[column] = df[column].astype(float)
            return df
