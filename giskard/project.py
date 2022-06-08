import json
import logging
from typing import Callable, Dict, Iterable, List, Optional, Union
import sys
sys.tracebacklimit = None # set to 0 for No stack trace

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
            classification_threshold: Optional[float] = None,
            classification_labels: Optional[List[str]] = None,
            validate_df: pd.DataFrame = None,
            target: Optional[List[str]] = None
    ):
        print(f"Initiating model upload to project '{self.project_key}'...")

        self._validate_model_type(model_type, classification_labels)
        self._validate_features(feature_names=feature_names, validate_df=validate_df)
        self._validate_prediction_function(prediction_function)

        if model_type == SupportedModelTypes.CLASSIFICATION.value:
            self._validate_classification_threshold_label(classification_labels, classification_threshold)
            classification_labels = self._validate_classification_labels(classification_labels, model_type)

        if validate_df is not None:
            self._validate_model_execution(prediction_function, validate_df, model_type, classification_labels)
            if target is not None:
                target_values = validate_df[target].unique()
                self._validate_label_with_target(classification_labels, target_values)

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
        print(f"Successfully uploaded model to project '{self.project_key}'")

    def upload_df(
            self,
            df: pd.DataFrame,
            column_types: Dict[str, str],
            target: str = None,
            name: str = "None",
    ) -> requests.Response:
        print(f"Initiating dataset upload to project '{self.project_key}'...")
        self._validate_features(column_types=column_types)
        if target is not None:
            self._validate_target(target, df.keys())
        self.validate_df(df, column_types)
        self._validate_input_types(column_types)

        data = compress(save_df(df))
        params = {
            "projectKey": self.project_key,
            "name": name,
            "featureTypes": column_types,
            "target": target
        }

        files = [
            ('metadata', (None, json.dumps(params), 'application/json')),
            ('file', data)
        ]

        print(f"Successfully uploaded dataset to project '{self.project_key}'")
        return self.session.post("project/data/upload", data={}, files=files)

    def upload_model_and_df(
            self,
            prediction_function: Callable[[pd.DataFrame], Iterable[Union[str, float, int]]],
            prediction_task: str,
            feature_names: List[str],
            df: pd.DataFrame,
            column_types: Dict[str, str],
            target: str = None,
            model_name: str = None,
            dataset_name: str = None,
            classification_threshold: Optional[float] = None,
            classification_labels: Optional[List[str]] = None,
    ) -> None:
        self.upload_model(prediction_function=prediction_function,
                          model_type=prediction_task,
                          feature_names=feature_names,
                          name=model_name,
                          classification_threshold=classification_threshold,
                          classification_labels=classification_labels,
                          validate_df=df,
                          target=target)
        self.upload_df(
            df=df,
            name=dataset_name,
            column_types=column_types,
            target=target)

    @staticmethod
    def _validate_model_type(prediction_task, classification_labels=None):
        if prediction_task not in {task.value for task in SupportedModelTypes}:
            raise ValueError(
                f"Invalid prediction_task parameter: {prediction_task}. "
                + f"Please choose one of {[task.value for task in SupportedModelTypes]}."
            )
        if prediction_task == SupportedModelTypes.REGRESSION.value and classification_labels is not None:
            raise ValueError(
                "Invalid Input parameter. Please do not pass 'classification_labels' for Regression Model.")

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
    def _validate_target(target, dataframe_keys):
        if target is not None and target not in dataframe_keys:
            raise ValueError(
                f"Invalid target parameter:"
                f" Select the target from the column names of the dataset:  {dataframe_keys}.\n"
                f" ATTENTION: If Dataframe does not contain target column, please do not pass target in the input")

    @staticmethod
    def _validate_features(feature_names=None, column_types=None, validate_df=None):
        if feature_names is not None:
            if not isinstance(feature_names, list):
                raise ValueError(
                    f"Invalid feature_names parameter. Please provide the feature names as a list."
                )
            if validate_df is not None:
                if not set(feature_names).issubset(set(validate_df.columns)):
                    missing_columns = set(feature_names) - set(validate_df.columns)
                    raise ValueError(
                        f"Value mentioned in  feature_names is  not available in validate_df: {missing_columns} ")

        if column_types is not None and not isinstance(column_types, dict):
            raise ValueError(
                f"Invalid column_types parameter. Please provide the feature names as a dictionary."
            )

    @staticmethod
    def _validate_classification_threshold_label(classification_labels, classification_threshold=None,
                                                 target_values=None):
        if classification_labels is None:
            raise ValueError(
                f"Missing classification_labels parameter for classification model."
            )
        if classification_threshold is not None and not isinstance(classification_threshold, (int, float)):
            raise ValueError(
                f"Invalid classification_threshold parameter: {classification_threshold}. Please specify valid number."
            )

        if classification_threshold is not None:
            if classification_threshold != 0.5:
                if len(classification_labels) != 2:
                    raise ValueError(
                        f"Invalid classification_threshold parameter:  {classification_threshold} value is applicable "
                        f"only for binary classification. "
                    )

    @staticmethod
    def _validate_label_with_target(classification_labels, target_values=None):
        if target_values is not None:
            if set(target_values) != set(classification_labels):
                raise ValueError(
                    f"Invalid classification_labels parameter: {classification_labels} do not match with"
                    f" target column values{target_values}."
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
    def _validate_model_execution(prediction_function, df: pd.DataFrame, model_type, classification_labels) -> None:
        prediction = prediction_function(df)
        if isinstance(prediction, np.ndarray) or isinstance(prediction, list):
            if model_type == SupportedModelTypes.CLASSIFICATION.value:
                if not any(isinstance(y, float) for x in prediction for y in x):
                    raise ValueError("Model prediction should return float values ")
            if model_type == SupportedModelTypes.REGRESSION.value:
                if not any(isinstance(x, float) for x in prediction):
                    raise ValueError("Model prediction should return float values ")
        else:
            raise ValueError("Model should return numpy array or a list")

        if model_type == SupportedModelTypes.CLASSIFICATION.value:
            if not np.all(np.sum(prediction, axis=1) == 1):
                raise ValueError("Invalid Classification Model prediction. Sum of all probabilities should be 1 ")
            if not prediction.shape[1] == len(classification_labels):
                raise ValueError("Prediction output label shape and classification_labels shape do not match")

    @staticmethod
    def validate_df(df: pd.DataFrame, input_types) -> pd.DataFrame:
        if not set(input_types.keys()).issubset(set(df.columns)):
            missing_columns = set(input_types.keys()) - set(df.columns)
            raise ValueError(f"Value mentioned in  column_types is not available in dataframe: {missing_columns} ")

        else:
            pandas_inferred_input_types = df.dtypes.to_dict()
            for column, dtype in pandas_inferred_input_types.items():
                if (
                        input_types.get(column) == SupportedColumnType.NUMERIC.value
                        and dtype == "object"
                ):
                    df[column] = df[column].astype(float)
            return df
