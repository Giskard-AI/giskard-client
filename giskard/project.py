import json
import warnings
from typing import Callable, Dict, Iterable, List, Optional, Union

import numpy as np
import pandas as pd
import requests
from requests import Session

from giskard.analytics_collector import GiskardAnalyticsCollector, anonymize
from giskard.io_utils import compress, pickle_dumps, save_df
from giskard.model import SupportedModelTypes, SupportedColumnType
from giskard.python_utils import get_python_requirements, get_python_version


class GiskardProject:
    def __init__(self, session: Session, project_key: str, analytics: GiskardAnalyticsCollector = None) -> None:
        self.project_key = project_key
        self._session = session
        self.url = self._session.base_url.replace("/api/v2/", "")
        self.analytics = analytics or GiskardAnalyticsCollector()

    @staticmethod
    def _serialize(prediction_function: Callable[
        [pd.DataFrame],
        Iterable[Union[str, float, int]],
    ]) -> bytes:
        compressed_pickle: bytes = compress(pickle_dumps(prediction_function))
        return compressed_pickle

    def upload_model(
            self,
            prediction_function: Callable[[pd.DataFrame], Iterable[Union[str, float, int]]],
            model_type: str,
            feature_names: List[str] = None,
            name: str = None,
            validate_df: pd.DataFrame = None,
            target: Optional[List[str]] = None,
            classification_threshold: Optional[float] = None,
            classification_labels: Optional[List[str]] = None,
    ):
        """
        Function to upload the Model to Giskard
        Args:
            prediction_function:
                The model you want to predict. It could be any Python function with the signature of
                predict_proba for classification: It returns the classification probabilities for all
                the classification labels
                predict for regression : It returns the predicted values for regression models.
            model_type:
                "classification" for classification model
                "regression" for regression model
            feature_names:
                 A list of the feature names of prediction_function. If provided, this list will be used to filter
                 the dataframe's columns before applying the model. By default, the dataframe is used as-is, meaning that
                 all of its columns are passed to the model.
                 Some important remarks:
                    Make sure these features are contained in df
                    Make sure that prediction_function(df[feature_names]) does not return an error message
                    Make sure these features have the same order as the ones used in the pipeline of prediction_function.
            name:
                The name of the model you want to upload
            validate_df:
                Dataset used to validate the model
            target:
                The column name in validate_df corresponding to the actual target variable (ground truth).
            classification_threshold:
                The probability threshold in the case of a binary classification model
            classification_labels:
                The classification labels of your prediction when prediction_task="classification".
                 Some important remarks:
                    If classification_labels is a list of n elements, make sure prediction_function is
                     also returning probabilities
                    Make sure the labels have the same order as the output of prediction_function
        """
        self._validate_model_type(model_type)
        self._validate_features(feature_names=feature_names, validate_df=validate_df)
        self._validate_prediction_function(prediction_function)
        classification_labels = self._validate_classification_labels(classification_labels, model_type)

        transformed_pred_func = self.transform_prediction_function(prediction_function, feature_names)

        if model_type == SupportedModelTypes.CLASSIFICATION.value:
            self._validate_classification_threshold_label(classification_labels, classification_threshold)

        if validate_df is not None:
            if model_type == SupportedModelTypes.REGRESSION.value:
                self._validate_model_execution(transformed_pred_func, validate_df, model_type, target=target)
            elif target is not None and model_type == SupportedModelTypes.CLASSIFICATION.value:
                self._validate_target(target, validate_df.keys())
                target_values = validate_df[target].unique()
                self._validate_label_with_target(classification_labels, target_values)
                self._validate_model_execution(transformed_pred_func, validate_df, model_type, classification_labels,
                                               target=target)
            else:
                self._validate_model_execution(transformed_pred_func, validate_df, model_type, classification_labels,
                                               target=target)

        model = self._serialize(transformed_pred_func)
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
        self._session.post('project/models/upload', data={}, files=files)
        self.analytics.track("Upload Model", {
            "name": anonymize(name),
            "projectKey": anonymize(self.project_key),
            "languageVersion": get_python_version(),
            "modelType": model_type,
            "threshold": classification_threshold,
            "featureNames": anonymize(feature_names),
            "language": "PYTHON",
            "classificationLabels": anonymize(classification_labels)
        })
        print(f"Model successfully uploaded to project key '{self.project_key}' and is available at {self.url} ")

    def upload_df(
            self,
            df: pd.DataFrame,
            column_types: Dict[str, str],
            target: str = None,
            name: str = None,
    ) -> requests.Response:
        """
        Function to upload Dataset to Giskard
        Args:
            df:
                Dataset you want to upload
            column_types:
                A dictionary of column names and their types (numeric, category or text) for all columns of df.
            target:
                The column name in df corresponding to the actual target variable (ground truth).
            name:
                The name of the dataset you want to upload
        Returns:
                Response of the upload
        """
        self._validate_features(column_types=column_types)
        if target is not None:
            self._validate_target(target, df.keys())
        self.validate_df(df, column_types)
        self._validate_column_types(column_types)
        self._verify_category_columns(df, column_types)

        raw_column_types = df.dtypes.apply(lambda x: x.name).to_dict()
        data = compress(save_df(df))
        params = {
            "projectKey": self.project_key,
            "name": name,
            "featureTypes": column_types,
            "columnTypes": raw_column_types,
            "target": target
        }

        files = [
            ('metadata', (None, json.dumps(params), 'application/json')),
            ('file', data)
        ]

        result = self._session.post("project/data/upload", data={}, files=files)
        print(f"Dataset successfully uploaded to project key '{self.project_key}' and is available at {self.url} ")
        self.analytics.track("Upload dataset", {
            "projectKey": anonymize(self.project_key),
            "name": anonymize(name),
            "featureTypes": anonymize(column_types),
            "target": anonymize(target)
        }
                             )
        return result

    def upload_model_and_df(
            self,
            prediction_function: Callable[[pd.DataFrame], Iterable[Union[str, float, int]]],
            model_type: str,
            df: pd.DataFrame,
            column_types: Dict[str, str],
            feature_names: List[str] = None,
            target: str = None,
            model_name: str = None,
            dataset_name: str = None,
            classification_threshold: Optional[float] = None,
            classification_labels: Optional[List[str]] = None,
    ) -> None:
        """
        Function to upload Dataset and model to Giskard
        Args:
            prediction_function:
                The model you want to predict. It could be any Python function with the signature of
                predict_proba for classification: It returns the classification probabilities for all
                the classification labels
                predict for regression : It returns the predicted values for regression models.
            model_type:
                "classification" for classification model
                "regression" for regression model
            df:
                Dataset you want to upload
            column_types:
                A dictionary of column names and their types (numeric, category or text) for all columns of df.
            feature_names:
                 A list of the feature names of prediction_function. If provided, this list will be used to filter
                 the dataframe's columns before applying the model. By default, the dataframe is used as-is, meaning that
                 all of its columns are passed to the model.
                 Some important remarks:
                    Make sure these features are contained in df
                    Make sure that prediction_function(df[feature_names]) does not return an error message
                    Make sure these features have the same order as the ones used in the pipeline of prediction_function.
            target:
                The column name in df corresponding to the actual target variable (ground truth).
            model_name:
                The name of the model you want to upload
            dataset_name:
                The name of the dataset you want to upload
            classification_threshold:
                The probability threshold in the case of a binary classification model
            classification_labels:
                The classification labels of your prediction when prediction_task="classification".
                 Some important remarks:
                    If classification_labels is a list of n elements, make sure prediction_function is
                     also returning probabilities
                    Make sure the labels have the same order as the output of prediction_function
        """
        self.analytics.track("Upload model and dataset")
        self.upload_model(prediction_function=prediction_function,
                          model_type=model_type,
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
    def _validate_model_type(model_type):
        if model_type not in {task.value for task in SupportedModelTypes}:
            raise ValueError(
                f"Invalid model_type parameter: {model_type}. "
                + f"Please choose one of {[task.value for task in SupportedModelTypes]}."
            )

    @staticmethod
    def _validate_column_types(column_types):
        if column_types and type(column_types) is dict:
            if not set(column_types.values()).issubset(set(column_type.value for column_type in SupportedColumnType)):
                raise ValueError(
                    f"Invalid column_types parameter: "
                    + f"Please choose types among {[column_type.value for column_type in SupportedColumnType]}."
                )
        else:
            raise ValueError(
                f"Invalid column_types parameter: {column_types}. Please specify non-empty dictionary."
            )

    @staticmethod
    def transform_prediction_function(prediction_function, feature_names=None):
        if feature_names:
            return lambda df: prediction_function(df[feature_names])
        else:
            return prediction_function

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
                f" {target} column is not present in the dataset with columns:  {dataframe_keys}")

    @staticmethod
    def _validate_features(feature_names=None, column_types=None, validate_df=None):
        if feature_names is not None:
            if not isinstance(feature_names, list):
                raise ValueError(
                    f"Invalid feature_names parameter. Please provide the feature names as a list."
                )
            if validate_df is not None:
                if not set(feature_names).issubset(set(validate_df.columns)):
                    missing_feature_names = set(feature_names) - set(validate_df.columns)
                    raise ValueError(
                        f"Value mentioned in  feature_names is  not available in validate_df: {missing_feature_names} ")

        if column_types is not None and not isinstance(column_types, dict):
            raise ValueError(
                f"Invalid column_types parameter. Please provide the feature names as a dictionary."
            )

    @staticmethod
    def _validate_classification_threshold_label(classification_labels, classification_threshold=None):
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
                        f"Invalid classification_threshold parameter: {classification_threshold} value is applicable "
                        f"only for binary classification. "
                    )

    @staticmethod
    def _validate_label_with_target(classification_labels, target_values=None):
        if target_values is not None:
            target_values = target_values if isinstance(target_values, str) else [str(label) for label in target_values]
            if not set(target_values).issubset(set(classification_labels)):
                invalid_target_values = set(target_values) - set(classification_labels)
                warnings.warn(f"Target column value {invalid_target_values} not declared in "
                              f"classification_labels parameter: {classification_labels}")

    @staticmethod
    def _validate_classification_labels(classification_labels, model_type):
        res = None
        if model_type == SupportedModelTypes.CLASSIFICATION.value:
            if (
                    classification_labels is not None
                    and hasattr(classification_labels, "__iter__")
                    and not isinstance(classification_labels, (str, dict))  # type: ignore
            ):
                if len(classification_labels) > 1:
                    res: Optional[List[str]] = [str(label) for label in classification_labels]
                else:
                    raise ValueError(
                        f"Invalid classification_labels parameter: {classification_labels}. "
                        f"Please specify more than 1 label."
                    )
            else:
                raise ValueError(
                    f"Invalid classification_labels parameter: {classification_labels}. "
                    f"Please specify valid list of strings."
                )
        if model_type == SupportedModelTypes.REGRESSION.value and classification_labels is not None:
            warnings.warn("'classification_labels' parameter is ignored for regression model")
            res = None
        return res

    @staticmethod
    def _validate_model_execution(prediction_function, df: pd.DataFrame, model_type,
                                  classification_labels=None, target=None) -> None:
        try:
            if target is not None and target in df.columns:
                df = df.drop(target, axis=1)
            prediction = prediction_function(df)
        except Exception:
            raise ValueError("Invalid prediction_function input.\n"
                             "Please make sure that prediction_function(df[feature_names]) does not return an error "
                             "message before uploading in Giskard")
        GiskardProject._verify_prediction_output(model_type, prediction)
        GiskardProject._validate_classification_prediction(classification_labels, model_type, prediction)

    @staticmethod
    def _verify_prediction_output(model_type, prediction):
        if isinstance(prediction, np.ndarray) or isinstance(prediction, list):
            if model_type == SupportedModelTypes.CLASSIFICATION.value:
                if not any(isinstance(y, (np.floating, float)) for x in prediction for y in x):
                    raise ValueError("Model prediction should return float values ")
            if model_type == SupportedModelTypes.REGRESSION.value:
                if not any(isinstance(x, (np.floating, float)) for x in prediction):
                    raise ValueError("Model prediction should return float values ")
        else:
            raise ValueError("Model should return numpy array or a list")

    @staticmethod
    def _validate_classification_prediction(classification_labels, model_type, prediction):
        if model_type == SupportedModelTypes.CLASSIFICATION.value:
            if not np.all(np.logical_and(prediction >= 0, prediction <= 1)):
                raise ValueError(
                    "Invalid Classification Model prediction. Output probabilities should be in range [0,1]")
            if not np.all(np.isclose(np.sum(prediction, axis=1), 1, atol=0.0000001)):
                raise ValueError("Invalid Classification Model prediction. Sum of all probabilities should be 1 ")
            if prediction.shape[1] != len(classification_labels):
                raise ValueError("Prediction output label shape and classification_labels shape do not match")

    @staticmethod
    def validate_df(df: pd.DataFrame, column_types) -> pd.DataFrame:
        if not set(column_types.keys()).issubset(set(df.columns)):
            missing_columns = set(column_types.keys()) - set(df.columns)
            raise ValueError(
                f"Missing columns in dataframe according to column_types: {missing_columns}"
            )
        elif not set(df.columns).issubset(set(column_types.keys())):
            missing_columns = set(df.columns) - set(column_types.keys())
            raise ValueError(f"Missing column_types for columns: {missing_columns}")
        else:
            pandas_inferred_column_types = df.dtypes.to_dict()
            for column, dtype in pandas_inferred_column_types.items():
                if (
                        column_types.get(column) == SupportedColumnType.NUMERIC.value
                        and dtype == "object"
                ):
                    try:
                        df[column] = df[column].astype(float)
                    except Exception as e:
                        raise ValueError(f"Failed to convert column '{column}' to float") from e
            return df

    @staticmethod
    def _verify_category_columns(df: pd.DataFrame, column_types):
        for name, types in column_types.items():
            if types == SupportedColumnType.CATEGORY.value and len(df[name].unique()) > 30:
                warnings.warn(f"Categorical feature '{name}' contains {len(df[name].unique())} distinct values. If "
                              f"necessary use 'numeric' or 'text' in column_types instead")

    def __repr__(self) -> str:
        return f"GiskardProject(project_key='{self.project_key}')"
