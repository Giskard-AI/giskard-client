"""Inspect Machine Learning models"""

from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import logging
import re
from enum import Enum
from io import StringIO
from urllib.parse import quote_plus, urlencode

import numpy as np
import pandas as pd
import requests
from apiclient import APIClient
from IPython.display import HTML, display
from ipywidgets import widgets

from .client import Client
from .io_utils import compress, pickle_dumps, save_df
from .python_utils import get_python_requirements, get_python_version


class SupportedPredictionTask(Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class SupportedColumnType(Enum):
    NUMERIC = "numeric"
    CATEGORY = "category"
    TEXT = "text"


class ModelInspector:
    def __init__(
        self,
        prediction_function: Callable[
            [pd.DataFrame],
            Iterable[Union[str, float, int]],
        ],
        prediction_task: str,
        input_types: Dict[str, str],
        classification_threshold: Optional[float] = 0.5,
        classification_labels: Optional[List[str]] = None,
    ):
        if prediction_task in {task.value for task in SupportedPredictionTask}:
            self.prediction_task = prediction_task
        else:
            raise ValueError(
                f"Invalid prediction_task parameter: {prediction_task}. "
                + f"Please choose one of {[task.value for task in SupportedPredictionTask]}."
            )
        if input_types and type(input_types) is dict:
            if {input_type for input_type in input_types.values()} <= {
                column_type.value for column_type in SupportedColumnType
            }:
                self.input_types = input_types
            else:
                raise ValueError(
                    f"Invalid input_types parameter: {input_types}. "
                    + f"Please choose types among {[column_type.value for column_type in SupportedColumnType]}."
                )
        else:
            raise ValueError(
                f"Invalid input_types parameter: {input_types}. Please specify non-empty dictionary."
            )
        if callable(prediction_function):
            self.prediction_function = lambda df: prediction_function(df[input_types.keys()])
        else:
            raise ValueError(
                f"Invalid prediction_function parameter: {prediction_function}. Please specify Python function."
            )
        if isinstance(classification_threshold, (int, float)):
            self.classification_threshold = float(classification_threshold)
        else:
            raise ValueError(
                f"Invalid classification_threshold parameter: {classification_threshold}. Please specify valid number."
            )
        if prediction_task == SupportedPredictionTask.CLASSIFICATION.value:
            if (
                classification_labels is not None
                and hasattr(classification_labels, "__iter__")
                and not isinstance(classification_labels, (str, dict))  # type: ignore
            ):
                if len(classification_labels) > 1:
                    self.classification_labels: Optional[List[str]] = [
                        str(label) for label in classification_labels
                    ]
                else:
                    raise ValueError(
                        f"Invalid classification_labels parameter: {classification_labels}. Please specify more than 1 label."
                    )
            else:
                raise ValueError(
                    f"Invalid classification_labels parameter: {classification_labels}. Please specify valid list of strings."
                )
        else:
            self.classification_labels = None
        self.train_df = None  # will be set after you use the fit method

    def fit(self, train_df: pd.DataFrame, method: str = "kmeans") -> None:
        raise NotImplementedError

    def _validate_model(self, df: pd.DataFrame) -> bool:
        prediction = self.prediction_function(df)
        try:
            if not isinstance(prediction, np.ndarray):
                raise ValueError("Model should return numpy array")
            return True
        except Exception as e:
            logging.exception(e)
            return False

    def _validate_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if set(self.input_types.values()) < set(df.columns):
            missing_columns = set(df.columns) - set(self.input_types.values())
            raise ValueError(f"Missing input_types for columns: {missing_columns}")
        elif set(self.input_types.values()) > set(df.columns):
            missing_columns = set(self.input_types.values()) - set(df.columns)
            raise ValueError(
                f"Missing columns in dataframe according to input_types: {missing_columns}"
            )
        else:
            pandas_inferred_input_types = df.dtypes.to_dict()
            for column, dtype in pandas_inferred_input_types.items():
                if (
                    self.input_types.get(column) == SupportedColumnType.NUMERIC.value
                    and dtype == "object"
                ):
                    df[column] = df[column].astype(float)
            return df

    def _validate_project_key(self, project_key: str) -> None:
        valid_project_key = re.match(r"^[a-z0-9-]*$", project_key)
        if not valid_project_key:
            raise ValueError(
                f"Invalid project_key parameter: {project_key}. "
                + "Please use only lowercase alphanumeric characters and hyphens (-)."
            )

    @staticmethod
    def transmogrify(text: str) -> str:
        """
        Generate a cleaner string, without non-ascii or special characters,
        that can be used as a key and displayed in URL
        """
        # filter non ascii
        ascii_text = text.encode("ascii", "ignore").decode()
        # filter special characters
        without_special = "".join(e for e in ascii_text if e.isalnum() or e.isspace() or e == "-")
        return without_special.strip().replace(" ", "-").lower()

    def inspect(self, df: pd.DataFrame) -> None:
        self._validate_model(df)
        df = self._validate_df(df)
        self._generate_inspector_widget(df)

    def _serialize(self) -> bytes:
        compressed_pickle: bytes = compress(pickle_dumps(self))
        return compressed_pickle

    def upload_model(
        self, client: APIClient, project_key: str = "my-project", model_name: str = "my-model"
    ) -> requests.Response:
        project_key = self.transmogrify(project_key)
        logging.info(f"Uploading model '{model_name}' to project '{project_key}'...")
        model = self._serialize()
        requirements = get_python_requirements()
        response: requests.Response = client.upload_model(
            model=model,
            requirements=StringIO(requirements),
            params={
                "project_key": project_key,
                "model_name": model_name,
                "python_version": get_python_version(),
            },
        )
        logging.info(f"Uploading model '{model_name}' to project '{project_key}': Done!")
        return response

    def upload_df(
        self,
        client: APIClient,
        df: pd.DataFrame,
        project_key: str = "my-project",
        dataset_name: str = "my-dataset",
    ) -> requests.Response:
        project_key = self.transmogrify(project_key)
        df = self._validate_df(df)
        logging.info(f"Uploading dataset '{dataset_name}' to project '{project_key}'...")
        response: requests.Response = client.upload_data(
            data=compress(save_df(df)),
            params={"project_key": project_key, "dataset_name": f"{dataset_name}.csv.zst"},
        )
        logging.info(f"Uploading dataset '{dataset_name}' to project '{project_key}': Done!")
        return response

    def upload_model_and_df(
        self,
        df: pd.DataFrame,
        url: str,
        api_token: str,
        target_column: Optional[str] = None,
        model_name: str = "my-model",
        dataset_name: str = "my-dataset",
        project_key: str = "my-project",
    ) -> Tuple[str, str]:
        try:
            if not api_token:
                raise ValueError("Please enter your API token")
            if not url:
                raise ValueError("Please enter your Giskard URL")
            if not project_key:
                raise ValueError("Please choose a project key")
            client = Client(url=url, token=api_token)
            model_upload_response = self.upload_model(client, project_key, model_name)
            model_upload_response_dict = model_upload_response.json()
            df_upload_response = self.upload_df(client, df, project_key, dataset_name)
            df_upload_response_dict = df_upload_response.json()
            payload = {
                "mid": str(model_upload_response_dict["id"]),
                "did": str(df_upload_response_dict["id"]),
            }
            if target_column:
                payload["tgt"] = str(target_column)
            project_id = model_upload_response_dict["project_id"]
            result = urlencode(payload, quote_via=quote_plus)  # type: ignore
            inspection_url = f"{url}/main/projects/{project_id}/inspect?{result}"
            return ("OK", f"""<a href="{inspection_url}">Open in browser</a>""")
        except Exception as error:
            return ("NOK", str(error))

    def _generate_inspector_widget(self, df: pd.DataFrame) -> None:
        title = widgets.HTML(value="<h2>Inspect your model</h2>")
        subtitle_input = widgets.HTML(value="<h3>Dataset settings</h3>")
        # subtitle_model = widgets.HTML(value="<h3>Model settings</h3>")
        subtitle_auth = widgets.HTML(value="<h3>Authentication</h3>")
        overall_style = {"description_width": "120px"}

        target_column_input = widgets.Dropdown(
            options=[""] + df.columns.tolist(),
            value="",
            description="Target column",
            disabled=False,
            style=overall_style,
        )

        # model_name_input = widgets.Text(
        #     value="my-model",
        #     placeholder="my-model",
        #     description="Model name",
        #     disabled=False,
        #     style=overall_style,
        # )
        # dataset_name_input = widgets.Text(
        #     value="my-dataset",
        #     placeholder="my-dataset",
        #     description="Dataset name",
        #     disabled=False,
        #     style=overall_style,
        # )
        project_key_input = widgets.Text(
            value="",
            placeholder="choose-a-project-key",
            description="Project key",
            disabled=False,
            style=overall_style,
        )
        url_input = widgets.Text(
            value="https://app.giskard.ai",
            placeholder="https://app.giskard.ai",
            description="Giskard URL",
            disabled=False,
            style=overall_style,
        )
        api_token_input = widgets.Password(
            value="",
            placeholder="",
            description="API token",
            disabled=False,
            style=overall_style,
        )
        output_upload_results = widgets.Output(
            layout=widgets.Layout(margin="6px"),
        )
        upload_button = widgets.Button(
            description="Upload",
            button_style="info",
        )
        upload_button.style.button_color = "#00897B"

        def on_button_clicked(button: widgets.Button) -> None:
            (upload_result, upload_result_message) = ("NOK", "")
            with output_upload_results:
                output_upload_results.clear_output()
                print("Uploading...")
                (upload_result, upload_result_message) = self.upload_model_and_df(
                    df=df,
                    # model_name=model_name_input.value,
                    # dataset_name=dataset_name_input.value,
                    project_key=project_key_input.value,
                    url=url_input.value,
                    api_token=api_token_input.value,
                    target_column=target_column_input.value,
                )
                output_upload_results.clear_output()
                print(f"Uploading... {upload_result}")
                display(HTML(upload_result_message))

        upload_button.on_click(on_button_clicked)
        display(
            widgets.VBox(
                [
                    title,
                    subtitle_input,
                    # dataset_name_input,
                    target_column_input,
                    # subtitle_model,
                    # model_name_input,
                    subtitle_auth,
                    url_input,
                    project_key_input,
                    api_token_input,
                    upload_button,
                    output_upload_results,
                ]
            )
        )
