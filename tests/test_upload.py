from io import BytesIO
import pytest
import httpretty
import numpy as np
import pandas as pd
import json
from requests_toolbelt.multipart import decoder

from giskard.giskard_client import GiskardClient
from giskard.io_utils import decompress, load_decompress
from giskard.model import GiskardModel
from giskard.project import GiskardProject

url = "http://giskard-host:12345"
token = "SECRET_TOKEN"
auth = 'Bearer SECRET_TOKEN'
content_type = 'multipart/form-data; boundary='
model_name = 'uploaded model'
b_content_type= b'application/json'


@httpretty.activate(verbose=True, allow_net_connect=False)
def test_upload_df(diabetes_dataset):
    httpretty.register_uri(
        httpretty.POST,
        "http://giskard-host:12345/api/v2/project/data/upload"
    )
    df, input_types, target = diabetes_dataset
    dataset_name = "diabetes dataset"
    client = GiskardClient(url, token)
    project = GiskardProject(client.session, "test-project")

    with pytest.raises(Exception):  # Error Scenario
        project.upload_df(
            df=df,
            column_types=input_types,
            target=target,
            name=dataset_name)
    with pytest.raises(Exception):  # Error Scenario
        project.upload_df(df=df,
                          column_types={"test": "test"},
                          name=dataset_name)

    project.upload_df(df=df,
                      column_types=input_types,
                      name=dataset_name)

    req = httpretty.last_request()
    assert req.headers.get('Authorization') == auth
    assert int(req.headers.get('Content-Length')) > 0
    assert req.headers.get('Content-Type').startswith(content_type)

    multipart_data = decoder.MultipartDecoder(req.body, req.headers.get('Content-Type'))
    assert len(multipart_data.parts) == 2
    meta, file = multipart_data.parts
    assert meta.headers.get(b'Content-Type') == b_content_type
    pd.testing.assert_frame_equal(df, pd.read_csv(BytesIO(decompress(file.content))))


@httpretty.activate(verbose=True, allow_net_connect=False)
def _test_upload_model(model: GiskardModel, data):
    httpretty.register_uri(
        httpretty.POST,
        "http://giskard-host:12345/api/v2/project/models/upload"
    )
    df, input_types, target = data

    client = GiskardClient(url, token)
    project = GiskardProject(client.session, "test-project")
    if model.model_type == 'regression':
        project.upload_model(
            prediction_function=model.prediction_function,
            model_type=model.model_type,
            feature_names=model.feature_names,
            name=model_name,
            validate_df=df
        )
    else:
        project.upload_model(
            prediction_function=model.prediction_function,
            model_type=model.model_type,
            feature_names=model.feature_names,
            name=model_name,
            validate_df=df,
            classification_labels=model.classification_labels
        )

    with pytest.raises(Exception):
        project.upload_model(
            prediction_function=model.prediction_function,
            model_type=model.model_type,
            feature_names=input_types,
            name=model_name,
            validate_df=df
        )
    req = httpretty.last_request()
    assert req.headers.get('Authorization') == auth
    assert int(req.headers.get('Content-Length')) > 0
    assert req.headers.get('Content-Type').startswith(content_type)

    multipart_data = decoder.MultipartDecoder(req.body, req.headers.get('Content-Type'))
    assert len(multipart_data.parts) == 3
    meta, model_file, requirements_file = multipart_data.parts

    if model.model_type == 'classification':
        metadata = json.loads(meta.content)
        assert np.array_equal(model.classification_labels, metadata.get('classificationLabels'))
    assert meta.headers.get(b'Content-Type') == b_content_type
    loaded_model = load_decompress(model_file.content)

    assert np.array_equal(loaded_model(df), model.prediction_function(df))
    assert requirements_file.content.decode()


def test_upload_regression_model(linear_regression_diabetes, diabetes_dataset):
    _test_upload_model(linear_regression_diabetes, diabetes_dataset)


def test_upload_classification_model(german_credit_model, german_credit_data):
    _test_upload_model(german_credit_model, german_credit_data)
