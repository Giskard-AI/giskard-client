import re
from io import BytesIO
import pytest
import httpretty
import numpy as np
import pandas as pd
from requests_toolbelt.multipart import decoder

from giskard.giskard_client import GiskardClient
from giskard.io_utils import decompress, load_decompress
from giskard.model import GiskardModel
from giskard.project import GiskardProject


@httpretty.activate(verbose=True, allow_net_connect=False)
def test_upload_df(diabetes_dataset):
    httpretty.register_uri(
        httpretty.POST,
        "http://giskard-host:12345/api/v2/project/data/upload"
    )

    df, input_types, target = diabetes_dataset
    client = GiskardClient("http://giskard-host:12345", "SECRET_TOKEN")
    project = GiskardProject(client.session, "test-project")

    with pytest.raises(Exception):  # Error Scenario
        project.upload_df(
            df=df,
            feature_types=input_types,
            target=target,
            name="diabetes dataset")
    with pytest.raises(Exception):  # Error Scenario
        project.upload_df(df=df,
                          feature_types={"test":"test"},
                          name="diabetes dataset")

    project.upload_df(df=df,
                      feature_types=input_types,
                      name="diabetes dataset")

    req = httpretty.last_request()
    assert req.headers.get('Authorization') == 'Bearer SECRET_TOKEN'
    assert int(req.headers.get('Content-Length')) > 0
    assert req.headers.get('Content-Type').startswith('multipart/form-data; boundary=')

    multipart_data = decoder.MultipartDecoder(req.body, req.headers.get('Content-Type'))
    assert len(multipart_data.parts) == 2
    meta, file = multipart_data.parts
    assert meta.headers.get(b'Content-Type') == b'application/json'
    pd.testing.assert_frame_equal(df, pd.read_csv(BytesIO(decompress(file.content))))


@httpretty.activate(verbose=True, allow_net_connect=False)
def test_upload_model(linear_regression_diabetes: GiskardModel, diabetes_dataset):
    httpretty.register_uri(
        httpretty.POST,
        "http://giskard-host:12345/api/v2/project/models/upload"
    )
    df, input_types, target = diabetes_dataset

    model = linear_regression_diabetes
    client = GiskardClient("http://giskard-host:12345", "SECRET_TOKEN")
    project = GiskardProject(client.session, "test-project")
    project.upload_model(
        prediction_function=model.prediction_function,
        model_type=model.model_type,
        feature_names=model.feature_names,
        name="uploaded model",
        validate_df=df
    )
    with pytest.raises(Exception):
        project.upload_model(
            prediction_function=model.prediction_function,
            model_type=model.model_type,
            feature_names=input_types,
            name="uploaded model",
            validate_df=df
        )
    req = httpretty.last_request()
    assert req.headers.get('Authorization') == 'Bearer SECRET_TOKEN'
    assert int(req.headers.get('Content-Length')) > 0
    assert req.headers.get('Content-Type').startswith('multipart/form-data; boundary=')

    multipart_data = decoder.MultipartDecoder(req.body, req.headers.get('Content-Type'))
    assert len(multipart_data.parts) == 3
    meta, model_file, requirements_file = multipart_data.parts
    assert meta.headers.get(b'Content-Type') == b'application/json'
    loaded_model = load_decompress(model_file.content)

    assert np.array_equal(loaded_model(df), model.prediction_function(df))
    assert requirements_file.content.decode()
    # assert re.search(r'scikit-learn==[\d\\.]+', requirements_file.content.decode())
    # assert re.search(r'pandas==[\d\\.]+', requirements_file.content.decode())
