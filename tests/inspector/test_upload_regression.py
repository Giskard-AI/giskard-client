import os

from dotenv import load_dotenv

from giskard_client import Client, ModelInspector

from .create_regression import categorical_columns, df_test, numeric_columns, prediction_pipeline

load_dotenv()

client = Client(url=str(os.getenv("GISKARD_URL")), token=str(os.getenv("API_ACCESS_TOKEN")))


def test_upload_regression():
    inspector = ModelInspector(
        prediction_function=prediction_pipeline.predict,
        prediction_task="regression",
        input_types={
            **{column: "numeric" for column in numeric_columns},
            **{column: "category" for column in categorical_columns},
        },
    )
    model_upload_response = inspector._upload_model(
        client, project_key="house-prices", model_name="house-prices-v2"
    )
    print(model_upload_response.json())
    assert model_upload_response.status_code == 200
    df_upload_response = inspector._upload_df(
        client, df_test, project_key="house-prices", dataset_name="british-house"
    )
    print(df_upload_response.json())
    assert df_upload_response.status_code == 200
