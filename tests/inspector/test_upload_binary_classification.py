import os

from dotenv import load_dotenv

from giskard import Client, ModelInspector

from .create_binary_classification import df, prediction_pipeline

load_dotenv()

client = Client(url=str(os.getenv("GISKARD_URL")), token=str(os.getenv("API_ACCESS_TOKEN")))


def test_upload_binary_classification():
    inspector = ModelInspector(
        prediction_function=prediction_pipeline.predict_proba,
        prediction_task="classification",
        input_types={"Age": "numeric", "Sex": "category", "Embarked": "category"},
        classification_labels=["dead", "alive"],
    )
    model_upload_response = inspector._upload_model(
        client, project_key="titanicé", model_name="titanic_v1"
    )
    print(model_upload_response.json())
    assert model_upload_response.status_code == 200
    df_upload_response = inspector._upload_df(
        client, df, project_key="titanicé", name="titanic_test"
    )
    print(df_upload_response.json())
    assert df_upload_response.status_code == 200
