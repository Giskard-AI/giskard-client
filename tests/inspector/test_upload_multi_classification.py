import os

from dotenv import load_dotenv

from giskard_client import Client, ModelInspector

from .create_multi_classification import df, prediction_pipeline

load_dotenv()

client = Client(url=str(os.getenv("GISKARD_URL")), token=str(os.getenv("API_ACCESS_TOKEN")))


def test_upload_multi_classification():
    inspector = ModelInspector(
        prediction_function=prediction_pipeline.predict_proba,
        prediction_task="classification",
        input_types={"Age": "numeric", "Sex": "category", "Embarked": "category"},
        classification_labels=["dead", "alive", "zombie"],
    )
    model_upload_response = inspector._upload_model(
        client, project_key="titanic-zombie", model_name="model_v1"
    )
    print(model_upload_response.json())
    assert model_upload_response.status_code == 200
    df_upload_response = inspector._upload_df(
        client, df, project_key="titanic-zombie", dataset_name="zombie_jack"
    )
    print(df_upload_response.json())
    assert df_upload_response.status_code == 200
