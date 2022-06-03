import os

from dotenv import load_dotenv

from giskard import Client, ModelInspector

from .create_text_classification import df_test, prediction_pipeline, twenty_train

load_dotenv()

client = Client(url=str(os.getenv("GISKARD_URL")), token=str(os.getenv("API_ACCESS_TOKEN")))


def test_upload_text_classification():
    inspector = ModelInspector(
        prediction_function=prediction_pipeline.predict_proba,
        prediction_task="classification",
        input_types={"data": "text"},
        classification_labels=twenty_train.target_names,
    )
    model_upload_response = inspector._upload_model(
        client, project_key="newspaper", model_name="predict_topic_v1"
    )
    print(model_upload_response.json())
    assert model_upload_response.status_code == 200
    df_upload_response = inspector._upload_df(
        client, df_test, project_key="newspaper", name="test_dataset"
    )
    print(df_upload_response.json())
    assert df_upload_response.status_code == 200
