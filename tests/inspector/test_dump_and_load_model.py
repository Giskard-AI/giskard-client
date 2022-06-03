from io import StringIO

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from giskard import ModelInspector
from giskard.io_utils import decompress, pickle_loads


def test_dump_and_load_dummy_lambda_model():
    model_inspector = ModelInspector(
        prediction_function=lambda x: [42],
        prediction_task="regression",
        input_types={"feature": "numeric"},
    )
    inspector_serialized = model_inspector._serialize()
    inspector_reloaded = pickle_loads(decompress(inspector_serialized))
    assert inspector_reloaded.prediction_function(pd.DataFrame({"feature": ["foo"]})) == [42]


def test_dump_and_load_binary_classification_model():
    df = pd.read_csv(
        StringIO(
            """
        PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
        1,0,3,"Braund, Mr. Owen Harris",male,22,1,0,A/5 21171,7.25,,S
        2,1,1,"Cumings, Mrs. John Bradley (Florence Briggs Thayer)",female,38,1,0,PC 17599,71.2833,C85,C
        3,1,3,"Heikkinen, Miss. Laina",female,26,0,0,STON/O2. 3101282,7.925,,S
        """
        )
    )
    one_hot_encoder = Pipeline(
        steps=[
            ("imputation", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    numeric_encoder = Pipeline(
        steps=[("imputation", SimpleImputer(strategy="median")), ("rescaling", MinMaxScaler())]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("category", one_hot_encoder, ["Sex", "Embarked"]),
            ("numeric", numeric_encoder, ["Age"]),
        ]
    )
    prediction_pipeline = Pipeline(
        steps=[("preprocessor", preprocessor), ("classifier", DecisionTreeClassifier())]
    )
    prediction_pipeline.fit(df[["Age", "Sex", "Embarked"]], df["Survived"])
    model_inspector = ModelInspector(
        prediction_function=prediction_pipeline.predict_proba,
        prediction_task="classification",
        input_types={"Age": "numeric", "Sex": "category", "Embarked": "category"},
        classification_labels=["dead", "alive"],
    )
    assert model_inspector._validate_model(df)
    inspector_serialized = model_inspector._serialize()
    inspector_reloaded = pickle_loads(decompress(inspector_serialized))
    np.testing.assert_array_equal(
        inspector_reloaded.prediction_function(df),
        np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 1.0],
            ]
        ),
    )
