import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

df = pd.read_csv("./sample_data/classification/titanic/titanic-train.csv")
selected_columns = ["Age", "Sex", "Embarked"]
df_selected_columns = df[selected_columns]
target = df["Survived"]

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
    steps=[("preprocessor", preprocessor), ("classifier", RandomForestClassifier())]
)

prediction_pipeline.fit(df_selected_columns, target)
