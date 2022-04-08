import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

df_train = pd.read_csv("./sample_data/regression/house-prices/train.csv")
df_test = pd.read_csv("./sample_data/regression/house-prices/test.csv")
numeric_columns = [
    "LotFrontage",
    "LotArea",
    "TotalBsmtSF",
    "1stFlrSF",
    "2ndFlrSF",
    "FullBath",
    "HalfBath",
    "BedroomAbvGr",
    "KitchenAbvGr",
    "TotRmsAbvGrd",
    "Fireplaces",
    "GarageYrBlt",
    "GarageCars",
]
categorical_columns = [
    "MSSubClass",
    "MSZoning",
    "Street",
    "Alley",
    "LotShape",
    "LandContour",
    "Utilities",
    "LotConfig",
    "LandSlope",
    "Neighborhood",
    "FireplaceQu",
]
selected_columns = numeric_columns + categorical_columns
target = df_train["SalePrice"]

categorical_encoder = Pipeline(
    steps=[
        ("imputation", SimpleImputer(strategy="constant", fill_value="unknown")),
        ("one_hot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

numeric_encoder = Pipeline(
    steps=[("imputation", SimpleImputer(strategy="median")), ("rescaling", RobustScaler())]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("category", categorical_encoder, categorical_columns),
        ("numeric", numeric_encoder, numeric_columns),
    ]
)

prediction_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", ElasticNet())])

prediction_pipeline.fit(df_train[selected_columns], target)
