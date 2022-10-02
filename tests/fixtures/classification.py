import logging
import time
from pathlib import Path

import pandas as pd
import pytest
from sklearn import model_selection
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from giskard.model import GiskardModel

input_types = {'account_check_status': "category",
               'duration_in_month': "numeric",
               'credit_history': "category",
               'purpose': "category",
               'credit_amount': "numeric",
               'savings': "category",
               'present_emp_since': "category",
               'installment_as_income_perc': "numeric",
               'sex': "category",
               'personal_status': "category",
               'other_debtors': "category",
               'present_res_since': "numeric",
               'property': "category",
               'age': "numeric",
               'other_installment_plans': "category",
               'housing': "category",
               'credits_this_bank': "numeric",
               'job': "category",
               'people_under_maintenance': "numeric",
               'telephone': "category",
               'foreign_worker': "category",
               'default': "category"}


@pytest.fixture()
def german_credit_data():
    logging.info("Reading german_credit_prepared.csv")
    data = pd.read_csv(Path(__file__).parent / '../test_data/german_credit_prepared.csv')
    target = 'default'
    return data, input_types, target


@pytest.fixture()
def german_credit_model(german_credit_data) -> GiskardModel:
    start = time.time()
    df, column_types, target = german_credit_data
    feature_types = {i: column_types[i] for i in column_types if i != 'default'}

    columns_to_scale = [key for key in feature_types.keys() if feature_types[key] == "numeric"]

    numeric_transformer = Pipeline([('imputer', SimpleImputer(strategy='median')),
                                    ('scaler', StandardScaler())])

    columns_to_encode = [key for key in feature_types.keys() if feature_types[key] == "category" and key != "default"]

    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, columns_to_scale),
            ('cat', categorical_transformer, columns_to_encode)
        ]
    )
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('classifier', LogisticRegression(max_iter=100))])

    Y = df['default']
    X = df.drop(columns="default")
    x_train, x_test, y_train, y_test = model_selection.train_test_split(X, Y,
                                                                        test_size=0.20,
                                                                        random_state=30,
                                                                        stratify=Y)
    clf.fit(x_train, y_train)

    train_time = time.time() - start
    model_score = clf.score(x_test, y_test)
    logging.info(f"Trained model with score: {model_score} in {round(train_time * 1000)} ms")

    return GiskardModel(
        prediction_function=clf.predict_proba,
        model_type='classification',
        feature_names=list(feature_types),
        classification_threshold=0.5,
        classification_labels=clf.classes_
    )
