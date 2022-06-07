import logging

import pytest
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

from giskard.model import GiskardModel, SupportedModelTypes


@pytest.fixture()
def diabetes_dataset():
    diabetes = datasets.load_diabetes(as_frame=True)

    feature_types = {feature: 'numeric' for feature in diabetes['feature_names']}
    target = 'target'
    return diabetes['data'], feature_types, target


@pytest.fixture()
def linear_regression_diabetes() -> GiskardModel:
    diabetes = datasets.load_diabetes()

    diabetes_x = diabetes['data']
    diabetes_y = diabetes['target']

    # Split the data into training/testing sets
    diabetes_x_train = diabetes_x[:-20]
    diabetes_x_test = diabetes_x[-20:]

    # Split the targets into training/testing sets
    diabetes_y_train = diabetes_y[:-20]
    diabetes_y_test = diabetes_y[-20:]

    # Create linear regression object
    regressor = linear_model.LinearRegression()

    # Train the model using the training sets
    regressor.fit(diabetes_x_train, diabetes_y_train)

    # Make predictions using the testing set
    diabetes_y_pred = regressor.predict(diabetes_x_test)

    logging.info(f"Model MSE: {mean_squared_error(diabetes_y_test, diabetes_y_pred)}")

    return GiskardModel(
        prediction_function=regressor.predict,
        model_type=SupportedModelTypes.REGRESSION.value,
        feature_names=diabetes['feature_names']
    )
