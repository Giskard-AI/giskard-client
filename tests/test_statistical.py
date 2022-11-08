from giskard.ml_worker.testing.functions import GiskardTestFunctions
from giskard.ml_worker.core.giskard_dataset import GiskardDataset
import numpy as np
import pytest

def test_disparate_impact(german_credit_data, german_credit_model):

    protected_df = german_credit_data.df[german_credit_data.df['sex'] == 'female']
    unprotected_df = german_credit_data.df[german_credit_data.df['sex'] == 'male']

    protected_ds = GiskardDataset(df=protected_df, target=german_credit_data.target, feature_types=german_credit_data.feature_types, column_types=german_credit_data.column_types)
    unprotected_ds = GiskardDataset(df=unprotected_df, target=german_credit_data.target, feature_types=german_credit_data.feature_types, column_types=german_credit_data.column_types)

    tests = GiskardTestFunctions()
    results = tests.statistical.test_disparate_impact(
        protected_ds=protected_ds,
        unprotected_ds=unprotected_ds,
        model=german_credit_model,
        positive_outcome="Default",
        threshold=0.8,
        classification_label=None
    )
    assert results.passed