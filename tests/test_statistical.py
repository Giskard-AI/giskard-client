from giskard.ml_worker.testing.functions import GiskardTestFunctions
from giskard.ml_worker.core.giskard_dataset import GiskardDataset
import numpy as np
import pytest

def test_disparate_impact(german_credit_data, german_credit_model):
    tests = GiskardTestFunctions()
    results = tests.statistical.test_disparate_impact(
        gsk_dataset=german_credit_data,
        protected_slice=lambda df: df[df.sex == "female"],
        unprotected_slice=lambda df: df[df.sex == "male"],
        model=german_credit_model,
        positive_outcome="Default",
        threshold=0.8,
        classification_label=None
    )
    assert results.passed