from giskard.ml_worker.testing.functions import GiskardTestFunctions


def test_metamorphic_invariance_no_change(german_credit_test_data, german_credit_model):
    perturbation = {}
    tests = GiskardTestFunctions()
    results = tests.metamorphic.test_metamorphic_invariance(
        df=german_credit_test_data,
        model=german_credit_model,
        perturbation_dict=perturbation,
        threshold=0.1,
    )
    assert results.actual_slices_size[0] == 1000
    assert results.passed


def _test_metamorphic_invariance_male_female(
    german_credit_test_data, german_credit_model, threshold
):
    perturbation = {"sex": lambda x: "female" if x.sex == "male" else "male"}
    tests = GiskardTestFunctions()
    results = tests.metamorphic.test_metamorphic_invariance(
        df=german_credit_test_data,
        model=german_credit_model,
        perturbation_dict=perturbation,
        threshold=threshold,
    )
    assert results.actual_slices_size[0] == len(german_credit_test_data)
    assert round(results.metric, 2) == 0.97
    return results.passed


def test_metamorphic_invariance_male_female(german_credit_test_data, german_credit_model):
    assert _test_metamorphic_invariance_male_female(
        german_credit_test_data, german_credit_model, 0.5
    )
    assert not _test_metamorphic_invariance_male_female(
        german_credit_test_data, german_credit_model, 1
    )


def test_metamorphic_invariance_2_perturbations(german_credit_test_data, german_credit_model):
    perturbation = {
        "duration_in_month": lambda x: x.duration_in_month * 2,
        "sex": lambda x: "female" if x.sex == "male" else "male",
    }
    tests = GiskardTestFunctions()
    results = tests.metamorphic.test_metamorphic_invariance(
        df=german_credit_test_data,
        model=german_credit_model,
        perturbation_dict=perturbation,
        threshold=0.1,
    )
    assert results.actual_slices_size[0] == len(german_credit_test_data)
    assert round(results.metric, 2) == 0.86


def test_metamorphic_invariance_some_rows(german_credit_test_data, german_credit_model):
    perturbation = {"sex": lambda x: "female" if x.sex == "male" else x.sex}
    tests = GiskardTestFunctions()
    results = tests.metamorphic.test_metamorphic_invariance(
        df=german_credit_test_data,
        model=german_credit_model,
        perturbation_dict=perturbation,
        threshold=0.1,
    )

    assert round(results.metric, 2) == 0.97


def test_metamorphic_invariance_regression(
    diabetes_dataset_with_target, linear_regression_diabetes
):
    perturbation = {"sex": lambda x: -0.044641636506989 if x.sex == 0.0506801187398187 else x.sex}
    tests = GiskardTestFunctions()
    results = tests.metamorphic.test_metamorphic_invariance(
        df=diabetes_dataset_with_target,
        model=linear_regression_diabetes,
        perturbation_dict=perturbation,
        output_sensitivity=0.1,
        threshold=0.1,
    )
    assert results.actual_slices_size[0] == len(diabetes_dataset_with_target)
    assert round(results.metric, 2) == 0.11
