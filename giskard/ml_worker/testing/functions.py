from typing import List

from giskard.ml_worker.generated.ml_worker_pb2 import NamedSingleTestResult, SingleTestResult
from giskard.ml_worker.testing.drift_tests import DriftTests
from giskard.ml_worker.testing.heuristic_tests import HeuristicTests
from giskard.ml_worker.testing.metamorphic_tests import MetamorphicTests
from giskard.ml_worker.testing.performance_tests import PerformanceTests

EMPTY_SINGLE_TEST_RESULT = SingleTestResult()


class GiskardTestFunctions:
    tests_results: List[NamedSingleTestResult]
    metamorphic: MetamorphicTests
    heuristic: HeuristicTests
    performance: PerformanceTests

    def __init__(self) -> None:
        self.tests_results = []
        self.metamorphic = MetamorphicTests(self.tests_results)
        self.heuristic = HeuristicTests(self.tests_results)
        self.performance = PerformanceTests(self.tests_results)
        self.drift = DriftTests(self.tests_results)
