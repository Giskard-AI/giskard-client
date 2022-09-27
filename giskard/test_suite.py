import requests.exceptions

from giskard.test_execution_result import GiskardTestExecutionResult
from giskard.test_execution_result import Status


class GiskardTestSuite:
    def __init__(self,
                 session,
                 id: int,
                 name: str,
                 project_id: int,
                 reference_dataset_id: int,
                 actual_dataset_id: int,
                 ) -> None:
        self.session = session
        self.id = id
        self.name = name
        self.project_id = project_id
        self.reference_dataset_id = reference_dataset_id
        self.actual_dataset_id = actual_dataset_id

    def execute(self):
        data = {
            "suiteId": self.id
        }
        headers = {
            "Content-Type": "application/json"
        }
        execute_res = ""
        try:
            execute_res = self.session.post("testing/suites/execute", json=data, headers=headers).json()
            print(execute_res)
        except requests.exceptions.RequestException as e:
            raise e
        return [GiskardTestExecutionResult(ter['testId'], Status(ter['status']), ter['result'][0]['name']) for ter in execute_res]

    def __repr__(self) -> str:
        return f"GiskardTestSuite(name='{self.name}')"





