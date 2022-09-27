import string
from enum import Enum


class Status(Enum):
    PASSED = 'PASSED'
    FAILED = 'FAILED'
    ERROR = 'ERROR'


class GiskardTestExecutionResult:
    def __init__(self,
                 id: int,
                 status: string,
                 name: string
                 ):
        self.id = id
        self.status = status
        self.name = name

    def __repr__(self) -> str:
        return f"GiskardTestExecutionRestult(name='{self.name}', status={self.status})"
