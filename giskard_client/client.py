"""API Client to interact with the Giskard app"""

from typing import Dict

from apiclient import APIClient
from apiclient.authentication_methods import HeaderAuthentication
from apiclient.response import Response


class Client(APIClient):
    def __init__(self, url: str, token: str):
        super().__init__(authentication_method=HeaderAuthentication(token=token))
        self.url = url

    def upload_model(self, model: bytes, requirements: str, params: Dict[str, str]) -> Response:
        endpoint = self.url + "/api/v1/third-party/models/upload"
        files = {"model": model, "requirements": requirements}
        return self.post(endpoint=endpoint, data={}, params=params, files=files)

    def upload_data(self, data: bytes, params: Dict[str, str]) -> Response:
        endpoint = self.url + "/api/v1/third-party/data/upload"
        files = {"data_file": data}
        return self.post(endpoint=endpoint, data={}, params=params, files=files)
