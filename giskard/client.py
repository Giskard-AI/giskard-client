"""API Client to interact with the Giskard app"""
import json
from io import BytesIO, StringIO
from typing import Dict, Union

from apiclient import APIClient
from apiclient.authentication_methods import HeaderAuthentication
from apiclient.response import Response


class Client(APIClient):
    def __init__(self, url: str, token: str):
        super().__init__(authentication_method=HeaderAuthentication(token=token))
        self.url = url

    def upload_model(self, model: bytes, requirements: Union[bytes, StringIO, BytesIO],
                     params: Dict[str, str]) -> Response:
        endpoint = self.url + f"/api/v2/project/models/upload"
        files = [
            ('metadata', (None, json.dumps(params), 'application/json')),
            ('modelFile', model),
            ('requirementsFile', requirements)
        ]
        return self.post(endpoint=endpoint, data={}, files=files)

    def upload_data(self, data: Union[bytes, StringIO, BytesIO], params: Dict[str, str]) -> Response:
        endpoint = self.url + "/api/v2/project/data/upload"
        files = [
            ('metadata', (None, json.dumps(params), 'application/json')),
            ('file', data)
        ]
        return self.post(endpoint=endpoint, data={}, files=files)
