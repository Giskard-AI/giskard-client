"""API Client to interact with the Giskard app"""
from typing import List
from urllib.parse import urljoin

from requests.adapters import HTTPAdapter
from requests_toolbelt import sessions

from giskard.project import GiskardProject


class GiskardError(Exception):
    pass


class ErrorHandlingAdapter(HTTPAdapter):
    def build_response(self, req, resp):
        response = super().build_response(req, resp)

        if not response.ok:
            giskard_error = None
            try:
                err_resp = response.json()
                giskard_error = GiskardError(
                    f"{err_resp.get('title', 'Unknown error')}: {err_resp.get('detail', 'no details')}")
            except:  # NOSONAR
                response.raise_for_status()
            raise giskard_error
        return response


class GiskardClient:
    def __init__(self, url: str, token: str):
        base_url = urljoin(url, "/api/v2/")
        self._session = sessions.BaseUrlSession(base_url=base_url)
        self._session.mount(base_url, ErrorHandlingAdapter())
        self._session.headers.update({'Authorization': f"Bearer {token}"})

    @property
    def session(self):
        return self._session

    def list_projects(self) -> List[GiskardProject]:
        response = self._session.get('projects').json()
        return [GiskardProject(self._session, p['key']) for p in response]

    def get_project(self, project_key: str):
        response = self._session.get(f'project', params={"key": project_key}).json()
        return GiskardProject(self._session, response['key'])

    def create_project(self, project_key: str, name: str, description: str = None):
        response = self._session.post('project', json={
            "description": description,
            "key": project_key,
            "name": name
        }).json()
        actual_project_key = response.get('key')
        if actual_project_key != project_key:
            print(f"Project created with a key : {actual_project_key}")
        return GiskardProject(self._session, actual_project_key)
