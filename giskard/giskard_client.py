"""API Client to interact with the Giskard app"""
from typing import List
from urllib.parse import urljoin

from requests_toolbelt import sessions

from giskard.project import GiskardProject


class GiskardClient:
    def __init__(self, url: str, token: str):
        self._session = sessions.BaseUrlSession(base_url=urljoin(url, "/api/v2/"))
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
