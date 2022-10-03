"""API Client to interact with the Giskard app"""
import logging
import warnings
from typing import List
from urllib.parse import urljoin

from requests.adapters import HTTPAdapter
from requests_toolbelt import sessions

import giskard
from giskard.analytics_collector import GiskardAnalyticsCollector, anonymize
from giskard.project import GiskardProject


class GiskardError(Exception):

    def __init__(self, message: str, status: int, code: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code


class ErrorHandlingAdapter(HTTPAdapter):
    def build_response(self, req, resp):
        response = super().build_response(req, resp)

        if not response.ok:
            giskard_error = None
            try:
                err_resp = response.json()
                giskard_error = GiskardError(
                    status=err_resp.get('status'),
                    code=err_resp.get('message'),
                    message=f"{err_resp.get('title', 'Unknown error')}: {err_resp.get('detail', 'no details')}")
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
        self.analytics = GiskardAnalyticsCollector()
        try:
            server_settings = self._session.get("settings").json()
            self.analytics.init(server_settings)
        except Exception:
            logging.warning(f"Failed to fetch server settings", exc_info=True)
        self.analytics.track("Init GiskardClient", {"client version": giskard.__version__})

    @property
    def session(self):
        return self._session

    def list_projects(self) -> List[GiskardProject]:
        self.analytics.track('List Projects')
        response = self._session.get('projects').json()
        return [GiskardProject(self._session, p['key'], analytics=self.analytics) for p in response]

    def get_project(self, project_key: str):
        """
        Function to get the project that belongs to the mentioned project key
        Args:
            project_key:
                The unique value of  project provided during project creation
        Returns:
            GiskardProject:
                The giskard project that belongs to the project key
        """
        self.analytics.track('Get Project', {'project_key': anonymize(project_key)})
        response = self._session.get(f'project', params={"key": project_key}).json()
        return GiskardProject(self._session, response['key'], analytics=self.analytics)

    def create_project(self, project_key: str, name: str, description: str = None):
        """
        Function to create a project in Giskard
        Args:
            project_key:
                The unique value of the project which will be used to identify  and fetch teh project in future
            name:
                The name of the project
            description:
                Describe your project
        Returns:
            GiskardProject:
                The project created in giskard
        """
        self.analytics.track('Create Project', {
            'project_key': anonymize(project_key),
            'description': anonymize(description),
            'name': anonymize(name),
        })
        try:
            response = self._session.post('project', json={
                "description": description,
                "key": project_key,
                "name": name
            }).json()
        except GiskardError as e:
            if e.code == 'error.http.409':
                warnings.warn("This project key already exists. "
                              "If you want to reuse existing project use get_project(“project_key”) instead")
            raise e
        actual_project_key = response.get('key')
        if actual_project_key != project_key:
            print(f"Project created with a key : {actual_project_key}")
        return GiskardProject(self._session, actual_project_key, analytics=self.analytics)
