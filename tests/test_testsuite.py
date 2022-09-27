import httpretty
import pytest

from giskard.giskard_client import  GiskardClient
from giskard.project import GiskardProject

url = "http://giskard-host:12345"
token = "SECRET_TOKEN"
auth = 'Bearer SECRET_TOKEN'
body_ts = '[{"id":1,"name":"test","project":{"id":2},"referenceDataset":{"id":3},"actualDataset":{"id":4},"model":{"id":5}},{"id":6,"name":"test2","project":{"id":7},"referenceDataset":{"id":8},"actualDataset":{"id":9},"model":{"id":10}}]'


@httpretty.activate()
def test_get_and_execute_testsuite():
    project_name = "test-project"
    httpretty.register_uri(
        httpretty.GET,
        f"http://giskard-host:12345/api/v2/project?key={project_name}",
        body='{"id": 1}',
        content_type="application/json",
        status=200
    )
    httpretty.register_uri(
        httpretty.GET,
        f"http://giskard-host:12345/api/v2/testing/suites/1",
        body=body_ts,
        content_type="application/json",
        status=200
    )
    httpretty.register_uri(
        httpretty.POST,
        f"http://giskard-host:12345/api/v2/testing/suites/execute",
        content_type="application/json",
        status=200
    )

    client = GiskardClient(url, token)
    project = GiskardProject(client.session, project_name)
    list_ts = project.list_test_suite()

    assert len(list_ts) == 2
    assert list_ts[0].name == "test"
    assert list_ts[0].id == 1
    assert list_ts[0].project_id == 2
    assert list_ts[0].reference_dataset_id == 3
    assert list_ts[0].actual_dataset_id == 4

    list_ts[0].execute()

    req = httpretty.last_request()
    assert req.headers.get('Authorization') == auth
