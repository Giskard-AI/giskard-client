import asyncio
import logging
import requests
import grpc

from urllib.parse import urlparse

from giskard.settings import settings
from giskard.ml_worker.testing.git_testing_repository import clone_git_testing_repository

logger = logging.getLogger(__name__)


async def _start_grpc_server(is_server=False):
    from giskard.ml_worker.generated.ml_worker_pb2_grpc import add_MLWorkerServicer_to_server
    from giskard.ml_worker.server.ml_worker_service import MLWorkerServiceImpl
    from giskard.ml_worker.utils.network import find_free_port

    server = grpc.aio.server(
        # interceptors=[ErrorInterceptor()],
        options=[
            ("grpc.max_send_message_length", settings.max_send_message_length_mb * 1024 ** 2),
            ("grpc.max_receive_message_length", settings.max_receive_message_length_mb * 1024 ** 2),
        ]
    )

    port = settings.port or find_free_port()
    add_MLWorkerServicer_to_server(MLWorkerServiceImpl(port, not is_server), server)
    server.add_insecure_port(f"{settings.host}:{port}")
    await server.start()
    logger.info(f"Started ML Worker server on port {port}")
    logger.debug(f"ML Worker settings: {settings}")
    return server, port


async def start_ml_worker(is_server=False, is_silent=False,remote_host=None, remote_port=None, token=None):
    from giskard.ml_worker.bridge.ml_worker_bridge import MLWorkerBridge
    url = f'{remote_host}/api/v2/settings/general'
    host_name = urlparse(remote_host).hostname
    res = requests.get(url, headers={"Authorization": f"Bearer {token}"})
    if res.status_code == 401:
        raise Exception("Wrong Token")
    res_json = res.json()
    if remote_port is None:
        remote_port = res_json['externalMlWorkerEntrypointPort']
    clone_git_testing_repository(res_json['generalSettings']['instanceId'], is_silent)
    tasks = []
    server, grpc_server_port = await _start_grpc_server(is_server)
    if not is_server:
        logger.info(
            "Remote server host and port are specified, connecting as an external ML Worker"
        )
        tunnel = MLWorkerBridge(grpc_server_port, host_name, remote_port)
        tasks.append(asyncio.create_task(tunnel.start()))

    tasks.append(asyncio.create_task(server.wait_for_termination()))
    await asyncio.wait(tasks)
