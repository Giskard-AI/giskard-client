import asyncio
import logging
import os
from pathlib import Path

import requests
import grpc

from urllib.parse import urlparse

from giskard.settings import settings, expand_env_var
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


async def start_ml_worker(server_instance=None, is_silent=False,remote_host=None, remote_port=None, token=None):
    from giskard.ml_worker.bridge.ml_worker_bridge import MLWorkerBridge
    url = f'{remote_host}/api/v2/settings/general'
    host_name = urlparse(remote_host).hostname
    res = requests.get(url, headers={"Authorization": f"Bearer {token}"})
    if res.status_code == 401:
        raise Exception("Invalid API Token")
    if res.status_code != 200:
        raise Exception("Failed to connect to Giskard Instance")
    res_json = res.json()
    if remote_port is None:
        remote_port = res_json['externalMlWorkerEntrypointPort']
    if server_instance is None:
        clone_git_testing_repository(res_json['generalSettings']['instanceId'], is_silent, remote_host)
    else:
        instance_path = Path(expand_env_var(settings.home)) / server_instance
        os.makedirs(instance_path, exist_ok=True)
        return
    tasks = []
    server, grpc_server_port = await _start_grpc_server(server_instance is not None)
    if server_instance is None:
        logger.info(
            "Remote server host and port are specified, connecting as an external ML Worker"
        )
        tunnel = MLWorkerBridge(grpc_server_port, host_name, remote_port)
        tasks.append(asyncio.create_task(tunnel.start()))

    tasks.append(asyncio.create_task(server.wait_for_termination()))
    await asyncio.wait(tasks)
