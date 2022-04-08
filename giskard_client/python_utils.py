"""Various utility functions to manage Python environments"""

import os
from platform import python_version


def get_python_requirements() -> str:
    poetry_requirements = os.popen("poetry export --without-hashes --with-credentials").read()
    if poetry_requirements:
        return poetry_requirements
    pipenv_requirements = os.popen("pipenv lock -r").read()
    if pipenv_requirements:
        return pipenv_requirements
    pip_requirements = os.popen("pip freeze").read()
    if pip_requirements:
        return pip_requirements
    else:
        raise RuntimeError(
            "Python requirements could not be resolved. "
            + "Please use one of the following Python package managers: "
            + "Poetry, Pipenv or Pip."
        )


def get_python_version() -> str:
    return python_version()
