# `python-base` sets up all our shared environment variables
FROM amd64/python:3.9-slim as python-base

    # python
ENV PYTHONUNBUFFERED=1 \
    # prevents python creating .pyc files
    PYTHONDONTWRITEBYTECODE=1 \
    \
    # pip
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    \
    # poetry
    # https://python-poetry.org/docs/configuration/#using-environment-variables
    POETRY_VERSION=1.2.1 \
    # make poetry install to this location
    POETRY_HOME="/opt/poetry" \
    # make poetry create the virtual environment in the project's root
    # it gets named `.venv`
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    # do not ask any interactive question
    POETRY_NO_INTERACTION=1 \
    \
    # paths
    # this is where our requirements + virtual environment will live
    PYSETUP_PATH="/opt/pysetup" \
    VENV_PATH="/opt/pysetup/.venv"

ENV PATH="$VENV_PATH/bin:$POETRY_HOME/bin:$PATH"


# `builder-base` stage is used to build deps + create our virtual environment
FROM python-base as builder-base

RUN apt-get update && \
    apt-get install --no-install-recommends -y \
        git \
        curl \
        build-essential

RUN pip install --upgrade pip

# install poetry - respects $POETRY_VERSION & $POETRY_HOME
RUN curl -sSL https://install.python-poetry.org | python3 -

# copy project requirement files here to ensure they will be cached.
WORKDIR $PYSETUP_PATH
COPY pyproject.toml poetry.lock ./

# install deps - uses $POETRY_VIRTUALENVS_IN_PROJECT internally
# Allow installing dev dependencies to run tests
ARG INSTALL_DEV=false
RUN bash -c "if [ $INSTALL_DEV == 'true' ] ; then poetry install --no-root ; else poetry install --only main ; fi"



# ------------------------------------------------------------------------
# Add your python dependencies here, for example
# RUN poetry add catboost transformers "tensorflow^2.9.1"
# ------------------------------------------------------------------------


FROM builder-base as proto-builder

WORKDIR $PYSETUP_PATH
# install dev dependencies needed to protobuf code generation
RUN poetry install --with main,dev

# copy makefile and proto to generate python sources
COPY ./giskard ./giskard
COPY ./ml-worker-proto ./ml-worker-proto
COPY ./scripts ./scripts
COPY ./Makefile ./

RUN make generate-proto


# `production` image used for runtime
FROM python-base as production
COPY --from=builder-base $VENV_PATH $VENV_PATH
COPY --from=proto-builder $PYSETUP_PATH/giskard $PYSETUP_PATH/giskard

WORKDIR $PYSETUP_PATH

ENV GSK_HOST=0.0.0.0
ENV PYTHONPATH=.

CMD ["python", "giskard/cli.py", "worker", "start"]
EXPOSE 50051
