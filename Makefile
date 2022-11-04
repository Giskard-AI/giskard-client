#* Variables
SHELL := /usr/bin/env bash
PYTHON := python

#* Docker variables
IMAGE := docker.io/giskardai/ml-worker
VERSION := dev


.PHONY: all
all: clean init-submodules poetry-download install-dependencies generate-proto test

#* Poetry
.PHONY: init-submodules
init-submodules:
	git submodule update --init --recursive -q

#* Poetry
.PHONY: poetry-download
poetry-download:
	curl -sSL https://install.python-poetry.org | $(PYTHON) -

.PHONY: poetry-remove
poetry-remove:
	curl -sSL https://install.python-poetry.org | $(PYTHON) - --uninstall

.PHONY: install-dependencies
install-dependencies:
	poetry install

#* Installation
.PHONY: install
install: install-dependencies generate-proto

.PHONY: build
build: install
	poetry build


GENERATED_OUT:=giskard/ml_worker/generated
.PHONY: generate-proto
generate-proto: proto-remove
	rm -rf $(GENERATED_OUT) && mkdir -p $(GENERATED_OUT) && \
	source .venv/bin/activate && \
	python -m grpc_tools.protoc \
      -Iml-worker-proto/proto \
      --python_out=$(GENERATED_OUT) \
      --grpc_python_out=$(GENERATED_OUT) \
      --mypy_out=$(GENERATED_OUT) \
      --mypy_grpc_out=$(GENERATED_OUT) \
      ml-worker-proto/proto/ml-worker.proto && \
    python scripts/fix_grpc_generated_imports.py $(GENERATED_OUT) giskard.ml_worker.generated

.PHONY: proto-remove
proto-remove:
	rm -rf $(GENERATED_OUT)

.PHONY: pre-commit-install
pre-commit-install:
	poetry run pre-commit install

#* Formatters
.PHONY: codestyle
codestyle:
	poetry run pyupgrade --exit-zero-even-if-changed --py37-plus **/*.py
	poetry run isort --settings-path pyproject.toml ./
	poetry run black --config pyproject.toml ./

.PHONY: formatting
formatting: codestyle

.PHONY: test
test:
	poetry run pytest -c pyproject.toml --cov=giskard tests --cov-report=xml

.PHONY: test-cov-html
test-cov-html:
	poetry run pytest -c pyproject.toml --cov=giskard tests --cov-report=html

.PHONY: check-codestyle
check-codestyle:
	poetry run isort --diff --check-only --settings-path pyproject.toml ./
	poetry run black --diff --check --config pyproject.toml ./
	poetry run darglint --verbosity 2 giskard tests

.PHONY: mypy
mypy:
	poetry run mypy --config-file pyproject.toml ./

.PHONY: check-safety
check-safety:
	poetry check
	poetry run safety check --full-report
	poetry run bandit -ll --recursive giskard tests

.PHONY: lint
lint: test check-codestyle mypy check-safety

#* Docker
# Example: make docker VERSION=dev
# Example: make docker IMAGE=some_name VERSION=0.1.0
.PHONY: docker-build
docker-build:
	@echo Building docker $(IMAGE):$(VERSION) ...
	docker build --platform linux/amd64 \
		-t $(IMAGE):$(VERSION) . \
		-f ./docker/Dockerfile

.PHONY: docker-build
docker-build-nocache:
	@echo Building docker $(IMAGE):$(VERSION) ...
	docker build \
		-t $(IMAGE):$(VERSION) . \
		-f ./docker/Dockerfile --no-cache

# Example: make clean_docker VERSION=dev
# Example: make clean_docker IMAGE=some_name VERSION=0.1.0
.PHONY: docker-remove
docker-remove:
	@echo Removing docker $(IMAGE):$(VERSION) ...
	docker rmi -f $(IMAGE):$(VERSION)

# Example: make clean_docker VERSION=dev
# Example: make clean_docker IMAGE=some_name VERSION=0.1.0
.PHONY: docker-push
docker-push:
	@echo Pushing docker $(IMAGE):$(VERSION) ...
	docker image push $(IMAGE):$(VERSION)

#* Cleaning
.PHONY: pycache-remove
pycache-remove:
	find . | grep -E "(__pycache__|\.pyc|\.pyo$$)" | xargs rm -rf

#* Cleaning
.PHONY: coverage-remove
coverage-remove:
	rm -rf htmlcov coverage.xml

.PHONY: build-remove
build-remove:
	rm -rf build dist

.PHONY: clean
clean: pycache-remove build-remove proto-remove coverage-remove

.PHONY: clean-all
clean-all: clean docker-remove
