from pathlib import Path

from giskard.settings import settings

run_dir = settings.home_dir / "run"
projects_dir = settings.home_dir / "projects"


def model_dir(project_key: str, file_name: str) -> Path:
    return projects_dir / project_key / "models" / file_name


def dataset_dir(project_key: str, file_name: str) -> Path:
    return projects_dir / project_key / "datasets" / file_name
