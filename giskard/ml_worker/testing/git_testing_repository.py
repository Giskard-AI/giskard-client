import os
import logging
from pathlib import Path

import click
from git import Repo
from git import GitCommandError

from giskard.settings import settings, expand_env_var

logger = logging.getLogger(__name__)


def clone_git_testing_repository(instance_id: int, is_silent: bool, remote_host: str):
    instance_path = Path(expand_env_var(settings.home)) / str(instance_id)
    os.makedirs(instance_path, exist_ok=True)
    repo_path = f'{instance_path}/project'
    if not os.path.isdir(repo_path):
        git_repo_path = f'{remote_host}/repository.git'
        Repo.clone_from(git_repo_path, to_path=f'{instance_path}/project')
        logger.info(f'Git testing repo cloned in {repo_path} You can now add some tests')
    else:
        repo = Repo(repo_path)
        repo.remotes.origin.fetch()
        actual_branch = repo.active_branch
        num_behind = None
        try:
            commits_diff = repo.git.rev_list('--left-right', '--count', f'{actual_branch}...{actual_branch}@{{u}}')
            _, num_behind = commits_diff.split('\t')
        except GitCommandError:
            repo.remotes.origin.pull()
        if is_silent:
            logger.info(
                f'You are currently on branch {actual_branch} and you are {num_behind} commits behind.')
        else:
            if int(num_behind) > 0:
                validation = click.confirm(
                    f'You are currently on branch {actual_branch} and you are {num_behind} commits behind.'
                    f'Do you want to pull?')
                if validation:
                    repo.remotes.origin.pull()
            else:
                logger.info(f'You are currently on branch {actual_branch} up to date')
