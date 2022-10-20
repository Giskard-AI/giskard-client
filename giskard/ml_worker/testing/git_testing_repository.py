import os
import logging
import click
from git import Repo

from giskard.settings import settings
logger = logging.getLogger(__name__)


def clone_git_testing_repository(instance_id: int):
    instance_path = os.path.expanduser(f'{settings.home}/{str(instance_id)}')
    os.makedirs(instance_path, exist_ok=True)
    repo_path = f'{instance_path}/project'
    if not os.path.isdir(repo_path):
        Repo.clone_from('http://localhost:3000/repository.git', to_path=f'{instance_path}/project')
        logger.info(f'Git testing repo cloned in {repo_path} You can now add some tests')
    else:
        repo = Repo(repo_path)
        repo.remotes.origin.fetch()
        actual_branch = repo.active_branch
        logger.warning(f'THE BRANCH IS {actual_branch}')
        commits_behind = repo.iter_commits(f'{actual_branch}...origin/main')
        nb_commit_behind = sum(1 for _ in commits_behind)
        if nb_commit_behind > 0:
            validation = click.confirm(f'You are currently on branch {actual_branch} and you are {nb_commit_behind} commits behind the origin/main. Do you want to pull from origin/main?')
            if validation:
                repo.remotes.origin.pull('main')
        else:
            logger.info(f'You are currently on branch {actual_branch} up to date')





