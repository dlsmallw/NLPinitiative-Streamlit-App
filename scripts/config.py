# Used for setting some constants for the project codebase
import toml
import typer
from pathlib import Path
from loguru import logger
from typing_extensions import Annotated

app = typer.Typer()

# Root Path
ROOT = Path(__file__).resolve().parents[1]

try:
    logger.info('Loading config.toml...')
    with open(ROOT / 'config.toml', 'r') as f:
        config = toml.load(f, dict)
        f.close()

    if 'repositories' not in config.keys() or \
        'bin_repo' not in config['repositories'] or \
        'ml_repo' not in config['repositories'] or \
        'ds_repo' not in config['repositories']:
        raise Exception('Malformed toml config file.')
    
    logger.success('config.toml loaded successfully.')
except Exception as e:
    logger.error(e)
    config = {
        'repositories': {
            'bin_repo': '',
            'ml_repo': '', 
            'ds_repo': ''
        }
    }

    with open(ROOT / 'config.toml', 'w') as f:
        toml.dump(config, f)
        f.close()


# HF Hub Repositories
BIN_REPO = config['repositories']['bin_repo']
ML_REPO = config['repositories']['ml_repo']
DATASET_REPO = config['repositories']['ds_repo']

@app.command()
def main(
    bin_repo: Annotated[str, typer.Option("--binary-repo", "-b")] = None,
    ml_repo: Annotated[str, typer.Option("--multilabel-regression-repo", "-m")] = None,
    ds_repo: Annotated[str, typer.Option("--dataset-repo", "-d")] = None
):
    toml_edited = False

    if bin_repo is not None and len(bin_repo) > 0:
        config['repositories']['bin_repo'] = bin_repo
        toml_edited = True
        logger.success(f'Successfully updated binary repository to {bin_repo}.')

    if ml_repo is not None and len(ml_repo) > 0:
        config['repositories']['ml_repo'] = ml_repo
        toml_edited = True
        logger.success(f'Successfully updated binary repository to {ml_repo}.')

    if ds_repo is not None and len(ds_repo) > 0:
        config['repositories']['ds_repo'] = ds_repo
        toml_edited = True
        logger.success(f'Successfully updated binary repository to "{ds_repo}".')

    if toml_edited:
        with open(ROOT / 'config.toml', 'w') as f:
            toml.dump(config, f)
            f.close()


if __name__ == "__main__":
    app()