#!/usr/bin/bash

build() {
    clean 

    python -m venv .venv
    source .venv/Scripts/activate
    pip install pipenv
    pipenv install
}

clean() {
    if [ -n "$VIRTUAL_ENV" ]; then
        deactivate
    fi

    if [ -d ./.venv ]; then
        rm -rf ./.venv
    fi

    find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
}

requirements() {
    pipenv requirements > requirements.txt
}

docs() {
    case $1 in
        build)
            mkdocs build
            ;;
        serve)
            mkdocs serve
            ;;
        deploy)
            mkdocs gh-deploy --force
        *)
            log_error "Specify 'build', 'serve' or 'deploy'. For example: docs build"
            ;;
    esac
}

set() {
    if [[ $# -lt 2 ]]; then
        echo "Command Requires Two Arguments."
    else
        case $1 in
            bin_repo)
                python ./scripts/config.py -b "$2"
                ;;
            ml_repo)
                python ./scripts/config.py -m "$2"
                ;;
            ds_repo)
                python ./scripts/config.py -d "$2"
                ;;
            *)
                log_error "Specify 'bin_repo', 'ml_repo' or 'ds_repo'. For example: set bin_repo <repo id>"
                ;;
        esac
    fi
}

run() {
    case $1 in
        dev)
            streamlit run app.py
            ;;
        *)
            log_error "Specify 'dev'. For example: run dev"
            ;;
    esac
}