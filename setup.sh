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
        find . -type d -name ".venv" -delete
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
        *)
            log_error "Specify 'build' or 'serve'. For example: docs build"
            ;;
    esac
}