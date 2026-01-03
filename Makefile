VENV=.venv
SRC_PATH=src/

.PHONY: setup install dist lint clean reset test

setup:
	uv venv

install: setup
	. $(VENV)/bin/activate && uv pip install -e ".[dev]"

build:
	. $(VENV)/bin/activate && python -m build

lint:
	uv run ruff format $(SRC_PATH)
	uv run ruff check --fix --unsafe-fixes $(SRC_PATH)

test:
	uv run pytest $(SRC_PATH) -v

clean:
	rm -rf $(VENV)
	find . -type f -name "*.DS_Store" -exec rm -f {} +
	find . -type d -name "build" -exec rm -rf {} +
	find . -type d -name "dist" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -exec rm -f {} +
	find . -type f -name "*.pyo" -exec rm -f {} +

reset: clean setup install
