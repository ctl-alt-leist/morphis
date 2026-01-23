SRC_PATH = src/

.PHONY: setup install lint test build clean reset publish

# Development setup
setup:
	uv venv
	uv sync --extra dev

install: setup
	uv run pre-commit install

# Code quality
lint:
	uv run ruff format $(SRC_PATH)
	uv run ruff check --fix --unsafe-fixes $(SRC_PATH)

test:
	uv run pytest $(SRC_PATH)

# Build and release
build:
	uv build

publish:
	@VERSION=$$(grep '^version' pyproject.toml | head -1 | sed 's/.*"\(.*\)"/\1/'); \
	if git rev-parse "v$$VERSION" >/dev/null 2>&1; then \
		echo "Error: Tag v$$VERSION already exists"; \
		echo "Update the version in pyproject.toml first"; \
		exit 1; \
	fi; \
	echo "Publishing version $$VERSION"; \
	git tag "v$$VERSION" && \
	git push origin main --tags && \
	echo "Tagged and pushed v$$VERSION - GitHub Actions will publish to PyPI"

# Cleanup
clean:
	rm -rf .venv dist/ build/
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name ".DS_Store" -delete 2>/dev/null || true

reset: clean setup install
