MDFILES := $(shell find . -name "*.md" -not -path "./.venv/*" -not -path "./build/*" -not -path "./.pytest_cache/*")

# run checks on all files for the repo
quality:
	@echo "Running quality checks"
	ruff check
	ruff format --check
	python -m mdformat --check $(MDFILES)
	mypy --check-untyped-defs

# style the code according to accepted standards for the repo
# Note: We run `ruff format` twice. Once to fix long lines before lint check
# and again to fix any formatting issues introduced by ruff check --fix
style:
	@echo "Running style fixes"
	ruff format
	ruff check --fix
	ruff format --silent
	python -m mdformat $(MDFILES)
