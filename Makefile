# Include tracked and new Markdown files using Git's standard ignore rules.
MDFILES = git ls-files -z --cached --others --exclude-standard -- '*.md'

# run checks on all files for the repo
quality:
	@echo "Running quality checks"
	ruff check
	ruff format --check
	@echo "mdformat --check"
	@$(MDFILES) | xargs -0 -r python -m mdformat --check --
	mypy --check-untyped-defs
	python scripts/sync_provenance.py --check

# style the code according to accepted standards for the repo
# Note: We run `ruff format` twice. Once to fix long lines before lint check
# and again to fix any formatting issues introduced by ruff check --fix
style:
	@echo "Running style fixes"
	ruff format
	ruff check --fix
	ruff format --silent
	@echo "mdformat"
	@$(MDFILES) | xargs -0 -r python -m mdformat --
	python scripts/sync_provenance.py

# clean up
clean:
	@echo "Running clean up"
	rm -fr build dist src/*.egg-info;
	find src setup.py | grep -E "(__pycache__|\.pyc|\.pyo|version\.py|version\.txt)" | xargs rm -fr;
