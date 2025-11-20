# Contributing to Speculators

Thank you for considering contributing to Speculators! We welcome contributions from the community to help improve and grow this project. This document outlines the process and guidelines for contributing.

## How Can You Contribute?

There are many ways to contribute to Speculators:

- **Reporting Bugs**: If you encounter a bug, please let us know by creating an issue.
- **Suggesting Features**: Have an idea for a new feature? Open an issue to discuss it.
- **Improving Documentation**: Help us improve our documentation by submitting pull requests.
- **Writing Code**: Contribute code to fix bugs, add features, or improve performance.
- **Reviewing Pull Requests**: Provide feedback on open pull requests to help maintain code quality.

## Getting Started

### Prerequisites

Before contributing, ensure you have the following installed:

- Python 3.10 or higher
- pip (Python package manager)
- Tox
- Git

### Setting Up the Repository

You can either clone the repository directly or fork it if you plan to contribute changes back:

#### Option 1: Cloning the Repository

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/neuralmagic/speculators.git
   cd speculators
   ```

#### Option 2: Forking the Repository

1. Fork the repository by clicking the "Fork" button on the repository's GitHub page.

2. Clone your forked repository to your local machine:

   ```bash
   git clone https://github.com/<your-username>/speculators.git
   cd speculators
   ```

### Installing Dependencies

To install the required dependencies for the package and development, run:

```bash
pip install -e .[dev]
```

The `-e` flag installs the package in editable mode, allowing you to make changes to the code without reinstalling it. The `[dev]` part installs additional dependencies needed for development, such as testing and linting tools.

Note: the data generation components off speculators (i.e. `src/speculators/data_generation`) require a vLLM installation. Code in this module should be run in a separate environment, with vllm. This can be done by installing the `datagen` extra using `pip install -e .[datagen]` (or `pip install -e .[datagen,dev]` for development).

## Code Style and Guidelines

We follow strict coding standards to ensure code quality and maintainability. Please adhere to the following guidelines:

- **Code Style**: Use [Ruff](https://github.com/astral-sh/ruff) for formatting and linting.
- **Type Checking**: Use [Mypy](http://mypy-lang.org/) for type checking.
- **Testing**: Write unit tests for new features and bug fixes. Use [pytest](https://docs.pytest.org/) for testing.
- **Documentation**: Update documentation for any changes to the codebase.

We use Tox to simplify running various tasks in isolated environments. Tox standardizes environments to ensure consistency across local development, CI/CD pipelines, and releases. This guarantees that the code behaves the same regardless of where it is executed.

Additionally, to ensure consistency and quality of the codebase, we use [ruff](https://github.com/astral-sh/ruff) for linting and styling, [mypy](https://github.com/python/mypy) for type checking, and [mdformat](https://github.com/hukkin/mdformat) for formatting Markdown files.

### Code Quality and Style

To check code quality, including linting and formatting:

```bash
tox -e quality
```

To automatically fix style issues, use:

```bash
tox -e style
```

### Type Checking

To ensure type safety using Mypy:

```bash
tox -e types
```

### Automating Quality Checks with Pre-Commit Hooks (Optional)

We use [pre-commit](https://pre-commit.com/) to automate quality checks before commits. Pre-commit hooks run checks like linting, formatting, and type checking, ensuring that only high-quality code is committed.

To install the pre-commit hooks, run:

```bash
pre-commit install
```

This will set up the hooks to run automatically before each commit. To manually run the hooks on all files, use:

```bash
pre-commit run --all-files
```

## Running Tests

For testing, we use [pytest](https://docs.pytest.org/) as our testing framework. We have different test suites for unit tests, integration tests, end-to-end tests, and data generation tests. To run the tests, you can use Tox, which will automatically create isolated environments for each test suite. Tox will also ensure that the tests are run in a consistent environment, regardless of where they are executed.

### Running All Tests

To run all tests:

```bash
tox
```

### Running Specific Tests

`tox` will set up the environment for the test environment for each test you run.

- Unit tests (focused on individual components with mocking):

  ```bash
  tox -e test-unit
  ```

- Integration tests (focused on interactions between components ideally without mocking):

  ```bash
  tox -e test-integration
  ```

- End-to-end tests (focused on the entire system and user interfaces):

  ```bash
  tox -e test-e2e
  ```

- Data generation tests (focused on the data generation process):

Note: This creates an environment with the `datagen` extra installed (including vllm) and executes the tests in this environment.

```bash
tox -e test-datagen
```

### Running Tests with Coverage

To ensure your changes are covered by tests, run:

```bash
tox -e test-unit -- --cov=speculators --cov-report=html
```

Review the coverage report to confirm that your new code is adequately tested.

## Submitting Changes

1. **Create a Branch**: Create a new branch for your changes:

   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Changes**: Implement your changes in the appropriate files. Commit your changes with clear and descriptive commit messages.

3. **Update Documentation**: Update or add documentation to reflect your changes. This includes updating README files, docstrings, and any relevant guides.

4. **Run Tests and Quality Checks**: Before submitting your changes, ensure all tests pass and code quality checks are satisfied:

   ```bash
   tox
   ```

5. **Push Changes**: Push your branch to your forked repository (if you forked):

   ```bash
   git push origin feature/your-feature-name
   ```

6. **Open a Pull Request**: From the fork repository, use the contribute button to open a pull request to the original repository's main branch. Provide a clear description of your changes and link any related issues.

## Reporting Issues

If you encounter a bug or have a feature request, please open an issue on GitHub. Include as much detail as possible, such as:

- Steps to reproduce the issue
- Expected and actual behavior
- Environment details (OS, Python version, etc.)

## Community Standards

We are committed to fostering a welcoming and inclusive community. Please read and adhere to our [Code of Conduct](https://github.com/neuralmagic/speculators/blob/main/CODE_OF_CONDUCT.md).

## Additional Resources

- [CODE_OF_CONDUCT.md](https://github.com/neuralmagic/speculators/blob/main/CODE_OF_CONDUCT.md): Our expectations for community behavior.
- [tox.ini](https://github.com/neuralmagic/speculators/blob/main/tox.ini): Configuration for Tox environments.
- [.pre-commit-config.yaml](https://github.com/neuralmagic/speculators/blob/main/.pre-commit-config.yaml): Configuration for pre-commit hooks.

## License

By contributing to Speculators, you agree that your contributions will be licensed under the [Apache License 2.0](https://github.com/neuralmagic/speculators/blob/main/LICENSE).
