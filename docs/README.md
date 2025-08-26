# Speculators Documentation

This directory contains the documentation for the Speculators project, built using [MkDocs](https://www.mkdocs.org/) with the [Material theme](https://squidfunk.github.io/mkdocs-material/).

## Prerequisites

- Python 3.9 or higher
- Install the development dependencies: `pip install -e .[dev]`

## Building the Documentation

### Local Development

To build and serve the documentation locally for development:

```bash
# Navigate to the project root
cd /path/to/speculators

# Start the development server with live reload
mkdocs serve
```

The documentation will be available at `http://127.0.0.1:8000` and will automatically reload when you make changes to any documentation files.

### Production Build

To build the static documentation files for production:

```bash
# Build the documentation
mkdocs build

# The static files will be generated in the site/ directory
```

## Configuration

The documentation is configured via `mkdocs.yml` in the project root. Key features include:

- **Material Theme**: Modern, responsive design with dark/light mode support
- **API Documentation**: Auto-generated from Python docstrings using `mkdocstrings`
- **Search**: Full-text search functionality
- **Navigation**: Automatic navigation generation with weight-based ordering
- **Extensions**: Support for code highlighting, math rendering (MathJax), and more

## Versioning

The documentation supports versioning using [mike](https://github.com/jimporter/mike):

```bash
# Deploy a specific version
mike deploy --push --update-aliases 1.0 latest

# Set the default version
mike set-default --push latest
```

## Deployment

The documentation is automatically deployed to GitHub Pages when changes are pushed to the main branch. Manual deployment can be done using:

```bash
# Deploy to GitHub Pages
mkdocs gh-deploy
```
