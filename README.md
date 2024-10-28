# Proteogram: Compression of Protein Structures into Image Data for Information Retrieval

This repo has the source code for the `proteogram` project and paper.

## Getting started

This project uses [Python Poetry](https://python-poetry.org/) to manage packages.  Using `poetry==1.8.3`, the following commands may be found useful.

To install all packages from the `pyproject.toml`:
```
poetry install
```

To add a package dependency to the environment (and add to the main dependencies section of the `pyproject.toml`):
```
poetry add <packagename>
```

To add a package dependency to the environment (and add to the dev dependencies section of the `pyproject.toml`):
```
poetry add <packagename> --dev
```

## Workflow

![](assets/Workflow-Structure-Compression.png)
