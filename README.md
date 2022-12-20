# CLS notebooks

This is a collection of educational [jupyter notebooks](https://jupyter.org) for the CLS (computational life sciences) course series.

## Packages and environments

We use [conda](https://conda.io) for package and environment management.
We provide an `environment.yml` file for easy setup.

Create the environment from the `environment.yml` file:
```
conda env create --file environment.yml
```

Activate the new environment:
```
conda activate cls_notebooks
```

Verify that the new environment was installed correctly:
```
conda env list
```

## Formatting et cetera

We use [black](https://github.com/psf/black) to format our python code.

We use [pydocstyle](http://www.pydocstyle.org) for checking compliance with python docstring conventions.

We use [pylint](https://pypi.org/project/pylint/) for static code analysis.

## Pre-commit hooks

We use [pre-commit](https://pre-commit.com) to manage our pre-commit hooks.
The first time you clone this project running `pre-commit install` should be the first thing you do.
If you want to manually run all pre-commit hooks, run `pre-commit run --all-files`.
