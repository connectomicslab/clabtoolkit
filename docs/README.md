# clabtoolkit documentation

Source of the documentation published at <https://clabtoolkit.readthedocs.io>. It is built with [Sphinx](https://www.sphinx-doc.org) and the Read the Docs theme.

## Build the documentation

From the root of the repository:

```bash
pip install -e ".[docs]"
cd docs
make html
```

The pages are written to `docs/build/html` (ignored by git). Open `docs/build/html/index.html` in a browser.

Nothing is mocked: the API pages import the real modules, so the package and all its dependencies must be installed (the `pip install -e` above does it). Python 3.11 or newer is required.

To check that the documentation builds without warnings, as it does on Read the Docs, run:

```bash
make html SPHINXOPTS="-W --keep-going -E"
```

`-E` rebuilds everything, and `-W` turns warnings into errors. Run `make clean` to remove the build folder.

## Structure

```text
docs/
├── Makefile, make.bat      Sphinx build commands
└── source/
    ├── conf.py             Sphinx configuration
    ├── index.rst           Landing page and table of contents
    ├── installation.rst    Installation and optional dependencies
    ├── quickstart.rst      Short examples with simulated data
    ├── examples.rst        Links to the notebooks of usage_examples/
    ├── api.rst             Table of contents of the module pages
    ├── contributing.rst    Includes CONTRIBUTING.rst
    ├── changelog.rst       Includes HISTORY.rst
    ├── modules/            One page per module of the package
    └── _static/            Logo, favicon and images
```

`contributing.rst` and `changelog.rst` only include the files of the repository root, so edit `CONTRIBUTING.rst` and `HISTORY.rst` there.

The reference of every function and class is generated from the docstrings (`autodoc` with `napoleon`, so numpy-style docstrings are supported). Each page in `source/modules/` adds an introduction, a list of the main functions and usage examples on top of the generated reference.

## Edit the documentation

- **Docstrings:** edit them in the code. Use the numpy style (`Parameters`, `Returns`, `Raises`, `Examples`) and leave a blank line before every list. The build fails with `-W` if a docstring is not valid reStructuredText.
- **Module pages:** edit `source/modules/<module>.rst`. Run the examples before you add them, and update the page when a function or an argument changes.
- **New module:** create `source/modules/<module>.rst`, copy the structure of an existing page (`automodule` directive, key features, main functions, usage examples), and add it to the list in `source/api.rst`. Add its notebook to `source/examples.rst` and to `usage_examples/README.md`.
- **Release number:** it is set in `release` in `source/conf.py`, and should match the version of `pyproject.toml`.

## Read the Docs

The configuration is in `.readthedocs.yml` in the root of the repository. It uses Python 3.12, installs `requirements-docs.txt` and the package (`pip install -e .`), and builds `docs/source/conf.py`.
