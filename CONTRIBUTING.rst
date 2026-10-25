.. highlight:: shell

============
Contributing
============

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given.

In short: fork the repository, install it with ``pip install -e .[dev,test]``, make your
change on a branch, run ``ruff check``, ``black --check`` and ``pytest``, and open a pull
request against ``main``. The maintainers review and merge pull requests and are the only
ones who prepare and publish releases (see `Releases`_).

By contributing, you agree that your contribution is licensed under the Apache-2.0
license of the project (see ``LICENSE``).

You can contribute in many ways:

Types of Contributions
----------------------

Report Bugs
~~~~~~~~~~~

Report bugs at https://github.com/connectomicslab/clabtoolkit/issues.

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

Fix Bugs
~~~~~~~~

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help
wanted" is open to whoever wants to implement it.

Implement Features
~~~~~~~~~~~~~~~~~~

Look through the GitHub issues for features. Anything tagged with "enhancement"
and "help wanted" is open to whoever wants to implement it.

Write Documentation
~~~~~~~~~~~~~~~~~~~

Connectomics Lab Toolkit could always use more documentation, whether as part of the
official Connectomics Lab Toolkit docs, in docstrings, or even on the web in blog posts,
articles, and such.

Submit Feedback
~~~~~~~~~~~~~~~

The best way to send feedback is to file an issue at https://github.com/connectomicslab/clabtoolkit/issues.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions
  are welcome :)

Get Started!
------------

Ready to contribute? Here's how to set up `clabtoolkit` for local development.

1. Fork the `clabtoolkit` repo on GitHub.
2. Clone your fork locally::

    $ git clone git@github.com:your_name_here/clabtoolkit.git

3. Install your local copy, with the development tools, into a virtual environment::

    $ cd clabtoolkit/
    $ python -m venv .venv
    $ source .venv/bin/activate
    $ pip install -e .[dev,test]

   A conda environment with the dependencies is also available in
   ``environment.yaml``.

4. Install the pre-commit hooks, so the formatting and lint checks run on every commit::

    $ pre-commit install

5. Add the main repository as ``upstream`` and branch from an up-to-date ``main``::

    $ git remote add upstream https://github.com/connectomicslab/clabtoolkit.git
    $ git fetch upstream
    $ git checkout -b name-of-your-bugfix-or-feature upstream/main

   Now you can make your changes locally. If ``main`` moves while you work, bring your
   branch up to date with ``git fetch upstream`` and ``git rebase upstream/main``.

6. When you're done making changes, check that your changes pass the linter, the
   formatting check and the tests. These are the same checks that run in CI::

    $ ruff check clabtoolkit tests
    $ black --check clabtoolkit tests
    $ pytest

   CI installs the latest ``black``, whose formatting can differ slightly from older
   releases, so keep it up to date (``pip install -U black``) before running the check.

7. Commit your changes and push your branch to GitHub::

    $ git add <the files you changed>
    $ git commit -m "docs: short description of your change"
    $ git push origin name-of-your-bugfix-or-feature

   Use a short, imperative summary line. A prefix such as ``feat:``, ``fix:``, ``docs:``,
   ``test:`` or ``refactor:`` is welcome, as in the project history.

8. Submit a pull request through the GitHub website, against the ``main`` branch.

Coding Standards
----------------

* Python 3.11 or newer, formatted with ``black`` and linted with ``ruff`` (line length 88).
  Both are configured in ``pyproject.toml``.
* Public functions, classes and methods need a docstring in the NumPy style (``Parameters``,
  ``Returns``, ``Raises``), like the rest of the code.
* Add type hints to new code.
* Keep the public API backward compatible. If a change breaks it, say so in the pull request
  so the maintainers can decide how to release it.

Building the Documentation
--------------------------

The documentation is built with Sphinx. To check your changes locally::

    $ pip install -e .[docs]
    $ make -C docs html

Open ``docs/build/html/index.html`` in a browser to see the result.

Pull Request Guidelines
-----------------------

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. If the pull request adds functionality, the docs should be updated. Put
   your new functionality into a function with a docstring, add the feature to
   the list in README.rst, and add an entry to the "Unreleased" section of
   HISTORY.rst (create the section at the top if it does not exist).
3. The pull request should work for Python 3.11 and newer. The tests run in CI on
   Linux, Windows and macOS, so check that they pass on all of them.
4. Keep the pull request focused on one change, and describe what it does and why.
5. Do not change the version number, create tags or touch the release files. Releases are
   decided and prepared by the maintainers (see `Releases`_).

Pull requests are merged by the maintainers once the CI is green and the changes have been
reviewed. Please be patient and respond to the review comments, this is a volunteer-driven
project.

Tips
----

The tests are in the ``tests`` folder, with one ``test_<module>.py`` file per module.
To run a subset of them::

    $ pytest tests/test_dicomtools.py
    $ pytest tests/test_imagetools.py -k morphology

The tests do not need a display, ANTs, FreeSurfer or any data. They use small synthetic
files and replace the external tools with fakes, so keep new tests self-contained.

Releases
--------

Releases are made only by the maintainers. Merging a pull request never publishes anything by
itself: a new version reaches PyPI only when a maintainer pushes a tag starting with ``v``.
Contributors should not bump the version or create tags in their pull requests, and
maintainers decide when enough changes have accumulated to justify a release.

The steps below are a reminder for the maintainers.

1. Check that ``main`` is green in the CI and that the "Unreleased" section of ``HISTORY.rst``
   describes everything that will ship.
2. Update the version in ``pyproject.toml``, ``clabtoolkit/__init__.py``, ``release`` in
   ``docs/source/conf.py``, the version line of ``DOCUMENTATION.md``, and ``version`` and
   ``date-released`` in ``CITATION.cff``. They must all match.
3. Move the "Unreleased" entries of ``HISTORY.rst`` under the new version and date.
4. Commit the changes and push to ``main``. The distribution is built and published to
   TestPyPI, which is a chance to check it before the real release.
5. Tag the release and push the tag. This is the step that publishes the distribution to PyPI,
   and it cannot be undone, since PyPI does not allow re-uploading a version::

    $ git tag vX.Y.Z
    $ git push origin vX.Y.Z

6. Publish a GitHub Release from the tag (Releases > Draft a new release). If the repository is
   enabled in Zenodo, publishing the release archives that version and creates its DOI. Add the
   DOI to ``CITATION.cff`` afterwards.
