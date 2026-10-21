.. highlight:: shell

============
Contributing
============

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given.

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

4. Create a branch for local development::

    $ git checkout -b name-of-your-bugfix-or-feature

   Now you can make your changes locally.

5. When you're done making changes, check that your changes pass the linter, the
   formatting check and the tests. These are the same checks that run in CI::

    $ ruff check clabtoolkit tests
    $ black --check clabtoolkit tests
    $ pytest

   CI installs the latest ``black``, whose formatting can differ slightly from older
   releases, so keep it up to date (``pip install -U black``) before running the check.

6. Commit your changes and push your branch to GitHub::

    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature

7. Submit a pull request through the GitHub website.

Pull Request Guidelines
-----------------------

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. If the pull request adds functionality, the docs should be updated. Put
   your new functionality into a function with a docstring, add the feature to
   the list in README.rst, and add an entry to the "Unreleased" section of
   HISTORY.rst.
3. The pull request should work for Python 3.11 and newer. The tests run in CI on
   Linux, Windows and macOS, so check that they pass on all of them.

Tips
----

The tests are in the ``tests`` folder, with one ``test_<module>.py`` file per module.
To run a subset of them::

    $ pytest tests/test_dicomtools.py
    $ pytest tests/test_imagetools.py -k morphology

The tests do not need a display, ANTs, FreeSurfer or any data. They use small synthetic
files and replace the external tools with fakes, so keep new tests self-contained.

Deploying
---------

A reminder for the maintainers on how to deploy.

1. Update the version in ``pyproject.toml``, ``clabtoolkit/__init__.py``, ``release`` in
   ``docs/source/conf.py``, the version line of ``DOCUMENTATION.md``, and ``version`` and
   ``date-released`` in ``CITATION.cff``. They must all match.
2. Move the "Unreleased" entries of ``HISTORY.rst`` under the new version and date.
3. Commit the changes, make sure the CI is green, and push to ``main``. The distribution is
   built and published to TestPyPI.
4. Tag the release and push the tag. Tags starting with ``v`` publish the distribution to PyPI::

    $ git tag vX.Y.Z
    $ git push origin vX.Y.Z

5. Publish a GitHub Release from the tag (Releases > Draft a new release). If the repository is
   enabled in Zenodo, publishing the release archives that version and creates its DOI. Add the
   DOI to ``CITATION.cff`` afterwards.
