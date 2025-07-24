========================
Connectomics Lab Toolkit
========================


.. image:: https://img.shields.io/pypi/v/clabtoolkit.svg
        :target: https://pypi.python.org/pypi/clabtoolkit

.. image:: https://github.com/connectomicslab/clabtoolkit/actions/workflows/ci.yml/badge.svg
        :target: https://github.com/connectomicslab/clabtoolkit/actions/workflows/ci.yml

.. image:: https://readthedocs.org/projects/clabtoolkit/badge/?version=latest
        :target: https://clabtoolkit.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status

.. image:: https://img.shields.io/pypi/pyversions/clabtoolkit.svg
        :target: https://pypi.python.org/pypi/clabtoolkit

.. image:: https://codecov.io/gh/connectomicslab/clabtoolkit/branch/main/graph/badge.svg
        :target: https://codecov.io/gh/connectomicslab/clabtoolkit


A comprehensive Python toolkit for neuroimaging data processing and analysis, specifically designed for working with brain connectivity data, BIDS datasets, and various neuroimaging formats.

* **Free software**: Apache Software License 2.0
* **Documentation**: https://clabtoolkit.readthedocs.io
* **Source Code**: https://github.com/connectomicslab/clabtoolkit
* **Python versions**: 3.9+

Installation
------------

Install from PyPI::

    pip install clabtoolkit

For development installation::

    git clone https://github.com/connectomicslab/clabtoolkit.git
    cd clabtoolkit
    pip install -e .[dev]

Features
--------

**BIDS Tools** (``clabtoolkit.bidstools``)
    * BIDS dataset validation and manipulation
    * Entity extraction from BIDS filenames
    * Conversion between BIDS formats
    * Metadata handling for neuroimaging datasets

**Connectivity Tools** (``clabtoolkit.connectivitytools``)
    * Brain connectivity matrix analysis
    * Network-based statistics
    * Graph theory metrics computation
    * Connectivity visualization utilities

**FreeSurfer Tools** (``clabtoolkit.freesurfertools``)
    * FreeSurfer output parsing and processing
    * Surface-based analysis utilities
    * Cortical thickness and morphometry tools
    * Integration with FreeSurfer workflows

**Image Processing Tools** (``clabtoolkit.imagetools``)
    * Neuroimaging data I/O operations
    * Image registration and transformation
    * Quality control and preprocessing utilities
    * Multi-modal image processing

**Parcellation Tools** (``clabtoolkit.parcellationtools``)
    * Brain parcellation scheme handling
    * Region-of-interest (ROI) extraction
    * Atlas-based analysis tools
    * Custom parcellation creation

**Surface Tools** (``clabtoolkit.surfacetools``)
    * Surface mesh processing and analysis
    * Cortical surface manipulation
    * Surface-based statistics
    * Visualization of surface data

**DWI Tools** (``clabtoolkit.dwitools``)
    * Diffusion-weighted imaging analysis
    * Tractography processing utilities
    * DTI and advanced diffusion modeling
    * White matter analysis tools

**Quality Control Tools** (``clabtoolkit.qcqatools``)
    * Automated quality assessment
    * Image artifact detection
    * Quality metrics computation
    * Reporting and visualization

**Visualization Tools** (``clabtoolkit.visualizationtools``)
    * Brain visualization utilities
    * Interactive plotting capabilities
    * Publication-ready figures
    * Multi-modal data visualization

Quick Start
-----------

.. code-block:: python

    import clabtoolkit.bidstools as bids
    import clabtoolkit.connectivitytools as conn
    
    # Load BIDS configuration
    config = bids.load_bids_json()
    
    # Extract entities from BIDS filename
    entities = bids.str2entity("sub-01_ses-M00_T1w.nii.gz")
    print(entities)  # {'sub': '01', 'ses': 'M00', 'suffix': 'T1w', 'extension': 'nii.gz'}
    
    # Process connectivity data
    # conn_matrix = conn.load_connectivity_matrix("path/to/connectivity.mat")

Contributing
------------

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (``git checkout -b feature/amazing-feature``)
3. Commit your changes (``git commit -m 'Add some amazing feature'``)
4. Push to the branch (``git push origin feature/amazing-feature``)
5. Open a Pull Request

Testing
-------

Run tests with::

    pytest

Run tests with coverage::

    pytest --cov=clabtoolkit

Changelog
---------

See `HISTORY.rst <HISTORY.rst>`_ for a detailed changelog.

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
