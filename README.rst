========================
Connectomics Lab Toolkit
========================

.. image:: docs/source/_static/clabtoolkit_logo.png
        :alt: clabtoolkit logo
        :align: center
        :width: 400


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
    * DWI volume removal by index or b-value and B0 extraction
    * Acquisition schemes from bvec/bval files or b-matrices
    * q-space visualization of the scheme (window, notebook, image or HTML)
    * Scalar maps from the tensor eigenvalues (FA, MD, AD, RD and more)

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

The examples below are self-contained: they use simulated data, so you can run them as they are.
More examples for every module are available in the ``usage_examples`` folder
(see ``usage_examples/clabtoolkit_examples.ipynb``).

**Filter a parcellation by region name** (``keep_by_name``)

.. code-block:: python

    import clabtoolkit.parcellationtools as cltparc

    # Simulate a parcellation with 10 regions and name them
    parc = cltparc.Parcellation.simulate_parcellation(n_regions=10, seed=42)
    parc.set_names(['ctx-rh-precentral', 'ctx-lh-precentral', 'ctx-rh-postcentral', 'ctx-lh-postcentral',
                    'ctx-rh-superiorfrontal', 'ctx-lh-superiorfrontal', 'ctx-rh-supramarginal',
                    'ctx-lh-supramarginal', 'subc-rh-amygdala', 'subc-lh-amygdala'])

    # Keep only the precentral regions (the object is modified in place)
    parc.keep_by_name(['ctx-rh-precentral', 'ctx-lh-precentral'])
    print(parc.name)  # ['ctx-rh-precentral', 'ctx-lh-precentral']

To load your own parcellation, use ``cltparc.Parcellation("parc.nii.gz", color_table="lut.txt")``.

**Sample an image along the streamlines of a tractogram** (``interpolate_on_tractogram``)

.. code-block:: python

    import clabtoolkit.imagetools as cltimg
    import clabtoolkit.tracttools as clttract

    # Simulate a reference image and a tractogram with 5 bundles
    cltimg.simulate_image(reference=(100, 100, 70), simulated_image="image.nii.gz")
    tract = clttract.Tractogram.simulate_tractogram(ref_image="image.nii.gz")

    # Sample the image at every point of every streamline
    tract.interpolate_on_tractogram("image.nii.gz", map_name="fa",
                                    storage_mode="data_per_point", interp_method="linear")

    print(tract.list_maps()["maps_per_point"])  # ['fa']
    print(tract.data_per_point["fa"][0][:5])    # values along the first streamline

Use ``storage_mode="data_per_streamline"`` with ``reduction="mean"`` to store a single value per streamline.

**Plot a surface with a scalar map** (``BrainPlotter``)

.. code-block:: python

    import clabtoolkit.surfacetools as cltsurf
    import clabtoolkit.visualizationtools as cltvis

    # Simulate a surface and a thickness map
    surf = cltsurf.Surface.simulate_surface(seed=42, shape="sphere")
    surf.simulate_map(map_name="thickness")

    # Plot two views of the map and save the figure
    plotter = cltvis.BrainPlotter()
    plotter.plot(objs2plot=surf, views=["lateral", "medial"], map_names="thickness",
                 colormaps="viridis", colorbar=True, save_path="/tmp/thickness.png")

To plot your own data, load a surface with ``cltsurf.Surface("lh.pial")`` and its map with
``surf.load_scalar_maps("lh.thickness", maps_names="thickness")``.

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
