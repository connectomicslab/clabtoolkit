Installation
============

Install from PyPI::

    pip install clabtoolkit

For development installation::

    git clone https://github.com/connectomicslab/clabtoolkit.git
    cd clabtoolkit
    pip install -e .[dev]

A conda environment with all the dependencies is also provided in the repository::

    conda env create -f environment.yaml
    conda activate clabtoolkit-env
    pip install clabtoolkit

Requirements and optional dependencies
--------------------------------------

* Python 3.11 or newer.
* 3D plotting (``visualizationtools``) uses PyVista and needs either a display or off-screen
  rendering (for example ``PYVISTA_OFF_SCREEN=true`` on a server).
* FreeSurfer is **not** required. The example notebooks retrieve the sample data of clabtoolkit from its
  `Zenodo repository <https://doi.org/10.5281/zenodo.23702131>`_ the first time they need it, which needs an
  internet connection. Only the examples that run FreeSurfer commands, such as ``annot2gcs``, need FreeSurfer.
* ANTs is needed only by ``segmentationtools.abased_parcellation``.
* The ``examples`` extra installs what the example notebooks need (``pip install clabtoolkit[examples]``).
