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
* FreeSurfer is **not** required. If it is installed and ``FREESURFER_HOME`` is set, the example
  notebooks use its sample subjects; otherwise they run on simulated data.
* ANTs is needed only by ``segmentationtools.abased_parcellation``.
* The ``examples`` extra installs what the example notebooks need (``pip install clabtoolkit[examples]``).
