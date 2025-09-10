.. clabtoolkit documentation master file, created by
   sphinx-quickstart on Thu Jul 11 08:31:28 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Connectomics Lab Toolkit Documentation
======================================

A comprehensive Python toolkit for neuroimaging data processing and analysis, specifically designed for working with brain connectivity data, BIDS datasets, and various neuroimaging formats.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   api
   examples
   contributing
   changelog

Installation
============

Install from PyPI::

    conda env create -f environment.yaml
    conda activate clabtoolkit-env
    pip install clabtoolkit

For development installation::

    git clone https://github.com/connectomicslab/clabtoolkit.git
    cd clabtoolkit
    pip install -e .[dev]

Quick Start
===========

Surface Visualization
=====================
.. code-block:: python

    import clabtoolkit.surfacetools as cltsurf

    # Reading a surface file
    surf_lh = cltsurf.Surface("/opt/freesurfer/subjects/fsaverage/surf/lh.pial")

    # Loading scalar maps
    surf_lh.load_scalar_maps("/opt/freesurfer/subjects/fsaverage/surf/lh.thickness", maps_names="Thickness")
    surf_lh.plot(overlay_name="Thickness", 
                views="4_views", 
                cmap= "jet", 
                colorbar_position="right")

API Reference
=============

.. toctree::
   :maxdepth: 2
   
   modules/bidstools
   modules/connectivitytools
   modules/dicomtools
   modules/dwitools
   modules/freesurfertools
   modules/imagetools
   modules/misctools
   modules/morphometrytools
   modules/networktools
   modules/parcellationtools
   modules/pipelinetools
   modules/plottools
   modules/qcqatools
   modules/segmentationtools
   modules/surfacetools
   modules/visualizationtools

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
