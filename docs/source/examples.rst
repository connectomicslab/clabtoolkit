Examples
========

The `usage_examples <https://github.com/connectomicslab/clabtoolkit/tree/main/usage_examples>`_
folder of the repository contains Jupyter notebooks with runnable examples. They retrieve the sample
data of clabtoolkit from its `Zenodo repository <https://doi.org/10.5281/zenodo.23702131>`_ the first time they
need it (see ``fetch_sample_data`` in :doc:`misctools <modules/misctools>`), so they do not need any input file.
The few images that the sample data lacks, such as functional images, are simulated.

General overview
----------------

The best place to start. Click the card to open the notebook on GitHub.

.. grid:: 1 2 2 3
   :gutter: 3
   :class-container: clab-grid

   .. grid-item-card:: clabtoolkit examples
      :img-top: _static/icons/clabtoolkit_examples.svg
      :link: https://github.com/connectomicslab/clabtoolkit/blob/main/usage_examples/clabtoolkit_examples.ipynb
      :link-type: url
      :class-card: clab-card sd-shadow-sm

      The core objects (``Parcellation``, ``Surface``, ``Tractogram`` and ``Connectome``), morphometry and the
      visualization of objects and scenes.

Representative workflows
------------------------

Three end-to-end workflows that combine several modules. Click the card to open the notebook on GitHub.

.. grid:: 1 2 2 3
   :gutter: 3
   :class-container: clab-grid

   .. grid-item-card:: Representative workflows
      :img-top: _static/icons/workflows.svg
      :link: https://github.com/connectomicslab/clabtoolkit/blob/main/usage_examples/examples-representative-workflows.ipynb
      :link-type: url
      :class-card: clab-card sd-shadow-sm

      Workflow A corrects a vertex-wise parcellation and computes regional statistics, workflow B selects the
      streamlines that connect a pair of regions, and workflow C computes the functional connectivity of the
      hippocampus and the temporal lobe.

Examples by module
------------------

One notebook per module, with the examples of each function and method. Click a card to open the notebook on GitHub.

.. rubric:: Data and formats

.. grid:: 1 2 2 3
   :gutter: 3
   :class-container: clab-grid

   .. grid-item-card:: bidstools
      :img-top: _static/icons/bidstools.svg
      :link: https://github.com/connectomicslab/clabtoolkit/blob/main/usage_examples/examples-bidstools-methods.ipynb
      :link-type: url
      :class-card: clab-card sd-shadow-sm

      Read, edit and summarize BIDS file names and whole datasets.

      +++
      :doc:`API reference <modules/bidstools>`

   .. grid-item-card:: dicomtools
      :img-top: _static/icons/dicomtools.svg
      :link: https://github.com/connectomicslab/clabtoolkit/blob/main/usage_examples/examples-dicomtools-methods.ipynb
      :link-type: url
      :class-card: clab-card sd-shadow-sm

      Organize, compress and uncompress raw DICOM files.

      +++
      :doc:`API reference <modules/dicomtools>`

   .. grid-item-card:: imagetools
      :img-top: _static/icons/imagetools.svg
      :link: https://github.com/connectomicslab/clabtoolkit/blob/main/usage_examples/examples-imagetools-methods.ipynb
      :link-type: url
      :class-card: clab-card sd-shadow-sm

      Morphological operations and manipulation of brain images.

      +++
      :doc:`API reference <modules/imagetools>`

   .. grid-item-card:: pipelinetools
      :img-top: _static/icons/pipelinetools.svg
      :link: https://github.com/connectomicslab/clabtoolkit/blob/main/usage_examples/examples-pipelinetools-methods.ipynb
      :link-type: url
      :class-card: clab-card sd-shadow-sm

      Track the processing status of BIDS derivatives.

      +++
      :doc:`API reference <modules/pipelinetools>`

   .. grid-item-card:: misctools
      :img-top: _static/icons/misctools.svg
      :link: https://github.com/connectomicslab/clabtoolkit/blob/main/usage_examples/examples-misctools-methods.ipynb
      :link-type: url
      :class-card: clab-card sd-shadow-sm

      General helpers: index parsing, conditions, files, JSON and containers.

      +++
      :doc:`API reference <modules/misctools>`


.. rubric:: Surfaces and parcellations

.. grid:: 1 2 2 3
   :gutter: 3
   :class-container: clab-grid

   .. grid-item-card:: surfacetools
      :img-top: _static/icons/surfacetools.svg
      :link: https://github.com/connectomicslab/clabtoolkit/blob/main/usage_examples/examples-surfacetools-methods.ipynb
      :link-type: url
      :class-card: clab-card sd-shadow-sm

      Surface meshes, scalar maps, annotations and 3D plots.

      +++
      :doc:`API reference <modules/surfacetools>`

   .. grid-item-card:: parcellationtools
      :img-top: _static/icons/parcellationtools.svg
      :link: https://github.com/connectomicslab/clabtoolkit/blob/main/usage_examples/examples-parcellationtools-methods.ipynb
      :link-type: url
      :class-card: clab-card sd-shadow-sm

      Load, edit, simulate and analyze brain parcellations.

      +++
      :doc:`API reference <modules/parcellationtools>`

   .. grid-item-card:: freesurfertools
      :img-top: _static/icons/freesurfertools.svg
      :link: https://github.com/connectomicslab/clabtoolkit/blob/main/usage_examples/examples-freesurfertools-methods.ipynb
      :link-type: url
      :class-card: clab-card sd-shadow-sm

      FreeSurfer subjects, annotations, GIFTI labels and statistics tables.

      +++
      :doc:`API reference <modules/freesurfertools>`

   .. grid-item-card:: segmentationtools
      :img-top: _static/icons/segmentationtools.svg
      :link: https://github.com/connectomicslab/clabtoolkit/blob/main/usage_examples/examples-segmentationtools-methods.ipynb
      :link-type: url
      :class-card: clab-card sd-shadow-sm

      Atlas-based parcellation with ANTs and tissue tables.

      +++
      :doc:`API reference <modules/segmentationtools>`

   .. grid-item-card:: morphometrytools
      :img-top: _static/icons/morphometrytools.svg
      :link: https://github.com/connectomicslab/clabtoolkit/blob/main/usage_examples/examples-morphometrytools-methods.ipynb
      :link-type: url
      :class-card: clab-card sd-shadow-sm

      Regional morphometry from surfaces, annotations and parcellations.

      +++
      :doc:`API reference <modules/morphometrytools>`


.. rubric:: Tractography and connectivity

.. grid:: 1 2 2 3
   :gutter: 3
   :class-container: clab-grid

   .. grid-item-card:: dwitools
      :img-top: _static/icons/dwitools.svg
      :link: https://github.com/connectomicslab/clabtoolkit/blob/main/usage_examples/examples-dwitools-methods.ipynb
      :link-type: url
      :class-card: clab-card sd-shadow-sm

      Diffusion MRI: b-values, gradients, volumes and tensor maps.

      +++
      :doc:`API reference <modules/dwitools>`

   .. grid-item-card:: tracttools
      :img-top: _static/icons/tracttools.svg
      :link: https://github.com/connectomicslab/clabtoolkit/blob/main/usage_examples/examples-tracttools-methods.ipynb
      :link-type: url
      :class-card: clab-card sd-shadow-sm

      Load, filter, cluster and plot tractograms (TRK and TCK).

      +++
      :doc:`API reference <modules/tracttools>`

   .. grid-item-card:: pointstools
      :img-top: _static/icons/pointstools.svg
      :link: https://github.com/connectomicslab/clabtoolkit/blob/main/usage_examples/examples-pointstools-methods.ipynb
      :link-type: url
      :class-card: clab-card sd-shadow-sm

      Point clouds with maps, filtering and 3D plots.

      +++
      :doc:`API reference <modules/pointstools>`

   .. grid-item-card:: connectivitytools
      :img-top: _static/icons/connectivitytools.svg
      :link: https://github.com/connectomicslab/clabtoolkit/blob/main/usage_examples/examples-connectivitytools-methods.ipynb
      :link-type: url
      :class-card: clab-card sd-shadow-sm

      The Connectome container: matrices, regions, files and plots.

      +++
      :doc:`API reference <modules/connectivitytools>`

   .. grid-item-card:: networktools
      :img-top: _static/icons/networktools.svg
      :link: https://github.com/connectomicslab/clabtoolkit/blob/main/usage_examples/examples-networktools-methods.ipynb
      :link-type: url
      :class-card: clab-card sd-shadow-sm

      Sparse graph tools for connectivity matrices and surface meshes.

      +++
      :doc:`API reference <modules/networktools>`


.. rubric:: Visualization and quality control

.. grid:: 1 2 2 3
   :gutter: 3
   :class-container: clab-grid

   .. grid-item-card:: visualizationtools
      :img-top: _static/icons/visualizationtools.svg
      :link: https://github.com/connectomicslab/clabtoolkit/blob/main/usage_examples/examples-visualizationtools-methods.ipynb
      :link-type: url
      :class-card: clab-card sd-shadow-sm

      BrainPlotter: multi-view figures of surfaces, tracts and points.

      +++
      :doc:`API reference <modules/visualizationtools>`

   .. grid-item-card:: plottools
      :img-top: _static/icons/plottools.svg
      :link: https://github.com/connectomicslab/clabtoolkit/blob/main/usage_examples/examples-plottools-methods.ipynb
      :link-type: url
      :class-card: clab-card sd-shadow-sm

      Layout, screen and font helpers for figures.

      +++
      :doc:`API reference <modules/plottools>`

   .. grid-item-card:: qcqatools
      :img-top: _static/icons/qcqatools.svg
      :link: https://github.com/connectomicslab/clabtoolkit/blob/main/usage_examples/examples-qcqatools-methods.ipynb
      :link-type: url
      :class-card: clab-card sd-shadow-sm

      Slice renders and HTML pages for visual quality control.

      +++
      :doc:`API reference <modules/qcqatools>`

   .. grid-item-card:: colorstools
      :img-top: _static/icons/colorstools.svg
      :link: https://github.com/connectomicslab/clabtoolkit/blob/main/usage_examples/examples-colorstools-methods.ipynb
      :link-type: url
      :class-card: clab-card sd-shadow-sm

      Color conversion, palettes and color tables (LUT, TSV, FSL, Nilearn).

      +++
      :doc:`API reference <modules/colorstools>`

