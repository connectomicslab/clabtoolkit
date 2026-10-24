API Reference
=============

Every module of CLABTOOLKIT, grouped by what it is used for. Click a card to open the reference of the
module, or use its *Examples* link to see it at work in a notebook.

Data and formats
----------------

Files, datasets and images: from raw DICOM to BIDS derivatives.

.. grid:: 1 2 2 3
   :gutter: 3
   :class-container: clab-grid

   .. grid-item-card:: bidstools
      :img-top: _static/icons/bidstools.svg
      :link: modules/bidstools
      :link-type: doc
      :class-card: clab-card sd-shadow-sm

      Read, edit and summarize BIDS file names and whole datasets.

      +++
      :doc:`API <modules/bidstools>` · `Examples <https://github.com/connectomicslab/clabtoolkit/blob/main/usage_examples/examples-bidstools-methods.ipynb>`__

   .. grid-item-card:: dicomtools
      :img-top: _static/icons/dicomtools.svg
      :link: modules/dicomtools
      :link-type: doc
      :class-card: clab-card sd-shadow-sm

      Organize, compress and uncompress raw DICOM files.

      +++
      :doc:`API <modules/dicomtools>` · `Examples <https://github.com/connectomicslab/clabtoolkit/blob/main/usage_examples/examples-dicomtools-methods.ipynb>`__

   .. grid-item-card:: imagetools
      :img-top: _static/icons/imagetools.svg
      :link: modules/imagetools
      :link-type: doc
      :class-card: clab-card sd-shadow-sm

      Morphological operations and manipulation of brain images.

      +++
      :doc:`API <modules/imagetools>` · `Examples <https://github.com/connectomicslab/clabtoolkit/blob/main/usage_examples/examples-imagetools-methods.ipynb>`__

   .. grid-item-card:: pipelinetools
      :img-top: _static/icons/pipelinetools.svg
      :link: modules/pipelinetools
      :link-type: doc
      :class-card: clab-card sd-shadow-sm

      Track the processing status of BIDS derivatives.

      +++
      :doc:`API <modules/pipelinetools>` · `Examples <https://github.com/connectomicslab/clabtoolkit/blob/main/usage_examples/examples-pipelinetools-methods.ipynb>`__

   .. grid-item-card:: misctools
      :img-top: _static/icons/misctools.svg
      :link: modules/misctools
      :link-type: doc
      :class-card: clab-card sd-shadow-sm

      General helpers: index parsing, conditions, files, JSON and containers.

      +++
      :doc:`API <modules/misctools>` · `Examples <https://github.com/connectomicslab/clabtoolkit/blob/main/usage_examples/examples-misctools-methods.ipynb>`__

Surfaces and parcellations
--------------------------

Cortical surfaces, atlases and the measures extracted from them.

.. grid:: 1 2 2 3
   :gutter: 3
   :class-container: clab-grid

   .. grid-item-card:: surfacetools
      :img-top: _static/icons/surfacetools.svg
      :link: modules/surfacetools
      :link-type: doc
      :class-card: clab-card sd-shadow-sm

      Surface meshes, scalar maps, annotations and 3D plots.

      +++
      :doc:`API <modules/surfacetools>` · `Examples <https://github.com/connectomicslab/clabtoolkit/blob/main/usage_examples/examples-surfacetools-methods.ipynb>`__

   .. grid-item-card:: parcellationtools
      :img-top: _static/icons/parcellationtools.svg
      :link: modules/parcellationtools
      :link-type: doc
      :class-card: clab-card sd-shadow-sm

      Load, edit, simulate and analyze brain parcellations.

      +++
      :doc:`API <modules/parcellationtools>` · `Examples <https://github.com/connectomicslab/clabtoolkit/blob/main/usage_examples/examples-parcellationtools-methods.ipynb>`__

   .. grid-item-card:: freesurfertools
      :img-top: _static/icons/freesurfertools.svg
      :link: modules/freesurfertools
      :link-type: doc
      :class-card: clab-card sd-shadow-sm

      FreeSurfer subjects, annotations, GIFTI labels and statistics tables.

      +++
      :doc:`API <modules/freesurfertools>` · `Examples <https://github.com/connectomicslab/clabtoolkit/blob/main/usage_examples/examples-freesurfertools-methods.ipynb>`__

   .. grid-item-card:: segmentationtools
      :img-top: _static/icons/segmentationtools.svg
      :link: modules/segmentationtools
      :link-type: doc
      :class-card: clab-card sd-shadow-sm

      Atlas-based parcellation with ANTs and tissue tables.

      +++
      :doc:`API <modules/segmentationtools>` · `Examples <https://github.com/connectomicslab/clabtoolkit/blob/main/usage_examples/examples-segmentationtools-methods.ipynb>`__

   .. grid-item-card:: morphometrytools
      :img-top: _static/icons/morphometrytools.svg
      :link: modules/morphometrytools
      :link-type: doc
      :class-card: clab-card sd-shadow-sm

      Regional morphometry from surfaces, annotations and parcellations.

      +++
      :doc:`API <modules/morphometrytools>` · `Examples <https://github.com/connectomicslab/clabtoolkit/blob/main/usage_examples/examples-morphometrytools-methods.ipynb>`__

Tractography and connectivity
-----------------------------

Diffusion data, tractograms, point clouds and connectomes.

.. grid:: 1 2 2 3
   :gutter: 3
   :class-container: clab-grid

   .. grid-item-card:: dwitools
      :img-top: _static/icons/dwitools.svg
      :link: modules/dwitools
      :link-type: doc
      :class-card: clab-card sd-shadow-sm

      Diffusion MRI: b-values, gradients, volumes and tensor maps.

      +++
      :doc:`API <modules/dwitools>` · `Examples <https://github.com/connectomicslab/clabtoolkit/blob/main/usage_examples/examples-dwitools-methods.ipynb>`__

   .. grid-item-card:: tracttools
      :img-top: _static/icons/tracttools.svg
      :link: modules/tracttools
      :link-type: doc
      :class-card: clab-card sd-shadow-sm

      Load, filter, cluster and plot tractograms (TRK and TCK).

      +++
      :doc:`API <modules/tracttools>` · `Examples <https://github.com/connectomicslab/clabtoolkit/blob/main/usage_examples/examples-tracttools-methods.ipynb>`__

   .. grid-item-card:: pointstools
      :img-top: _static/icons/pointstools.svg
      :link: modules/pointstools
      :link-type: doc
      :class-card: clab-card sd-shadow-sm

      Point clouds with maps, filtering and 3D plots.

      +++
      :doc:`API <modules/pointstools>` · `Examples <https://github.com/connectomicslab/clabtoolkit/blob/main/usage_examples/examples-pointstools-methods.ipynb>`__

   .. grid-item-card:: connectivitytools
      :img-top: _static/icons/connectivitytools.svg
      :link: modules/connectivitytools
      :link-type: doc
      :class-card: clab-card sd-shadow-sm

      The Connectome container: matrices, regions, files and plots.

      +++
      :doc:`API <modules/connectivitytools>` · `Examples <https://github.com/connectomicslab/clabtoolkit/blob/main/usage_examples/examples-connectivitytools-methods.ipynb>`__

   .. grid-item-card:: networktools
      :img-top: _static/icons/networktools.svg
      :link: modules/networktools
      :link-type: doc
      :class-card: clab-card sd-shadow-sm

      Sparse graph tools for connectivity matrices and surface meshes.

      +++
      :doc:`API <modules/networktools>` · `Examples <https://github.com/connectomicslab/clabtoolkit/blob/main/usage_examples/examples-networktools-methods.ipynb>`__

Visualization and quality control
---------------------------------

Figures, colors and visual checks of your data.

.. grid:: 1 2 2 3
   :gutter: 3
   :class-container: clab-grid

   .. grid-item-card:: visualizationtools
      :img-top: _static/icons/visualizationtools.svg
      :link: modules/visualizationtools
      :link-type: doc
      :class-card: clab-card sd-shadow-sm

      BrainPlotter: multi-view figures of surfaces, tracts and points.

      +++
      :doc:`API <modules/visualizationtools>` · `Examples <https://github.com/connectomicslab/clabtoolkit/blob/main/usage_examples/examples-visualizationtools-methods.ipynb>`__

   .. grid-item-card:: plottools
      :img-top: _static/icons/plottools.svg
      :link: modules/plottools
      :link-type: doc
      :class-card: clab-card sd-shadow-sm

      Layout, screen and font helpers for figures.

      +++
      :doc:`API <modules/plottools>` · `Examples <https://github.com/connectomicslab/clabtoolkit/blob/main/usage_examples/examples-plottools-methods.ipynb>`__

   .. grid-item-card:: qcqatools
      :img-top: _static/icons/qcqatools.svg
      :link: modules/qcqatools
      :link-type: doc
      :class-card: clab-card sd-shadow-sm

      Slice renders and HTML pages for visual quality control.

      +++
      :doc:`API <modules/qcqatools>` · `Examples <https://github.com/connectomicslab/clabtoolkit/blob/main/usage_examples/examples-qcqatools-methods.ipynb>`__

   .. grid-item-card:: colorstools
      :img-top: _static/icons/colorstools.svg
      :link: modules/colorstools
      :link-type: doc
      :class-card: clab-card sd-shadow-sm

      Color conversion, palettes and color tables (LUT, TSV, FSL, Nilearn).

      +++
      :doc:`API <modules/colorstools>` · `Examples <https://github.com/connectomicslab/clabtoolkit/blob/main/usage_examples/examples-colorstools-methods.ipynb>`__

.. toctree::
   :hidden:
   :maxdepth: 2

   modules/bidstools
   modules/colorstools
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
   modules/pointstools
   modules/qcqatools
   modules/segmentationtools
   modules/surfacetools
   modules/tracttools
   modules/visualizationtools
