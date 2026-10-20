# Usage examples

Jupyter notebooks with runnable examples of **clabtoolkit**. There is one notebook per module, plus an overview notebook that covers the core objects (`Parcellation`, `Surface`, `Tractogram` and `Connectome`).

The examples are organized as numbered tests of every method of the module, each one with its code and printed output. The full reference is in the [documentation](https://clabtoolkit.readthedocs.io).

## Getting started

```bash
pip install "clabtoolkit[examples]" jupyterlab
git clone https://github.com/connectomicslab/clabtoolkit.git
cd clabtoolkit/usage_examples
jupyter lab
```

The `examples` extra installs `itables`, which some notebooks use to display tables. If you are new to the toolkit, start with `clabtoolkit_examples.ipynb`.

## Requirements

- **No input data is needed.** The notebooks simulate the images, surfaces, tractograms and connectomes they use, and write the simulated files to `/tmp`.
- **FreeSurfer is optional.** The notebooks that read `FREESURFER_HOME` use the sample subjects (`bert`, `fsaverage`) when it is set, and simulated data otherwise.
- **3D plots** use PyVista and need a display. On a server, set `PYVISTA_OFF_SCREEN=true` or run the notebooks in Jupyter with a notebook backend.
- **ANTs** is needed only for `abased_parcellation` (`segmentationtools`). That example is not executed in the notebook.

## Notebooks

### Overview

| Notebook | Content |
| --- | --- |
| [`clabtoolkit_examples.ipynb`](clabtoolkit_examples.ipynb) | Core objects (`Parcellation`, `Surface`, `Tractogram`, `Connectome`), morphometry and the plotting of objects and scenes |

### Data organization

| Notebook | Module | Content |
| --- | --- | --- |
| [`examples-bidstools-methods.ipynb`](examples-bidstools-methods.ipynb) | `bidstools` | BIDS filenames and entities, dataset tables and trees |
| [`examples-dicomtools-methods.ipynb`](examples-dicomtools-methods.ipynb) | `dicomtools` | DICOM organization, compression and metadata |

### Images and diffusion

| Notebook | Module | Content |
| --- | --- | --- |
| [`examples-imagetools-methods.ipynb`](examples-imagetools-methods.ipynb) | `imagetools` | Voxel/world conversion, cropping, morphology, interpolation and image simulation |
| [`examples-dwitools-methods.ipynb`](examples-dwitools-methods.ipynb) | `dwitools` | Removal of volumes, B0 extraction, acquisition schemes and tensor-derived maps |
| [`examples-tracttools-methods.ipynb`](examples-tracttools-methods.ipynb) | `tracttools` | Tractograms: filtering, resampling, interpolation of images, merging and format conversion |
| [`examples-segmentationtools-methods.ipynb`](examples-segmentationtools-methods.ipynb) | `segmentationtools` | Atlas-based parcellation and the tissue segmentation table |

### Surfaces and parcellations

| Notebook | Module | Content |
| --- | --- | --- |
| [`examples-surfacetools-methods.ipynb`](examples-surfacetools-methods.ipynb) | `surfacetools` | Surfaces: scalar maps, annotations, volume-to-surface mapping and export |
| [`examples-parcellationtools-methods.ipynb`](examples-parcellationtools-methods.ipynb) | `parcellationtools` | Parcellations: region selection, centroids, volumes, time series and functional connectivity |
| [`examples-freesurfertools-methods.ipynb`](examples-freesurfertools-methods.ipynb) | `freesurfertools` | FreeSurfer subjects, annotations and statistics tables |
| [`examples-morphometrytools-methods.ipynb`](examples-morphometrytools-methods.ipynb) | `morphometrytools` | Regional area, volume and Euler number, and parsing of FreeSurfer statistics |
| [`examples-pointstools-methods.ipynb`](examples-pointstools-methods.ipynb) | `pointstools` | Point clouds: filtering, transformations and merging |

### Connectivity and networks

| Notebook | Module | Content |
| --- | --- | --- |
| [`examples-connectivitytools-methods.ipynb`](examples-connectivitytools-methods.ipynb) | `connectivitytools` | The `Connectome` class: thresholding, metrics, plotting and I/O |
| [`examples-networktools-methods.ipynb`](examples-networktools-methods.ipynb) | `networktools` | Sparse graphs and connected components |

### Visualization and utilities

| Notebook | Module | Content |
| --- | --- | --- |
| [`examples-visualizationtools-methods.ipynb`](examples-visualizationtools-methods.ipynb) | `visualizationtools` | `BrainPlotter`: surfaces, tractograms, multi-view layouts and carpet plots |
| [`examples-plottools-methods.ipynb`](examples-plottools-methods.ipynb) | `plottools` | Layout, screen and font helpers for the figures |
| [`examples-colorstools-methods.ipynb`](examples-colorstools-methods.ipynb) | `colorstools` | Color conversion, palettes and color tables |
| [`examples-qcqatools-methods.ipynb`](examples-qcqatools-methods.ipynb) | `qcqatools` | Slice images and HTML pages for visual quality control |
| [`examples-misctools-methods.ipynb`](examples-misctools-methods.ipynb) | `misctools` | Indices and conditions, list and string helpers, files and tables |

## Citing

The notebooks reference the toolkit paper: [arXiv:2607.02638](https://arxiv.org/abs/2607.02638).
