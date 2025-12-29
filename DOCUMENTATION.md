# Connectomics Lab Toolkit (clabtoolkit)

## Overview

**clabtoolkit** is a comprehensive Python toolkit designed for neuroimaging data processing and analysis, with specialized focus on brain connectivity data, BIDS datasets, and various neuroimaging formats. Developed by Yasser Alemán-Gómez, this toolkit provides an end-to-end solution for connectomics research and surface-based brain analysis.

**Version**: 0.4.0  
**License**: Apache Software License 2.0  
**Python Support**: 3.9+  
**Documentation**: https://clabtoolkit.readthedocs.io  
**Repository**: https://github.com/connectomicslab/clabtoolkit

## Installation

### From PyPI

```bash
pip install clabtoolkit
```

### Development Installation with Conda

```bash
git clone https://github.com/connectomicslab/clabtoolkit.git
cd clabtoolkit
conda env create -f environment.yaml
conda activate clabtoolkit
pip install -e .[dev]
```

## Quick Start

```python
import clabtoolkit.bidstools as cltbids
import clabtoolkit.surfacetools as cltsurf

# Extract entities from BIDS filename
entities = cltbids.str2entity("sub-01_ses-M00_T1w.nii.gz")
print(entities)  # {'sub': '01', 'ses': 'M00', 'suffix': 'T1w', 'extension': 'nii.gz'}

# Load and visualize a brain surface
surface = cltsurf.Surface("/path/to/surface.pial")
surface.plot()
```

## Package Architecture

The toolkit follows a modular, layered architecture designed for scalability and ease of use:

-   **Foundation Layer**: Core utilities and plotting infrastructure
-   **Data Layer**: BIDS compliance and image processing
-   **Analysis Layer**: Specialized neuroimaging analysis tools
-   **Workflow Layer**: Pipeline management and quality control

## Module Reference

### Core Data Management

#### bidstools - BIDS Dataset Management

**Purpose**: Comprehensive BIDS (Brain Imaging Data Structure) compliance and dataset management

**Key Features**:

-   BIDS entity manipulation and validation
-   Dataset organization and structure analysis
-   Batch processing utilities
-   Automated database table generation from BIDS datasets

**Key Functions**:

```python
# Convert BIDS filename to entity dictionary
entities = cltbids.str2entity("sub-01_ses-M00_acq-3T_T1w.nii.gz")

# Manipulate BIDS entities
cltbids.replace_entity_value(filename, 'ses', 'new_session')
cltbids.insert_entity(filename, 'run', '01', prev_entity='acq')

# Generate and display BIDS dataset structure
tree = cltbids.generate_bids_tree("/path/to/bids/dataset")
print(tree)

# Get all subjects from a BIDS dataset
subjects = cltbids.get_subjects("/path/to/bids/dataset")

# Generate comprehensive dataset overview
database = cltbids.get_bids_database_table("/path/to/bids/dataset")
print(database.head())
```

#### imagetools - Image Processing Engine

**Purpose**: Neuroimaging operations and morphological processing

**Key Classes**:

-   `MorphologicalOperations`: Binary image morphology (erosion, dilation, opening, closing)

**Key Features**:

-   2D/3D morphological operations
-   Volume filtering and hole filling
-   Image resampling and transformation
-   Quality control utilities

**Usage Example**:

```python
from clabtoolkit.imagetools import MorphologicalOperations

morph = MorphologicalOperations()
# Perform morphological closing on binary image
filled = morph.closing(image_with_holes, iterations=1)
```

### FreeSurfer Integration

#### freesurfertools - FreeSurfer Ecosystem Interface

**Purpose**: Complete interface with FreeSurfer outputs and cortical surface analysis

**Key Classes**:

-   `AnnotParcellation`: Advanced annotation file management
    -   Load/save annotation files (.annot, .gcs formats)
    -   Parcellation correction and validation
    -   Format conversion between annotation types

**Key Features**:

-   FreeSurfer stats file parsing
-   Surface-based morphometry computation
-   Container technology integration
-   Annotation file correction and processing

**Usage Example**:

```python
import clabtoolkit.freesurfertools as cltfree
import os

# Get FreeSurfer environment variables
freesurfer_home = os.environ.get('FREESURFER_HOME')
fs_subject_dir = os.path.join(freesurfer_home, 'subjects')
fs_fullid = 'bert'

# Load the Subject object
subject = cltfree.FreeSurferSubject(fs_fullid, fs_subject_dir)

# Obtain dictionaries containing hemisphere-specific FreeSurfer file paths
surf, maps, parc, stats = subject.get_hemi_dicts(fs_subject_dir, 'lh')

# Get the processing status
print(f"Processing status for subject {fs_fullid}:")
subject.get_proc_status()
print(subject.pstatus)

# Load and process annotation file
annot = cltfree.AnnotParcellation("lh.aparc.a2009s.annot")
annot.correct_parcellation()  # Fix unlabeled vertices
```

### Analysis and Processing

#### parcellationtools - Brain Parcellation Management

**Purpose**: Comprehensive brain parcellation handling and regional analysis

**Key Classes**:

-   `Parcellation`: Complete parcellation ecosystem
    -   Load parcellations with lookup tables
    -   Regional filtering and grouping
    -   Volume calculations and statistics
    -   Multi-format export capabilities

**Key Features**:

-   Flexible parcellation filtering and modification
-   Regional statistics computation
-   Atlas integration and validation
-   BIDS-compliant output generation

**Usage Example**:

```python
import clabtoolkit.parcellationtools as cltparc

# Load parcellation (automatically detects and loads lookup table if available)
vol_parc = cltparc.Parcellation("/path/to/parcellation/parcellation.nii.gz")
```

If a lookup table with the same filename and extension (.lut or .tsv) exists, it will be loaded automatically. Otherwise, load it manually:

```python
# Load the LUT file manually
vol_parc.load_colortable("/path/to/lookuptable/lookuptable.lut", lut_type="lut")
```

Compute regional volumes for all structures in the parcellation:

```python
# Compute regional volumes
volumes = vol_parc.compute_regional_volumes()
```

Filter structures by name or code if only specific regions are of interest:

```python
# Select structures to keep by their name
names_to_keep = ["thalamus", "cerebellum"]

# Keep only the specified structures
vol_parc.keep_by_name(names2look=names_to_keep)

# Save the modified parcellation
out_parc_path = "/tmp/sub-test_desc-tha+cer_dseg.nii.gz"
vol_parc.save_parcellation(out_parc_path, save_lut=True)
```

#### surfacetools - Surface Geometry Processing

**Purpose**: Brain surface mesh processing and advanced visualization

**Key Classes**:

-   `Surface`: Comprehensive surface management
    -   FreeSurfer surface file support (.pial, .white, .inflated)
    -   Scalar data overlay and visualization
    -   Parcellation integration
    -   PyVista-powered 3D rendering

**Key Features**:

-   Multi-format surface loading
-   Scalar map management and visualization
-   Interactive 3D plotting
-   Surface-based analysis tools

**Usage Example**:

```python
from clabtoolkit.surfacetools import Surface

# Load surface with scalar data
surface = Surface("/path/to/lh.pial")
surface.load_scalar_data("/path/to/thickness.mgh", maps_names="Thickness")
surface.plot(overlay_name="Thickness", cmap='viridis', views=["lateral", "medial"])

# Load annotations
surface.load_annotation("/path/to/lh.aparc.annot", 'aparc')
surface.plot(overlay_name="Thickness", cmap='viridis', views="8_views")
```

#### morphometrytools - Morphometric Analysis

**Purpose**: Surface-based morphometric computations and statistics

**Key Features**:

-   Regional value extraction from surface annotations
-   Multi-hemisphere morphometric analysis
-   Statistical summary generation
-   Integration with parcellation workflows

**Usage Examples**:

Extract regional values by combining vertex-wise surface metrics with anatomical parcellation data:

```python
import clabtoolkit.morphometrytools as morpho

metric_file = '/path/to/metric/lh.thickness'
parc_file = '/path/to/annotation/lh.atlas.annot'

df_region, metric_values, _ = morpho.compute_reg_val_fromannot(
    metric_file, parc_file, hemi, metric=metric_name, include_global=False
)
```

Compute regional statistics from a volumetric metric map and a parcellation:

```python
import clabtoolkit.morphometrytools as morpho

metric_file = '/path/to/metric/fa.nii.gz'
parc_file = '/path/to/parcellation/parc.nii.gz'

# Compute regional statistics
df, metric_values, _ = morpho.compute_reg_val_fromparcellation(
    metric_file, parc_file, metric='intensity'
)
```

### Specialized Analysis

#### dwitools - Diffusion MRI Analysis

**Purpose**: Diffusion-weighted imaging analysis and tractography processing

**Key Features**:

-   DWI volume manipulation and quality control
-   Tractography file processing (.trk, .tck formats)
-   Bundle analysis and clustering

**Usage Example**:

```python
from clabtoolkit.dwitools import delete_dwi_volumes

# Remove specific DWI volumes
delete_dwi_volumes(
    dwi_file="dwi.nii.gz",
    bval_file="dwi.bval",
    bvec_file="dwi.bvec",
    volumes_to_delete=[0, 5, 10]  # Remove specific volumes
)
```

#### networktools - Graph Analysis

**Purpose**: Network analysis and graph theory applications for brain connectivity

**Key Features**:

-   Graph representation creation from brain meshes
-   Sparse matrix operations for large-scale networks
-   Connectivity analysis utilities

**Usage Example**:

```python
from clabtoolkit.networktools import triangulated_mesh_to_csr

# Convert mesh to graph representation
graph = triangulated_mesh_to_csr(vertices, faces)
```

#### segmentationtools - Image Segmentation

**Purpose**: Atlas-based and automated image segmentation

**Key Features**:

-   Atlas-based parcellation using ANTs
-   Template registration workflows
-   Multi-atlas segmentation support

**Usage Example**:

```python
from clabtoolkit.segmentationtools import abased_parcellation

# Perform atlas-based segmentation
abased_parcellation(
    moving_image="T1w.nii.gz",
    atlas_image="template.nii.gz",
    atlas_labels="template_labels.nii.gz"
)
```

### Quality Control and Visualization

#### qcqatools - Quality Control Framework

**Purpose**: Comprehensive quality assessment for neuroimaging data

**Key Features**:

-   Automated slice selection for quality control
-   Multi-modal data validation
-   Visual quality assessment tools
-   Report generation capabilities

#### visualizationtools - Advanced Brain Visualization

**Purpose**: Publication-quality brain visualization system

**Key Classes**:

-   `BrainPlotter`: Multi-view brain surface visualization
    -   Configurable view layouts
    -   Custom colormap support
    -   Publication-ready output
    -   Multi-hemisphere rendering

**Key Features**:

-   JSON-based view configuration system
-   Flexible layout management
-   High-quality rendering for publications
-   Interactive visualization capabilities

### Utilities and Infrastructure

#### misctools - Core Utilities

**Purpose**: Foundation utility functions supporting all modules

**Key Features**:

-   Enhanced command-line argument parsing
-   File system operations
-   Color processing utilities
-   Documentation generation helpers

#### pipelinetools - Workflow Management

**Purpose**: Pipeline orchestration and batch processing

**Key Features**:

-   Subject ID management for batch workflows
-   Parallel processing utilities
-   Progress tracking and monitoring

#### plottools - Plotting Infrastructure

**Purpose**: Low-level plotting support and layout calculations

**Key Features**:

-   Dynamic subplot grid calculation
-   Screen size detection
-   Multi-monitor support

#### dicomtools - DICOM Processing

**Purpose**: DICOM file organization and BIDS conversion

**Key Features**:

-   Multi-threaded DICOM organization
-   BIDS conversion workflows
-   Demographics integration
-   Session management

## Configuration System

clabtoolkit uses a sophisticated JSON-based configuration system located in `clabtoolkit/config/`:

-   **bids.json**: BIDS entity definitions and validation rules
-   **viz_views.json**: Visualization layout configurations
-   **lobes.json**: Anatomical lobe definitions for parcellation
-   **stats_mapping.json**: Statistical measure mappings and metadata

## Dependencies

### Core Dependencies

-   **nibabel**: Neuroimaging file I/O
-   **numpy**: Numerical computing
-   **pandas**: Data manipulation
-   **scipy**: Scientific computing
-   **matplotlib**: Basic plotting

### Specialized Dependencies

-   **pyvista**: 3D visualization and mesh processing
-   **rich**: Enhanced console output
-   **dipy**: Diffusion MRI processing
-   **h5py**: HDF5 file support

### Environment

All dependencies are specified in the **environment.yaml** file for reproducible conda environments.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Support and Documentation

-   **GitHub Issues**: Report bugs and request features
-   **Documentation**: Comprehensive API documentation with examples
-   **Test Suite**: Extensive test coverage with example data
-   **Jupyter Notebooks**: Interactive examples and tutorials

## Research Applications

clabtoolkit is particularly well-suited for:

-   **Connectomics Research**: Brain connectivity analysis and visualization
-   **Surface-based Analysis**: Cortical thickness, area, and curvature studies
-   **BIDS Data Management**: Large-scale neuroimaging dataset organization
-   **Multi-modal Integration**: Combined structural and diffusion MRI analysis
-   **Quality Control**: Automated quality assessment for neuroimaging data
-   **Publication Visualization**: High-quality brain visualizations for research papers

---

_This toolkit represents a comprehensive solution for modern neuroimaging research, combining robust data management with advanced analysis capabilities and publication-quality visualization tools._
