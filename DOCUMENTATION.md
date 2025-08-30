# Connectomics Lab Toolkit (clabtoolkit)

## Overview

**clabtoolkit** is a comprehensive Python toolkit for neuroimaging data processing and analysis, specifically designed for working with brain connectivity data, BIDS datasets, and various neuroimaging formats. Developed by Yasser Alemán-Gómez, this toolkit provides a complete solution for connectomics research and surface-based brain analysis.

**Version**: 0.3.1  
**License**: Apache Software License 2.0  
**Python Support**: 3.9+  
**Documentation**: https://clabtoolkit.readthedocs.io  
**Repository**: https://github.com/connectomicslab/clabtoolkit

## Installation

### From PyPI

```bash
pip install clabtoolkit
```

### Development Installation

```bash
git clone https://github.com/connectomicslab/clabtoolkit.git
cd clabtoolkit
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

The toolkit follows a modular, layered architecture:

-   **Foundation Layer**: Core utilities and plotting infrastructure
-   **Data Layer**: BIDS compliance and image processing
-   **Analysis Layer**: Specialized neuroimaging analysis tools
-   **Workflow Layer**: Pipeline management and quality control

## Module Reference

### Core Data Management

#### bidstools - BIDS Dataset Management

**Purpose**: Complete BIDS (Brain Imaging Data Structure) compliance and dataset management

**Key Features**:

-   BIDS entity manipulation (`str2entity`, `entity2str`)
-   Dataset organization and validation
-   Batch processing utilities
-   Database table generation from BIDS datasets

**Key Functions**:

```python
# Convert BIDS filename to entity dictionary
entities = cltbids.str2entity("sub-01_ses-M00_acq-3T_T1w.nii.gz")

# Manipulate BIDS entities
cltbids.replace_entity_value(filename, 'ses', 'new_session')
cltbids.insert_entity(filename, 'run', '01', after='acq')

# Get all subjects from a BIDS dataset
subjects = cltbids.get_subjects("/path/to/bids/dataset")

# Generate comprehensive dataset overview
database = cltbids.get_bids_database_table("/path/to/bids/dataset")
```

#### imagetools - Image Processing Engine

**Purpose**: Advanced neuroimaging operations and morphological processing

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
result = morph.closing(binary_image, structuring_element)
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
from clabtoolkit.freesurfertools import AnnotParcellation

# Load and process annotation file
annot = AnnotParcellation("lh.aparc.a2009s.annot")
annot.correct_parcellation()  # Fix unlabeled vertices
annot.save_as_gcs("output.gcs")  # Convert format
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
from clabtoolkit.parcellationtools import Parcellation

# Load parcellation with lookup table
parc = Parcellation()
parc.load_from_file("/path/to/parcellation.nii.gz", "/path/to/lut.lut")

# Filter specific regions
parc.filter_regions(['cortex', 'cerebellum'])

# Compute regional volumes
volumes = parc.compute_regional_volumes()
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
surface.load_scalar_data("/path/to/thickness.mgh")
surface.plot(colormap='viridis', views = ["lateral", "medial"])
```

#### morphometrytools - Morphometric Analysis

**Purpose**: Surface-based morphometric computations and statistics

**Key Features**:

-   Regional value extraction from surface annotations
-   Multi-hemisphere morphometric analysis
-   Statistical summary generation
-   Integration with parcellation workflows

**Usage Example**:

```python
from clabtoolkit.morphometrytools import compute_reg_val_fromannot

# Extract regional cortical thickness values
stats = compute_reg_val_fromannot(
    scalar_file="lh.thickness.mgh",
    annot_file="lh.aparc.annot",
    lut_file="aparc.lut"
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

-   `SurfacePlotter`: Multi-view brain surface visualization
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

## Testing

Run the test suite:

```bash
pytest
```

Run with coverage:

```bash
pytest --cov=clabtoolkit
```

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

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Common Workflows

### 1. BIDS Dataset Processing

```python
import clabtoolkit.bidstools as bids

# Get dataset overview
subjects = bids.get_subjects("/path/to/bids")
database = bids.get_bids_database_table("/path/to/bids")

# Process filenames
for filename in dataset_files:
    entities = bids.str2entity(filename)
    # Process based on entities
```

### 2. FreeSurfer Surface Analysis

```python
import clabtoolkit.surfacetools as surf
import clabtoolkit.freesurfertools as fs

# Load surface and annotation
surface = surf.Surface("lh.pial")
annot = fs.AnnotParcellation("lh.aparc.annot")

# Load scalar data and visualize
surface.load_scalar_data("lh.thickness.mgh")
surface.plot(scalar_map=True)
```

### 3. Multi-View Brain Visualization

```python
from clabtoolkit.visualizationtools import SurfacePlotter

# Create publication-quality multi-view plots
plotter = SurfacePlotter(config_file="custom_views.json")
plotter.plot_surface_with_parcellation(
    surface_files=["lh.pial", "rh.pial"],
    scalar_files=["lh.thickness.mgh", "rh.thickness.mgh"]
)
```

### 4. Atlas-based Analysis

```python
from clabtoolkit.parcellationtools import Parcellation
from clabtoolkit.morphometrytools import compute_reg_val_fromannot

# Load and process parcellation
parc = Parcellation()
parc.load_from_file("parcellation.nii.gz", "lookup_table.lut")

# Extract regional morphometry
stats = compute_reg_val_fromannot(
    scalar_file="cortical_thickness.mgh",
    annot_file="parcellation.annot",
    lut_file="lookup_table.lut"
)
```

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
