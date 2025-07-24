"""Pytest configuration and fixtures for clabtoolkit tests."""

import pytest
import tempfile
import shutil
from pathlib import Path
import numpy as np
import pandas as pd


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_bids_filename():
    """Sample BIDS filename for testing."""
    return "sub-01_ses-M00_acq-3T_dir-AP_run-01_T1w.nii.gz"


@pytest.fixture
def sample_bids_entities():
    """Expected entities from sample BIDS filename."""
    return {
        'sub': '01',
        'ses': 'M00', 
        'acq': '3T',
        'dir': 'AP',
        'run': '01',
        'suffix': 'T1w',
        'extension': 'nii.gz'
    }


@pytest.fixture
def sample_connectivity_matrix():
    """Sample connectivity matrix for testing."""
    np.random.seed(42)
    return np.random.rand(100, 100)


@pytest.fixture
def sample_dataframe():
    """Sample pandas DataFrame for testing."""
    return pd.DataFrame({
        'subject': ['sub-01', 'sub-02', 'sub-03'],
        'session': ['ses-M00', 'ses-M00', 'ses-M06'],
        'value': [1.5, 2.3, 1.8]
    })


@pytest.fixture
def mock_nifti_file(temp_dir):
    """Create a mock NIfTI file for testing."""
    nifti_path = temp_dir / "test.nii.gz"
    # Create empty file (in real tests, you'd use nibabel to create proper NIfTI)
    nifti_path.touch()
    return nifti_path


@pytest.fixture
def mock_freesurfer_stats(temp_dir):
    """Create mock FreeSurfer stats file."""
    stats_content = """# FreeSurfer stats file
# Measure Cortex, NumVert, Number of Vertices, 100000, unitless
# Measure Cortex, WhiteSurfArea, White Surface Total Area, 80000, mm^2
# Measure Cortex, MeanThickness, Mean Thickness, 2.5, mm
"""
    stats_file = temp_dir / "lh.aparc.stats"
    stats_file.write_text(stats_content)
    return stats_file