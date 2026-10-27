"""Tests for the sample data helpers of clabtoolkit.misctools."""

import hashlib
import re
import zipfile
from pathlib import Path

import pytest

import clabtoolkit.freesurfertools as cltfree
import clabtoolkit.misctools as cltmisc


@pytest.fixture
def cached_archive(tmp_path, monkeypatch):
    """A cache that already holds a valid 'surfaces' archive, so nothing is downloaded."""
    monkeypatch.delenv("CLABTOOLKIT_DATA_DIR", raising=False)
    cache = tmp_path / "cache"
    cache.mkdir()
    archive = cache / "surfaces.zip"
    with zipfile.ZipFile(archive, "w") as zip_file:
        zip_file.writestr("surfaces/pial.txt", "pial")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    monkeypatch.setattr(cltmisc, "ZENODO_DOI", "10.5281/zenodo.1")
    monkeypatch.setattr(
        cltmisc, "SAMPLE_DATA_REGISTRY", {"surfaces.zip": f"sha256:{digest}"}
    )
    return cache


class TestRegistry:
    def test_list_sample_data(self):
        assert cltmisc.list_sample_data() == [
            "annotations",
            "corticalmaps",
            "parcellations",
            "surfaces",
            "tractograms",
        ]

    def test_checksums_are_sha256(self):
        for name, checksum in cltmisc.SAMPLE_DATA_REGISTRY.items():
            assert name.endswith(".zip")
            assert re.fullmatch(r"sha256:[0-9a-f]{64}", checksum)


class TestFetchSampleData:
    def test_unknown_folder(self, tmp_path):
        with pytest.raises(ValueError, match="nope"):
            cltmisc.fetch_sample_data("nope", path=tmp_path)

    def test_record_doi_not_set(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cltmisc, "ZENODO_DOI", "10.5281/zenodo.REPLACE_ME")
        with pytest.raises(RuntimeError, match="DOI"):
            cltmisc.fetch_sample_data("surfaces", path=tmp_path)

    def test_unzips_a_cached_archive(self, cached_archive):
        root = cltmisc.fetch_sample_data("surfaces", path=cached_archive)
        assert (root / "surfaces" / "pial.txt").read_text() == "pial"

    def test_does_not_extract_twice(self, cached_archive):
        root = cltmisc.fetch_sample_data(["surfaces"], path=cached_archive)
        (root / "surfaces" / "pial.txt").write_text("edited")
        cltmisc.fetch_sample_data("surfaces", path=cached_archive)
        assert (root / "surfaces" / "pial.txt").read_text() == "edited"


class TestGetSampleDataDir:
    def test_uses_the_folder_bundled_in_the_package(self, tmp_path, monkeypatch):
        (tmp_path / "surfaces").mkdir()
        (tmp_path / "surfaces" / "pial.txt").write_text("pial")
        monkeypatch.setattr(cltmisc, "PACKAGE_DATA_DIR", tmp_path)
        # Nothing is fetched, which would fail with the placeholder DOI
        assert cltmisc.get_sample_data_dir("surfaces") == tmp_path / "surfaces"

    def test_fetches_a_folder_that_is_not_bundled(self, cached_archive, monkeypatch):
        monkeypatch.setattr(cltmisc, "PACKAGE_DATA_DIR", Path("/nonexistent"))
        folder = cltmisc.get_sample_data_dir("surfaces", path=cached_archive)
        assert (folder / "pial.txt").is_file()

    def test_bundled_parcellations(self):
        folder = cltmisc.get_sample_data_dir("parcellations")
        assert any(folder.glob("*_dseg.nii.gz"))

    def test_bundled_talairach_transform(self):
        folder = cltmisc.get_sample_data_dir("transforms")
        lta_file = folder / "sub-colin27_ses-01_space-tallin_desc-talairach_xfm.lta"
        assert lta_file.is_file()
        assert cltfree.get_cras_coordinates(str(lta_file)) == pytest.approx(
            [0.5, -17.5, 18.5]
        )
