"""Tests for the dicomtools module.

The tests follow usage_examples/examples-dicomtools-methods.ipynb and use the same
small synthetic dataset: two subjects, three sessions and seven series of 32 x 32 MR
slices with the header fields used by dicomtools.
"""

import os
import shutil
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import pytest
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

import clabtoolkit.dicomtools as cltdicom

# {folder: [(PatientID, StudyDate, StudyTime, [(SeriesNumber, SeriesDescription, n_slices)])]}
PLAN = {
    "sub-01": [
        (
            "sub-01",
            "20240312",
            "091530",
            [(1, "T1w_MPRAGE", 6), (2, "DWI_b1000_30dir", 4), (3, "rs-fMRI_BOLD", 4)],
        ),
        ("sub-01", "20240920", "140210", [(1, "T1w_MPRAGE", 6), (2, "FLAIR_3D", 5)]),
    ],
    "sub-02": [
        ("sub-02", "20240402", "104000", [(1, "T1w_MPRAGE", 6), (2, "DWI_b1000_30dir", 4)]),
    ],
}

# Expected <session>/<series>: number of files, for each subject
EXPECTED = {
    "sub-01": {
        "ses-20240312091530": {
            "0001-T1w-MPRAGE": 6,
            "0002-DWI-b1000-30dir": 4,
            "0003-rs-fMRI-BOLD": 4,
        },
        "ses-20240920140210": {"0001-T1w-MPRAGE": 6, "0002-FLAIR-3D": 5},
    },
    "sub-02": {
        "ses-20240402104000": {"0001-T1w-MPRAGE": 6, "0002-DWI-b1000-30dir": 4},
    },
}


def _write_slice(path, patient_id, study_date, study_time, study_uid, series_uid, ser_num, ser_desc, inst):
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = MRImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = Dataset()
    ds.file_meta = meta
    ds.SOPClassUID = MRImageStorage
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.PatientID, ds.PatientName = patient_id, f"Synthetic^{patient_id}"
    ds.StudyDate = ds.SeriesDate = ds.AcquisitionDate = study_date
    ds.StudyTime = ds.SeriesTime = study_time
    ds.StudyInstanceUID, ds.SeriesInstanceUID = study_uid, series_uid
    ds.Modality, ds.Manufacturer = "MR", "SyntheticScanner"
    ds.SeriesNumber, ds.InstanceNumber = ser_num, inst
    ds.SeriesDescription = ds.ProtocolName = ser_desc
    ds.Rows = ds.Columns = 32
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 12, 11, 0
    ds.PixelData = np.full((32, 32), 100 * ser_num + inst, dtype=np.uint16).tobytes()
    ds.save_as(path, enforce_file_format=True)


def _simulate_dicom_dataset(root: Path) -> Path:
    """Write unorganized DICOMs. sub-01 files are flat, sub-02 files are nested."""
    count = 0
    for folder, studies in PLAN.items():
        for patient_id, sdate, stime, series in studies:
            study_uid = generate_uid()
            for ser_num, ser_desc, n_slices in series:
                series_uid = generate_uid()
                for inst in range(1, n_slices + 1):
                    count += 1
                    if folder == "sub-01":
                        sub_dir = root / folder
                    else:
                        sub_dir = root / folder / "DICOM" / f"ST{sdate[-4:]}" / f"SE{ser_num:04d}"
                    sub_dir.mkdir(parents=True, exist_ok=True)
                    _write_slice(
                        sub_dir / f"IM_{count:08d}.dcm",
                        patient_id,
                        sdate,
                        stime,
                        study_uid,
                        series_uid,
                        ser_num,
                        ser_desc,
                        inst,
                    )
    return root


@pytest.fixture(scope="module")
def _raw_master(tmp_path_factory):
    return _simulate_dicom_dataset(tmp_path_factory.mktemp("dicom_raw"))


@pytest.fixture
def raw_dir(_raw_master, tmp_path):
    """Fresh copy of the unorganized dataset (tests may add or remove files)."""
    dest = tmp_path / "raw"
    shutil.copytree(_raw_master, dest)
    return dest


@pytest.fixture
def out_dir(tmp_path):
    return tmp_path / "out"


@pytest.fixture
def dicom_file(raw_dir):
    return sorted((raw_dir / "sub-01").iterdir())[0]


@pytest.fixture
def demog_file(tmp_path):
    path = tmp_path / "demographics.csv"
    pd.DataFrame(
        {
            "participant_id": ["sub-01", "sub-01", "sub-02"],
            "session_id": ["V1", "V2", "V1"],
            "acq_date": ["03/12/2024", "09/20/2024", "04/02/2024"],
        }
    ).to_csv(path, index=False)
    return path


def _layout(root: Path) -> dict:
    """{subject: {session: {series: n_files}}} of an organized folder."""
    layout = {}
    for subj in sorted(p for p in root.iterdir() if p.is_dir()):
        layout[subj.name] = {
            ses.name: {ser.name: len(list(ser.iterdir())) for ser in sorted(ses.iterdir())}
            for ses in sorted(subj.iterdir())
        }
    return layout


def _dataset(**fields) -> Dataset:
    ds = Dataset()
    for key, value in fields.items():
        setattr(ds, key, value)
    return ds


######################################################################################################
class TestSelectSubjectIds:
    SUBJECTS = ["sub-01", "sub-02", "sub-10"]

    def test_comma_separated_ids_with_and_without_prefix(self):
        assert cltdicom._select_subject_ids(self.SUBJECTS, "02,sub-10", False) == [
            "sub-02",
            "sub-10",
        ]

    def test_ids_are_matched_exactly(self):
        # "01" must not select "sub-10" nor the other way round
        assert cltdicom._select_subject_ids(self.SUBJECTS, "1") == []

    def test_ids_from_text_file(self, tmp_path):
        ids = tmp_path / "ids.txt"
        ids.write_text("sub-01\n\n 02 \n")
        assert cltdicom._select_subject_ids(self.SUBJECTS, str(ids)) == [
            "sub-01",
            "sub-02",
        ]

    def test_nosub_requires_exact_folder_names(self):
        assert cltdicom._select_subject_ids(["01", "02"], "02", nosub=True) == ["02"]
        assert cltdicom._select_subject_ids(["sub-01"], "01", nosub=True) == []


class TestParseAcqDate:
    @pytest.mark.parametrize("value", ["03/12/2024", "2024-03-12", "2024-03-12 09:15:30"])
    def test_supported_formats(self, value):
        assert cltdicom._parse_acq_date(value).date().isoformat() == "2024-03-12"

    def test_invalid_date_raises(self):
        with pytest.raises(ValueError):
            cltdicom._parse_acq_date("12-03-2024")


######################################################################################################
class TestCreateSessionSeriesNames:
    def base(self, **extra):
        fields = dict(
            StudyDate="20240312",
            StudyTime="091530",
            SeriesNumber=1,
            SeriesDescription="T1w_MPRAGE",
        )
        fields.update(extra)
        return _dataset(**fields)

    def test_basic_names(self):
        assert cltdicom.create_session_series_names(self.base()) == (
            "20240312091530",
            "0001-T1w-MPRAGE",
        )

    def test_names_of_a_real_file(self, dicom_file):
        ses, ser = cltdicom.create_session_series_names(pydicom.dcmread(dicom_file))
        assert (ses, ser) == ("20240312091530", "0001-T1w-MPRAGE")

    def test_special_characters_are_cleaned(self):
        ds = self.base(SeriesDescription="T1w (MPRAGE) 1mm/iso*")
        assert cltdicom.create_session_series_names(ds)[1] == "0001-T1w-MPRAGE-1mm-iso"

    def test_protocol_name_used_without_series_description(self):
        ds = self.base(ProtocolName="t1_mprage_sag")
        del ds.SeriesDescription
        assert cltdicom.create_session_series_names(ds)[1] == "0001-t1-mprage-sag"

    def test_sequence_name_has_priority_over_protocol_name(self):
        ds = self.base(SequenceName="*tfl3d1", ProtocolName="other")
        del ds.SeriesDescription
        assert cltdicom.create_session_series_names(ds)[1] == "0001-tfl3d1"

    def test_scanning_sequence_and_variant(self):
        ds = self.base(ScanningSequence="GR", SequenceVariant="SP")
        del ds.SeriesDescription
        assert cltdicom.create_session_series_names(ds)[1] == "0001-GR-SP"

    def test_no_name_available(self):
        ds = self.base()
        del ds.SeriesDescription
        assert cltdicom.create_session_series_names(ds)[1] == "0001-NoSerName"

    def test_series_number_omitted_when_missing(self):
        ds = self.base()
        del ds.SeriesNumber
        assert cltdicom.create_session_series_names(ds)[1] == "T1w-MPRAGE"

    @pytest.mark.parametrize(
        "study_time, expected",
        [
            ("091530.250000", "20240312091530"),
            ("0915", "20240312091500"),
            ("09", "20240312090000"),
            ("09:15:30", "20240312091530"),
            ("", "20240312000000"),
        ],
    )
    def test_study_time_formats(self, study_time, expected):
        ds = self.base(StudyTime=study_time)
        assert cltdicom.create_session_series_names(ds)[0] == expected

    def test_missing_study_date_and_time(self):
        ds = self.base()
        del ds.StudyDate, ds.StudyTime
        assert cltdicom.create_session_series_names(ds)[0] == "00000000000000"


######################################################################################################
class TestGetDicomInfo:
    def test_all_tags(self, dicom_file):
        info = cltdicom.get_dicom_info(str(dicom_file))
        assert info["PatientID"] == "sub-01"
        assert info["SeriesDescription"] == "T1w_MPRAGE"
        assert "PixelData" in info

    def test_selected_tags(self, dicom_file):
        info = cltdicom.get_dicom_info(str(dicom_file), tags=["PatientID", "Modality"])
        assert info == {"PatientID": "sub-01", "Modality": "MR"}

    def test_single_tag_as_string(self, dicom_file):
        assert cltdicom.get_dicom_info(str(dicom_file), tags="Modality") == {"Modality": "MR"}

    def test_missing_tag_ignore(self, dicom_file):
        info = cltdicom.get_dicom_info(
            str(dicom_file), tags=["PatientID", "EchoTime"], missing_tag_behavior="ignore"
        )
        assert info == {"PatientID": "sub-01"}

    def test_missing_tag_warn_prints(self, dicom_file, capsys):
        info = cltdicom.get_dicom_info(str(dicom_file), tags=["EchoTime"])
        assert info == {}
        assert "EchoTime" in capsys.readouterr().out

    def test_missing_tag_raise(self, dicom_file):
        with pytest.raises(AttributeError):
            cltdicom.get_dicom_info(str(dicom_file), tags=["EchoTime"], missing_tag_behavior="raise")

    def test_invalid_missing_tag_behavior(self, dicom_file):
        with pytest.raises(ValueError):
            cltdicom.get_dicom_info(str(dicom_file), missing_tag_behavior="skip")

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            cltdicom.get_dicom_info(str(tmp_path / "nope.dcm"))

    def test_not_a_dicom(self, tmp_path):
        bad = tmp_path / "notes.dcm"
        bad.write_text("not a DICOM file")
        with pytest.raises(Exception):
            cltdicom.get_dicom_info(str(bad))

    @pytest.mark.parametrize("tags", [5, ("PatientID",), [1]])
    def test_invalid_tags_type(self, dicom_file, tags):
        with pytest.raises(TypeError):
            cltdicom.get_dicom_info(str(dicom_file), tags=tags)


######################################################################################################
class TestCopyDicomFile:
    def test_copy_into_hierarchy(self, dicom_file, out_dir):
        dest = cltdicom.copy_dicom_file(str(dicom_file), "sub-01", str(out_dir))
        assert Path(dest) == out_dir / "sub-01" / "ses-20240312091530" / "0001-T1w-MPRAGE"
        assert (Path(dest) / dicom_file.name).is_file()

    def test_ses_id_overrides_the_session(self, dicom_file, out_dir):
        dest = cltdicom.copy_dicom_file(str(dicom_file), "sub-01", str(out_dir), ses_id="baseline")
        assert Path(dest).parent.name == "ses-baseline"

    def test_ses_id_with_prefix_is_not_duplicated(self, dicom_file, out_dir):
        dest = cltdicom.copy_dicom_file(str(dicom_file), "sub-01", str(out_dir), ses_id="ses-M00")
        assert Path(dest).parent.name == "ses-M00"

    def test_demographics_visit_is_appended(self, dicom_file, out_dir):
        demog = pd.DataFrame({"session_id": ["V1", "V2"]})
        dates = [pd.Timestamp("2024-03-12"), pd.Timestamp("2024-09-20")]
        dest = cltdicom.copy_dicom_file(
            str(dicom_file),
            "sub-01",
            str(out_dir),
            date_times=dates,
            demogbool=True,
            demog_tab=demog,
        )
        assert Path(dest).parent.name == "ses-20240312091530V1"

    def test_overwrite(self, dicom_file, out_dir):
        dest = Path(cltdicom.copy_dicom_file(str(dicom_file), "sub-01", str(out_dir)))
        copied = dest / dicom_file.name
        copied.write_bytes(b"damaged")

        cltdicom.copy_dicom_file(str(dicom_file), "sub-01", str(out_dir))
        assert copied.read_bytes() == b"damaged"

        cltdicom.copy_dicom_file(str(dicom_file), "sub-01", str(out_dir), overwrite=True)
        assert copied.stat().st_size == dicom_file.stat().st_size

    def test_not_a_dicom_is_skipped(self, tmp_path, out_dir, capsys):
        bad = tmp_path / "notes.txt"
        bad.write_text("Exported from the scanner")
        assert cltdicom.copy_dicom_file(str(bad), "sub-01", str(out_dir)) is None
        assert "not a valid DICOM" in capsys.readouterr().out
        assert not out_dir.exists()


######################################################################################################
class TestOrgConvDicoms:
    def test_organizes_all_subjects(self, raw_dir, out_dir):
        cltdicom.org_conv_dicoms(str(raw_dir), str(out_dir), nthreads=1)
        assert _layout(out_dir) == EXPECTED

    def test_input_is_left_untouched(self, raw_dir, out_dir):
        before = sum(len(f) for _, _, f in os.walk(raw_dir))
        cltdicom.org_conv_dicoms(str(raw_dir), str(out_dir), nthreads=1)
        assert sum(len(f) for _, _, f in os.walk(raw_dir)) == before

    def test_demographics_define_the_visit(self, raw_dir, out_dir, demog_file):
        cltdicom.org_conv_dicoms(str(raw_dir), str(out_dir), demog_file=str(demog_file), nthreads=1)
        assert sorted(p.name for p in (out_dir / "sub-01").iterdir()) == [
            "ses-20240312091530V1",
            "ses-20240920140210V2",
        ]
        assert [p.name for p in (out_dir / "sub-02").iterdir()] == ["ses-20240402104000V1"]

    @pytest.mark.parametrize("ids", ["02", "sub-02"])
    def test_subset_of_subjects(self, raw_dir, out_dir, ids):
        cltdicom.org_conv_dicoms(str(raw_dir), str(out_dir), ids_file=ids, nthreads=2)
        assert _layout(out_dir) == {"sub-02": EXPECTED["sub-02"]}

    def test_ids_from_file(self, raw_dir, out_dir, tmp_path):
        ids = tmp_path / "ids.txt"
        ids.write_text("sub-01\n")
        cltdicom.org_conv_dicoms(str(raw_dir), str(out_dir), ids_file=str(ids), nthreads=1)
        assert list(_layout(out_dir)) == ["sub-01"]

    def test_ses_id_forces_the_session(self, raw_dir, out_dir):
        cltdicom.org_conv_dicoms(
            str(raw_dir), str(out_dir), ses_id="baseline", ids_file="sub-02", nthreads=1
        )
        assert _layout(out_dir) == {
            "sub-02": {"ses-baseline": EXPECTED["sub-02"]["ses-20240402104000"]}
        }

    def test_compression(self, raw_dir, out_dir):
        cltdicom.org_conv_dicoms(str(raw_dir), str(out_dir), boolcomp=True, nthreads=1)
        assert sorted(p.name for p in (out_dir / "sub-01").iterdir()) == [
            "ses-20240312091530.tar.gz",
            "ses-20240920140210.tar.gz",
        ]
        assert [p.name for p in (out_dir / "sub-02").iterdir()] == ["ses-20240402104000.tar.gz"]

    def test_missing_input_folder(self, tmp_path, out_dir):
        with pytest.raises(FileNotFoundError):
            cltdicom.org_conv_dicoms(str(tmp_path / "missing"), str(out_dir))


class TestOrganizeDicomFiles:
    def test_default_is_org_conv_dicoms(self, raw_dir, out_dir):
        cltdicom.organize_dicom_files(str(raw_dir), str(out_dir), nthreads=1)
        assert _layout(out_dir) == EXPECTED

    def test_no_sub_folder_uses_patient_id(self, raw_dir, out_dir):
        cltdicom.organize_dicom_files(str(raw_dir), str(out_dir), no_sub_folder=True, nthreads=1)
        assert _layout(out_dir) == EXPECTED

    def test_arguments_are_forwarded(self, monkeypatch, raw_dir, out_dir):
        calls = {}
        monkeypatch.setattr(cltdicom, "org_conv_dicoms", lambda **kw: calls.update(kw))
        cltdicom.organize_dicom_files(str(raw_dir), str(out_dir), ses_id="M00", overwrite=True)
        assert calls["ses_id"] == "M00" and calls["overwrite"] is True


######################################################################################################
class TestOrgDicomFolder:
    @pytest.fixture
    def mixed_dir(self, raw_dir, tmp_path):
        mixed = tmp_path / "mixed"
        mixed.mkdir()
        for root, _, files in os.walk(raw_dir):
            for f in files:
                shutil.copy(os.path.join(root, f), mixed)
        return mixed

    def test_subjects_named_after_patient_id(self, mixed_dir, out_dir):
        cltdicom.org_dicom_folder(str(mixed_dir), str(out_dir), nthreads=1)
        assert _layout(out_dir) == EXPECTED

    def test_non_dicom_files_are_skipped(self, mixed_dir, out_dir):
        (mixed_dir / "README.txt").write_text("Exported from the scanner")
        cltdicom.org_dicom_folder(str(mixed_dir), str(out_dir), nthreads=2)
        assert _layout(out_dir) == EXPECTED

    def test_compression(self, mixed_dir, out_dir):
        cltdicom.org_dicom_folder(str(mixed_dir), str(out_dir), boolcomp=True, nthreads=1)
        assert sorted(p.name for p in (out_dir / "sub-01").iterdir()) == [
            "ses-20240312091530.tar.gz",
            "ses-20240920140210.tar.gz",
        ]

    def test_missing_input_folder(self, tmp_path, out_dir):
        with pytest.raises(FileNotFoundError):
            cltdicom.org_dicom_folder(str(tmp_path / "missing"), str(out_dir))


######################################################################################################
class TestCompression:
    @pytest.fixture
    def organized(self, raw_dir, out_dir):
        cltdicom.org_conv_dicoms(str(raw_dir), str(out_dir), nthreads=1)
        return out_dir

    def test_compress_replaces_the_session_folders(self, organized):
        failed = cltdicom.compress_dicom_session(str(organized))
        assert not failed
        assert sorted(p.name for p in (organized / "sub-01").iterdir()) == [
            "ses-20240312091530.tar.gz",
            "ses-20240920140210.tar.gz",
        ]

    def test_compress_one_subject_keeping_originals(self, organized):
        cltdicom.compress_dicom_session(str(organized), subj_ids=["sub-02"], remove_original=False)
        assert sorted(p.name for p in (organized / "sub-02").iterdir()) == [
            "ses-20240402104000",
            "ses-20240402104000.tar.gz",
        ]
        # the other subject is not touched
        assert not list((organized / "sub-01").glob("*.tar.gz"))

    def test_archive_content(self, organized):
        cltdicom.compress_dicom_session(str(organized))
        tar_path = organized / "sub-01" / "ses-20240312091530.tar.gz"
        with tarfile.open(tar_path) as tar:
            names = tar.getnames()
        series = {n.split("/")[1] for n in names if n.count("/") == 1}
        assert series == {"0001-T1w-MPRAGE", "0002-DWI-b1000-30dir", "0003-rs-fMRI-BOLD"}
        assert sum(n.endswith(".dcm") for n in names) == 14

    def test_existing_archives_are_skipped(self, organized, capsys):
        cltdicom.compress_dicom_session(str(organized), remove_original=False)
        cltdicom.compress_dicom_session(str(organized), remove_original=False)
        assert "already exists" in capsys.readouterr().out

    def test_round_trip(self, organized):
        cltdicom.compress_dicom_session(str(organized))
        failed = cltdicom.uncompress_dicom_session(str(organized))
        assert not failed
        for subj, sessions in EXPECTED.items():
            for ses, series in sessions.items():
                for ser, n_files in series.items():
                    assert len(list((organized / subj / ses / ser).iterdir())) == n_files
        # tar.gz files are kept by default
        assert (organized / "sub-02" / "ses-20240402104000.tar.gz").is_file()

    def test_uncompress_one_subject(self, raw_dir, out_dir):
        cltdicom.org_conv_dicoms(str(raw_dir), str(out_dir), boolcomp=True, nthreads=1)
        cltdicom.uncompress_dicom_session(str(out_dir), subj_ids=["sub-02"])
        assert (out_dir / "sub-02" / "ses-20240402104000").is_dir()
        assert not (out_dir / "sub-01" / "ses-20240312091530").exists()

    def test_uncompress_removing_archives(self, raw_dir, out_dir):
        cltdicom.org_conv_dicoms(str(raw_dir), str(out_dir), boolcomp=True, nthreads=1)
        cltdicom.uncompress_dicom_session(str(out_dir), boolrmtar=True)
        assert not list(out_dir.rglob("*.tar.gz"))
        assert _layout(out_dir) == EXPECTED

    def test_uncompress_ids_from_file(self, raw_dir, out_dir, tmp_path):
        cltdicom.org_conv_dicoms(str(raw_dir), str(out_dir), boolcomp=True, nthreads=1)
        ids = tmp_path / "subjects.txt"
        ids.write_text("sub-01\n")
        cltdicom.uncompress_dicom_session(str(out_dir), subj_ids=str(ids), boolrmtar=True)
        assert (out_dir / "sub-01" / "ses-20240312091530").is_dir()
        assert list((out_dir / "sub-02").glob("*.tar.gz"))


######################################################################################################
class TestHelpers:
    def test_safe_unlink_removes_file(self, tmp_path):
        f = tmp_path / "x.tar.gz"
        f.write_text("x")
        cltdicom._safe_unlink(f)
        assert not f.exists()

    def test_safe_unlink_ignores_missing_file(self, tmp_path):
        cltdicom._safe_unlink(tmp_path / "missing")
