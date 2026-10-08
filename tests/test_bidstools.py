"""Tests for clabtoolkit.bidstools."""

import os
from pathlib import Path

import pandas as pd
import pytest

import clabtoolkit.bidstools as cltbids

T1W = "sub-01_ses-M00_acq-3T_dir-AP_run-01_T1w.nii.gz"


def nifti_files(root) -> list[str]:
    """Relative POSIX paths of the NIfTI files inside root."""
    return sorted(
        Path(os.path.relpath(os.path.join(r, f), root)).as_posix()
        for r, _, files in os.walk(root)
        for f in files
        if f.endswith(".nii.gz")
    )


@pytest.fixture
def bids_root(tmp_path):
    """Two subjects with two sessions each (anat and dwi), plus fmriprep derivatives."""
    return cltbids.create_a_simulated_bids_dataset(
        tmp_path / "ds",
        n_subjects=2,
        n_visits=2,
        modalities=["anat", "dwi"],
        image_shape=(2, 2, 2),
        n_directions_dwi=2,
        include_derivatives="fmriprep",
        show_progress=False,
    )


@pytest.fixture
def nested_bids(tmp_path):
    """
    One subject with two sessions whose T1w images carry an acq entity, stored inside
    a parent folder named with BIDS entities. Renames must never leave the dataset root.
    """

    def _make(parent: str = "ses-01_acq-run_project") -> Path:
        root = cltbids.create_a_simulated_bids_dataset(
            tmp_path / parent / "ds",
            n_subjects=1,
            n_visits=2,
            modalities=["anat"],
            image_shape=(2, 2, 2),
            show_progress=False,
        )
        for r, _, files in os.walk(root):
            for f in files:
                if "_T1w" in f:
                    new = f.replace("_T1w", "_acq-mprage_T1w")
                    os.rename(os.path.join(r, f), os.path.join(r, new))
        return root

    return _make


####################################################################################################
# Section 1: BIDS naming conventions
####################################################################################################
class TestEntityConversion:
    def test_str2entity(self, sample_bids_entities):
        assert cltbids.str2entity(T1W) == sample_bids_entities
        assert cltbids.str2entity("sub-01_ses-M00") == {"sub": "01", "ses": "M00"}
        assert cltbids.str2entity("sub-01_desc-a-b_T1w") == {
            "sub": "01",
            "desc": "a-b",  # Only the first dash separates key and value
            "suffix": "T1w",
        }

    def test_entity2str(self, sample_bids_entities):
        assert cltbids.entity2str(sample_bids_entities) == T1W
        assert cltbids.entity2str({"sub": "01", "run": 1}) == "sub-01_run-1"
        original = dict(sample_bids_entities)
        cltbids.entity2str(sample_bids_entities)
        assert sample_bids_entities == original  # The input is not modified

    def test_round_trip(self):
        name = "sub-01_task-rest_space-MNI_desc-preproc_bold.nii.gz"
        assert cltbids.entity2str(cltbids.str2entity(name)) == name


class TestEntityEditing:
    def test_delete_entity(self):
        assert (
            cltbids.delete_entity(T1W, "acq")
            == "sub-01_ses-M00_dir-AP_run-01_T1w.nii.gz"
        )
        assert (
            cltbids.delete_entity(T1W, ["acq", "dir", "acq"])
            == "sub-01_ses-M00_run-01_T1w.nii.gz"
        )
        assert cltbids.delete_entity({"sub": "01", "run": "01"}, "run") == {"sub": "01"}
        assert cltbids.delete_entity(T1W, "echo") == T1W  # Absent entity

    def test_delete_entity_by_value(self):
        # With a dict, only the entities holding exactly that value are removed
        assert cltbids.delete_entity(T1W, {"acq": "3T"}) == cltbids.delete_entity(
            T1W, "acq"
        )
        assert cltbids.delete_entity(T1W, {"acq": "7T"}) == T1W
        assert cltbids.delete_entity("sub-01_acq-3_T1w.nii.gz", {"acq": "3T"}) == (
            "sub-01_acq-3_T1w.nii.gz"
        )
        assert cltbids.delete_entity(
            T1W, {"acq": ["1T", "3T"]}
        ) == cltbids.delete_entity(T1W, "acq")

    def test_delete_entity_errors(self):
        with pytest.raises(ValueError):
            cltbids.delete_entity(5, "acq")
        with pytest.raises(ValueError):
            cltbids.delete_entity(T1W, 5)

    def test_replace_entity_value(self, capsys):
        assert (
            cltbids.replace_entity_value(T1W, {"acq": "7T"})
            == "sub-01_ses-M00_acq-7T_dir-AP_run-01_T1w.nii.gz"
        )
        assert cltbids.replace_entity_value(T1W, "acq-7T_run-02") == (
            "sub-01_ses-M00_acq-7T_dir-AP_run-02_T1w.nii.gz"
        )
        assert cltbids.replace_entity_value({"sub": "01"}, {"sub": ""}) == {"sub": "01"}
        assert cltbids.replace_entity_value(T1W, {"echo": "2"}, verbose=True) == T1W
        assert "not found" in capsys.readouterr().out
        with pytest.raises(ValueError):
            cltbids.replace_entity_value(5, {"acq": "7T"})

    def test_replace_entity_key(self, capsys):
        assert (
            cltbids.replace_entity_key(T1W, {"acq": "TESTrep1", "dir": "TESTrep2"})
            == "sub-01_ses-M00_TESTrep1-3T_TESTrep2-AP_run-01_T1w.nii.gz"
        )
        assert cltbids.replace_entity_key(
            {"sub": "01", "run": "1"}, {"run": "echo"}
        ) == {
            "sub": "01",
            "echo": "1",
        }
        with pytest.raises(ValueError):
            cltbids.replace_entity_key(T1W, "acq")
        with pytest.raises(ValueError):
            cltbids.replace_entity_key(5, {"acq": "rec"})

    def test_replace_entity_key_verbose_warns_for_missing_keys(self, capsys):
        cltbids.replace_entity_key("sub-01_T1w", {"acq": "rec"}, verbose=True)
        assert "acq" in capsys.readouterr().out

    def test_insert_entity(self):
        assert (
            cltbids.insert_entity(T1W, {"task": "rest"})
            == "sub-01_ses-M00_acq-3T_dir-AP_run-01_task-rest_T1w.nii.gz"
        )
        assert (
            cltbids.insert_entity(T1W, {"task": "rest"}, prev_entity="ses")
            == "sub-01_ses-M00_task-rest_acq-3T_dir-AP_run-01_T1w.nii.gz"
        )
        # Entities that already exist are not added again
        assert cltbids.insert_entity(T1W, {"acq": "7T"}) == T1W
        out = cltbids.insert_entity({"sub": "01", "suffix": "T1w"}, {"run": "01"})
        assert out == {"sub": "01", "run": "01", "suffix": "T1w"}
        with pytest.raises(ValueError, match="Reference entity"):
            cltbids.insert_entity(T1W, {"task": "rest"}, prev_entity="echo")
        with pytest.raises(ValueError):
            cltbids.insert_entity(5, {"task": "rest"})


class TestFilenameValidation:
    @pytest.mark.parametrize(
        "name, expected",
        [
            ("sub-01_ses-pre_task-rest_bold.nii.gz", True),
            ("sub-01_ses-pre_task-rest_bold", True),
            ("sub-01_ses-pre_task-rest.nii.gz", True),
            ("sub-01_ses-M00_run-01", True),
            ("sub-01", True),
            ("sub-01_ses-pre_task-rest_bold_extra.nii.gz", False),
            ("bert", False),
            ("", False),
            ("sub-01_bad-value-x_T1w", False),
        ],
    )
    def test_is_bids_filename(self, name, expected):
        assert cltbids.is_bids_filename(name) is expected

    def test_is_bids_filename_extensive(self):
        assert cltbids.is_bids_filename("sub-01_ses-pre_T1w.nii.gz", extensive=True)
        assert not cltbids.is_bids_filename("sub-01_foo-bar_T1w", extensive=True)
        assert not cltbids.is_bids_filename("sub-01_notasuffix", extensive=True)


class TestEntitiesTables:
    def test_entities4table(self, tmp_path):
        all_entities = cltbids.entities4table()
        assert (
            all_entities["sub"] == "Participant"
            and all_entities["desc"] == "Description"
        )
        assert cltbids.entities4table(selected_entities="sub,ses") == {
            "sub": "Participant",
            "ses": "Session",
        }
        assert list(cltbids.entities4table(selected_entities=["run"])) == ["run"]
        assert list(cltbids.entities4table(selected_entities={"acq": None})) == ["acq"]
        assert list(cltbids.entities4table(selected_entities="sub-01_ses-02_T1w")) == [
            "sub",
            "ses",
        ]
        custom = tmp_path / "entities.json"
        custom.write_text('{"a": {"sub": "Subject"}}')
        assert cltbids.entities4table(str(custom)) == {"sub": "Subject"}

    def test_entities4table_errors(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            cltbids.entities4table(str(tmp_path / "missing.json"))
        with pytest.raises(TypeError):
            cltbids.entities4table(5)

    def test_entities_to_table(self):
        df = cltbids.entities_to_table(
            "/data/sub-01/ses-pre/sub-01_ses-pre_task-rest_bold.nii.gz", ["sub", "ses"]
        )
        assert df.to_dict("records") == [{"Participant": "01", "Session": "pre"}]
        df = cltbids.entities_to_table(
            "/d/sub-01_ses-pre_bold.nii.gz", {"sub": "Subject"}
        )
        assert df.columns.tolist() == ["Subject"]
        everything = cltbids.entities_to_table("/d/sub-01_run-02_echo-1_T1w.nii.gz")
        assert everything.columns.tolist() == ["Participant", "Run", "Echo"]

    def test_entities_to_table_type_column_is_last(self):
        df = cltbids.entities_to_table(
            "/data/sub-01_ses-pre_task-rest_bold.nii.gz",
            ["sub", "ses"],
            include_suffix=True,
        )
        assert df.columns.tolist() == ["Participant", "Session", "Type"]
        assert df["Type"].tolist() == ["bold"]

    def test_entities_to_table_special_entities(self):
        row = cltbids.entities_to_table(
            "/d/sub-01_atlas-chimeraLFMIIIFIF_desc-grow1mm_dseg.nii.gz"
        ).iloc[0]
        assert row["Atlas"] == "chimera" and row["ChimeraCode"] == "LFMIIIFIF"
        assert row["Description"] == "grow1mm" and row["GrowIntoWM"] == "1mm"
        other = cltbids.entities_to_table("/d/sub-01_atlas-aparc_dseg.nii.gz").iloc[0]
        assert other["Atlas"] == "aparc" and other["ChimeraCode"] == ""

    def test_entities_to_table_non_bids(self):
        df = cltbids.entities_to_table("/subjects/bert", include_suffix=True)
        assert df.to_dict("records") == [{"Participant": "bert", "Type": ""}]
        with pytest.raises(TypeError):
            cltbids.entities_to_table(None)


####################################################################################################
# Section 2: datasets on disk
####################################################################################################
class TestDatasetQueries:
    def test_simulated_dataset(self, bids_root):
        assert (
            nifti_files(bids_root)[0]
            == "derivatives/fmriprep/sub-01/ses-01/anat/sub-01_ses-01_T1w.nii.gz"
        )
        assert (bids_root / "dataset_description.json").is_file()

    def test_get_subjects(self, bids_root):
        (bids_root / "code").mkdir()
        assert cltbids.get_subjects(str(bids_root)) == ["sub-01", "sub-02"]

    def test_get_all_entities(self, bids_root):
        entities, suffixes = cltbids.get_all_entities(str(bids_root))
        assert entities == {"sub": "Participant", "ses": "Session"}
        assert suffixes == ["T1w", "dwi"]
        with pytest.raises(ValueError):
            cltbids.get_all_entities(str(bids_root / "missing"))

    def test_get_derivatives_folders(self, bids_root):
        (bids_root / "derivatives" / "empty_pipeline").mkdir()
        (bids_root / "derivatives" / ".hidden").mkdir()
        assert cltbids.get_derivatives_folders(str(bids_root / "derivatives")) == [
            "fmriprep"
        ]
        with pytest.raises(ValueError):
            cltbids.get_derivatives_folders(str(bids_root / "missing"))

    def test_get_individual_files_and_folders(self, bids_root):
        deriv = str(bids_root / "derivatives" / "fmriprep")
        files = cltbids.get_individual_files_and_folders(deriv, "sub-01")
        assert files and all("sub-01" in os.path.basename(f) for f in files)
        ses = cltbids.get_individual_files_and_folders(
            deriv, {"sub": "01", "ses": "02"}
        )
        assert ses and all("ses-02" in os.path.basename(f) for f in ses)
        assert cltbids.get_individual_files_and_folders(deriv, "sub-09") == []
        with pytest.raises(ValueError):
            cltbids.get_individual_files_and_folders(
                str(bids_root / "missing"), "sub-01"
            )
        with pytest.raises(TypeError):
            cltbids.get_individual_files_and_folders(Path(deriv), "sub-01")

    def test_get_individual_files_and_folders_bids_string(self, bids_root):
        deriv = str(bids_root / "derivatives" / "fmriprep")
        files = cltbids.get_individual_files_and_folders(deriv, "sub-01_ses-02")
        assert files and all("sub-01_ses-02" in os.path.basename(f) for f in files)

    def test_validate_bids_structure(self, bids_root, tmp_path):
        assert cltbids.validate_bids_structure(str(bids_root)) == [
            "Derivatives directory found"
        ]
        assert cltbids.validate_bids_structure(str(tmp_path)) == [
            "Missing required file: dataset_description.json",
            "No subject directories found (sub-*)",
        ]
        with pytest.raises(FileNotFoundError):
            cltbids.validate_bids_structure(str(tmp_path / "missing"))
        with pytest.raises(NotADirectoryError):
            cltbids.validate_bids_structure(str(bids_root / "README"))

    def test_load_bids_json(self, tmp_path):
        assert "bids_entities" in cltbids.load_bids_json()
        with pytest.raises(ValueError):
            cltbids.load_bids_json(str(tmp_path / "missing.json"))
        bad = tmp_path / "bad.json"
        bad.write_text("{")
        with pytest.raises(ValueError):
            cltbids.load_bids_json(str(bad))


class TestTrees:
    def test_generate_bids_tree(self, bids_root, tmp_path):
        tree = cltbids.generate_bids_tree(str(bids_root), max_depth=2)
        lines = tree.splitlines()
        assert lines[0] == "ds/"
        # Subjects first, then other folders, then files
        assert lines[1] == "├── sub-01/" and "├── derivatives/" in lines
        assert lines.index("├── derivatives/") < lines.index(
            "├── dataset_description.json"
        )
        assert "│   ├── ses-01/" in lines and not any("anat" in line for line in lines)
        out = tmp_path / "tree.txt"
        cltbids.generate_bids_tree(str(bids_root), save_to_file=str(out))
        assert "sub-01_ses-01_T1w.nii.gz" in out.read_text(encoding="utf-8")

    def test_generate_bids_tree_hidden_and_excluded(self, bids_root):
        (bids_root / ".git").mkdir()
        (bids_root / ".bidsignore").write_text("")
        assert ".bidsignore" not in cltbids.generate_bids_tree(
            str(bids_root), max_depth=1
        )
        tree = cltbids.generate_bids_tree(str(bids_root), max_depth=1, show_hidden=True)
        assert (
            ".bidsignore" in tree and ".git" not in tree
        )  # .git is excluded by default
        tree = cltbids.generate_bids_tree(
            str(bids_root), max_depth=1, exclude_patterns={"README"}
        )
        assert "README" not in tree

    def test_generate_bids_tree_errors(self, bids_root):
        with pytest.raises(FileNotFoundError):
            cltbids.generate_bids_tree(str(bids_root / "missing"))
        with pytest.raises(NotADirectoryError):
            cltbids.generate_bids_tree(str(bids_root / "README"))

    def test_generate_bids_tree_with_stats(self, bids_root):
        text = cltbids.generate_bids_tree_with_stats(str(bids_root), max_depth=1)
        n_files = sum(len(f) for _, _, f in os.walk(bids_root))
        n_dirs = sum(len(d) for _, d, _ in os.walk(bids_root))
        assert text.endswith(f"├── Directories: {n_dirs}\n└── Files: {n_files}")


class TestDatabaseTable:
    def test_get_bids_database_table(self, bids_root, tmp_path):
        out = tmp_path / "table.csv"
        df = cltbids.get_bids_database_table(
            str(bids_root), output_table=str(out), n_jobs=2
        )
        assert df.columns.tolist() == ["Participant", "Session", "suffix", "N"]
        assert len(df) == 8  # 2 subjects x 2 sessions x 2 image types
        assert out.is_file() and len(pd.read_csv(out)) == 8
        nii = cltbids.get_bids_database_table(
            str(bids_root), valid_extensions=".nii.gz"
        )
        assert nii["N"].tolist() == [1] * 8

    def test_counts_only_images_by_default(self, bids_root):
        # Sidecars (.json, .bvec, .bval) and other files are not counted
        anat = bids_root / "sub-01" / "ses-01" / "anat"
        (anat / "sub-01_ses-01_T1w.png").write_text("x")
        df = cltbids.get_bids_database_table(str(bids_root))
        assert df["N"].tolist() == [1] * 8
        with_json = cltbids.get_bids_database_table(
            str(bids_root), valid_extensions=[".nii.gz", ".json"]
        )
        assert with_json["N"].tolist() == [2] * 8

    def test_get_bids_database_table_errors(self, bids_root, tmp_path):
        with pytest.raises(FileNotFoundError):
            cltbids.get_bids_database_table(str(tmp_path / "missing"))
        with pytest.raises(NotADirectoryError):
            cltbids.get_bids_database_table(str(bids_root / "README"))
        (tmp_path / "empty").mkdir()
        with pytest.raises(ValueError, match="No subjects"):
            cltbids.get_bids_database_table(str(tmp_path / "empty"))


class TestCopyBidsFolder:
    def test_copy_subject(self, bids_root, tmp_path):
        out = tmp_path / "out"
        out.mkdir()
        cltbids.copy_bids_folder(str(bids_root), str(out), subjects_to_copy="01")
        assert nifti_files(out) == [
            "sub-01/ses-01/anat/sub-01_ses-01_T1w.nii.gz",
            "sub-01/ses-01/dwi/sub-01_ses-01_dwi.nii.gz",
            "sub-01/ses-02/anat/sub-01_ses-02_T1w.nii.gz",
            "sub-01/ses-02/dwi/sub-01_ses-02_dwi.nii.gz",
        ]
        assert (out / "sub-01" / "ses-01" / "dwi" / "sub-01_ses-01_dwi.bval").is_file()

    def test_copy_session_folder_and_derivatives(self, bids_root, tmp_path):
        out = tmp_path / "out"
        out.mkdir()
        cltbids.copy_bids_folder(
            str(bids_root),
            str(out),
            subjects_to_copy=["sub-02_ses-01"],
            folders_to_copy="anat",
            include_derivatives="fmriprep",
        )
        assert nifti_files(out) == [
            "derivatives/fmriprep/sub-02/ses-01/anat/sub-02_ses-01_T1w.nii.gz",
            "derivatives/fmriprep/sub-02/ses-01/dwi/sub-02_ses-01_dwi.nii.gz",
            "sub-02/ses-01/anat/sub-02_ses-01_T1w.nii.gz",
        ]

    def test_copy_all(self, bids_root, tmp_path):
        out = tmp_path / "out"
        out.mkdir()
        cltbids.copy_bids_folder(str(bids_root), str(out), include_derivatives="all")
        assert nifti_files(out) == nifti_files(bids_root)

    def test_copy_errors(self, bids_root, tmp_path, capsys):
        with pytest.raises(FileNotFoundError):
            cltbids.copy_bids_folder(str(tmp_path / "missing"), str(tmp_path))
        with pytest.raises(FileNotFoundError):
            cltbids.copy_bids_folder(str(bids_root), str(tmp_path / "missing"))
        with pytest.raises(FileNotFoundError):
            cltbids.copy_bids_folder(
                str(bids_root), str(tmp_path), deriv_dir=str(tmp_path / "x")
            )
        out = tmp_path / "out"
        out.mkdir()
        cltbids.copy_bids_folder(
            str(bids_root), str(out), "01", include_derivatives="nope"
        )
        assert "No derivatives folders" in capsys.readouterr().out


####################################################################################################
# Recursive renaming inside a dataset
####################################################################################################
class TestRecursiveRenaming:
    def test_replace_entity_value(self, nested_bids):
        root = nested_bids()
        cltbids.recursively_replace_entity_value(
            str(root), {"ses": "01"}, {"ses": "baseline"}
        )
        assert nifti_files(root) == [
            "sub-01/ses-02/anat/sub-01_ses-02_acq-mprage_T1w.nii.gz",
            "sub-01/ses-baseline/anat/sub-01_ses-baseline_acq-mprage_T1w.nii.gz",
        ]

    def test_replace_entity_value_with_strings(self, nested_bids):
        root = nested_bids()
        cltbids.recursively_replace_entity_value(str(root), "acq-mprage", "acq-spgr")
        assert all("acq-spgr" in f for f in nifti_files(root))

    def test_replace_entity_key(self, nested_bids):
        root = nested_bids()
        cltbids.recursively_replace_entity_key(str(root), {"acq": "rec"})
        assert nifti_files(root) == [
            "sub-01/ses-01/anat/sub-01_ses-01_rec-mprage_T1w.nii.gz",
            "sub-01/ses-02/anat/sub-01_ses-02_rec-mprage_T1w.nii.gz",
        ]

    def test_replace_entity_key_keeps_values(self, nested_bids):
        root = nested_bids()
        for r, _, files in os.walk(root):
            for f in files:
                new = f.replace("_acq-mprage", "_task-rerun_run-01")
                os.rename(os.path.join(r, f), os.path.join(r, new))
        cltbids.recursively_replace_entity_key(str(root), {"run": "echo"})
        assert (
            nifti_files(root)[0]
            == "sub-01/ses-01/anat/sub-01_ses-01_task-rerun_echo-01_T1w.nii.gz"
        )

    def test_delete_entity(self, nested_bids):
        root = nested_bids()
        cltbids.recursively_delete_entity(str(root), "acq")
        assert nifti_files(root) == [
            "sub-01/ses-01/anat/sub-01_ses-01_T1w.nii.gz",
            "sub-01/ses-02/anat/sub-01_ses-02_T1w.nii.gz",
        ]

    def test_delete_entity_keeps_folder_names(self, nested_bids):
        root = nested_bids()
        cltbids.recursively_delete_entity(str(root), "ses")
        assert nifti_files(root) == [
            "sub-01/ses-01/anat/sub-01_acq-mprage_T1w.nii.gz",
            "sub-01/ses-02/anat/sub-01_acq-mprage_T1w.nii.gz",
        ]

    def test_insert_entity(self, nested_bids):
        root = nested_bids()
        cltbids.recursively_insert_entity(str(root), {"rec": "norm"}, prev_entity="ses")
        assert nifti_files(root) == [
            "sub-01/ses-01/anat/sub-01_ses-01_rec-norm_acq-mprage_T1w.nii.gz",
            "sub-01/ses-02/anat/sub-01_ses-02_rec-norm_acq-mprage_T1w.nii.gz",
        ]

    @pytest.mark.parametrize(
        "rename",
        [
            lambda root: cltbids.recursively_replace_entity_value(
                root, {"ses": "01"}, {"ses": "baseline"}
            ),
            lambda root: cltbids.recursively_replace_entity_key(root, {"acq": "rec"}),
            lambda root: cltbids.recursively_delete_entity(root, "acq"),
            lambda root: cltbids.recursively_delete_entity(root, {"acq": "mprage"}),
        ],
        ids=["replace_value", "replace_key", "delete", "delete_by_value"],
    )
    def test_nothing_outside_the_dataset_is_renamed(
        self, nested_bids, tmp_path, rename
    ):
        root = nested_bids("ses-01_acq-run_project")
        rename(str(root))
        assert sorted(os.listdir(tmp_path)) == ["ses-01_acq-run_project"]
        assert (tmp_path / "ses-01_acq-run_project" / "ds").is_dir()

    def test_errors_and_no_matches(self, nested_bids, tmp_path, capsys):
        root = str(nested_bids())
        for func, args in [
            (cltbids.recursively_replace_entity_value, ({"ses": "1"}, {"ses": "2"})),
            (cltbids.recursively_replace_entity_key, ({"acq": "rec"},)),
            (cltbids.recursively_delete_entity, ("acq",)),
            (cltbids.recursively_insert_entity, ({"rec": "x"},)),
        ]:
            with pytest.raises(ValueError):
                func(str(tmp_path / "missing"), *args)
        cltbids.recursively_replace_entity_value(root, {"ses": "09"}, {"ses": "10"})
        assert "No files found" in capsys.readouterr().out
