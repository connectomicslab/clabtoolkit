"""Tests for clabtoolkit.pipelinetools."""

import io
import json
import os
import shutil
import sqlite3
from pathlib import Path

import pandas as pd
import pytest
from rich.console import Console
from rich.progress import Progress

import clabtoolkit.bidstools as cltbids
import clabtoolkit.pipelinetools as cltpipe

PIPELINES = ["thalseg-cnn", "chimera", "fsl-first"]
N_SUB = 3
N_SES = 2
ALL_FILES_PER_ID = 2  # each simulated derivative has a .json and a .nii.gz


####################################################################################################
# Fixtures
####################################################################################################
def _simulate(bids_dir: Path, n_subjects=N_SUB, n_visits=N_SES, modalities=("anat",)):
    cltbids.create_a_simulated_bids_dataset(
        output_dir=str(bids_dir),
        n_subjects=n_subjects,
        n_visits=n_visits,
        modalities=list(modalities),
        include_derivatives=list(PIPELINES),
        runs_per_modality=1,
        overwrite=True,
    )
    return bids_dir


@pytest.fixture(scope="session")
def _pristine_bids(tmp_path_factory):
    """Simulated BIDS dataset built once and never modified by the tests."""
    return _simulate(tmp_path_factory.mktemp("pristine") / "bids")


@pytest.fixture
def bids_dir(_pristine_bids, tmp_path):
    """A private, modifiable copy of the simulated BIDS dataset."""
    dst = tmp_path / "bids"
    shutil.copytree(_pristine_bids, dst)
    return dst


@pytest.fixture
def deriv_dir(bids_dir):
    return bids_dir / "derivatives"


@pytest.fixture
def full_ids(bids_dir):
    return cltpipe.get_fullids2process(in_dir=str(bids_dir), source="files")


@pytest.fixture
def quiet_progress():
    """Rich Progress that writes into a buffer instead of the terminal."""
    console = Console(file=io.StringIO(), force_terminal=False)
    with Progress(console=console) as progress:
        yield progress


def _remove_file(deriv_dir: Path, pipe: str, sub: str, ses: str):
    """Remove the NIfTI of one subject/session in one pipeline."""
    os.remove(
        deriv_dir / pipe / f"sub-{sub}" / f"ses-{ses}" / "anat" / f"sub-{sub}_ses-{ses}_T1w.nii.gz"
    )


@pytest.fixture
def broken_deriv_dir(deriv_dir):
    """Derivatives with 2 files missing in chimera and 1 in fsl-first (as in the notebook)."""
    _remove_file(deriv_dir, "chimera", "01", "01")
    _remove_file(deriv_dir, "chimera", "01", "02")
    _remove_file(deriv_dir, "fsl-first", "03", "02")
    return deriv_dir


@pytest.fixture
def status_table(broken_deriv_dir, full_ids):
    df, _ = cltpipe.create_processing_status_table(str(broken_deriv_dir), full_ids)
    return df


####################################################################################################
# Section 1: get_fullids2process
####################################################################################################
class TestGetFullIds2Process:
    def test_invalid_source(self):
        with pytest.raises(ValueError, match="source must be"):
            cltpipe.get_fullids2process(["sub-01"], source="nope")

    @pytest.mark.parametrize("in_dir", [None, "", "/this/does/not/exist"])
    def test_none_requires_valid_in_dir(self, in_dir):
        with pytest.raises(ValueError, match="Valid in_dir"):
            cltpipe.get_fullids2process(None, in_dir=in_dir)

    @pytest.mark.parametrize("bad", [5, 1.5, {"a": 1}, ("sub-01",)])
    def test_invalid_type(self, bad):
        with pytest.raises(ValueError, match="ids must be"):
            cltpipe.get_fullids2process(bad)

    def test_list_is_cleaned_and_sorted(self):
        out = cltpipe.get_fullids2process([" sub-02 ", "sub-01", "", "   "])
        assert out == ["sub-01", "sub-02"]

    def test_list_non_string_items_are_cast(self):
        assert cltpipe.get_fullids2process([2, 1]) == ["1", "2"]

    def test_empty_list(self):
        assert cltpipe.get_fullids2process([]) == []

    def test_comma_separated_string(self):
        out = cltpipe.get_fullids2process(
            "sub-003_ses-V1_acq-mprage, sub-001_ses-V1,sub-002_ses-V1,, "
        )
        assert out == ["sub-001_ses-V1", "sub-002_ses-V1", "sub-003_ses-V1_acq-mprage"]

    def test_single_id(self):
        assert cltpipe.get_fullids2process("sub-001") == ["sub-001"]

    @pytest.mark.parametrize("empty", ["", "   ", "\n"])
    def test_empty_string(self, empty):
        assert cltpipe.get_fullids2process(empty) == []

    def test_text_file(self, tmp_path):
        f = tmp_path / "ids.txt"
        f.write_text("sub-003\n\n  sub-001  \nsub-002\n\n")
        assert cltpipe.get_fullids2process(str(f)) == ["sub-001", "sub-002", "sub-003"]

    def test_discover_folders(self, bids_dir):
        out = cltpipe.get_fullids2process(None, in_dir=str(bids_dir), source="folders")
        assert out == [f"sub-0{i}" for i in range(1, N_SUB + 1)]

    def test_discover_folders_ignores_non_subject_entries(self, tmp_path):
        (tmp_path / "sub-01").mkdir()
        (tmp_path / "derivatives").mkdir()
        (tmp_path / "README").write_text("x")
        assert cltpipe.get_fullids2process(None, in_dir=str(tmp_path)) == ["sub-01"]

    def test_discover_files(self, bids_dir):
        out = cltpipe.get_fullids2process(None, in_dir=str(bids_dir), source="files")
        expected = [
            f"sub-0{s}_ses-0{v}" for s in range(1, N_SUB + 1) for v in range(1, N_SES + 1)
        ]
        assert out == expected

    def test_discover_files_has_no_duplicates_with_several_runs(self, tmp_path):
        bids = _simulate(tmp_path / "bids", n_subjects=2, n_visits=1)
        out = cltpipe.get_fullids2process(None, in_dir=str(bids), source="files")
        assert out == sorted(set(out))

    def test_discover_files_with_filter_without_match(self, bids_dir):
        out = cltpipe.get_fullids2process(
            None, in_dir=str(bids_dir), source="files", or_filter="dwi"
        )
        assert out == []

    def test_discover_files_with_filter(self, tmp_path):
        bids = _simulate(tmp_path / "bids", n_subjects=2, n_visits=1, modalities=("anat", "dwi"))
        out = cltpipe.get_fullids2process(
            None, in_dir=str(bids), source="files", or_filter="dwi"
        )
        # a single visit gives IDs without the session entity
        assert out == ["sub-01", "sub-02"]


####################################################################################################
# Section 2: small private helpers
####################################################################################################
class TestLoadFullIds:
    def test_list(self):
        assert cltpipe._load_full_ids([" a ", "", "b"]) == ["a", "b"]

    def test_list_keeps_order(self):
        assert cltpipe._load_full_ids(["b", "a"]) == ["b", "a"]

    def test_file(self, tmp_path):
        f = tmp_path / "ids.txt"
        f.write_text("sub-01\n\nsub-02 \n")
        assert cltpipe._load_full_ids(str(f)) == ["sub-01", "sub-02"]

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            cltpipe._load_full_ids(str(tmp_path / "nope.txt"))

    def test_wrong_type(self):
        with pytest.raises(TypeError):
            cltpipe._load_full_ids(5)

    @pytest.mark.parametrize("empty", [[], ["", "  "]])
    def test_empty_list(self, empty):
        with pytest.raises(ValueError, match="empty"):
            cltpipe._load_full_ids(empty)

    def test_empty_file(self, tmp_path):
        f = tmp_path / "ids.txt"
        f.write_text("\n\n")
        with pytest.raises(ValueError, match="empty"):
            cltpipe._load_full_ids(str(f))


class TestLoadStatusTable:
    def test_dataframe_is_copied(self):
        df = pd.DataFrame({"a": [1]})
        out = cltpipe._load_status_table(df)
        assert out.equals(df) and out is not df
        out.loc[0, "a"] = 99
        assert df.loc[0, "a"] == 1

    @pytest.mark.parametrize("ext,sep", [(".tsv", "\t"), (".csv", ","), (".txt", ";")])
    def test_read_files_keeps_strings(self, tmp_path, ext, sep):
        f = tmp_path / f"table{ext}"
        pd.DataFrame({"Participant": ["01"], "chimera": [2]}).to_csv(f, sep=sep, index=False)
        out = cltpipe._load_status_table(f)
        assert out.loc[0, "Participant"] == "01"
        assert out.loc[0, "chimera"] == "2"

    def test_accepts_str_and_path(self, tmp_path):
        f = tmp_path / "t.tsv"
        pd.DataFrame({"a": [1]}).to_csv(f, sep="\t", index=False)
        assert cltpipe._load_status_table(str(f)).equals(cltpipe._load_status_table(f))

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            cltpipe._load_status_table(str(tmp_path / "nope.tsv"))

    @pytest.mark.parametrize("bad", [None, 5, ["a"]])
    def test_wrong_type(self, bad):
        with pytest.raises(TypeError):
            cltpipe._load_status_table(bad)


class TestResolvePipeDirs:
    def test_missing_dir(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            cltpipe._resolve_pipe_dirs(str(tmp_path / "nope"), None)

    def test_no_pipelines(self, tmp_path):
        with pytest.raises(ValueError, match="No derivatives folders"):
            cltpipe._resolve_pipe_dirs(str(tmp_path), None)

    def test_all_pipelines(self, deriv_dir):
        _, all_pipes, selected = cltpipe._resolve_pipe_dirs(str(deriv_dir), None)
        assert set(all_pipes) == set(PIPELINES)
        assert selected == all_pipes
        assert selected is not all_pipes

    def test_trailing_separator_is_removed(self, deriv_dir):
        out_dir, _, _ = cltpipe._resolve_pipe_dirs(str(deriv_dir) + os.sep, None)
        assert out_dir == str(deriv_dir)

    def test_single_string(self, deriv_dir):
        _, _, selected = cltpipe._resolve_pipe_dirs(str(deriv_dir), "chimera")
        assert selected == ["chimera"]

    def test_list_exact(self, deriv_dir):
        _, _, selected = cltpipe._resolve_pipe_dirs(
            str(deriv_dir), ["chimera", "fsl-first"]
        )
        assert sorted(selected) == ["chimera", "fsl-first"]

    def test_substring_fallback(self, deriv_dir):
        _, _, selected = cltpipe._resolve_pipe_dirs(str(deriv_dir), ["thalseg"])
        assert selected == ["thalseg-cnn"]

    def test_exact_and_substring_without_duplicates(self, deriv_dir):
        _, _, selected = cltpipe._resolve_pipe_dirs(
            str(deriv_dir), ["chimera", "chim"]
        )
        assert selected == ["chimera"]

    def test_unknown_pipeline(self, deriv_dir):
        with pytest.raises(ValueError, match="None of the requested"):
            cltpipe._resolve_pipe_dirs(str(deriv_dir), ["does-not-exist"])


class TestRowEntities:
    ENT_LIST = {"sub": "Participant", "ses": "Session", "acq": "Acquisition"}

    def test_builds_ordered_dict(self):
        row = pd.Series({"Participant": "01", "Session": "02", "Acquisition": "mp"})
        out = cltpipe._row_entities(row, self.ENT_LIST, list(row.index))
        assert out == {"sub": "sub-01", "ses": "ses-02", "acq": "acq-mp"}
        assert list(out) == ["sub", "ses", "acq"]

    def test_skips_nan_and_empty(self):
        row = pd.Series({"Participant": "01", "Session": float("nan"), "Acquisition": "  "})
        out = cltpipe._row_entities(row, self.ENT_LIST, list(row.index))
        assert out == {"sub": "sub-01"}

    def test_skips_columns_not_in_table(self):
        row = pd.Series({"Participant": "01"})
        out = cltpipe._row_entities(row, self.ENT_LIST, ["Participant"])
        assert out == {"sub": "sub-01"}

    def test_strips_values(self):
        row = pd.Series({"Participant": " 01 "})
        assert cltpipe._row_entities(row, self.ENT_LIST, ["Participant"]) == {
            "sub": "sub-01"
        }


class TestFindFullId:
    SUBJ = ["sub-010_ses-01", "sub-01_ses-01_acq-a", "sub-01_ses-01", "sub-01_ses-02"]

    def test_exact_match_preferred_over_partial(self):
        assert (
            cltpipe._find_full_id(self.SUBJ, ["sub-01", "ses-01"], "fb") == "sub-01_ses-01"
        )

    def test_whole_token_matching(self):
        # 'sub-01' must not match 'sub-010'
        assert cltpipe._find_full_id(["sub-010"], ["sub-01"], "fb") == "fb"

    def test_partial_match_when_no_exact(self):
        assert (
            cltpipe._find_full_id(["sub-01_ses-01_acq-a"], ["sub-01", "ses-01"], "fb")
            == "sub-01_ses-01_acq-a"
        )

    def test_order_of_entities_is_irrelevant(self):
        assert (
            cltpipe._find_full_id(self.SUBJ, ["ses-02", "sub-01"], "fb") == "sub-01_ses-02"
        )

    def test_empty_entities_returns_fallback(self):
        assert cltpipe._find_full_id(self.SUBJ, [], "fb") == "fb"

    def test_no_match_returns_fallback(self):
        assert cltpipe._find_full_id(self.SUBJ, ["sub-99"], "fb") == "fb"


class TestFinalizeSummary:
    SUMMARY = {
        "pipe": {
            "ref_fullid": "sub-01",
            "missmatch_fullid": {
                "sub-02": {"missing_files": ["a"], "extra_files": []},
            },
        }
    }

    def test_full_summary_is_unchanged(self):
        assert cltpipe._finalize_summary(self.SUMMARY, False, None) == self.SUMMARY

    def test_only_ids(self):
        out = cltpipe._finalize_summary(self.SUMMARY, True, None)
        assert out == {"pipe": {"ref_fullid": "sub-01", "missmatch_fullid": ["sub-02"]}}

    def test_json_is_saved(self, tmp_path):
        out_json = tmp_path / "out.json"
        cltpipe._finalize_summary(self.SUMMARY, False, str(out_json))
        assert json.loads(out_json.read_text()) == self.SUMMARY

    def test_json_dir_must_exist(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            cltpipe._finalize_summary(self.SUMMARY, False, str(tmp_path / "x" / "o.json"))


####################################################################################################
# Section 3: file parsing and scanning
####################################################################################################
class TestProcessFile:
    def test_valid_bids_file(self):
        table, failed = cltpipe.process_file(
            "/data/derivatives/chimera/sub-01/ses-01/anat/sub-01_ses-01_T1w.nii.gz"
        )
        assert failed is None
        assert isinstance(table, pd.DataFrame) and not table.empty
        assert table.iloc[0]["Participant"] == "01"
        assert table.iloc[0]["Session"] == "01"
        assert table.iloc[0]["Type"] == "T1w"

    def test_non_bids_name_is_reported_as_failed(self):
        path = "/data/derivatives/chimera/dataset_description.json"
        assert cltpipe.process_file(path) == (None, path)

    def test_path_objects_are_parsed(self):
        # entities_to_table accepts Path objects since a49d160
        table, failed = cltpipe.process_file(Path("/x/sub-01_T1w.nii.gz"))
        assert failed is None
        assert table.iloc[0]["Participant"] == "01"
        assert table.iloc[0]["Type"] == "T1w"

    def test_parser_exception_is_reported_as_failed(self, monkeypatch):
        def boom(*a, **k):
            raise RuntimeError("boom")

        monkeypatch.setattr(cltpipe.cltbids, "entities_to_table", boom)
        assert cltpipe.process_file("sub-01_T1w.nii.gz") == (None, "sub-01_T1w.nii.gz")

    @pytest.mark.parametrize("value", [None, pd.DataFrame()])
    def test_empty_result_is_reported_as_failed(self, monkeypatch, value):
        monkeypatch.setattr(cltpipe.cltbids, "entities_to_table", lambda *a, **k: value)
        assert cltpipe.process_file("sub-01_T1w.nii.gz") == (None, "sub-01_T1w.nii.gz")


class TestProcessFreesurferSubject:
    def test_missing_folder_is_failed_and_not_created(self, tmp_path):
        out = cltpipe.process_freesurfer_subject(("sub-01", str(tmp_path)))
        assert out == (None, "sub-01")
        assert not (tmp_path / "sub-01").exists()

    def test_success_with_fake_subject(self, tmp_path, monkeypatch):
        (tmp_path / "sub-01_ses-01").mkdir()

        class FakeSubject:
            def __init__(self, fs_id, pipe_dir):
                self.fs_files_count = {"mri": 3, "surf": 5}

        monkeypatch.setattr(cltpipe.cltfree, "FreeSurferSubject", FakeSubject)
        table, failed = cltpipe.process_freesurfer_subject(
            ("sub-01_ses-01", str(tmp_path))
        )
        assert failed is None
        assert table["Type"].tolist() == ["mri", "surf"]
        assert table["Count"].tolist() == [3, 5]
        assert "Participant" in table.columns

    def test_exception_is_failed(self, tmp_path, monkeypatch):
        (tmp_path / "sub-01").mkdir()

        def boom(*a, **k):
            raise RuntimeError("boom")

        monkeypatch.setattr(cltpipe.cltfree, "FreeSurferSubject", boom)
        assert cltpipe.process_freesurfer_subject(("sub-01", str(tmp_path))) == (
            None,
            "sub-01",
        )


class TestScanDerivatives:
    def test_scan_all_subjects(self, deriv_dir):
        files = cltpipe.scan_derivatives(str(deriv_dir / "chimera"))
        assert len(files) == N_SUB * N_SES
        assert all(f.endswith(".nii.gz") for f in files)
        assert files == sorted(files)

    def test_single_id_as_string(self, deriv_dir):
        files = cltpipe.scan_derivatives(
            str(deriv_dir / "chimera"), full_ids="sub-01_ses-01"
        )
        assert [os.path.basename(f) for f in files] == ["sub-01_ses-01_T1w.nii.gz"]

    def test_list_of_ids(self, deriv_dir):
        files = cltpipe.scan_derivatives(
            str(deriv_dir / "chimera"), full_ids=["sub-01_ses-01", "sub-02_ses-02"]
        )
        assert len(files) == 2

    def test_unknown_id_gives_nothing(self, deriv_dir):
        assert cltpipe.scan_derivatives(str(deriv_dir / "chimera"), ["sub-99"]) == []

    def test_custom_extensions(self, deriv_dir):
        files = cltpipe.scan_derivatives(
            str(deriv_dir / "chimera"), extensions=[".json"]
        )
        assert len(files) == N_SUB * N_SES
        assert all(f.endswith(".json") for f in files)

    def test_extension_without_matches(self, deriv_dir):
        assert cltpipe.scan_derivatives(str(deriv_dir / "chimera"), extensions=[".mgz"]) == []


class TestRunParallel:
    def test_collects_results_and_failures(self, quiet_progress):
        task = quiet_progress.add_task("t", total=4)

        def work(x):
            return (x * 2, None) if x % 2 == 0 else (None, x)

        results, failed = cltpipe._run_parallel([1, 2, 3, 4], work, quiet_progress, task, 2)
        assert sorted(results) == [4, 8]
        assert sorted(failed) == [1, 3]
        assert quiet_progress.tasks[0].completed == 4

    def test_exceptions_are_counted_as_failed(self, quiet_progress, capsys):
        task = quiet_progress.add_task("t", total=2)

        def work(x):
            if x == 1:
                raise RuntimeError("boom")
            return x, None

        results, failed = cltpipe._run_parallel([1, 2], work, quiet_progress, task, 2)
        assert results == [2]
        assert failed == [1]
        assert "boom" in capsys.readouterr().out

    def test_empty_items(self, quiet_progress):
        task = quiet_progress.add_task("t", total=0)
        assert cltpipe._run_parallel([], lambda x: (x, None), quiet_progress, task, 2) == (
            [],
            [],
        )


####################################################################################################
# Section 4: inventories
####################################################################################################
class TestBuildInventory:
    def test_missing_pipeline_returns_empty(self, deriv_dir, quiet_progress):
        df = cltpipe.build_inventory(
            str(deriv_dir), "nope", 1, 1, quiet_progress
        )
        assert isinstance(df, pd.DataFrame) and df.empty

    def test_bids_path(self, deriv_dir, quiet_progress):
        df = cltpipe.build_inventory(str(deriv_dir), "chimera", 1, 3, quiet_progress)
        assert len(df) == N_SUB * N_SES
        assert (df["Count"] == 1).all()
        assert set(df["Type"]) == {"T1w"}
        # entity columns come first, bookkeeping columns after
        cols = list(df.columns)
        assert cols.index("Participant") < cols.index("Session") < cols.index("Count")
        # sorted by participant/session
        assert df[["Participant", "Session"]].apply(tuple, axis=1).is_monotonic_increasing

    def test_restricted_to_ids(self, deriv_dir, quiet_progress):
        df = cltpipe.build_inventory(
            str(deriv_dir), "chimera", 1, 1, quiet_progress, full_ids=["sub-02_ses-01"]
        )
        assert len(df) == 1
        assert df.iloc[0]["Participant"] == "02" and df.iloc[0]["Session"] == "01"

    def test_custom_extensions(self, deriv_dir, quiet_progress):
        df = cltpipe.build_inventory(
            str(deriv_dir), "chimera", 1, 1, quiet_progress, extensions=[".json"]
        )
        assert len(df) == N_SUB * N_SES

    def test_save_csv_adds_extension(self, deriv_dir, quiet_progress, tmp_path):
        out = tmp_path / "inventory"
        df = cltpipe.build_inventory(
            str(deriv_dir), "chimera", 1, 1, quiet_progress, output_csv=out
        )
        saved = tmp_path / "inventory.csv"
        assert saved.is_file()
        assert len(pd.read_csv(saved)) == len(df)

    def test_save_csv_with_extension(self, deriv_dir, quiet_progress, tmp_path):
        out = tmp_path / "inv.csv"
        cltpipe.build_inventory(
            str(deriv_dir), "chimera", 1, 1, quiet_progress, output_csv=str(out)
        )
        assert out.is_file()

    def test_save_in_missing_dir_is_skipped(self, deriv_dir, quiet_progress, tmp_path):
        out = tmp_path / "nope" / "inv.csv"
        df = cltpipe.build_inventory(
            str(deriv_dir), "chimera", 1, 1, quiet_progress, output_csv=out
        )
        assert not out.exists()
        assert len(df) == N_SUB * N_SES

    def test_unparseable_files_give_empty_table(
        self, tmp_path, quiet_progress, monkeypatch
    ):
        pipe = tmp_path / "pipe" / "sub-01"
        pipe.mkdir(parents=True)
        (pipe / "sub-01_T1w.nii.gz").write_text("x")
        monkeypatch.setattr(cltpipe, "process_file", lambda f: (None, f))
        df = cltpipe.build_inventory(
            str(tmp_path), "pipe", 1, 1, quiet_progress, full_ids=["sub-01"]
        )
        assert df.empty

    def test_freesurfer_fallback(self, tmp_path, quiet_progress, monkeypatch):
        for sub in ("sub-01", "sub-02"):
            (tmp_path / "freesurfer" / sub).mkdir(parents=True)
            # an unmatched extension makes the scan empty -> FreeSurfer fallback
            (tmp_path / "freesurfer" / sub / "note.txt").write_text("x")

        class FakeSubject:
            def __init__(self, fs_id, pipe_dir):
                self.fs_files_count = {"mri": 3, "surf": 5}

        monkeypatch.setattr(cltpipe.cltfree, "FreeSurferSubject", FakeSubject)
        df = cltpipe.build_inventory(str(tmp_path), "freesurfer", 1, 1, quiet_progress)
        assert len(df) == 4
        assert set(df["Type"]) == {"mri", "surf"}
        assert set(df["Participant"]) == {"01", "02"}

    def test_freesurfer_fallback_without_readable_subjects(self, tmp_path, quiet_progress):
        (tmp_path / "freesurfer").mkdir()
        df = cltpipe.build_inventory(str(tmp_path), "freesurfer", 1, 1, quiet_progress)
        assert df.empty


class TestBuildDerivativesInventory:
    def test_all_pipelines(self, deriv_dir):
        df = cltpipe.build_derivatives_inventory(str(deriv_dir))
        assert df.columns[0] == "Pipeline"
        assert set(df["Pipeline"]) == set(PIPELINES)
        assert len(df) == len(PIPELINES) * N_SUB * N_SES

    def test_selected_pipelines(self, deriv_dir):
        df = cltpipe.build_derivatives_inventory(
            str(deriv_dir), pipe_dirs=["chimera", "fsl-first"]
        )
        assert set(df["Pipeline"]) == {"chimera", "fsl-first"}

    def test_restricted_to_ids(self, deriv_dir):
        df = cltpipe.build_derivatives_inventory(
            str(deriv_dir), full_ids=["sub-01_ses-01"]
        )
        assert len(df) == len(PIPELINES)
        assert set(df["Participant"]) == {"01"}

    def test_no_pipelines(self, tmp_path):
        with pytest.raises(ValueError, match="No derivatives folders"):
            cltpipe.build_derivatives_inventory(str(tmp_path))

    def test_empty_pipe_dirs_list(self, deriv_dir):
        with pytest.raises(ValueError, match="No derivatives folders"):
            cltpipe.build_derivatives_inventory(str(deriv_dir), pipe_dirs=[])

    def test_save_table_is_tab_separated(self, deriv_dir, tmp_path):
        out = tmp_path / "inv.tsv"
        df = cltpipe.build_derivatives_inventory(str(deriv_dir), output_table=out)
        saved = pd.read_csv(out, sep="\t", dtype=str)
        assert len(saved) == len(df)
        assert saved.columns[0] == "Pipeline"

    def test_existing_table_requires_overwrite(self, deriv_dir, tmp_path):
        out = tmp_path / "inv.tsv"
        out.write_text("old")
        with pytest.raises(ValueError, match="already exists"):
            cltpipe.build_derivatives_inventory(str(deriv_dir), output_table=out)
        assert out.read_text() == "old"

        cltpipe.build_derivatives_inventory(
            str(deriv_dir), output_table=out, overwrite=True
        )
        assert out.read_text() != "old"

    def test_output_dir_must_exist(self, deriv_dir, tmp_path):
        with pytest.raises(ValueError, match="does not exist"):
            cltpipe.build_derivatives_inventory(
                str(deriv_dir), output_table=tmp_path / "nope" / "inv.tsv"
            )


####################################################################################################
# Section 5: processing status table
####################################################################################################
class TestCreateProcessingStatusTable:
    def test_complete_dataset(self, deriv_dir, full_ids):
        df, path = cltpipe.create_processing_status_table(str(deriv_dir), full_ids)
        assert path is None
        assert len(df) == N_SUB * N_SES
        assert set(PIPELINES) <= set(df.columns)
        assert (df[PIPELINES].astype(int) == ALL_FILES_PER_ID).all().all()
        assert set(df["Participant"]) == {"01", "02", "03"}
        assert set(df["Session"]) == {"01", "02"}

    def test_detects_missing_files(self, broken_deriv_dir, full_ids):
        df, _ = cltpipe.create_processing_status_table(str(broken_deriv_dir), full_ids)
        counts = df.set_index(["Participant", "Session"])
        assert counts.loc[("01", "01"), "chimera"] == 1
        assert counts.loc[("01", "02"), "chimera"] == 1
        assert counts.loc[("03", "02"), "fsl-first"] == 1
        assert counts.loc[("02", "01"), "chimera"] == 2
        assert (counts["thalseg-cnn"].astype(int) == ALL_FILES_PER_ID).all()

    def test_subject_absent_from_pipeline_counts_zero(self, deriv_dir):
        df, _ = cltpipe.create_processing_status_table(
            str(deriv_dir), ["sub-01_ses-01", "sub-99_ses-01"]
        )
        row = df[df["Participant"] == "99"].iloc[0]
        assert all(int(row[p]) == 0 for p in PIPELINES)

    def test_sequential_equals_parallel(self, deriv_dir, full_ids):
        a, _ = cltpipe.create_processing_status_table(str(deriv_dir), full_ids, n_jobs=1)
        b, _ = cltpipe.create_processing_status_table(str(deriv_dir), full_ids, n_jobs=2)
        pd.testing.assert_frame_equal(
            a.sort_values(["Participant", "Session"]).reset_index(drop=True),
            b.sort_values(["Participant", "Session"]).reset_index(drop=True),
        )

    def test_ids_from_text_file(self, deriv_dir, tmp_path):
        f = tmp_path / "ids.txt"
        f.write_text("sub-01_ses-01\n\nsub-02_ses-01\n")
        df, _ = cltpipe.create_processing_status_table(str(deriv_dir), str(f))
        assert len(df) == 2

    def test_save_creates_missing_directory(self, deriv_dir, full_ids, tmp_path):
        out = tmp_path / "new" / "dir" / "status.tsv"
        df, path = cltpipe.create_processing_status_table(
            str(deriv_dir), full_ids, output_table=str(out)
        )
        assert path == str(out)
        saved = pd.read_csv(out, sep="\t", dtype=str)
        assert len(saved) == len(df)
        assert saved["Participant"].tolist() == df["Participant"].tolist()

    def test_trailing_separator_in_deriv_dir(self, deriv_dir, full_ids):
        df, _ = cltpipe.create_processing_status_table(
            str(deriv_dir) + os.sep, full_ids
        )
        assert len(df) == N_SUB * N_SES

    def test_missing_deriv_dir(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            cltpipe.create_processing_status_table(str(tmp_path / "nope"), ["sub-01"])

    def test_no_pipelines(self, tmp_path):
        with pytest.raises(ValueError, match="No derivatives folders"):
            cltpipe.create_processing_status_table(str(tmp_path), ["sub-01"])

    def test_missing_ids_file(self, deriv_dir, tmp_path):
        with pytest.raises(FileNotFoundError):
            cltpipe.create_processing_status_table(
                str(deriv_dir), str(tmp_path / "nope.txt")
            )

    def test_empty_ids(self, deriv_dir):
        with pytest.raises(ValueError, match="empty"):
            cltpipe.create_processing_status_table(str(deriv_dir), [])

    def test_wrong_ids_type(self, deriv_dir):
        with pytest.raises(TypeError):
            cltpipe.create_processing_status_table(str(deriv_dir), 5)


####################################################################################################
# Section 6: processing status details (JSON)
####################################################################################################
class TestProcessingStatusDetailsJson:
    def test_complete_dataset_has_no_mismatches(self, deriv_dir, full_ids):
        table, _ = cltpipe.create_processing_status_table(str(deriv_dir), full_ids)
        summary, out_json = cltpipe.get_processing_status_details_json(
            table, full_ids, str(deriv_dir)
        )
        assert out_json is None
        assert set(summary) == set(PIPELINES)
        for info in summary.values():
            assert info["missmatch_fullid"] == {}
            assert info["ref_fullid"] in full_ids

    def test_only_ids(self, status_table, full_ids, broken_deriv_dir):
        summary, _ = cltpipe.get_processing_status_details_json(
            status_table,
            full_ids,
            str(broken_deriv_dir),
            pipe_dirs=["chimera"],
            only_ids=True,
        )
        assert list(summary) == ["chimera"]
        assert sorted(summary["chimera"]["missmatch_fullid"]) == [
            "sub-01_ses-01",
            "sub-01_ses-02",
        ]

    def test_details_report_missing_files(self, status_table, full_ids, broken_deriv_dir):
        summary, _ = cltpipe.get_processing_status_details_json(
            status_table, full_ids, str(broken_deriv_dir), pipe_dirs=["chimera", "fsl-first"]
        )
        chimera = summary["chimera"]["missmatch_fullid"]
        assert set(chimera) == {"sub-01_ses-01", "sub-01_ses-02"}
        for full_id, details in chimera.items():
            assert details["extra_files"] == []
            assert len(details["missing_files"]) == 1
            assert details["missing_files"][0].endswith(f"{full_id}_T1w.nii.gz")
            assert not os.path.exists(details["missing_files"][0])

        first = summary["fsl-first"]["missmatch_fullid"]
        assert set(first) == {"sub-03_ses-02"}
        assert first["sub-03_ses-02"]["missing_files"][0].endswith(
            "sub-03_ses-02_T1w.nii.gz"
        )

    def test_pipeline_without_problems(self, status_table, full_ids, broken_deriv_dir):
        summary, _ = cltpipe.get_processing_status_details_json(
            status_table, full_ids, str(broken_deriv_dir), pipe_dirs="thalseg-cnn"
        )
        assert summary["thalseg-cnn"]["missmatch_fullid"] == {}

    def test_reference_is_a_complete_subject(self, status_table, full_ids, broken_deriv_dir):
        summary, _ = cltpipe.get_processing_status_details_json(
            status_table, full_ids, str(broken_deriv_dir), pipe_dirs=["chimera"]
        )
        ref = summary["chimera"]["ref_fullid"]
        assert ref not in summary["chimera"]["missmatch_fullid"]

    def test_extra_files_are_reported(self, deriv_dir, full_ids):
        extra = deriv_dir / "chimera" / "sub-02" / "ses-01" / "anat" / "sub-02_ses-01_extra.json"
        extra.write_text("{}")
        table, _ = cltpipe.create_processing_status_table(str(deriv_dir), full_ids)
        summary, _ = cltpipe.get_processing_status_details_json(
            table, full_ids, str(deriv_dir), pipe_dirs=["chimera"]
        )
        details = summary["chimera"]["missmatch_fullid"]["sub-02_ses-01"]
        assert details["missing_files"] == []
        assert details["extra_files"] == [str(extra)]

    def test_subject_without_any_file(self, deriv_dir, full_ids):
        shutil.rmtree(deriv_dir / "chimera" / "sub-02" / "ses-02")
        table, _ = cltpipe.create_processing_status_table(str(deriv_dir), full_ids)
        summary, _ = cltpipe.get_processing_status_details_json(
            table, full_ids, str(deriv_dir), pipe_dirs=["chimera"]
        )
        details = summary["chimera"]["missmatch_fullid"]["sub-02_ses-02"]
        assert len(details["missing_files"]) == ALL_FILES_PER_ID
        assert details["extra_files"] == []
        assert all("sub-02" in f and "ses-02" in f for f in details["missing_files"])

    def test_table_from_file_keeps_leading_zeros(
        self, broken_deriv_dir, full_ids, tmp_path
    ):
        table, _ = cltpipe.create_processing_status_table(str(broken_deriv_dir), full_ids)
        for name in ("status.tsv", "status.csv"):
            path = tmp_path / name
            table.to_csv(path, sep="\t" if name.endswith("tsv") else ",", index=False)
            summary, _ = cltpipe.get_processing_status_details_json(
                str(path), full_ids, str(broken_deriv_dir), pipe_dirs=["chimera"], only_ids=True
            )
            assert sorted(summary["chimera"]["missmatch_fullid"]) == [
                "sub-01_ses-01",
                "sub-01_ses-02",
            ]

    def test_ids_from_file(self, status_table, full_ids, broken_deriv_dir, tmp_path):
        ids_file = tmp_path / "ids.txt"
        ids_file.write_text("\n".join(full_ids))
        summary, _ = cltpipe.get_processing_status_details_json(
            status_table, str(ids_file), str(broken_deriv_dir), pipe_dirs="chimera", only_ids=True
        )
        assert len(summary["chimera"]["missmatch_fullid"]) == 2

    def test_json_output(self, status_table, full_ids, broken_deriv_dir, tmp_path):
        out = tmp_path / "details.json"
        summary, path = cltpipe.get_processing_status_details_json(
            status_table, full_ids, str(broken_deriv_dir), out_json=str(out)
        )
        assert path == str(out)
        assert json.loads(out.read_text()) == summary

    def test_json_output_dir_must_exist(self, status_table, full_ids, broken_deriv_dir, tmp_path):
        with pytest.raises(FileNotFoundError):
            cltpipe.get_processing_status_details_json(
                status_table,
                full_ids,
                str(broken_deriv_dir),
                out_json=str(tmp_path / "nope" / "d.json"),
            )

    def test_input_table_is_not_modified(self, status_table, full_ids, broken_deriv_dir):
        before = status_table.copy()
        cltpipe.get_processing_status_details_json(
            status_table, full_ids, str(broken_deriv_dir)
        )
        pd.testing.assert_frame_equal(status_table, before)

    def test_table_without_pipeline_column(self, status_table, full_ids, broken_deriv_dir):
        with pytest.raises(ValueError, match="no column"):
            cltpipe.get_processing_status_details_json(
                status_table.drop(columns=["chimera"]),
                full_ids,
                str(broken_deriv_dir),
                pipe_dirs=["chimera"],
            )

    def test_unknown_pipeline(self, status_table, full_ids, broken_deriv_dir):
        with pytest.raises(ValueError, match="None of the requested"):
            cltpipe.get_processing_status_details_json(
                status_table, full_ids, str(broken_deriv_dir), pipe_dirs=["nope"]
            )

    def test_missing_table_file(self, full_ids, deriv_dir, tmp_path):
        with pytest.raises(FileNotFoundError):
            cltpipe.get_processing_status_details_json(
                str(tmp_path / "nope.tsv"), full_ids, str(deriv_dir)
            )

    def test_missing_deriv_dir(self, status_table, full_ids, tmp_path):
        with pytest.raises(FileNotFoundError):
            cltpipe.get_processing_status_details_json(
                status_table, full_ids, str(tmp_path / "nope")
            )

    def test_wrong_table_type(self, full_ids, deriv_dir):
        with pytest.raises(TypeError):
            cltpipe.get_processing_status_details_json(5, full_ids, str(deriv_dir))


####################################################################################################
# Section 7: processing status details (SQLite) and queries
####################################################################################################
@pytest.fixture
def status_db(status_table, full_ids, broken_deriv_dir, tmp_path):
    db = tmp_path / "status.db"
    summary, _ = cltpipe.get_processing_status_details_sqlite3(
        status_table,
        full_ids,
        str(broken_deriv_dir),
        pipe_dirs=["chimera", "fsl-first"],
        db_path=str(db),
    )
    return db, summary


class TestProcessingStatusDetailsSqlite3:
    def test_same_summary_as_json(self, status_table, full_ids, broken_deriv_dir):
        a, _ = cltpipe.get_processing_status_details_json(
            status_table, full_ids, str(broken_deriv_dir)
        )
        b, _ = cltpipe.get_processing_status_details_sqlite3(
            status_table, full_ids, str(broken_deriv_dir)
        )
        assert a == b

    def test_no_db_by_default(self, status_table, full_ids, broken_deriv_dir, tmp_path):
        cltpipe.get_processing_status_details_sqlite3(
            status_table, full_ids, str(broken_deriv_dir)
        )
        assert list(tmp_path.glob("*.db")) == []

    def test_db_schema_and_content(self, status_db):
        db, _ = status_db
        assert db.is_file()
        conn = sqlite3.connect(db)
        try:
            tables = {
                r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            }
            assert {"pipelines", "mismatches", "file_details"} <= tables
            assert {r[0] for r in conn.execute("SELECT pipeline_id FROM pipelines")} == {
                "chimera",
                "fsl-first",
            }
            assert conn.execute("SELECT COUNT(*) FROM mismatches").fetchone()[0] == 3
            statuses = {r[0] for r in conn.execute("SELECT DISTINCT status FROM file_details")}
            assert statuses == {"missing"}
        finally:
            conn.close()

    def test_db_receives_full_details_when_only_ids(
        self, status_table, full_ids, broken_deriv_dir, tmp_path
    ):
        db = tmp_path / "ids.db"
        summary, _ = cltpipe.get_processing_status_details_sqlite3(
            status_table, full_ids, str(broken_deriv_dir), only_ids=True, db_path=str(db)
        )
        assert isinstance(summary["chimera"]["missmatch_fullid"], list)
        conn = sqlite3.connect(db)
        try:
            assert conn.execute("SELECT COUNT(*) FROM file_details").fetchone()[0] == 3
        finally:
            conn.close()

    def test_existing_db_is_replaced(self, status_db, status_table, full_ids, broken_deriv_dir):
        db, _ = status_db
        cltpipe.get_processing_status_details_sqlite3(
            status_table,
            full_ids,
            str(broken_deriv_dir),
            pipe_dirs=["thalseg-cnn"],
            db_path=str(db),
        )
        conn = sqlite3.connect(db)
        try:
            assert [r[0] for r in conn.execute("SELECT pipeline_id FROM pipelines")] == [
                "thalseg-cnn"
            ]
            assert conn.execute("SELECT COUNT(*) FROM mismatches").fetchone()[0] == 0
        finally:
            conn.close()

    def test_db_dir_must_exist(self, status_table, full_ids, broken_deriv_dir, tmp_path):
        with pytest.raises(FileNotFoundError):
            cltpipe.get_processing_status_details_sqlite3(
                status_table,
                full_ids,
                str(broken_deriv_dir),
                db_path=str(tmp_path / "nope" / "x.db"),
            )

    def test_json_output(self, status_table, full_ids, broken_deriv_dir, tmp_path):
        out = tmp_path / "d.json"
        summary, path = cltpipe.get_processing_status_details_sqlite3(
            status_table, full_ids, str(broken_deriv_dir), out_json=str(out)
        )
        assert path == str(out)
        assert json.loads(out.read_text()) == summary


class TestQueryProcessingStatusDb:
    def test_invalid_query(self, status_db):
        db, _ = status_db
        with pytest.raises(ValueError, match="Unknown query type"):
            cltpipe.query_processing_status_db(str(db), "nope")

    def test_missing_db(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            cltpipe.query_processing_status_db(str(tmp_path / "nope.db"))

    def test_subjects_with_mismatches(self, status_db):
        db, _ = status_db
        df = cltpipe.query_processing_status_db(str(db), "subjects_with_mismatches")
        assert df["subject_fullid"].tolist() == [
            "sub-01_ses-01",
            "sub-01_ses-02",
            "sub-03_ses-02",
        ]
        assert df.columns.tolist() == ["subject_fullid", "pipelines"]

    def test_subjects_with_mismatches_for_pipeline(self, status_db):
        db, _ = status_db
        df = cltpipe.query_processing_status_db(
            str(db), "subjects_with_mismatches", pipeline="fsl-first"
        )
        assert df["subject_fullid"].tolist() == ["sub-03_ses-02"]
        assert set(df["pipeline_id"]) == {"fsl-first"}

    def test_pipelines_with_mismatches(self, status_db):
        db, _ = status_db
        df = cltpipe.query_processing_status_db(str(db), "pipelines_with_mismatches")
        assert dict(zip(df["pipeline_id"], df["subject_count"])) == {
            "chimera": 2,
            "fsl-first": 1,
        }
        assert df.iloc[0]["pipeline_id"] == "chimera"  # ordered by count, descending

    def test_missing_files_count(self, status_db):
        db, _ = status_db
        df = cltpipe.query_processing_status_db(str(db), "missing_files_count")
        assert len(df) == 3
        assert (df["missing_count"] == 1).all()
        assert {"subject_fullid", "pipeline_id", "missing_count"} <= set(df.columns)

    def test_missing_files_count_for_pipeline(self, status_db):
        db, _ = status_db
        df = cltpipe.query_processing_status_db(
            str(db), "missing_files_count", pipeline="chimera"
        )
        assert sorted(df["subject_fullid"]) == ["sub-01_ses-01", "sub-01_ses-02"]
        assert "pipeline_id" not in df.columns

    def test_pipeline_without_mismatches(self, status_db):
        db, _ = status_db
        df = cltpipe.query_processing_status_db(
            str(db), "missing_files_count", pipeline="thalseg-cnn"
        )
        assert df.empty

    def test_extra_files_count(self, deriv_dir, full_ids, tmp_path):
        (deriv_dir / "chimera" / "sub-02" / "ses-01" / "anat" / "sub-02_ses-01_extra.json").write_text(
            "{}"
        )
        table, _ = cltpipe.create_processing_status_table(str(deriv_dir), full_ids)
        db = tmp_path / "extra.db"
        cltpipe.get_processing_status_details_sqlite3(
            table, full_ids, str(deriv_dir), pipe_dirs=["chimera"], db_path=str(db)
        )
        df = cltpipe.query_processing_status_db(str(db), "extra_files_count")
        assert df["subject_fullid"].tolist() == ["sub-02_ses-01"]
        assert df["extra_count"].tolist() == [1]

        df = cltpipe.query_processing_status_db(
            str(db), "extra_files_count", pipeline="chimera"
        )
        assert df["extra_count"].tolist() == [1]

        # no missing files in this database
        assert cltpipe.query_processing_status_db(str(db), "missing_files_count").empty


class TestExportDbToJson:
    def test_roundtrip_matches_summary(self, status_db, tmp_path):
        db, summary = status_db
        out = tmp_path / "export.json"
        exported = cltpipe.export_db_to_json(str(db), str(out))
        assert json.loads(out.read_text()) == exported

        assert set(exported) == set(summary)
        for pipe, info in summary.items():
            assert exported[pipe]["ref_fullid"] == info["ref_fullid"]
            assert set(exported[pipe]["missmatch_fullid"]) == set(info["missmatch_fullid"])
            for full_id, details in info["missmatch_fullid"].items():
                got = exported[pipe]["missmatch_fullid"][full_id]
                assert sorted(got["missing_files"]) == sorted(details["missing_files"])
                assert sorted(got["extra_files"]) == sorted(details["extra_files"])

    def test_missing_db(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            cltpipe.export_db_to_json(str(tmp_path / "nope.db"), str(tmp_path / "o.json"))
