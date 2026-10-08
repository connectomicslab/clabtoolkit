"""Tests for clabtoolkit.misctools."""

import argparse
import inspect
import json
import os
import re
import types

import h5py
import numpy as np
import pandas as pd
import pytest

import clabtoolkit.misctools as cltmisc

ANSI = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi(text: str) -> str:
    return ANSI.sub("", text)


####################################################################################################
# Section 1: indices, conditions and searches in lists
####################################################################################################
class TestBuildIndices:
    def test_docstring_example(self):
        vec = [1, (2, 5), [6, 7], np.array([0, 0, 0]), "8-10", "11:13", "14:2:22",
               "1, 2, 4:10, 16-20, 25, 0"]
        expected = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 22, 25]
        assert cltmisc.build_indices(vec) == expected
        assert cltmisc.build_indices(vec, nonzeros=False) == [0] + expected

    @pytest.mark.parametrize(
        "expr, expected",
        [
            ("5", [5]),
            ("8-10", [8, 9, 10]),
            ("11:13", [11, 12, 13]),
            ("14:2:22", [14, 16, 18, 20, 22]),
            ("1, 2, 3", [1, 2, 3]),
        ],
    )
    def test_string_formats(self, expr, expected):
        assert cltmisc.build_indices([expr]) == expected

    def test_scalar_inputs_are_wrapped(self):
        assert cltmisc.build_indices(7) == [7]
        assert cltmisc.build_indices(np.int64(3)) == [3]
        assert cltmisc.build_indices("2-4") == [2, 3, 4]
        assert cltmisc.build_indices(np.array(5)) == [5]

    def test_tuples(self):
        assert cltmisc.build_indices([(2, 5)]) == [2, 3, 4, 5]          # 2 items: range
        assert cltmisc.build_indices([(9, 1, 4)]) == [1, 4, 9]          # Other: explicit list

    def test_sorted_and_unique(self):
        assert cltmisc.build_indices([5, 3, 5, [3, 1]]) == [1, 3, 5]

    def test_zeros(self):
        assert cltmisc.build_indices([0, 1]) == [1]
        assert cltmisc.build_indices([0, 1], nonzeros=False) == [0, 1]

    @pytest.mark.parametrize("bad", [["a-b"], ["1:2:3:4"], [1.5j], [{"a": 1}]])
    def test_invalid_items(self, bad):
        with pytest.raises(ValueError):
            cltmisc.build_indices(bad)


class TestConditions:
    bvals = np.array([0, 500, 1000, 2000, 3000])

    def test_indices_simple(self):
        idx = cltmisc.get_indices_by_condition("bvals > 1000", bvals=self.bvals)
        assert idx.tolist() == [3, 4]

    def test_indices_chained_with_variables(self):
        idx = cltmisc.get_indices_by_condition(
            "bmin <= bvals <= bmax", bvals=self.bvals, bmin=800, bmax=2500
        )
        assert idx.tolist() == [2, 3]

    def test_indices_chained_with_literals(self):
        idx = cltmisc.get_indices_by_condition("-1 < bvals < 600.5", bvals=self.bvals)
        assert idx.tolist() == [0, 1]

    def test_indices_list_input(self):
        idx = cltmisc.get_indices_by_condition("x != 0", x=[0, 1, 0, 2])
        assert idx.tolist() == [1, 3]

    def test_indices_missing_variable(self):
        with pytest.raises(ValueError, match="Missing variable"):
            cltmisc.get_indices_by_condition("bvals > thr", bvals=self.bvals)

    def test_indices_needs_exactly_one_array(self):
        with pytest.raises(ValueError, match="Exactly one"):
            cltmisc.get_indices_by_condition("a > 1", a=[1, 2], b=[3, 4])
        with pytest.raises(ValueError, match="Exactly one"):
            cltmisc.get_indices_by_condition("a > 1", a=1)

    def test_indices_non_boolean_result(self):
        with pytest.raises(ValueError):
            cltmisc.get_indices_by_condition("bvals + 1", bvals=self.bvals)

    def test_values(self):
        assert cltmisc.get_values_by_condition("bvals > 1000", bvals=self.bvals) == [2000, 3000]
        assert cltmisc.get_values_by_condition(
            "bmin <= bvals <= bmax", bvals=self.bvals, bmin=800, bmax=2500
        ) == [1000, 2000]

    def test_values_list_input_unique_in_order(self):
        assert cltmisc.get_values_by_condition("b > 0", b=[30, 10, 30, 0, 20]) == [30, 10, 20]

    def test_build_indices_with_conditions(self):
        data = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 45])
        assert cltmisc.build_indices_with_conditions(["1:4", "5-7", "8:2:10"], nonzeros=False) == [
            1, 2, 3, 4, 5, 6, 7, 8, 10]
        assert cltmisc.build_indices_with_conditions(["5<=data<=20"], data=data) == [1, 2, 3, 4]
        # Index 0 (where data == 0) is dropped by nonzeros=True
        assert cltmisc.build_indices_with_conditions([0, "2:4", "data == 0", 9], data=data) == [
            2, 3, 4, 9]
        assert cltmisc.build_indices_with_conditions(
            [0, "data == 0", 9], nonzeros=False, data=data) == [0, 9]
        assert cltmisc.build_indices_with_conditions(
            ["data > thr", "1:2", np.array([8])], data=data, thr=35) == [1, 2, 8, 9]

    def test_build_indices_with_conditions_invalid(self):
        with pytest.raises(ValueError):
            cltmisc.build_indices_with_conditions(["data > missing"], data=[1, 2])

    def test_build_values_with_conditions(self):
        bvals = np.array([0, 1000, 1000, 2000, 3000])
        assert cltmisc.build_values_with_conditions(["bvals >= 2000"], bvals=bvals) == [2000, 3000]
        assert cltmisc.build_values_with_conditions([3000, "bvals < 1500"], bvals=bvals,
                                                    nonzeros=False) == [0, 1000, 3000]

    @pytest.mark.parametrize(
        "condition, var, limits",
        [
            ("bmin <= bvals <= bmax", "bvals", ["bmin", "bmax"]),
            ("bvals > 1000", "bvals", ["1000"]),
            ("1000 < bvals <= 2000", "bvals", ["1000", "2000"]),
            ("1000 < bvals", "bvals", ["1000"]),
            ("bvals != bval", "bvals", ["bval"]),
            ("invalid condition", None, []),
        ],
    )
    def test_parse_condition(self, condition, var, limits):
        assert cltmisc.parse_condition(condition) == (var, limits)

    def test_analyze_condition(self):
        info = cltmisc.analyze_condition("bmin <= bvals <= bmax")
        assert info["main_variable"] == "bvals"
        assert info["limit_variables"] == ["bmin", "bmax"]
        assert info["is_chained"] and info["is_valid"]
        assert "<=" in info["operators"]
        assert not cltmisc.analyze_condition("nothing here")["is_valid"]

    @pytest.mark.parametrize("s, expected", [("1", True), ("-2.5", True), ("1e3", True),
                                             ("abc", False), ("", False)])
    def test_is_numeric(self, s, expected):
        assert cltmisc.is_numeric(s) is expected


class TestListSearches:
    fruits = ["apple", "banana", "cherry", "date", "grape"]

    def test_remove_duplicates_keeps_order(self):
        assert cltmisc.remove_duplicates([3, 1, 3, 2, 1]) == [3, 1, 2]
        assert cltmisc.remove_duplicates([]) == []

    def test_select_ids_from_file(self, tmp_path):
        ids_file = tmp_path / "ids.txt"
        ids_file.write_text("sub-01\nsub-03\n")
        subj = ["sub-01", "sub-02", "sub-03"]
        assert cltmisc.select_ids_from_file(subj, str(ids_file)) == ["sub-01", "sub-03"]
        assert cltmisc.select_ids_from_file(subj, ["sub-02"]) == ["sub-02"]
        assert cltmisc.select_ids_from_file(subj, str(tmp_path / "missing.txt")) == []

    def test_indexes_by_substring_or(self):
        assert cltmisc.get_indexes_by_substring(self.fruits, ["app", "ch"]) == [0, 2]

    def test_indexes_by_substring_and(self):
        # Elements containing "e" AND "a": apple, date, grape
        assert cltmisc.get_indexes_by_substring(self.fruits, "e", and_filter="a") == [0, 3, 4]

    def test_indexes_by_substring_invert_and_none(self):
        assert cltmisc.get_indexes_by_substring(self.fruits, ["app", "ch"], invert=True) == [1, 3, 4]
        assert cltmisc.get_indexes_by_substring(self.fruits, None, and_filter="an") == [1]

    def test_indexes_by_substring_case(self):
        items = ["Apple", "apple"]
        assert cltmisc.get_indexes_by_substring(items, "APP") == [0, 1]
        assert cltmisc.get_indexes_by_substring(items, "App", bool_case=True) == [0]

    def test_indexes_by_substring_whole_word(self):
        items = ["app", "apple", "application", "the app"]
        assert cltmisc.get_indexes_by_substring(items, "app", match_entire_word=True) == [0, 3]

    def test_indexes_by_substring_whole_word_case_insensitive(self):
        # bool_case=False (default) must also apply to whole-word matching
        items = ["App", "the APP", "apple"]
        assert cltmisc.get_indexes_by_substring(items, "app", match_entire_word=True) == [0, 1]

    def test_indexes_by_substring_errors(self):
        with pytest.raises(ValueError):
            cltmisc.get_indexes_by_substring(("a", "b"), "a")
        with pytest.raises(ValueError):
            cltmisc.get_indexes_by_substring(["a", 1], "a")

    def test_filter_by_substring(self):
        items = ["apple", "banana", "cherry", "date", "Apple Pie", "apple"]
        assert cltmisc.filter_by_substring(items, ["app", "ch"]) == ["apple", "cherry", "Apple Pie"]
        assert cltmisc.filter_by_substring(items, "app", and_filter="pie") == ["Apple Pie"]
        assert cltmisc.filter_by_substring("sub-01_T1w", "T1w") == ["sub-01_T1w"]

    def test_remove_substrings(self):
        assert cltmisc.remove_substrings(["hello_world", "test_world", "worldwide"], "world") == [
            "hello_", "test_", "wide"]
        assert cltmisc.remove_substrings(["apple_pie", "cherry_pie"], ["pie", "_"]) == ["apple", "cherry"]
        mixed = ["Hello_WORLD", "test_World", "WORLDWIDE"]
        assert cltmisc.remove_substrings(mixed, "world") == ["Hello_", "test_", "WIDE"]
        assert cltmisc.remove_substrings(mixed, "WORLD", bool_case=True) == ["Hello_", "test_World", "WIDE"]
        assert cltmisc.remove_substrings("a.b", ".") == ["ab"]     # Not a regex

    def test_remove_substrings_errors(self):
        with pytest.raises(TypeError):
            cltmisc.remove_substrings([1, 2], "a")
        with pytest.raises(TypeError):
            cltmisc.remove_substrings(["a"], [1])

    def test_replace_substrings(self):
        assert cltmisc.replace_substrings("Hello_World", {"World": "Earth"}) == ["Hello_Earth"]
        assert cltmisc.replace_substrings(["abc123", "ABC123"], {"abc": "xyz", "123": "789"},
                                          bool_case=False) == ["xyz789", "xyz789"]
        assert cltmisc.replace_substrings(["Hello_World", "hello_world"], {"hello": "Hi"}) == [
            "Hello_World", "Hi_world"]
        assert cltmisc.replace_substrings("run-01_run-02", {r"run-\d+": "run-X"}) == ["run-X_run-X"]

    def test_replace_substrings_errors(self):
        with pytest.raises(TypeError):
            cltmisc.replace_substrings([1], {"a": "b"})
        with pytest.raises(TypeError):
            cltmisc.replace_substrings("a", [("a", "b")])
        with pytest.raises(TypeError):
            cltmisc.replace_substrings("a", {"a": 1})
        with pytest.raises(ValueError):
            cltmisc.replace_substrings("a", {})

    @pytest.mark.parametrize(
        "item, expected",
        [(5, [5]), ([1, 2], [1, 2]), ("hello", ["hello"]), ((1, 2, 3), [1, 2, 3]),
         (range(3), [0, 1, 2]), (None, [None])],
    )
    def test_to_list(self, item, expected):
        assert cltmisc.to_list(item) == expected

    def test_list_intercept(self):
        assert cltmisc.list_intercept([1, 2, 3, 4, 5], [3, 4, 5, 6]) == [3, 4, 5]
        assert cltmisc.list_intercept([5, 3, 4, 1], [1, 2, 3, 4, 5]) == [5, 3, 4, 1]
        assert cltmisc.list_intercept([1, 2, 2, 3], [2, 3]) == [2, 2, 3]
        with pytest.raises(ValueError):
            cltmisc.list_intercept((1, 2), [1])

    def test_ismember(self):
        assert cltmisc.ismember([1, 3, 2, 3, 5], [3, 5, 7]) == ([3, 3, 5], [1, 3, 4])
        assert cltmisc.ismember(np.array([1, 2]), np.array([2])) == ([2], [1])
        with pytest.raises(ValueError):
            cltmisc.ismember("abc", ["a"])


####################################################################################################
# Section 2: dates
####################################################################################################
class TestFindClosestDate:
    def test_docstring_example(self):
        assert cltmisc.find_closest_date(["20230101", "20230201", "20230301"], "20230215") == (
            "20230201", 1, 14)

    def test_custom_format_and_tie(self):
        dates = ["2023-01-01", "2023-01-05"]
        # A tie keeps the first date
        assert cltmisc.find_closest_date(dates, "2023-01-03", date_fmt="%Y-%m-%d") == ("2023-01-01", 0, 2)

    def test_errors(self):
        with pytest.raises(TypeError):
            cltmisc.find_closest_date("20230101", "20230101")
        with pytest.raises(ValueError):
            cltmisc.find_closest_date([], "20230101")
        with pytest.raises(ValueError, match="target_date"):
            cltmisc.find_closest_date(["20230101"], "2023-01-01")
        with pytest.raises(ValueError, match="index 1"):
            cltmisc.find_closest_date(["20230101", "bad"], "20230101")


####################################################################################################
# Section 3: directories and files
####################################################################################################
@pytest.fixture
def file_tree(tmp_path):
    """sub-01/anat/{T1w, T2w}, sub-01/dwi/dwi, sub-02/anat/T1w_preproc and a top-level README."""
    files = [
        "sub-01/anat/sub-01_T1w.nii.gz",
        "sub-01/anat/sub-01_T2w.nii.gz",
        "sub-01/dwi/sub-01_dwi.nii.gz",
        "sub-02/anat/sub-02_T1w_preproc.nii.gz",
        "README.txt",
    ]
    for f in files:
        p = tmp_path / f
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x")
    (tmp_path / "empty_a" / "empty_b").mkdir(parents=True)
    return tmp_path


class TestDirectories:
    def test_get_leaf_directories(self, file_tree):
        leaves = sorted(os.path.relpath(p, file_tree) for p in cltmisc.get_leaf_directories(str(file_tree)))
        assert leaves == ["empty_a/empty_b", "sub-01/anat", "sub-01/dwi", "sub-02/anat"]
        with pytest.raises(ValueError):
            cltmisc.get_leaf_directories(str(file_tree / "missing"))

    @pytest.mark.parametrize("path, expected", [("/path/to/dir///", "/path/to/dir"),
                                                ("/path/to/dir", "/path/to/dir"), ("///", "/")])
    def test_remove_trailing_separators(self, path, expected):
        assert cltmisc.remove_trailing_separators(path) == expected

    def test_get_all_files(self, file_tree):
        names = lambda files: sorted(os.path.basename(f) for f in files)  # noqa: E731
        everything = cltmisc.get_all_files(file_tree)
        assert len(everything) == 5 and all(os.path.isabs(f) for f in everything)
        assert names(cltmisc.get_all_files(str(file_tree), recursive=False)) == ["README.txt"]
        assert names(cltmisc.get_all_files(file_tree, or_filter=["T1w", "T2w"])) == [
            "sub-01_T1w.nii.gz", "sub-01_T2w.nii.gz", "sub-02_T1w_preproc.nii.gz"]
        assert names(cltmisc.get_all_files(file_tree, or_filter="T1w", and_filter="preproc")) == [
            "sub-02_T1w_preproc.nii.gz"]
        assert names(cltmisc.get_all_files(file_tree, or_filter="t1w", bool_case=True)) == []
        # Match against the full path instead of the file name
        assert names(cltmisc.get_all_files(file_tree, or_filter="dwi/", just_files=False)) == [
            "sub-01_dwi.nii.gz"]

    def test_get_all_files_errors(self, file_tree, monkeypatch):
        with pytest.raises(ValueError, match="does not exist"):
            cltmisc.get_all_files(file_tree / "missing")
        with pytest.raises(ValueError, match="is a file"):
            cltmisc.get_all_files(file_tree / "README.txt")
        with pytest.raises(ValueError, match="empty"):
            cltmisc.get_all_files(file_tree / "empty_a" / "empty_b")
        with pytest.raises(TypeError):
            cltmisc.get_all_files(123)
        with pytest.raises(TypeError):
            cltmisc.get_all_files(file_tree, or_filter=5)
        (file_tree / "link").symlink_to(file_tree / "sub-01")
        with pytest.raises(ValueError, match="symlink"):
            cltmisc.get_all_files(file_tree / "link")
        monkeypatch.chdir(file_tree)
        with pytest.raises(ValueError, match="absolute"):
            cltmisc.get_all_files("sub-01")

    def test_rename_folders_simulate(self, capsys):
        planned = cltmisc.rename_folders(["/data/sub-01/session1"], {"sub-": "subject-", "session": "ses"},
                                         simulate=True)
        assert planned == [("/data/sub-01", "/data/subject-01"),
                           ("/data/sub-01/session1", "/data/subject-01/ses1")]
        assert "SIMULATION" in capsys.readouterr().out

    def test_rename_folders_nested(self, tmp_path):
        (tmp_path / "sub-01" / "session1").mkdir(parents=True)
        (tmp_path / "sub-01" / "session1" / "keep.txt").write_text("x")
        renamed = cltmisc.rename_folders([str(tmp_path / "sub-01" / "session1")],
                                         {"sub-": "subject-", "session": "ses"})
        assert len(renamed) == 2
        assert (tmp_path / "subject-01" / "ses1" / "keep.txt").is_file()
        assert not (tmp_path / "sub-01").exists()

    def test_rename_folders_case_insensitive(self, tmp_path):
        (tmp_path / "SUB-01").mkdir()
        cltmisc.rename_folders([str(tmp_path / "SUB-01")], {"sub-": "subject-"}, bool_case=False)
        assert (tmp_path / "subject-01").is_dir()

    def test_remove_empty_folders(self, file_tree):
        would = cltmisc.remove_empty_folders(str(file_tree / "empty_a"), simulate=True)
        # Simulation only reports folders that are already empty
        assert would == [str(file_tree / "empty_a" / "empty_b")]
        assert (file_tree / "empty_a" / "empty_b").is_dir()

        deleted = cltmisc.remove_empty_folders(str(file_tree))
        assert str(file_tree / "empty_a" / "empty_b") in deleted
        assert str(file_tree / "empty_a") in deleted
        assert not (file_tree / "empty_a").exists()
        assert (file_tree / "sub-01" / "anat" / "sub-01_T1w.nii.gz").is_file()

    def test_create_temporary_filename(self, tmp_path):
        f = cltmisc.create_temporary_filename(tmp_dir=str(tmp_path), prefix="tmp", suffix="corr")
        name = os.path.basename(f)
        assert os.path.dirname(f) == str(tmp_path)
        assert name.startswith("tmp_") and name.endswith("_corr.nii.gz")
        assert not os.path.exists(f)
        assert cltmisc.create_temporary_filename(tmp_dir=str(tmp_path), extension="csv").endswith(".csv")
        assert os.path.basename(cltmisc.create_temporary_filename(tmp_dir=str(tmp_path), prefix="")).count("_") == 0
        assert f != cltmisc.create_temporary_filename(tmp_dir=str(tmp_path), prefix="tmp", suffix="corr")
        with pytest.raises(ValueError):
            cltmisc.create_temporary_filename(tmp_dir=str(tmp_path / "missing"))


####################################################################################################
# Section 4: strings
####################################################################################################
class TestStrings:
    @pytest.mark.parametrize(
        "text, char, expected",
        [
            ("hello__world", "_", "hello_world"),
            ("hello__world//test", ["_", "/"], "hello_world/test"),
            (["path//to", "file///here"], "/", ["path/to", "file/here"]),
            ("aabbccdd__ee", None, "abcd_e"),
            ("aa__bb", "_", "aa_bb"),
            ("", "_", ""),
        ],
    )
    def test_remove_consecutive_duplicates(self, text, char, expected):
        assert cltmisc.remove_consecutive_duplicates(text, char) == expected

    def test_create_names_from_indices(self):
        assert cltmisc.create_names_from_indices([1, 2]) == ["auto-roi-000001", "auto-roi-000002"]
        assert cltmisc.create_names_from_indices(np.array([10, 20]), suffix="lh", padding=4) == [
            "auto-roi-0010-lh", "auto-roi-0020-lh"]
        assert cltmisc.create_names_from_indices([1], prefix="ctx", padding=2, sep="_") == ["ctx_01"]
        assert cltmisc.create_names_from_indices(5, padding=8) == ["auto-roi-00000005"]

    @pytest.mark.parametrize(
        "kwargs",
        [dict(indices=[1], padding=0), dict(indices=[1], sep=1), dict(indices="1"),
         dict(indices=[1.5])],
    )
    def test_create_names_from_indices_errors(self, kwargs):
        with pytest.raises(ValueError):
            cltmisc.create_names_from_indices(**kwargs)

    def test_correct_names(self):
        assert cltmisc.correct_names(["CTX-LH-1"], case="lower") == ["ctx-lh-1"]
        assert cltmisc.correct_names(["CTX-LH-1"], case="title") == ["Ctx-Lh-1"]
        assert cltmisc.correct_names(["hello world"], case="capitalize") == ["Hello world"]
        assert cltmisc.correct_names(["a"], case="upper") == ["A"]
        assert cltmisc.correct_names(["ctx-lh-1", "ctx-rh-2"], remove=["ctx-"],
                                     replacements={"lh": "left", "rh": "right"}) == ["left-1", "right-2"]
        assert cltmisc.correct_names(["ctx-lh-1"], replacements=[["lh", "left"]]) == ["ctx-left-1"]
        assert cltmisc.correct_names(["  region1 "], prefix="ctx-", suffix="-lh") == ["ctx-region1-lh"]
        assert cltmisc.correct_names(["path//to", "a__b"], remove_consecutive=["/", "_"]) == ["path/to", "a_b"]
        assert cltmisc.correct_names(["CTX__LH__1"], remove=["CTX"], replacements={"LH": "left"},
                                     remove_consecutive="_", case="lower", prefix="cortex-") == [
            "cortex-_left_1"]

    def test_correct_names_prefix_suffix_skip(self):
        assert cltmisc.correct_names(["ctx-a-lh"], prefix="ctx-", suffix="-lh") == ["ctx-a-lh"]
        assert cltmisc.correct_names(["ctx-a"], prefix="ctx-", skip_existing_prefix=False) == ["ctx-ctx-a"]
        assert cltmisc.correct_names(["a-lh"], suffix="-lh", skip_existing_suffix=False) == ["a-lh-lh"]
        assert cltmisc.correct_names([" a "], strip=False) == [" a "]

    def test_correct_names_errors(self):
        with pytest.raises(TypeError):
            cltmisc.correct_names("abc")
        with pytest.raises(TypeError):
            cltmisc.correct_names([1])
        with pytest.raises(ValueError):
            cltmisc.correct_names(["a"], case="snake")
        with pytest.raises(TypeError):
            cltmisc.correct_names(["a"], remove="a")
        with pytest.raises(ValueError):
            cltmisc.correct_names(["a"], replacements=[["a"]])
        with pytest.raises(TypeError):
            cltmisc.correct_names(["a"], replacements="a")

    @pytest.mark.parametrize(
        "name, expected",
        [("/path/to/image.nii.gz", "image"), ("image.jpg", "image"), ("archive.tar.gz", "archive.tar"),
         ("README", "README"), ("sub-01_T1w.nii", "sub-01_T1w")],
    )
    def test_get_real_basename(self, name, expected):
        assert cltmisc.get_real_basename(name) == expected


####################################################################################################
# Section 5: dictionaries and dataframes
####################################################################################################
class TestJsonAndDicts:
    def test_json_round_trip(self, tmp_path):
        data = {"a": 1, "b": [1, 2], "c": {"d": "x"}}
        path = str(tmp_path / "data.json")
        cltmisc.save_dictionary_to_json(data, path)
        assert cltmisc.load_json(path) == data
        assert cltmisc.load_json(tmp_path / "data.json") == data

    def test_json_errors(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            cltmisc.load_json(tmp_path / "missing.json")
        with pytest.raises(ValueError):
            cltmisc.load_json(123)
        bad = tmp_path / "bad.json"
        bad.write_text("{not json")
        with pytest.raises(ValueError):
            cltmisc.load_json(bad)
        with pytest.raises(ValueError):
            cltmisc.save_dictionary_to_json({}, str(tmp_path / "data.txt"))
        with pytest.raises(ValueError):
            cltmisc.save_dictionary_to_json([1], str(tmp_path / "data.json"))

    def test_remove_empty_keys_or_values(self):
        d = {"key1": "value1", "key2": "", "": "value3", "   ": "value4", "key5": None,
             "key6": 0, "key7": False, "key8": [], "key9": {}}
        assert cltmisc.remove_empty_keys_or_values(d) == {"key1": "value1", "key5": None, "key6": 0,
                                                          "key7": False}
        assert cltmisc.remove_empty_keys_or_values(d, remove_none=True) == {"key1": "value1", "key6": 0,
                                                                            "key7": False}
        assert cltmisc.remove_empty_keys_or_values(d, remove_empty_collections=False) == {
            "key1": "value1", "key5": None, "key6": 0, "key7": False, "key8": [], "key9": {}}
        assert len(d) == 9                                      # The input is not modified
        cltmisc.remove_empty_keys_or_values(d, in_place=True)
        assert "key2" not in d

    def test_compare_dicts(self):
        d1 = {"sub": "sub-002", "ses": "V1", "atlas": "A", "only1": 1}
        d2 = {"sub": "sub-002", "ses": "V1", "atlas": "B", "only2": 2}
        res = cltmisc.compare_dicts(d1, d2)
        assert res["differing"] == {"atlas": ("A", "B")}
        assert res["identical"] == ["sub", "ses"]
        assert res["only_in_first"] == ["only1"] and res["only_in_second"] == ["only2"]
        assert cltmisc.compare_dicts(d1, d2, ignore_keys=["atlas", "only2"])["differing"] == {}

    def test_compare_dicts_comparator(self):
        res = cltmisc.compare_dicts({"a": np.array([1, 2])}, {"a": np.array([1, 2])}, comparator=np.array_equal)
        assert res["identical"] == ["a"]
        with pytest.raises(TypeError):
            cltmisc.compare_dicts({}, [], )
        with pytest.raises(TypeError):
            cltmisc.compare_dicts({}, {}, comparator="eq")

    def test_extract_string_values(self, tmp_path):
        data = {"a": {"b": "value1", "c": {"d": "value2"}}, "e": ["list"], "f": "value3", "g": 5}
        assert cltmisc.extract_string_values(data) == {"b": "value1", "d": "value2", "f": "value3"}
        assert cltmisc.extract_string_values(data, only_last_key=False) == {
            "a.b": "value1", "a.c.d": "value2", "f": "value3"}
        # Colliding leaf keys fall back to the full path
        assert cltmisc.extract_string_values({"a": {"name": "v1"}, "b": {"name": "v2"}}) == {
            "name": "v1", "b.name": "v2"}
        path = tmp_path / "data.json"
        path.write_text(json.dumps(data))
        assert cltmisc.extract_string_values(str(path))["f"] == "value3"

    def test_extract_string_values_errors(self, tmp_path):
        with pytest.raises(ValueError):
            cltmisc.extract_string_values(str(tmp_path / "missing.json"))
        with pytest.raises(TypeError):
            cltmisc.extract_string_values([1, 2])

    def test_update_dict(self):
        orig = {"name": "John", "items": [1, 2], "cfg": {"a": 1, "b": 2}}
        out = cltmisc.update_dict(orig, {"name": "Jane", "age": 30, "items": [3, 4], "cfg": {"b": 5}},
                                  merge_lists=True, allow_new_keys=True)
        assert out is orig
        assert orig == {"name": "Jane", "items": [1, 2, 3, 4], "cfg": {"a": 1, "b": 5}, "age": 30}
        assert cltmisc.update_dict({"items": [1]}, {"items": [2]})["items"] == [2]

    def test_update_dict_errors(self):
        with pytest.raises(TypeError, match="Type mismatch"):
            cltmisc.update_dict({"a": 1}, {"a": "1"})
        with pytest.raises(KeyError):
            cltmisc.update_dict({"a": 1}, {"b": 1})
        with pytest.raises(KeyError):
            cltmisc.update_dict({"cfg": {"a": 1}}, {"cfg": {"new": 1}})

    def test_explorer_dict_is_reexported(self):
        assert "ExplorerDict" in cltmisc.__all__ and hasattr(cltmisc, "ExplorerDict")


class TestTables:
    def test_read_file_with_separator_detection(self, tmp_path):
        csv = tmp_path / "a.csv"
        csv.write_text("x,y\n1,2\n3,4\n")
        tsv = tmp_path / "a.tsv"
        tsv.write_text("x\ty\n1\t2\n")
        spaces = tmp_path / "a.txt"
        spaces.write_text("x y\n1 2\n")
        for f in (csv, tsv, spaces):
            df = cltmisc.read_file_with_separator_detection(f)
            assert df.columns.tolist() == ["x", "y"] and df.iloc[0].tolist() == [1, 2]

    def test_read_file_with_separator_detection_errors(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            cltmisc.read_file_with_separator_detection(tmp_path / "missing.csv")
        empty = tmp_path / "empty.csv"
        empty.write_text("")
        with pytest.raises(ValueError, match="empty"):
            cltmisc.read_file_with_separator_detection(empty)
        single = tmp_path / "single.txt"
        single.write_text("value\n")
        with pytest.raises(ValueError, match="separator"):
            cltmisc.read_file_with_separator_detection(single)

    def test_smart_read_table(self, tmp_path):
        semicolon = tmp_path / "a.csv"
        semicolon.write_text("participant_id;age\nsub-01;30\nsub-02;41\n")
        df = cltmisc.smart_read_table(semicolon)
        assert df.columns.tolist() == ["participant_id", "age"] and df["age"].tolist() == [30, 41]

    def test_smart_read_table_keeps_run_as_text(self, tmp_path):
        f = tmp_path / "runs.tsv"
        f.write_text("Run\tvalue\n01\t1.5\n02\t2.5\n")
        assert cltmisc.smart_read_table(f)["Run"].tolist() == ["01", "02"]
        with pytest.raises(FileNotFoundError):
            cltmisc.smart_read_table(tmp_path / "missing.tsv")

    def test_drop_empty_columns(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [np.nan] * 3, "c": ["", "", ""],
                           "d": ["x", "", None], "e": [" ", "  ", None]})
        assert cltmisc.drop_empty_columns(df).columns.tolist() == ["a", "d"]
        assert cltmisc.drop_empty_columns(df, treat_whitespace_as_empty=False).columns.tolist() == [
            "a", "d", "e"]
        assert df.shape[1] == 5                                 # The input is not modified
        out = cltmisc.drop_empty_columns(df, inplace=True)
        assert out is df and df.columns.tolist() == ["a", "d"]

    def test_drop_empty_columns_edge_cases(self):
        empty_rows = pd.DataFrame({"a": pd.Series([], dtype=float)})
        assert cltmisc.drop_empty_columns(empty_rows).columns.tolist() == ["a"]
        df = pd.DataFrame({"flag": [False, False], "n": [0, 0], "t": pd.to_datetime(["2024-01-01", None])})
        assert cltmisc.drop_empty_columns(df).columns.tolist() == ["flag", "n", "t"]
        with pytest.raises(TypeError):
            cltmisc.drop_empty_columns([1, 2])

    def test_expand_and_concatenate(self):
        df_add = pd.DataFrame({"participant_id": ["sub-01"], "session_id": ["ses-01"]})
        df = pd.DataFrame({"region": ["a", "b", "c"], "value": [1, 2, 3]}, index=[10, 11, 12])
        out = cltmisc.expand_and_concatenate(df_add, df)
        assert out.columns.tolist() == ["participant_id", "session_id", "region", "value"]
        assert out["participant_id"].tolist() == ["sub-01"] * 3 and out["value"].tolist() == [1, 2, 3]

    def test_expand_and_concatenate_shared_column(self):
        df_add = pd.DataFrame({"participant_id": ["sub-01"], "value": [0]})
        df = pd.DataFrame({"value": [1, 2]})
        out = cltmisc.expand_and_concatenate(df_add, df)
        assert out.columns.tolist() == ["participant_id", "value"] and out["value"].tolist() == [1, 2]


####################################################################################################
# Section 6: containers
####################################################################################################
class TestContainerCommand:
    def test_local(self):
        assert cltmisc.generate_container_command(["ls", "-l"]) == ["ls", "-l"]
        assert cltmisc.generate_container_command("echo 'a b'") == ["echo", "a b"]

    @pytest.mark.parametrize("tech, flag", [("singularity", "--bind"), ("docker", "-v")])
    def test_container(self, tmp_path, tech, flag):
        image = tmp_path / "image.sif"
        image.write_text("x")
        data = tmp_path / "data"
        data.mkdir()
        cmd = cltmisc.generate_container_command(["mri_info", str(data / "T1.mgz")], technology=tech,
                                                 image_path=image)
        assert cmd[:2] == [tech, "run"]
        assert cmd[2:4] == [flag, f"{data}:{data}"]
        assert cmd[4:] == [str(image), "mri_info", str(data / "T1.mgz")]

    def test_errors(self, tmp_path):
        with pytest.raises(ValueError, match="Docker"):
            cltmisc.generate_container_command(["ls"], technology="docker")
        with pytest.raises(ValueError, match="does not exist"):
            cltmisc.generate_container_command(["ls"], technology="singularity",
                                               image_path=str(tmp_path / "missing.sif"))


####################################################################################################
# Section 7: printing and inspection
####################################################################################################
def _example_function(name: str, age: int = 25, *args, **kwargs):
    """Example function used to test the signature formatting."""


class _Example:
    """Example class with a configuration method."""

    value = 3

    def load_config(self, path: str):
        """Load the configuration from a file."""

    def save(self):
        """Save the results."""

    @property
    def size(self):
        return 1


class TestPrinting:
    def test_is_notebook_outside_jupyter(self):
        assert cltmisc.is_notebook() is False

    def test_format_signature(self):
        sig = inspect.signature(_example_function)
        text = strip_ansi(cltmisc.format_signature(sig))
        assert all(s in text for s in ["name", "str", "age", "25", "args", "kwargs"])
        html = cltmisc.format_signature(sig, notebook_mode=True)
        assert "<span" in html and "name" in html

    def test_show_module_contents(self, capsys):
        module = types.ModuleType("fake_module", "A fake module.")
        module.example_function = _example_function
        module.Example = _Example
        _example_function.__module__ = _Example.__module__ = "fake_module"
        cltmisc.show_module_contents(module)
        out = strip_ansi(capsys.readouterr().out)
        assert "example_function" in out and "Example" in out and "load_config" in out

    def test_show_module_contents_by_name(self, capsys):
        cltmisc.show_module_contents("clabtoolkit.misctools")
        assert "build_indices" in strip_ansi(capsys.readouterr().out)

    def test_show_object_content(self, capsys):
        cltmisc.show_object_content(_Example())
        out = strip_ansi(capsys.readouterr().out)
        assert "load_config" in out and "size" in out and "value" in out

    def test_search_methods(self, capsys):
        cltmisc.search_methods(_Example, "config")
        out = strip_ansi(capsys.readouterr().out)
        assert "load_config" in out and "save" not in out and "Found 1 match" in out
        cltmisc.search_methods(_Example, "results")              # Found in the docstring
        assert "save" in strip_ansi(capsys.readouterr().out)
        cltmisc.search_methods(_Example, "nothing_like_this")
        assert "No matches" in strip_ansi(capsys.readouterr().out)

    def test_h5explorer(self, tmp_path, capsys):
        path = tmp_path / "data.h5"
        with h5py.File(path, "w") as f:
            f.create_dataset("measurements", data=np.zeros((10, 4)))
            grp = f.create_group("metadata")
            grp.create_dataset("info", data=np.arange(3))
            grp.attrs["units"] = "volts"
        stats = cltmisc.h5explorer(str(path))
        assert stats["groups"] == 1 and stats["datasets"] == 2
        out = strip_ansi(capsys.readouterr().out)
        assert "measurements" in out and "metadata" in out and "units" in out
        cltmisc.h5explorer_simple(str(path))
        assert "info" in capsys.readouterr().out

    def test_h5explorer_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            cltmisc.h5explorer(str(tmp_path / "missing.h5"))

    def test_printprogressbar(self, capsys):
        cltmisc.printprogressbar(5, 10, prefix="Progress", suffix="done", length=10)
        out = capsys.readouterr().out
        assert "Progress |█████-----| 50.0% done" in out
        cltmisc.printprogressbar(10, 10, length=4)
        out = capsys.readouterr().out
        assert "|████| 100.0%" in out and out.endswith("\n")   # A new line is printed when complete

    def test_smart_formatter_raw_text(self):
        parser = argparse.ArgumentParser(prog="tool", formatter_class=cltmisc.SmartFormatter)
        parser.add_argument("--mode", help="R|first line\nsecond line")
        parser.add_argument("--other", help="normal help text that is wrapped")
        help_text = parser.format_help()
        lines = [line.strip() for line in help_text.splitlines()]
        # The raw help keeps its line break: the second line is printed on its own line
        assert any(line.startswith("--mode MODE") and line.endswith("first line") for line in lines)
        assert "second line" in lines
        assert "R|" not in help_text
