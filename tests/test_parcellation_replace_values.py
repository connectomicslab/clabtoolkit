"""Tests for ``clabtoolkit.parcellationtools.Parcellation.replace_values``."""

import numpy as np
import pytest

from clabtoolkit.parcellationtools import Parcellation


@pytest.fixture
def parc():
    """A tiny parcellation holding the codes 0-5."""

    def _make():
        data = np.array([[[0, 1, 2, 3, 4, 5]]], dtype=np.int32)
        return Parcellation(data)

    return _make


def codes(parcellation):
    return list(map(int, parcellation.data.ravel()))


class TestFlatListInput:
    def test_replaces_each_code_positionally(self, parc):
        p = parc()
        p.replace_values([1, 2], [10, 20])
        assert codes(p) == [0, 10, 20, 3, 4, 5]

    def test_preserves_pairing_when_new_codes_descend(self, parc):
        """Regression: new_codes used to be sorted, so 1 was relabelled 10."""
        p = parc()
        p.replace_values([1, 2], [20, 10])
        assert codes(p) == [0, 20, 10, 3, 4, 5]

    def test_allows_repeated_new_codes(self, parc):
        """Regression: de-duplication used to shrink new_codes below n_groups."""
        p = parc()
        p.replace_values([1, 2], [7, 7])
        assert codes(p) == [0, 7, 7, 3, 4, 5]

    def test_updates_the_region_index(self, parc):
        p = parc()
        p.replace_values([1], [42])
        assert 42 in p.index
        assert 1 not in p.index

    def test_leaves_untouched_codes_alone(self, parc):
        p = parc()
        p.replace_values([1], [99])
        assert codes(p)[3:] == [3, 4, 5]


class TestGroupedInput:
    def test_collapses_each_group_to_one_code(self, parc):
        p = parc()
        p.replace_values([[1, 2], [3, 4]], [100, 200])
        assert codes(p) == [0, 100, 100, 200, 200, 5]

    def test_preserves_group_order(self, parc):
        p = parc()
        p.replace_values([[1, 2], [3, 4]], [200, 100])
        assert codes(p) == [0, 200, 200, 100, 100, 5]


class TestDictInput:
    def test_maps_each_key_to_its_own_value(self, parc):
        p = parc()
        p.replace_values({1: 8, 2: 9})
        assert codes(p) == [0, 8, 9, 3, 4, 5]

    def test_preserves_pairing_when_values_descend(self, parc):
        """Regression: keys and values were sorted separately, so {1: 9, 2: 8}
        relabelled 1 as 8 and 2 as 9."""
        p = parc()
        p.replace_values({1: 9, 2: 8})
        assert codes(p) == [0, 9, 8, 3, 4, 5]

    def test_accepts_a_range_string_as_key(self, parc):
        p = parc()
        p.replace_values({"1-2": 9})
        assert codes(p) == [0, 9, 9, 3, 4, 5]

    def test_accepts_a_tuple_range_as_key(self, parc):
        p = parc()
        p.replace_values({(1, 2): 9, 3: 30})
        assert codes(p) == [0, 9, 9, 30, 4, 5]


class TestNumpyInput:
    def test_accepts_a_1d_array(self, parc):
        p = parc()
        p.replace_values(np.array([1, 2]), [10, 20])
        assert codes(p) == [0, 10, 20, 3, 4, 5]

    def test_rejects_a_2d_array(self, parc):
        with pytest.raises(TypeError):
            parc().replace_values(np.array([[1, 2], [3, 4]]), [10, 20])


class TestInvalidInput:
    def test_rejects_length_mismatch(self, parc):
        with pytest.raises(ValueError, match="must equal"):
            parc().replace_values([1, 2], [10])

    def test_rejects_empty_codes(self, parc):
        with pytest.raises(ValueError, match="cannot be empty"):
            parc().replace_values([], [])

    def test_rejects_unsupported_type(self, parc):
        with pytest.raises(TypeError, match="list or numpy array"):
            parc().replace_values("not-a-code-list", [1])

    def test_rejects_mixed_flat_and_grouped_codes(self, parc):
        with pytest.raises(TypeError, match="list of ints or a list of lists"):
            parc().replace_values([1, [2, 3]], [10, 20])
