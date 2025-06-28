import pandas as pd
import pytest

from src.utils import sort_column_by_keywords


@pytest.mark.parametrize(
    "data, keywords, expected_result",
    [
        (
            pd.DataFrame(
                {"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9], "D": [10, 11, 12]}
            ),
            ["B", "C"],
            pd.DataFrame(
                {"B": [4, 5, 6], "C": [7, 8, 9], "A": [1, 2, 3], "D": [10, 11, 12]}
            ),
        ),
        (
            pd.DataFrame(
                {"Ak": [1, 2, 3], "Bl": [4, 5, 6], "Cm": [7, 8, 9], "Dn": [10, 11, 12]}
            ),
            ["Ak", "Dn"],
            pd.DataFrame(
                {"Ak": [1, 2, 3], "Dn": [10, 11, 12], "Bl": [4, 5, 6], "Cm": [7, 8, 9]}
            ),
        ),
    ],
)
def test_sort_column_by_keywords_first(data, keywords, expected_result):
    # Test sorting with keywords first
    sorted_data = sort_column_by_keywords(data, keywords, first=True)
    pd.testing.assert_frame_equal(sorted_data, expected_result)


@pytest.mark.parametrize(
    "data, keywords, expected_result",
    [
        (
            pd.DataFrame(
                {"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9], "D": [10, 11, 12]}
            ),
            ["B", "C"],
            pd.DataFrame(
                {"A": [1, 2, 3], "D": [10, 11, 12], "B": [4, 5, 6], "C": [7, 8, 9]}
            ),
        ),
        (
            pd.DataFrame(
                {"Ak": [1, 2, 3], "Bl": [4, 5, 6], "Cm": [7, 8, 9], "Dn": [10, 11, 12]}
            ),
            ["Ak", "Dn"],
            pd.DataFrame(
                {"Bl": [4, 5, 6], "Cm": [7, 8, 9], "Ak": [1, 2, 3], "Dn": [10, 11, 12]}
            ),
        ),
    ],
)
def test_sort_column_by_keywords_last(data, keywords, expected_result):
    # Test sorting with keywords last
    sorted_data = sort_column_by_keywords(data, keywords, first=False)
    pd.testing.assert_frame_equal(sorted_data, expected_result)
