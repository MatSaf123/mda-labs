from .normalization import (
    normalize_min_max,
    normalize_standardize,
    print_matrix,
    transpose_matrix,
)
import pytest


def test_normalize_min_max():

    # Same data that can be found in the file
    matrix = [[10, 20, 30], [15, 12, 45], [100, 400, 350], [2, 4, 2], [6, 12, 1]]

    # Result I got from sklearn.preprocessing.MinMaxScaler
    expected = [
        [
            0.08163265,
            0.04040404,
            0.08309456,
        ],
        [
            0.13265306,
            0.02020202,
            0.1260745,
        ],
        [
            1,
            1,
            1,
        ],
        [
            0,
            0,
            0.00286533,
        ],
        [0.04081633, 0.02020202, 0],
    ]

    result = normalize_min_max(matrix)

    # The results of function above are more precise when compared to sklearn,
    # so we need to wrap the result.

    for e, r in zip(expected, result):
        assert e == pytest.approx(r, 0.0001)


def test_small_normalize_standardize():

    # This small example is brought to you by https://www.statisticshowto.com/standardized-values-examples/

    # Our input is this way cause we operate on columns by transposing matrix at the entry
    matrix = [[3], [3], [4], [4], [6]]

    expected = [
        [-0.9128709291752769],
        [-0.9128709291752769],
        [0.0],
        [0.0],
        [1.8257418583505538],
    ]

    assert normalize_standardize(matrix) == expected


def test_normalize_standardize():

    # Same data that can be found in the file
    matrix = [[10, 20, 30], [15, 12, 45], [100, 400, 350], [2, 4, 2], [6, 12, 1]]

    # Result I got from sklearn.preprocessing.StandardScaler

    expected = [
        [-0.44923313038674617, -0.4482154870559686, -0.41722463162823936],
        [-0.3139219465353166, -0.4997345085566547, -0.30466402957026106],
        [1.986368178938986, 1.9989380342266188, 1.9840682122752964],
        [-0.6657310245490334, -0.5512535300573407, -0.6273377554697988],
        [-0.5574820774678898, -0.4997345085566547, -0.6348417956069973],
    ]

    result = normalize_standardize(matrix)

    for e, r in zip(expected, result):
        assert e == pytest.approx(r, 0.0001)
