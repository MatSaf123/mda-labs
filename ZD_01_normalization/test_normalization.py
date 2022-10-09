from .normalization import normalize_min_max
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

def test_normalize_standardize():
    raise NotImplementedError