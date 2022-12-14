from typing import List
from pathlib import Path
import math
import csv


def print_matrix(matrix: List[List[int]], name: str = "") -> None:
    """Print a matrix in a nice way. I just didn't like the
    way pprint did it."""

    print(f"\n{name}")
    for row in matrix:
        print(row)


def transpose_matrix(matrix: List[List[int]]) -> List[List[int]]:
    """Swap columns with rows."""
    return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]


def normalize_min_max(matrix: List[List[int]]) -> List[List[int]]:
    """Normalize data in matrix' columns with min-max formula."""

    # Transpose input matrix
    t_matrix = transpose_matrix(matrix)

    # Create a r(esult)_matrix of None values with same dimensions as original matrix
    r_matrix = [[None for _ in row] for row in t_matrix]

    for i in range(len(t_matrix)):
        row = t_matrix[i]
        column_max, column_min = max(row), min(row)
        denominator = column_max - column_min

        for j in range(len(row)):
            x = row[j]
            r_matrix[i][j] = (x - column_min) / denominator

    # Transpose matrix back to it's original columns/rows state
    r_matrix = transpose_matrix(r_matrix)

    return r_matrix


def normalize_standardize(matrix: List[List[int]]) -> List[List[int]]:
    """Normalize data in matrix' column with standardization formula."""

    # Transpose input matrix
    t_matrix = transpose_matrix(matrix)

    # Create a r(esult)_matrix of None values with same dimensions as original matrix
    r_matrix = [[None for _ in row] for row in t_matrix]

    for i in range(len(t_matrix)):
        row = t_matrix[i]
        mean = sum(row) / len(row)

        # Compute standard deviation for this row
        s_d = math.sqrt(sum([math.pow(x - mean, 2) for x in row]) / len(row))

        for j in range(len(row)):
            x = row[j]
            r_matrix[i][j] = (x - mean) / s_d

    # Transpose matrix back to it's original columns/rows state
    r_matrix = transpose_matrix(r_matrix)

    return r_matrix


if __name__ == "__main__":

    data: List[List[int]] = []

    # Read data from csv
    data_path = Path("ZD_01_normalization", "data.csv").resolve()

    with open(data_path) as csv_file:
        reader = csv.reader(csv_file, delimiter=",")
        for row in reader:
            data.append(row)

    # Skip first row cause it's labels, convert to ints
    matrix = [list(map(int, row)) for row in data[1:]]

    print_matrix(matrix, "Input:")

    result_min_max = normalize_min_max(matrix)
    print_matrix(result_min_max, "Output min-max:")

    # # result_standardize = normalize_standardize(matrix)
    # print_matrix(result_standardize, "Output standardize:")
