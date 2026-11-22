from math import isclose

import pytest
from main import cosine_similarity


run_cases = [
    pytest.param([1, 2, 3], [1, 2, 3], 1.0),  # Identical vectors → 1.0
    pytest.param([1, 0, 0], [0, 1, 0], 0.0),  # Orthogonal vectors → 0.0
    pytest.param([1, 0], [-1, 0], -1.0),  # Opposite direction → -1.0
    pytest.param([1, 2], [10, 20], 1.0),  # Scaling invariance → 1.0
    pytest.param([1, -1, 0], [1, 1, 0], 0.0),  # Mixed signs but orthogonal → 0.0
    # Known 60° angle → cos = 0.5
    pytest.param([1.0, 0.0], [0.5, 3**0.5 / 2], 0.5),
    # Mismatched lengths → exception
    pytest.param([1, 2], [1, 2, 3], "exception"),
]

submit_cases = [
    # Zero vector present → 0.0
    pytest.param([0, 0, 0], [1, 2, 3], 0.0, marks=pytest.mark.submit),
    # Float proportional vectors → 1.0
    pytest.param([0.1, 0.2], [0.3, 0.6], 1.0, marks=pytest.mark.submit),
    # Nearly orthogonal with tiny component
    pytest.param([1.0, 1e-12], [0.0, 1.0], 0.0, marks=pytest.mark.submit),
    # 120° angle → cos = -0.5
    pytest.param(
        [1.0, 0.0],
        [-0.5, 3**0.5 / 2],
        -0.5,
        marks=pytest.mark.submit,
    ),
]


@pytest.mark.parametrize(("vec1", "vec2", "expected"), run_cases + submit_cases)
def test_cosine_similarity(vec1, vec2, expected):
    print("\n---------------------------------")
    print(f"Vector 1: {vec1}")
    print(f"Vector 2: {vec2}")
    print(
        f"\nExpected result: {expected:.2f}"
        if isinstance(expected, float)
        else f"\nExpected result: {expected}"
    )

    try:
        result = cosine_similarity(vec1, vec2)
    except ValueError:
        result = "exception"

    print(
        f"Actual result:   {result:.2f}"
        if isinstance(result, (int, float))
        else f"Actual result:   {result}"
    )
    if isinstance(expected, float):
        assert isinstance(result, (int, float))
        assert isclose(result, expected, abs_tol=1e-12)
    else:
        assert result == expected
