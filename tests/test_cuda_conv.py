# test_cuda_conv.py
import pytest
import numpy as np
from hypothesis import given, settings
from hypothesis.strategies import floats
import minitorch
from minitorch import Tensor
from .strategies import assert_close
from .tensor_strategies import tensors

small_floats = floats(
    min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False
)

cuda_backend = minitorch.TensorBackend(minitorch.CudaOps)
simple_backend = minitorch.TensorBackend(minitorch.SimpleOps)


@pytest.mark.task4_4b
def test_conv1d_cuda_simple():
    # Simple deterministic test
    input_data = [1, 1, 1, 1]
    weight_data = [1, 1, 4]

    input_cuda = Tensor.make(np.array(input_data), (1, 1, 4), backend=cuda_backend)
    weight_cuda = Tensor.make(np.array(weight_data), (1, 1, 3), backend=cuda_backend)

    input_simple = Tensor.make(np.array(input_data), (1, 1, 4), backend=simple_backend)
    weight_simple = Tensor.make(np.array(weight_data), (1, 1, 3), backend=simple_backend)

    output_cuda = minitorch.conv1d(input_cuda, weight_cuda)
    output_simple = minitorch.conv1d(input_simple, weight_simple)

    for i in range(output_simple.size):
        assert_close(output_cuda._tensor._storage[i], output_simple._tensor._storage[i])


@pytest.mark.task4_4b
def test_conv1d_zero_weight_cuda_simple():
    input_data = [1, 2, 3, 4]
    weight_data = [0, 0, 0]

    input_cuda = Tensor.make(np.array(input_data), (1, 1, 4), backend=cuda_backend)
    weight_cuda = Tensor.make(np.array(weight_data), (1, 1, 3), backend=cuda_backend)

    input_simple = Tensor.make(np.array(input_data), (1, 1, 4), backend=simple_backend)
    weight_simple = Tensor.make(np.array(weight_data), (1, 1, 3), backend=simple_backend)

    output_cuda = minitorch.conv1d(input_cuda, weight_cuda)
    output_simple = minitorch.conv1d(input_simple, weight_simple)

    for i in range(output_simple.size):
        assert_close(output_cuda._tensor._storage[i], output_simple._tensor._storage[i])


@pytest.mark.task4_4b
def test_conv1d_cuda_cases():
    # Various input/weight pairs
    cases = [
        ([1, 2, 3, 4], [0, 0, 0]),
        ([1, 1, 1, 1], [1, 1, 1]),
        ([1, -1, 1, -1], [1, -1, 1]),
    ]
    for input_data, weight_data in cases:
        input_cuda = Tensor.make(np.array(input_data), (1, 1, len(input_data)), backend=cuda_backend)
        weight_cuda = Tensor.make(np.array(weight_data), (1, 1, len(weight_data)), backend=cuda_backend)

        input_simple = Tensor.make(np.array(input_data), (1, 1, len(input_data)), backend=simple_backend)
        weight_simple = Tensor.make(np.array(weight_data), (1, 1, len(weight_data)), backend=simple_backend)

        output_cuda = minitorch.conv1d(input_cuda, weight_cuda)
        output_simple = minitorch.conv1d(input_simple, weight_simple)

        for i in range(output_simple.size):
            assert_close(output_cuda._tensor._storage[i], output_simple._tensor._storage[i])


@pytest.mark.task4_4b
def test_conv2d_cuda_simple():
    # Test a simple 2D convolution scenario
    input_data = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
    ]
    weight_data = [
        [1, 0],
        [0, -1],
    ]

    input_cuda = Tensor.make(np.array(input_data), (1, 1, 4, 4), backend=cuda_backend)
    weight_cuda = Tensor.make(np.array(weight_data), (1, 1, 2, 2), backend=cuda_backend)

    input_simple = Tensor.make(np.array(input_data), (1, 1, 4, 4), backend=simple_backend)
    weight_simple = Tensor.make(np.array(weight_data), (1, 1, 2, 2), backend=simple_backend)

    output_cuda = minitorch.conv2d(input_cuda, weight_cuda)
    output_simple = minitorch.conv2d(input_simple, weight_simple)

    for i in range(output_simple.size):
        assert_close(output_cuda._tensor._storage[i], output_simple._tensor._storage[i])

