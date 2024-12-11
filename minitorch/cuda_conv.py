# cuda_conv.py
from typing import Tuple, TypeVar, Any
import numba
from numba import cuda
from numba.cuda import jit as _jit

import minitorch
from .autodiff import Context
from .tensor import Tensor
from .tensor_data import (
    Shape,
    Strides,
    Storage,
    index_to_position,
    to_index,
    broadcast_index,
    MAX_DIMS,
)
from .tensor_functions import Function

Fn = TypeVar("Fn")

# Backend for CUDA
cuda_backend = minitorch.TensorBackend(minitorch.CudaOps)

# You may adjust this block dimension for optimization
BLOCK_DIM = 32

def device_jit(fn: Fn, **kwargs: Any) -> Fn:
    """JIT-compile a function for CUDA device."""
    return _jit(device=True, **kwargs)(fn)

def jit(fn: Fn, **kwargs: Any) -> Any:
    """JIT-compile a CUDA kernel function."""
    return _jit(**kwargs)(fn)

# JIT compile indexing helper functions
to_index = device_jit(to_index)
index_to_position = device_jit(index_to_position)
broadcast_index = device_jit(broadcast_index)

@cuda.jit
def _tensor_conv1d(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    input: Storage,
    input_shape: Shape,
    input_strides: Strides,
    weight: Storage,
    weight_shape: Shape,
    weight_strides: Strides,
    reverse: bool,
) -> None:
    """
    CUDA kernel for 1D convolution.
    """

    batch_, out_channels, out_width = out_shape
    batch, in_channels, width = input_shape
    out_channels_, in_channels_, kw = weight_shape

    # Check shape consistency
    if not (batch == batch_ and in_channels == in_channels_ and out_channels == out_channels_):
        return

    # Compute global thread index
    idx = cuda.grid(1)
    if idx >= out_size:
        return

    # Convert flat index idx to (b, oc, w)
    b = idx // (out_channels * out_width)
    remainder = idx % (out_channels * out_width)
    oc = remainder // out_width
    w = remainder % out_width

    acc_val = 0.0
    for ic in range(in_channels):
        for k in range(kw):
            if reverse:
                in_w = w - kw + 1 + k
            else:
                in_w = w + k

            if 0 <= in_w < width:
                input_pos = b * input_strides[0] + ic * input_strides[1] + in_w * input_strides[2]
                weight_pos = oc * weight_strides[0] + ic * weight_strides[1] + k * weight_strides[2]
                acc_val += input[input_pos] * weight[weight_pos]

    out_pos = b * out_strides[0] + oc * out_strides[1] + w * out_strides[2]
    out[out_pos] = acc_val

tensor_conv1d = jit(_tensor_conv1d)


class Conv1dCudaFun(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        """
        Forward pass for CUDA-based 1D convolution.
        """
        ctx.save_for_backward(input, weight)
        batch, in_channels, width = input.shape
        out_channels, in_channels2, kw = weight.shape
        assert in_channels == in_channels2

        out = input.zeros((batch, out_channels, width))
        threadsperblock = 128
        blockspergrid = (out.size + threadsperblock - 1) // threadsperblock

        tensor_conv1d[blockspergrid, threadsperblock](
            out._tensor._storage, out.shape, out._tensor._strides, out.size,
            input._tensor._storage, input.shape, input._tensor._strides,
            weight._tensor._storage, weight.shape, weight._tensor._strides,
            False,
        )
        return out

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Backward pass for CUDA-based 1D convolution.
        """
        input, weight = ctx.saved_values
        batch, in_channels, width = input.shape
        out_channels, in_channels_wt, kw = weight.shape

        # Grad w.r.t. weight
        grad_weight = grad_output.zeros((in_channels, out_channels, kw))
        new_input = input.permute(1, 0, 2)
        new_grad_out = grad_output.permute(1, 0, 2)

        threadsperblock = 128
        blockspergrid = (grad_weight.size + threadsperblock - 1) // threadsperblock
        tensor_conv1d[blockspergrid, threadsperblock](
            grad_weight._tensor._storage, grad_weight.shape, grad_weight._tensor._strides, grad_weight.size,
            new_input._tensor._storage, new_input.shape, new_input._tensor._strides,
            new_grad_out._tensor._storage, new_grad_out.shape, new_grad_out._tensor._strides,
            False,
        )
        grad_weight = grad_weight.permute(1, 0, 2)

        # Grad w.r.t input
        grad_input = input.zeros((batch, in_channels, width))
        new_weight = weight.permute(1, 0, 2)
        blockspergrid = (grad_input.size + threadsperblock - 1) // threadsperblock
        tensor_conv1d[blockspergrid, threadsperblock](
            grad_input._tensor._storage, grad_input.shape, grad_input._tensor._strides, grad_input.size,
            grad_output._tensor._storage, grad_output.shape, grad_output._tensor._strides,
            new_weight._tensor._storage, new_weight.shape, new_weight._tensor._strides,
            True,
        )

        return grad_input, grad_weight


conv1d = Conv1dCudaFun.apply


# Similar approach for conv2d:
# Adjust MAX_KERNEL_HEIGHT, MAX_KERNEL_WIDTH, and MAX_IN_CHANNELS as needed.
MAX_KERNEL_HEIGHT = 7
MAX_KERNEL_WIDTH = 7
MAX_IN_CHANNELS = 64

@cuda.jit
def _tensor_conv2d(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    input: Storage,
    input_shape: Shape,
    input_strides: Strides,
    weight: Storage,
    weight_shape: Shape,
    weight_strides: Strides,
    reverse: bool,
) -> None:
    """
    CUDA kernel for 2D convolution.
    """
    batch_, out_channels, out_h, out_w = out_shape
    batch, in_channels, h, w = input_shape
    out_c_wt, in_c_wt, kh, kw = weight_shape

    if not (batch == batch_ and in_channels == in_c_wt and out_channels == out_c_wt):
        return

    idx = cuda.grid(1)
    if idx >= out_size:
        return

    # Compute (b, oc, oh, ow) from idx
    b = idx // (out_channels * out_h * out_w)
    remainder = idx % (out_channels * out_h * out_w)
    oc = remainder // (out_h * out_w)
    remainder = remainder % (out_h * out_w)
    oh = remainder // out_w
    ow = remainder % out_w

    acc_val = 0.0
    for ic in range(in_channels):
        for kh_idx in range(kh):
            for kw_idx in range(kw):
                if reverse:
                    in_h = oh - kh + 1 + kh_idx
                    in_w = ow - kw + 1 + kw_idx
                else:
                    in_h = oh + kh_idx
                    in_w = ow + kw_idx

                if 0 <= in_h < h and 0 <= in_w < w:
                    input_pos = b * input_strides[0] + ic * input_strides[1] + in_h * input_strides[2] + in_w * input_strides[3]
                    weight_pos = oc * weight_strides[0] + ic * weight_strides[1] + kh_idx * weight_strides[2] + kw_idx * weight_strides[3]
                    acc_val += input[input_pos] * weight[weight_pos]

    out_pos = b * out_strides[0] + oc * out_strides[1] + oh * out_strides[2] + ow * out_strides[3]
    out[out_pos] = acc_val

tensor_conv2d = jit(_tensor_conv2d)


class Conv2dCudaFun(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        """
        Forward pass for CUDA-based 2D convolution.
        """
        ctx.save_for_backward(input, weight)
        batch, in_channels, h, w = input.shape
        out_channels, in_c_wt, kh, kw = weight.shape
        assert in_channels == in_c_wt

        out_h = h - kh + 1
        out_w = w - kw + 1
        out = input.zeros((batch, out_channels, out_h, out_w))

        threadsperblock = 128
        blockspergrid = (out.size + threadsperblock - 1) // threadsperblock

        tensor_conv2d[blockspergrid, threadsperblock](
            out._tensor._storage, out.shape, out._tensor._strides, out.size,
            input._tensor._storage, input.shape, input._tensor._strides,
            weight._tensor._storage, weight.shape, weight._tensor._strides,
            False,
        )
        return out

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Backward pass for CUDA-based 2D convolution.
        """
        input, weight = ctx.saved_values
        batch, in_channels, h, w = input.shape
        out_channels, in_c_wt, kh, kw = weight.shape
        _, _, out_h, out_w = grad_output.shape

        # grad_weight
        grad_weight = grad_output.zeros((in_channels, out_channels, kh, kw))
        new_input = input.permute(1, 0, 2, 3)
        new_grad_out = grad_output.permute(1, 0, 2, 3)

        threadsperblock = 128
        blockspergrid = (grad_weight.size + threadsperblock - 1) // threadsperblock
        tensor_conv2d[blockspergrid, threadsperblock](
            grad_weight._tensor._storage, grad_weight.shape, grad_weight._tensor._strides, grad_weight.size,
            new_input._tensor._storage, new_input.shape, new_input._tensor._strides,
            new_grad_out._tensor._storage, new_grad_out.shape, new_grad_out._tensor._strides,
            False,
        )
        grad_weight = grad_weight.permute(1, 0, 2, 3)

        # grad_input
        grad_input = input.zeros((batch, in_channels, h, w))
        new_weight = weight.permute(1, 0, 2, 3)
        blockspergrid = (grad_input.size + threadsperblock - 1) // threadsperblock
        tensor_conv2d[blockspergrid, threadsperblock](
            grad_input._tensor._storage, grad_input.shape, grad_input._tensor._strides, grad_input.size,
            grad_output._tensor._storage, grad_output.shape, grad_output._tensor._strides,
            new_weight._tensor._storage, new_weight.shape, new_weight._tensor._strides,
            True,
        )

        return grad_input, grad_weight

conv2d = Conv2dCudaFun.apply
