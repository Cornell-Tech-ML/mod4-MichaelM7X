from typing import Tuple

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand, tensor


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off

def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0, "Height must be divisible by kernel height."
    assert width % kw == 0, "Width must be divisible by kernel width."

    new_height = height // kh
    new_width = width // kw

    # Reshape the tensor to extract pooling regions
    reshaped = input.view(batch, channel, new_height, kh, new_width, kw)
    permuted = reshaped.permute(0, 1, 2, 4, 3, 5)
    tiled = permuted.contiguous().view(batch, channel, new_height, new_width, kh * kw)

    return tiled, new_height, new_width

def max(input: Tensor, axis: int) -> Tensor:
    """Compute the max along a specified axis.

    Args:
    ----
        input: Input tensor.
        axis: Axis along which to compute the max.

    Returns:
    -------
        Tensor containing the max values along the specified axis.

    """
    return input.max(axis=axis, keepdims=False)

def softmax(input: Tensor, axis: int) -> Tensor:
    """Compute the softmax function.

    Args:
    ----
        input: Input tensor.
        axis: Axis along which to compute the softmax.

    Returns:
    -------
        Tensor containing the softmax values.

    """
    exp_values = (input - input.max(axis=axis, keepdims=True)).exp()
    return exp_values / exp_values.sum(axis=axis, keepdims=True)

def logsoftmax(input: Tensor, axis: int) -> Tensor:
    """Compute the log of the softmax function.

    Args:
    ----
        input: Input tensor.
        axis: Axis along which to compute the logsoftmax.

    Returns:
    -------
        Tensor containing the logsoftmax values.

    """
    max_values = input.max(axis=axis, keepdims=True)
    log_sum_exp = (input - max_values).exp().sum(axis=axis, keepdims=True).log()
    return input - max_values - log_sum_exp

def dropout(input: Tensor, p: float = 0.5, training: bool = True) -> Tensor:
    """Apply dropout to the input tensor.

    Args:
    ----
        input: Input tensor.
        p: Probability of dropping a unit.
        training: Whether to apply dropout (default: True).

    Returns:
    -------
        Tensor with dropout applied during training or unchanged during evaluation.

    """
    if not training:
        return input

    mask = rand(input.shape) > p
    return input * mask / (1.0 - p)
