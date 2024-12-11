from typing import Tuple

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand, tensor

# Reduce operation for max provided by FastOps.
max_reduce = FastOps.reduce(operators.max, -float("inf"))


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: (kernel_height, kernel_width) of pooling

    Returns:
    -------
        Tensor of size (batch x channel x new_height x new_width x kernel_height*kernel_width),
        along with new_height and new_width.

    """
    input = input.contiguous()  # Ensure input is contiguous before view

    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0, "Height must be divisible by kernel height."
    assert width % kw == 0, "Width must be divisible by kernel width."

    new_height = height // kh
    new_width = width // kw

    # Reshape the tensor to extract pooling regions
    # shape: (batch, channel, new_height, kh, new_width, kw)
    reshaped = input.view(batch, channel, new_height, kh, new_width, kw)
    # permute to (batch, channel, new_height, new_width, kh, kw)
    permuted = reshaped.permute(0, 1, 2, 4, 3, 5)
    # flatten kh * kw dimension
    tiled = permuted.contiguous().view(batch, channel, new_height, new_width, kh * kw)

    return tiled, new_height, new_width


def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Perform 2D average pooling using the tile function.

    Args:
    ----
        input: Tensor (batch, channel, height, width)
        kernel: (kernel_height, kernel_width)

    Returns:
    -------
        (batch, channel, new_height, new_width) after average pooling.

    """
    batch, channel, _, _ = input.shape
    tiled, new_height, new_width = tile(input, kernel)
    # mean over the last dimension (kernel)
    return tiled.mean(dim=4).view(batch, channel, new_height, new_width)


def argmax(input: Tensor, dim: int) -> Tensor:
    """Compute the argmax of the input tensor along a specified dimension as a 1-hot tensor."""
    max_vals = max_reduce(input, dim)
    shape = list(input.shape)
    shape[dim] = 1
    max_vals = max_vals.view(*shape)
    return input == max_vals


class Max(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, dim: Tensor) -> Tensor:
        """Forward pass for Max function.

        Args:
        ----
            ctx: Context for saving values needed in backward pass
            t1: Input tensor
            dim: Dimension along which to compute maximum

        Returns:
        -------
            Tensor containing maximum values along specified dimension

        """
        d = int(dim.item())
        ctx.save_for_backward(t1, d)
        out = max_reduce(t1, d)

        # Now explicitly remove the reduced dimension
        shape = list(t1.shape)
        shape.pop(d)  # Remove the dimension 'd' from the shape
        out = out.view(*shape)  # Reshape out to remove the reduced dimension

        return out

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Backward pass for Max function.

        Args:
        ----
            ctx: Context containing saved tensors from forward pass
            grad_output: Gradient of the loss with respect to the output

        Returns:
        -------
            Tuple of (gradient with respect to input, gradient with respect to dimension)

        """
        t1, d = ctx.saved_values
        original_shape = list(t1.shape)
        
        # Insert a 1 dimension at 'd' to match original shape except at dimension d
        grad_output = grad_output.view(*original_shape[:d], 1, *original_shape[d+1:])
        
        # Trigger broadcasting by arithmetic with a zeros tensor of t1's shape
        # minitorch arithmetic supports broadcasting, so adding zeros with shape of t1
        # will broadcast grad_output to match t1.
        zeros_for_broadcast = t1.zeros(t1.shape)
        grad_output = grad_output + zeros_for_broadcast

        grad_input = argmax(t1, d) * grad_output

        return grad_input, 0.0


def max(input: Tensor, dim: int) -> Tensor:
    """Compute the maximum value of the input tensor along a specified dimension."""
    dim_tensor = tensor([dim], requires_grad=False)
    return Max.apply(input, dim_tensor)


def softmax(input: Tensor, dim: int) -> Tensor:
    """Compute the softmax along a specified dimension."""
    max_vals = max(input, dim)
    max_shape = list(input.shape)
    max_shape[dim] = 1
    max_vals = max_vals.view(*max_shape)
    exp_values = (input - max_vals).exp()
    sum_exp = exp_values.sum(dim=dim)
    sum_exp = sum_exp.view(*max_shape)
    return exp_values / sum_exp


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    """Compute the log-softmax along a specified dimension."""
    max_vals = max(input, dim)
    max_shape = list(input.shape)
    max_shape[dim] = 1
    max_vals = max_vals.view(*max_shape)

    log_sum_exp = (input - max_vals).exp().sum(dim=dim).log()
    log_sum_exp = log_sum_exp.view(*max_shape)
    return input - max_vals - log_sum_exp


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Perform 2D max pooling."""
    batch, channel, _, _ = input.shape
    tiled, new_height, new_width = tile(input, kernel)
    max_pooled = max(tiled, 4)
    return max_pooled.view(batch, channel, new_height, new_width)


def dropout(input: Tensor, prob: float = 0.5, ignore: bool = False) -> Tensor:
    """Apply dropout to the input tensor."""
    if ignore:
        return input
    if prob == 1.0:
        return input * 0.0
    if prob == 0.0:
        return input

    mask = rand(input.shape) > prob
    return input * mask / (1.0 - prob)
