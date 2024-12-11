from __future__ import annotations

from typing import TYPE_CHECKING

import minitorch

from . import operators
from .autodiff import Context

if TYPE_CHECKING:
    from typing import Tuple

    from .scalar import Scalar, ScalarLike


def wrap_tuple(x: float | Tuple[float, ...]) -> Tuple[float, ...]:
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


class ScalarFunction:
    """A wrapper for a mathematical function that processes and produces
    Scalar variables.

    This is a static class and is never instantiated. We use `class`
    here to group together the `forward` and `backward` code.
    """

    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> Tuple[float, ...]:
        return wrap_tuple(cls.backward(ctx, d_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: float) -> float:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: ScalarLike) -> Scalar:
        """Converts input values to standard numbers to call forward.
        Then creates a new Scalar from the result with a new history.

        Args:
        ----
            vals (ScalarLike): The input values.

        Returns:
        -------
            Scalar: The result of the scalar function.

        """
        raw_vals = []
        scalars = []
        for v in vals:
            if isinstance(v, minitorch.scalar.Scalar):
                scalars.append(v)
                raw_vals.append(v.data)
            else:
                scalars.append(minitorch.scalar.Scalar(v))
                raw_vals.append(v)

        # Create the context.
        ctx = Context(False)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        assert isinstance(c, float), "Expected return type float got %s" % (type(c))

        # Create a new variable from the result with a new history.
        back = minitorch.scalar.ScalarHistory(cls, ctx, scalars)
        return minitorch.scalar.Scalar(c, back)


# Examples
class Add(ScalarFunction):
    """Addition function $f(x, y) = x + y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Compute the sum of two floating-point numbers.

        Args:
        ----
            ctx (Context): The context class used to store information.
            a (float): The first number.
            b (float): The second number.

        Returns:
        -------
            float: The sum of a and b.

        """
        return a + b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Compute the gradient of the addition function.

        Args:
        ----
            ctx (Context): The context (unused).
            d_output (float): The derivative of the previous output.

        Returns:
        -------
            Tuple of partial gradients for both inputs

        """
        return d_output, d_output


class Log(ScalarFunction):
    """Log function $f(x) = log(x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Compute the natural logarithm of a floating-point number.
        Save the input for backward computation.

        Args:
        ----
            ctx (Context): The context class used to store information.
            a (float): The number to compute the logarithm of.

        Returns:
        -------
            The natural logarithm of a.

        """
        ctx.save_for_backward(a)
        return operators.log(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute the gradient of the log function.

        Args:
        ----
            ctx (Context): The context with saved values.
            d_output (float): The derivative of the previous output.

        Returns:
        -------
            The gradient with respect to the input.

        """
        (a,) = ctx.saved_values
        return operators.log_back(a, d_output)


class Mul(ScalarFunction):
    """Multiplication function $f(x, y) = x * y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Multiply two floating-point numbers.
        Save the two numbers for backward computation.

        Args:
        ----
            ctx (Context): The context class is used to store information.
            a (float): The first number.
            b (float): The second number.

        Returns:
        -------
            float: The product of a and b.

        """
        ctx.save_for_backward(a, b)
        c = a * b
        return c

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Compute the gradient of the multiplication function.

        Args:
        ----
            ctx (Context): The context with saved values.
            d_output (float): The derivative of the previous output.

        Returns:
        -------
            Tuple of partial gradients for both inputs.

        """
        a, b = ctx.saved_values
        return b * d_output, a * d_output


class Inv(ScalarFunction):
    """Inverse function $f(x) = 1 / x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Compute the inverse of a floating-point number.
        Save the input for backward computation.

        Args:
        ----
            ctx (Context): The context class is used to store information.
            a (float): The number to compute the inverse of.

        Returns:
        -------
            float: The inverse of a.

        """
        ctx.save_for_backward(a)
        return operators.inv(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute the gradient of the inverse function.

        Args:
        ----
            ctx (Context): The context with saved values.
            d_output (float): The derivative of the previous output.

        Returns:
        -------
            The gradient with respect to the input.

        """
        (a,) = ctx.saved_values
        return operators.inv_back(a, d_output)


class Neg(ScalarFunction):
    """Negation function $f(x) = -x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Compute the negation of a floating-point number.

        Args:
        ----
            ctx (Context): The context class is used to store information.
            a (float): The number to compute the negation of.

        Returns:
        -------
            float: The negation of a.

        """
        return -a

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute the gradient of the negation function.

        Args:
        ----
            ctx (Context): The context with saved values (unused in this case).
            d_output (float): The derivative of the previous output.

        Returns:
        -------
            The gradient with respect to the input.

        """
        return -d_output


class Sigmoid(ScalarFunction):
    """Sigmoid function $f(x) = 1.0/(1.0 + e^{-x}) if x >=0 else e^x/(1.0 + e^{x})$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Compute the sigmoid of a floating-point number.
        Save the input for backward computation.

        Args:
        ----
            ctx (Context): The context class is used to store information.
            a (float): The number to compute the sigmoid of.

        Returns:
        -------
            float: The sigmoid of a.

        """
        out = operators.sigmoid(a)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute the gradient of the sigmoid function.

        Args:
        ----
            ctx (Context): The context with saved values.
            d_output (float): The derivative of the previous output.

        Returns:
        -------
            The gradient with respect to the input.

        """
        sigma: float = ctx.saved_values[0]
        return sigma * (1.0 - sigma) * d_output


class ReLU(ScalarFunction):
    """ReLU function $f(x) = max(0, x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Compute the ReLU of a floating-point number.
        Save the input for backward computation.

        Args:
        ----
            ctx (Context): The context class is used to store information.
            a (float): The number to compute the ReLU of.

        Returns:
        -------
            float: The ReLU of a.

        """
        ctx.save_for_backward(a)
        return operators.relu(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute the gradient of the ReLU function.

        Args:
        ----
            ctx (Context): The context with saved values.
            d_output (float): The derivative of the previous output.

        Returns:
        -------
            The gradient with respect to the input.

        """
        (a,) = ctx.saved_values
        return operators.relu_back(a, d_output)


class Exp(ScalarFunction):
    """Exp function $f(x) = e^x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Compute the exponential of a floating-point number.
        Save the input for backward computation.

        Args:
        ----
            ctx (Context): The context class is used to store information.
            a (float): The number to compute the exponential of.

        Returns:
        -------
            float: The exponential of a.

        """
        out = operators.exp(a)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute the gradient of the exponential function.

        Args:
        ----
            ctx (Context): The context with saved values.
            d_output (float): The derivative of the previous output.

        Returns:
        -------
            The gradient with respect to the input.

        """
        out: float = ctx.saved_values[0]
        return d_output * out


class LT(ScalarFunction):
    """Less than function $f(x, y) = 1 if x < y else 0$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Check if the first number is less than the second number.

        Args:
        ----
            ctx (Context): The context class is used to store information.
            a (float): The first number.
            b (float): The second number.

        Returns:
        -------
            float: 1.0 if a is less than b, 0.0 otherwise.

        """
        return 1.0 if a < b else 0.0

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Compute the gradient of the less than function.

        Args:
        ----
            ctx (Context): The context (unused).
            d_output (float): The derivative of the previous output.

        Returns:
        -------
            Tuple of gradients for both inputs, which are always zero.

        """
        return 0.0, 0.0  # No gradient for LT


class EQ(ScalarFunction):
    """Equality function $f(x, y) = 1 if x == y else 0$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Check if two floating-point numbers are equal.

        Args:
        ----
            ctx (Context): The context class is used to store information.
            a (float): The first number.
            b (float): The second number.

        Returns:
        -------
            float: 1.0 if a equals b, 0.0 otherwise.

        """
        return 1.0 if a == b else 0.0

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Compute the gradient of the equality function.

        Args:
        ----
            ctx (Context): The context (unused).
            d_output (float): The derivative of the previous output.

        Returns:
        -------
            Tuple of gradients for both inputs, which are always zero.

        """
        return 0.0, 0.0
