"""
Reversed Huber penalty.
"""

import cvxpy.utilities as u
import cvxpy.lin_ops.lin_utils as lu
from cvxpy.expressions.constants.parameter import Parameter
from cvxpy.atoms.elementwise.elementwise import Elementwise
from cvxpy.atoms.elementwise.abs import abs
import numpy as np
from .power import power
from fractions import Fraction

class reversed_huber(Elementwise):
    """The reversed Huber function

    ReversedHuber(x) = |x| for |x| <= 1
                       (|x|^2 + 1)/2 else

    Parameters
    ----------
    x : Expression
        A CVXPY expression.
    """

    def __init__(self, x):
        super(reversed_huber, self).__init__(x)

    @Elementwise.numpy_numeric
    def numeric(self, values):
        """Returns the huber function applied elementwise to x.
        """
        x = values[0]
        output = np.zeros(x.shape)
        for row in range(x.shape[0]):
            for col in range(x.shape[1]):
                if np.abs(x[row, col]) <= 1.0:
                    output[row, col] = np.abs(x[row, col])
                else:
                    output[row, col] = 0.5 * (np.abs(x[row, col])*np.abs(x[row, col]) + 1.0)
        return output

    def sign_from_args(self):
        """Always positive.
        """
        return u.Sign.POSITIVE

    def func_curvature(self):
        """Default curvature.
        """
        return u.Curvature.CONVEX

    def monotonicity(self):
        """Increasing for positive arg, decreasing for negative.
        """
        return [u.monotonicity.SIGNED]
"""
    @staticmethod
    def graph_implementation(arg_objs, size, data=None):
        Reduces the atom to an affine expression and list of constraints.

        minimize n^2 + 2M|s|
        subject to s + n = x

        Parameters
        ----------
        arg_objs : list
            LinExpr for each argument.
        size : tuple
            The size of the resulting expression.
        data :
            Additional data required by the atom.

        Returns
        -------
        tuple
            (LinOp for objective, list of constraints)
   
        M = data[0]
        x = arg_objs[0]
        n = lu.create_var(size)
        s = lu.create_var(size)
        two = lu.create_const(2, (1, 1))
        if isinstance(M, Parameter):
            M = lu.create_param(M, (1, 1))
        else: # M is constant.
            M = lu.create_const(M.value, (1, 1))

        # n**2 + 2*M*|s|
        n2, constr_sq = power.graph_implementation([n], size, (2, (Fraction(1, 2), Fraction(1, 2))))
        abs_s, constr_abs = abs.graph_implementation([s], size)
        M_abs_s = lu.mul_expr(M, abs_s, size)
        obj = lu.sum_expr([n2, lu.mul_expr(two, M_abs_s, size)])
        # x == s + n
        constraints = constr_sq + constr_abs
        constraints.append(lu.create_eq(x, lu.sum_expr([n, s])))
        return (obj, constraints)
"""
