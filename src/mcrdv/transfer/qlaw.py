"""Transfer layer based on an analytic Q-law estimate."""

import math
from typing import NamedTuple

import numpy as np
from numba import njit

from ..kernel import KERNEL, TransferKernel
from ..orbit import mee_at, mee_max_rates, pack
from ..planner import TransferLayer


class QlawContext(NamedTuple):
    # Packed MEE states of the problem instance, shape `(n, 6)`.
    table: np.ndarray
    # Thrust acceleration of the chaser [m/s^2].
    thrust: float
    # Dimensionless factor scaling the estimate.
    factor: float
    # Cost factor for the transfer time.
    factor_time: float
    # Cost factor for the transfer fuel.
    factor_fuel: float


@njit(**KERNEL)
def _qlaw_transfer_evaluate(kernel, i, j, t):
    context = kernel.context

    sa, sf, sg, sh, sk, sl = mee_at(context.table, i, t)
    da, df, dg, dh, dk, _ = mee_at(context.table, j, t)
    ra, rf, rg, rh, rk, _ = mee_max_rates(sa, sf, sg, sh, sk, sl)

    xa = (sa - da) / ra
    xf = (sf - df) / rf
    xg = (sg - dg) / rg
    xh = (sh - dh) / rh
    xk = (sk - dk) / rk

    dv = context.factor * math.sqrt(
        xa * xa + xf * xf + xg * xg + xh * xh + xk * xk
    )
    dt = dv / context.thrust
    dc = context.factor_time * dt + context.factor_fuel * dv

    return dt, dc


class QlawTransfer(TransferLayer):
    """Transfer layer built on the Q-law estimate.

    Parameters
    ----------
    thrust : float
        Thrust acceleration of a chaser [m/s^2].
    factor : float
        Dimensionless factor scaling the estimate.
    factor_time : float
        Cost factor for the transfer time.
    factor_fuel : float
        Cost factor for the transfer fuel.
    **kwargs : dict
        Keyword arguments passed to the :class:`TransferLayer` class.
    """

    def __init__(
        self,
        *,
        thrust: float = 3e-3,
        factor: float = 1.5,
        factor_time: float = 0.0,
        factor_fuel: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.thrust = float(thrust)
        self.factor = float(factor)
        self.factor_time = float(factor_time)
        self.factor_fuel = float(factor_fuel)

        if self.thrust <= 0.0:
            raise ValueError("thrust must be positive")
        if self.factor <= 0.0:
            raise ValueError("factor must be positive")
        if self.factor_time < 0.0:
            raise ValueError("factor_time must be non-negative")
        if self.factor_fuel < 0.0:
            raise ValueError("factor_fuel must be non-negative")

    def _kernel(self) -> TransferKernel:
        context = QlawContext(
            pack(self.problem.states),
            self.thrust,
            self.factor,
            self.factor_time,
            self.factor_fuel,
        )
        return TransferKernel(context, _qlaw_transfer_evaluate)
