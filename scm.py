"""
Common Structural Causal Model (SCM)
====================================

Nominal exogenous variables:
U1, U2, U3, U4, Uy ~ N(0, 1), independently

Structural equations:
X1 = U1
X2 = 0.8 * X1 + U2
X3 = -0.6 * X1 + U3
X4 = 0.7 * X2 - 0.5 * X3 + U4

Y = 0.7 * X2 - 0.5 * X3 + 0.6 * X4
    + 0.8 * tanh(0.5 * X1 * X4)
    + 0.3 * Uy

- The SCM is fixed across all environments.
- OOD shifts affect only U1, U2, U3, and U4.
- Uy remains N(0, 1).
"""


from typing import Any, Tuple

import numpy as np

try:
    import torch
except ImportError:
    torch = None


# ---------------------------------------------------------------------
# Fixed SCM coefficients
# ---------------------------------------------------------------------

C_X1_TO_X2 = 0.8
C_X1_TO_X3 = -0.6
C_X2_TO_X4 = 0.7
C_X3_TO_X4 = -0.5

C_X2_TO_Y = 0.7
C_X3_TO_Y = -0.5
C_X4_TO_Y = 0.6
C_INTERACTION_TO_Y = 0.8
C_INTERACTION_SCALE = 0.5
C_UY_TO_Y = 0.3


def _is_torch_tensor(x: Any) -> bool:
    return torch is not None and torch.is_tensor(x)


def _tanh(x: Any) -> Any:
    if _is_torch_tensor(x):
        return torch.tanh(x)
    return np.tanh(x)


def _stack_features(x1: Any, x2: Any, x3: Any, x4: Any) -> Any:
    if _is_torch_tensor(x1):
        return torch.stack((x1, x2, x3, x4), dim=-1)
    return np.stack((x1, x2, x3, x4), axis=-1)


def _ensure_column(y: Any) -> Any:
    if _is_torch_tensor(y):
        if y.ndim == 0:
            return y.reshape(1, 1)
        if y.ndim == 1:
            return y.unsqueeze(-1)
        return y

    y = np.asarray(y)
    if y.ndim == 0:
        return y.reshape(1, 1)
    if y.ndim == 1:
        return y[:, None]
    return y


def target_mechanism(
    X1: Any,
    X2: Any,
    X3: Any,
    X4: Any,
) -> Any:

    """
    Deterministic part of the target mechanism E[Y | X].

    Returns:
        f(X1, X2, X3, X4), before adding 0.3 * Uy
    """

    return (
        C_X2_TO_Y * X2
        + C_X3_TO_Y * X3
        + C_X4_TO_Y * X4
        + C_INTERACTION_TO_Y
        * _tanh(C_INTERACTION_SCALE * X1 * X4)
    )


def forward_scm(
    U1: Any,
    U2: Any,
    U3: Any,
    U4: Any,
    Uy: Any,
) -> Tuple[Any, Any]:

    """
    Forward structural map:
        (U1, U2, U3, U4, Uy) -> (X, y)

    Inputs should be one-dim arrays/tensors of equal length N

    Returns:
        X: shape (N, 4)
        y: shape (N, 1)
    """

    X1 = U1
    X2 = C_X1_TO_X2 * X1 + U2
    X3 = C_X1_TO_X3 * X1 + U3
    X4 = C_X2_TO_X4 * X2 + C_X3_TO_X4 * X3 + U4

    Y = target_mechanism(X1, X2, X3, X4) + C_UY_TO_Y * Uy

    X = _stack_features(X1, X2, X3, X4)
    y = _ensure_column(Y)

    return X, y


def inverse_scm(
    X: Any,
    y: Any,
) -> Tuple[Any, Any, Any, Any, Any]:

    """
    Analytic inverse structural map:
        (X, y) -> (U1, U2, U3, U4, Uy)

    Args:
        X: shape (N, 4)
        y: shape (N, 1) or (N,)

    Returns:
        U1, U2, U3, U4, Uy
        Each has shape (N,)
    """

    X1 = X[..., 0]
    X2 = X[..., 1]
    X3 = X[..., 2]
    X4 = X[..., 3]

    if _is_torch_tensor(y):
        Y = y.squeeze(-1) if y.ndim > 1 else y
    else:
        y = np.asarray(y)
        Y = np.squeeze(y, axis=-1) if y.ndim > 1 else y

    U1 = X1
    U2 = X2 - C_X1_TO_X2 * X1
    U3 = X3 - C_X1_TO_X3 * X1
    U4 = X4 - C_X2_TO_X4 * X2 - C_X3_TO_X4 * X3

    deterministic_y = target_mechanism(X1, X2, X3, X4)
    Uy = (Y - deterministic_y) / C_UY_TO_Y

    return U1, U2, U3, U4, Uy


if __name__ == "__main__":
    # NumPy sanity check: forward -> inverse must recover U exactly
    rng = np.random.default_rng(0)
    n = 128

    U = [rng.normal(size=n) for _ in range(5)]

    X, y = forward_scm(*U)
    U_hat = inverse_scm(X, y)

    assert X.shape == (n, 4)
    assert y.shape == (n, 1)

    for u, u_hat in zip(U, U_hat):
        assert np.allclose(u, u_hat)

    print("SCM sanity check passed.")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
