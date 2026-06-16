# ----------------------------------------------------------------------------
# Copyright (c) 2013--, scikit-bio development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE.txt, distributed with this software.
# ----------------------------------------------------------------------------

r"""OptSpace Matrix Completion Algorithm.

This module provides the OptSpace algorithm for low-rank matrix completion
from partially observed entries. It is used by the Robust PCA (RPCA)
ordination method.

The algorithm minimizes the objective:

.. math::

    \min_{U, V, S} \|P_\Omega(Y - USV^T)\|_F^2

where :math:`Y` is the observed matrix, :math:`U` and :math:`V` are the
left and right singular vector matrices, :math:`S` is the diagonal matrix
of singular values, and :math:`P_\Omega` projects onto the observed entries.

References
----------
.. [1] Keshavan RH, Montanari A, Oh S. 2010. Matrix Completion from a
   Few Entries. IEEE Transactions on Information Theory 56(6):2980-2998.

.. [2] Martino C, Morton JT, Marotz CA, Thompson LR, Tripathi A,
   Knight R, Zengler K. 2019. A Novel Sparse Compositional Technique
   Reveals Microbial Perturbations. mSystems 4:e00016-19.

"""

import numpy as np
from scipy.sparse.linalg import svds
from scipy.linalg import svd
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import cg


def trim(X, observed_mask, m, n, n_observed):
    """Trim over-represented rows and columns.

    Any row or column with more than half the average observed entries per
    row or column respectively is set to zero per Keshavan et al. (2010).
    This makes the low-rank structure more pronounced.)"""

    n_observed_rows = np.sum(observed_mask, axis=1)
    n_observed_cols = np.sum(observed_mask, axis=0)

    row_threshold = 2 * n_observed / m
    col_threshold = 2 * n_observed / n

    valid_rows = n_observed_rows <= row_threshold
    valid_cols = n_observed_cols <= col_threshold

    trim_mask = np.outer(valid_rows, valid_cols)

    return np.where(trim_mask & observed_mask, X, 0.0)


def _svd_sort(U, S, V):
    """Sort SVD components by descending singular values.

    Also applies sign correction for reproducibility.

    Parameters
    ----------
    U : ndarray
        Left singular vectors.
    S : ndarray
        Singular values.
    V : ndarray
        Right singular vectors.

    Returns
    -------
    tuple
        Sorted (U, S, V) with sign correction applied.
    """
    # Sort by descending singular values
    idx = np.argsort(S)[::-1]
    S = S[idx]
    U = U[:, idx]
    V = V[:, idx]

    # Apply sign correction for reproducibility
    # Make the largest element in each column of U positive
    for i in range(U.shape[1]):
        if np.abs(U[:, i].min()) > np.abs(U[:, i].max()):
            U[:, i] *= -1
            V[:, i] *= -1

    return U, S, V


def estimate_rank(S, epsilon):
    """Estimate rank r\\hat by minimizing the cost function of singular
    values from Keshavan et al. (2010).
    """

    # Indices i = 1, 2, ... , len(S) - 1
    # The last element is excluded because S[i] would be out of bounds
    i = np.arange(1, len(S))  # 1 , ... , k-1
    cost = (S[0] * np.sqrt(i / epsilon) + S[i]) / S[i - 1]
    # print(f"Index {i}") # Delete
    # print(f"S[i]: {S[i]}")
    # print(f"S[i - 1]: {S[i - 1]}")
    # print(f"epsilon: {epsilon}")
    # print(f"Cost: {cost}") # Delete

    return i[np.argmin(cost)]


def solve_S(U, V, M_observed, observed_mask):
    """Compute optimal S given U and V.

    Solves the least squares problem to find the optimal S
    that minimizes reconstruction difference on the observed entries:

    arg min_S ||P_\\Omega(U V S^T - M_observed)||_F^2

    where P_\\Omega is the projection onto the observed entries.
    Since only the observed entries are considered, the problem
    can be vectorized and solved efficiently using least squares:

    arg min_S \\sum_{(i,j) \\in \\Omega} (U[i] S V[j]^T - M_observed[i, j])^2
    """

    r = U.shape[1]

    n_observed = np.sum(observed_mask)

    rows, cols = np.where(observed_mask)

    A = np.einsum("ni,nj->nij", U[rows], V[cols]).reshape(len(rows), -1)

    b = M_observed[rows, cols]

    s = np.linalg.lstsq(A, b, rcond=None)[0]

    S = s.reshape(r, r)

    return S


def residual(U, V, S, M_obs):
    """Compute the residual R = P_Ω(U S V^T - M_obs)."""
    R = U @ S @ V.T - M_obs
    R = np.nan_to_num(R, nan=0.0)

    return R

    # (U_i S V_j^T) evaluated only on observed entries
    rows, cols = np.where(observed_mask)
    Ui = U[rows]  # (n_obs, r)
    Vj = V[cols]  # (n_obs, r)

    R = np.einsum("ir,rs,is->i", Ui, S, Vj)  # (n_obs,)
    R -= M_obs[rows, cols]  # (n_obs,)

    return R


def jacobian_action(U, V, S, dU, dV, observed_mask):
    """Compute J(dU, dV).

    The Jacobian action is a mapping from the tangent space of (U, V) to
    the space of observed entries. It computes how changes in U and V (dU, dV)
    affect the observed reconstruction error.

    J(dU, dV) = dU S V^T + U S dV^T"""

    W = dU @ S @ V.T
    W += U @ S @ dV.T

    return W * observed_mask


def jacobian_adjoint(U, V, S, W):
    """Compute J*(W).

    The Jacobian adjoint is defined with respect to the inner product by

    <J(dU, dV), W> = <(dU, dV), J*(W)>

    It defines a mapping from the space of observed entries back to the
    tangent space of (U, V). With J known,

    J*(W) = (W V S^T, W^T U S)

    This is projected back to the tangent space of (U, V):

    J*(W)_U = (I - U U^T) W V S^T
    J*(W)_V = (I - V V^T) W^T U S"""

    dU = W @ V @ S.T
    dV = W.T @ U @ S

    dU -= U @ (U.T @ dU)
    dV -= V @ (V.T @ dV)

    return dU, dV


def gauss_newton_hvp(U, V, S, dU, dV, observed_mask, damping=0.0):
    """Apply (J*J + λI) to a tangent vector.

    The Hessian-Vector Product is used in the Conjugate Gradient
    solver to compute the Gauss-Newton step. This avoids explicitly
    forming the Hessian matrix."""

    W = jacobian_action(U, V, S, dU, dV, observed_mask)
    Hu, Hv = jacobian_adjoint(U, V, S, W)

    return Hu, Hv


def pack(dU, dV):
    """Pack dU and dV to a single vector η"""
    return np.concatenate([dU.ravel(), dV.ravel()])


def unpack(x, U_shape, V_shape):
    """Unpack the vector η back to dU and dV."""
    nu = np.prod(U_shape)

    dU = x[:nu].reshape(U_shape)
    dV = x[nu:].reshape(V_shape)

    return dU, dV


def solve_gauss_newton_step(U, V, S, M_observed, observed_mask, R, damping=1e-1):
    """Solve (J*J+λI)η=-J*R.

    The Gauss-Newton step is the vector η = (dU, dV), where dU and dV are
    tangent vectors in their respective Grassmann manifolds. The step is computed
    by solving the linear system using the conjugate gradient method."""

    # Right-hand side

    bU, bV = jacobian_adjoint(U, V, S, R)

    b = pack(-bU, -bV)

    nvars = b.size

    # Matvec is passed to the LinearOperator, which computes (J*J + λI)η
    # for a given η = (dU, dV)

    def matvec(x):
        dU, dV = unpack(x, U.shape, V.shape)

        Hu, Hv = gauss_newton_hvp(U, V, S, dU, dV, observed_mask, damping=damping)

        return pack(Hu, Hv)

    A = LinearOperator((nvars, nvars), matvec=matvec, dtype=U.dtype)

    # Solve the system by Conjugate Gradient

    step, info = cg(A, b, rtol=1e-6, maxiter=50)

    dU, dV = unpack(step, U.shape, V.shape)

    # dU -= U @ (U.T @ dU)
    # dV -= V @ (V.T @ dV)

    return dU, dV


def retract_grassmann(X, dX):
    """Retract the updated matrix X + dX back to the Grassmann manifold."""
    Q, _ = np.linalg.qr(X + dX)
    return Q[:, : X.shape[1]]


class OptSpace:
    r"""Matrix completion using the OptSpace algorithm.

    OptSpace is an algorithm for recovering a low-rank matrix from a
    subset of observed entries. It uses gradient descent on the
    Grassmann manifold to find the optimal low-rank approximation.

    Parameters
    ----------
    n_components : int, optional
        The rank of the matrix to recover. Default is 3.
    max_iterations : int, optional
        Maximum number of iterations. Default is 5.
    tol : float, optional
        Convergence tolerance. Default is 1e-5.

    Attributes
    ----------
    U : ndarray
        Left singular vectors after fitting.
    S : ndarray
        Singular values (as diagonal matrix) after fitting.
    V : ndarray
        Right singular vectors after fitting.
    converged : bool
        Whether the algorithm converged.

    See Also
    --------
    rpca

    Notes
    -----
    The algorithm proceeds as follows:

    1. Initialize U, V using trimmed SVD of the observed matrix
    2. Iteratively:
       a. Compute optimal S given current U, V
       b. Compute gradients with respect to U, V
       c. Update U, V using gradient descent with line search
       d. Project U, V back to Grassmann manifold

    References
    ----------
    .. [1] Keshavan RH, Montanari A, Oh S. 2010. Matrix Completion from a
       Few Entries. IEEE Transactions on Information Theory 56(6):2980-2998.

    Examples
    --------
    >>> import numpy as np
    >>> from skbio.stats.ordination import OptSpace
    >>> # Create a low-rank matrix with missing entries
    >>> np.random.seed(42)
    >>> true_U = np.random.randn(10, 2)
    >>> true_V = np.random.randn(8, 2)
    >>> true_M = true_U.dot(true_V.T)
    >>> # Mask some entries
    >>> M_observed = true_M.copy()
    >>> M_observed[::2, ::2] = np.nan  # Hide some entries
    >>> # Recover the matrix
    >>> opt = OptSpace(n_components=2, max_iterations=10)
    >>> M_recovered = opt.fit_transform(M_observed)

    """

    def __init__(self, n_components=3, max_iterations=5, tol=1e-5):
        self.n_components = n_components
        self.max_iterations = max_iterations
        self.tol = tol
        self.U = None
        self.S = None
        self.V = None
        self.converged = False

    def fit(self, X):
        """Fit the OptSpace model to the observed matrix.

        Parameters
        ----------
        X : ndarray
            A 2D array with observed values and NaN for missing entries.

        Returns
        -------
        self
            The fitted OptSpace instance.

        Raises
        ------
        ValueError
            If input is not 2D or n_components exceeds matrix dimensions.
        """
        X = np.asarray(X, dtype=np.float64)

        if X.ndim != 2:
            raise ValueError(f"Input must be 2D, got {X.ndim}D array.")

        m, n = X.shape
        r = self.n_components

        if r > min(m, n):
            raise ValueError(
                f"n_components ({r}) cannot exceed min matrix dimension ({min(m, n)})."
            )

        # Create observed mask (1 for observed, 0 for missing)
        observed_mask = ~np.isnan(X)
        n_observed = np.sum(observed_mask)

        # Trim over-represented rows and columns
        X_trimmed = trim(X, observed_mask, m, n, n_observed)

        # Compute sparsity for rescaling
        sparsity = n_observed / (n * m)
        epsilon = n_observed / np.sqrt(m * n)

        # Rescale observed values for sparse initialization
        X_trimmed *= sparsity  # max(sparsity, 1e-10)

        """
        # Estimate rank
        U, s, Vt = svd(X_trimmed, full_matrices=False)
        #U, s, Vt = svds(X_trimmed, k=min(m, n) * 0.2)
        V = Vt.T
        U, s, V = _svd_sort(U, s, V)
        print(s) # Delete

        rhat = estimate_rank(s, epsilon)

        #print(f"Estimated rank: {rhat}") # Delete
        #print(f"Size of U and V: {U.shape}, {V.shape}, s: {s.shape}") # Delete

        # Compute the rank-rhat projection of the trimmed matrix
        U = U[:, :rhat]
        s = s[:rhat]
        V = V[:, :rhat]

        #print(f"Estimated rank: {rhat}") # Delete
        #print(f"Size of U and V: {U.shape}, {V.shape}, s: {s.shape}") # Delete
        """

        # Initialize with truncated SVD
        try:
            if r < min(n, m) - 1:
                U, s, Vt = svds(X_trimmed, k=r)
                V = Vt.T
            else:
                U, s, Vt = svd(X_trimmed, full_matrices=False)
                U = U[:, :r]
                s = s[:r]
                V = Vt[:r, :].T
        except Exception:
            # Fallback to random initialization
            U = np.random.randn(n, r)
            V = np.random.randn(m, r)
            U, _ = np.linalg.qr(U)
            V, _ = np.linalg.qr(V)
            s = np.ones(r)

        # Initialize S as diagonal matrix
        # S = np.diag(s) / sparsity  # Scale back

        prev_obj = np.inf

        for iteration in range(self.max_iterations):
            # Compute optimal S given current U, V
            S = solve_S(U, V, X, observed_mask)

            # Compute current error
            R = residual(U, V, S, X)

            # Current objective (Frobenius norm of error)
            obj = np.sum(R**2)
            # print(f"Iteration {iteration}, objective: {obj:.6f}")  # Delete

            # Check convergence
            if abs(prev_obj - obj) < self.tol:
                self.converged = True
                """print(
                    f"Converged at iteration {iteration}, objective: {obj:.6f}"
                )  # Delete"""
                break

            prev_obj = obj

            # Compute Gauss-Newton step
            dU, dV = solve_gauss_newton_step(U, V, S, X, observed_mask, R)

            # Retract updates back to Grassmann manifold
            U = retract_grassmann(U, dU)
            V = retract_grassmann(V, dV)

        # Final sort
        s_diag = np.diag(S)
        U, s_diag, V = _svd_sort(U, s_diag, V)
        S = np.diag(s_diag)

        self.U = U
        self.S = S
        self.V = V

        return self

    def transform(self, X=None):
        """Reconstruct the complete matrix.

        Parameters
        ----------
        X : ndarray, optional
            Not used, present for API compatibility.

        Returns
        -------
        ndarray
            The reconstructed low-rank matrix.

        Raises
        ------
        ValueError
            If the model has not been fitted.
        """
        if self.U is None:
            raise ValueError("Model has not been fitted. Call fit() first.")

        return self.U.dot(self.S).dot(self.V.T)

    def fit_transform(self, X):
        """Fit the model and return the reconstructed matrix.

        Parameters
        ----------
        X : ndarray
            A 2D array with observed values and NaN for missing entries.

        Returns
        -------
        ndarray
            The reconstructed low-rank matrix.
        """
        self.fit(X)
        return self.transform()

    def get_loadings(self):
        """Get sample and feature loadings.

        Returns
        -------
        tuple
            (sample_loadings, feature_loadings) where sample_loadings
            has shape (n_samples, n_components) and feature_loadings
            has shape (n_features, n_components).

        Raises
        ------
        ValueError
            If the model has not been fitted.
        """
        if self.U is None:
            raise ValueError("Model has not been fitted. Call fit() first.")

        s = np.sqrt(np.diag(self.S))
        sample_loadings = self.U * s
        feature_loadings = self.V * s

        return sample_loadings, feature_loadings
