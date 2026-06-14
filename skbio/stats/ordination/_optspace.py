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
    # Estimate rank r\hat by minimizing the cost function of singular
    # values from Keshavan et al. (2010)

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


def _compute_gradient(E, U, V, S, observed_mask):
    """Compute gradient for OptSpace optimization.

    Parameters
    ----------
    M : ndarray
        Reconstructed matrix.
    E : ndarray
        Error matrix.
    U : ndarray
        Left singular vectors.
    V : ndarray
        Right singular vectors.
    n : int
        Number of rows.
    m : int
        Number of columns.
    observed_mask : ndarray
        Binary mask indicating observed entries.

    Returns
    -------
    tuple
        Gradients for U and V.
    """
    # Apply mask to error
    E_masked = E * observed_mask

    # Compute gradients
    grad_U = E_masked.dot(V).dot(S.T)
    grad_V = E_masked.T.dot(U).dot(S)

    return grad_U, grad_V


####

# def _sym(A):
#     return 0.5 * (A + A.T)
#
#
# def _euclidean_grad(U, V, S, M_obs, Omega):
#     M = U @ S @ V.T
#     R = Omega * (M - M_obs)
#     R = np.nan_to_num(R, nan=0.0)
#
#     grad_U = R @ V @ S.T
#     grad_V = R.T @ U @ S
#
#     return R, grad_U, grad_V
#
#
# def _riemannian_grad(U, V, grad_U, grad_V):
#     # Stiefel projections
#     grad_U = grad_U - U @ _sym(U.T @ grad_U)
#     grad_V = grad_V - V @ _sym(V.T @ grad_V)
#     return grad_U, grad_V
#
#
# def _newton_direction(U, V, grad_U, grad_V, damping=1e-3):
#     """
#     Gauss-Newton style preconditioned step.
#     """
#     # simple curvature-aware scaling (cheap Hessian approximation)
#     H_U = np.linalg.norm(U, axis=0, keepdims=True).T + damping
#     H_V = np.linalg.norm(V, axis=0, keepdims=True).T + damping
#
#     dU = -grad_U / H_U.T
#     dV = -grad_V / H_V.T
#
#     return dU, dV
#
#
# def _retract_qr(X):
#     Q, _ = np.linalg.qr(X)
#     return Q
#
#
# def _compute_S(U, V, M_obs, Omega):
#     # same idea as OptSpace: least squares over observed entries
#     return _compute_singular_values(U, V, M_obs, Omega)
#
#
# def _objective(U, V, S, M_obs, Omega):
#     M = U @ S @ V.T
#     R = Omega * (M - M_obs)
#     R = np.nan_to_num(R, nan=0.0)
#     return 0.5 * np.sum(R * R)
#
#
# def _line_search(
#     U,
#     V,
#     S,
#     M_obs,
#     Omega,
#     step_size=1.0,
#     max_iter=20,
#     tol=1e-6,
# ):
#     current_obj = _objective(U, V, S, M_obs, Omega)
#
#     for _ in range(max_iter):
#
#         # --- gradients ---
#         R, gU, gV = _euclidean_grad(U, V, S, M_obs, Omega)
#         gU, gV = _riemannian_grad(U, V, gU, gV)
#
#         # --- Newton direction (Gauss-Newton approx) ---
#         dU, dV = _newton_direction(U, V, gU, gV)
#
#         # --- trial step ---
#         U_new = _retract_qr(U + step_size * dU)
#         V_new = _retract_qr(V + step_size * dV)
#
#         # --- re-solve S exactly (block minimization) ---
#         S_new = _compute_S(U_new, V_new, M_obs, Omega)
#
#         new_obj = _objective(U_new, V_new, S_new, M_obs, Omega)
#
#         # --- Armijo-like acceptance ---
#         if new_obj < current_obj:
#             return U_new, V_new, S_new, step_size
#
#         step_size *= 0.5
#
#         if step_size < tol:
#             break
#
#     return U, V, S, step_size

####


def _line_search(
    U,
    V,
    S,
    grad_U,
    grad_V,
    M_observed,
    observed_mask,
    m,
    n,
    step_size=1.0,
    max_iter=50,
    tol=1e-6,
):
    """Perform line search for step size optimization.

    Parameters
    ----------
    U, V, S : ndarray
        Current SVD components.
    grad_U, grad_V : ndarray
        Gradients for U and V.
    M_observed : ndarray
        Original observed matrix (with NaN for unobserved).
    observed_mask : ndarray
        Binary mask for observed entries.
    n, m : int
        Matrix dimensions.
    step_size : float
        Initial step size.
    max_iter : int
        Maximum line search iterations.
    tol : float
        Convergence tolerance.

    Returns
    -------
    tuple
        Updated (U, V, step_size).
    """
    # Current objective
    M = U.dot(S).dot(V.T)
    E = (M_observed - M) * observed_mask
    E = np.nan_to_num(E, nan=0.0)
    current_obj = np.sum(E**2)

    for _ in range(max_iter):
        # Trial update
        U_new = U - step_size * grad_U
        V_new = V - step_size * grad_V

        # Orthonormalize via QR
        U_new, _ = np.linalg.qr(U_new)
        V_new, _ = np.linalg.qr(V_new)

        # Compute new objective
        # First compute optimal S for new U, V
        S_new = _compute_singular_values(U_new, V_new, M_observed, observed_mask)

        M_new = U_new.dot(S_new).dot(V_new.T)
        E_new = (M_observed - M_new) * observed_mask
        E_new = np.nan_to_num(E_new, nan=0.0)
        new_obj = np.sum(E_new**2)

        if new_obj < current_obj:
            return U_new, V_new, S_new, step_size
        else:
            step_size *= 0.5

        if step_size < tol:
            break

    return U, V, S, step_size


def _compute_singular_values(U, V, M_observed, observed_mask):
    """Compute optimal singular values given U and V.

    Solves the least squares problem to find optimal S.

    Parameters
    ----------
    U : ndarray
        Left singular vectors (n x r).
    V : ndarray
        Right singular vectors (m x r).
    M_observed : ndarray
        Observed matrix values.
    observed_mask : ndarray
        Binary mask for observed entries.

    Returns
    -------
    ndarray
        Optimal diagonal singular value matrix (r x r).
    """

    r = U.shape[1]

    rows, cols = np.where(observed_mask)

    A = np.einsum("ni,nj->nij", U[rows], V[cols]).reshape(len(rows), -1)

    b = M_observed[rows, cols]

    s = np.linalg.lstsq(A, b, rcond=None)[0]

    S = s.reshape(r, r)

    return S

    """
    This approach solves for a diagonal S via the reduced rxr system. However,
    unless the basis vectors of U and V align with the singular vectors of the
    observed matrix, S will not in general be diagonal, and the true singular
    values will instead be mixed across the entire matrix. So, we solve for this
    full S.

    G = np.zeros((r, r))
    c = np.zeros(r)

    rows, cols = np.where(observed_mask)

    A = U[rows] * V[cols]
    b = M_observed[rows, cols]

    G = A.T @ A
    c = A.T @ b

    s = np.linalg.lstsq(G, c, rcond=None)[0]
    """

    """

    # Build and solve the linear system for S
    # For each observed entry (i,j): M[i,j] = sum_k U[i,k] * S[k,k] * V[j,k]
    # This is a least squares problem

    # For diagonal S, we solve independently for each singular value
    S = np.zeros((r, r))

    for k in range(r):
        # Build the linear system for s_k
        uv_k = np.outer(U[:, k], V[:, k])
        uv_k_masked = uv_k * observed_mask

        # Flatten and solve
        a = uv_k_masked.flatten()
        b = (M_observed * observed_mask).flatten()
        b = np.nan_to_num(b, nan=0.0)

        # Filter to observed entries
        observed_flat = observed_mask.flatten() > 0
        if np.sum(observed_flat) > 0:
            # Least squares solution
            a_obs = a[observed_flat]
            b_obs = b[observed_flat]

            if np.sum(a_obs**2) > 1e-10:
                S[k, k] = np.dot(a_obs, b_obs) / np.dot(a_obs, a_obs)
    """

    return np.diag(s)


###


def jacobian_action(U, V, S, dU, dV, observed_mask):
    """Compute J(dU, dV)."""

    W = dU @ S @ V.T
    W += U @ S @ dV.T

    return W * observed_mask


def jacobian_adjoint(U, V, S, W):
    """Compute J*(W)."""

    dU = W @ V @ S.T
    dV = W.T @ U @ S

    dU -= U @ (U.T @ dU)
    dV -= V @ (V.T @ dV)

    return dU, dV


def gauss_newton_hvp(
    U,
    V,
    S,
    dU,
    dV,
    observed_mask,
    damping=0.0,
):
    """Apply (J*J + λI) to a tangent vector."""

    W = jacobian_action(U, V, S, dU, dV, observed_mask)
    Hu, Hv = jacobian_adjoint(U, V, S, W)

    # if damping:
    #     r = S.shape[0]
    #
    #     # scale by geometry + dimension
    #     lambda_scale = damping * (observed_mask.sum() / (U.shape[0] * U.shape[1]))
    #
    #     lambda_U = lambda_scale * (np.trace(V.T @ V) / r)
    #     lambda_V = lambda_scale * (np.trace(U.T @ U) / r)
    #
    #     Hu += lambda_U * dU
    #     Hv += lambda_V * dV

    # if damping:
    #     Hu += damping * dU
    #     Hv += damping * dV

    return Hu, Hv


def gauss_newton_rhs(
    U,
    V,
    S,
    M_observed,
    observed_mask,
):
    """Compute -J*(R)."""

    R = U @ S @ V.T - M_observed
    R = np.nan_to_num(R, nan=0.0)

    gU, gV = jacobian_adjoint(U, V, S, R)

    return -gU, -gV


def pack(dU, dV):
    return np.concatenate([dU.ravel(), dV.ravel()])


def unpack(x, U_shape, V_shape):
    nu = np.prod(U_shape)

    dU = x[:nu].reshape(U_shape)
    dV = x[nu:].reshape(V_shape)

    return dU, dV


def solve_gauss_newton_step(
    U,
    V,
    S,
    M_observed,
    observed_mask,
    damping=1e-1,
):
    """Solve (J*J+λI)η=-J*R."""

    bU, bV = gauss_newton_rhs(U, V, S, M_observed, observed_mask)

    b = pack(bU, bV)

    nvars = b.size

    def matvec(x):
        dU, dV = unpack(x, U.shape, V.shape)

        Hu, Hv = gauss_newton_hvp(
            U,
            V,
            S,
            dU,
            dV,
            observed_mask,
            damping=damping,
        )

        return pack(Hu, Hv)

    A = LinearOperator(
        (nvars, nvars),
        matvec=matvec,
        dtype=U.dtype,
    )

    step, info = cg(
        A,
        b,
        rtol=1e-6,
        maxiter=50,
    )

    dU, dV = unpack(step, U.shape, V.shape)

    # dU -= U @ (U.T @ dU)
    # dV -= V @ (V.T @ dV)

    return dU, dV


def retract_grassmann(X, dX):
    Q, _ = np.linalg.qr(X + dX)
    return Q[:, : X.shape[1]]


###


def trim(X, observed_mask, m, n, n_observed):
    """Trim over-represented rows and columns.

    (More than half the average observed entries)"""

    n_observed_rows = np.sum(observed_mask, axis=1)
    n_observed_cols = np.sum(observed_mask, axis=0)

    row_threshold = 2 * n_observed / m
    col_threshold = 2 * n_observed / n

    valid_rows = n_observed_rows <= row_threshold
    valid_cols = n_observed_cols <= col_threshold

    trim_mask = np.outer(valid_rows, valid_cols)

    return np.where(trim_mask & observed_mask, X, 0.0)


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
        S = _compute_singular_values(U, V, X, observed_mask)

        ###

        prev_obj = np.inf

        for iteration in range(self.max_iterations):
            # Compute current reconstruction and error
            M = U.dot(S).dot(V.T)
            E = (M - X) * observed_mask
            E = np.nan_to_num(E, nan=0.0)

            # Current objective (Frobenius norm of error)
            obj = np.sum(E**2)
            print(f"Iteration {iteration}, objective: {obj:.6f}")  # Delete

            # Check convergence
            if abs(prev_obj - obj) < self.tol:
                self.converged = True
                """print(
                    f"Converged at iteration {iteration}, objective: {obj:.6f}"
                )  # Delete"""
                break

            prev_obj = obj

            ###

            noise_var = obj / n_observed

            ###

            dU, dV = solve_gauss_newton_step(U, V, S, X, observed_mask)

            U = retract_grassmann(U, dU)
            V = retract_grassmann(V, dV)

            S = _compute_singular_values(U, V, X, observed_mask)

        ###

        """
        # Optimization loop
        step_size = 1.0
        prev_obj = np.inf

        for iteration in range(self.max_iterations):
            # Compute current reconstruction and error
            M = U.dot(S).dot(V.T)
            E = (M - X) * observed_mask
            E = np.nan_to_num(E, nan=0.0)

            # Current objective (Frobenius norm of error)
            obj = np.sum(E**2)

            # Check convergence
            if abs(prev_obj - obj) < self.tol:
                self.converged = True
                print(
                    f"Converged at iteration {iteration}, objective: {obj:.6f}"
                )  # Delete
                break

            prev_obj = obj

            # Compute gradients
            grad_U, grad_V = _compute_gradient(E, U, V, S, observed_mask)

            # Update with line search
            U, V, S, step_size = _line_search(
                U, V, S, grad_U, grad_V, X, observed_mask, m, n, step_size=step_size
            )

            # # Update with line search
            # U, V, S, step_size = _line_search(
            #     U, V, S, X, observed_mask, step_size=step_size
            # )

        # Recompute final S
        S = _compute_singular_values(U, V, X, observed_mask)
        """

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
