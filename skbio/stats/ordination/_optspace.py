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
from scipy.sparse.linalg import lsmr


def _trim(X, observed_mask, m, n, n_observed):
    """Trim over-represented rows and columns.

    Any row or column with more than half the average observed entries per
    row or column respectively is set to zero per Keshavan et al. (2010).
    This makes the low-rank structure more pronounced."""

    n_observed_rows = np.sum(observed_mask, axis=1)
    n_observed_cols = np.sum(observed_mask, axis=0)

    row_threshold = 2 * n_observed / m
    col_threshold = 2 * n_observed / n

    valid_rows = n_observed_rows <= row_threshold
    valid_cols = n_observed_cols <= col_threshold

    trim_mask = np.outer(valid_rows, valid_cols)

    return np.where(trim_mask & observed_mask, X, 0.0)


def _estimate_rank(S, epsilon):
    """Estimate rank r\\hat by minimizing the cost function of singular
    values from Keshavan et al. (2010).
    """

    # Indices i = 1, 2, ... , len(S) - 1
    # The last element is excluded because S[i] would be out of bounds
    i = np.arange(1, len(S))  # 1 , ... , k-1
    cost = (S[0] * np.sqrt(i / epsilon) + S[i]) / S[i - 1]

    return i[np.argmin(cost)]


def _solve_S(A, U, V, b, rows, cols):
    """Compute optimal S given U and V.

    Solves the least squares problem to find the optimal S that
    minimizes the reconstruction difference on the observed entries:

    arg min_S ||P_\\Omega(U V S^T - M_observed)||_F^2

    where P_\\Omega is the projection onto the observed entries.
    Since only the observed entries are considered, the problem
    can be vectorized and solved efficiently using least squares:

    arg min_S \\sum_{(i,j) \\in \\Omega} (U[i] S V[j]^T - M_observed[i, j])^2
    """

    r = U.shape[1]
    n_observed = len(rows)

    # Matrix A such that (USV^T)_ij = A_(i,j)(k,l) vec(S)_(k,l)
    # A = np.einsum("ni,nj->nij", U[rows], V[cols]).reshape(n_observed, -1)
    A = A.reshape(n_observed, r, r)
    np.einsum("ni,nj->nij", U[rows], V[cols], out=A)
    A = A.reshape(n_observed, r**2)

    # Solve least-squares problem As = b
    s = np.linalg.solve(A.T @ A, A.T @ b)

    return s.reshape(r, r)


def _update_residual(R, U, S, V, M_obs):
    """Compute the residual R = P_Ω(U S V^T - M_obs)."""
    np.matmul(U @ S, V.T, out=R)
    R -= M_obs
    np.nan_to_num(R, nan=0.0, copy=False)

    """
    Note:
    A potentially big optimization may be found in computing R only on the observed
    entries. The reasoning is as follows:
    We must compute the matrix A with dimension n_observed x r^2 to solve the system
    As = m,
    where s is the vector (size r^2) containing the entries of the rxr square
    matrix S, and m is the vector (size n_observed) containing the observed entries
    of M_obs.
    Once s is found, the same matrix A can be used to compute
    m_hat = As.
    This is the vector (size n_observed) of the reconstructed matrix USV^T, *only on
    the observed entries*. This immediately gives a way of computing the residual
    vector,
    r =  m_hat - m,
    which is also size n_observed.

    The Jacobian J gives a map from the tangent space of pairs (U, V) onto the space
    of projected entries. The output takes the form of a matrix W (size mxn), which is
    projected onto the space of observed entries: P_omega (W). In other words, this
    is a dense mxn matrix, where m * n - n_observed entries are just zero.
    For the Jacobian J, it is no problem to simply truncate dU and dV to instead return
    a vector with only the observed entries, rather than the entire sparse matrix W.

    The problem comes from the Jacobian adjoint, which must reconstruct (dU, dV) from
    a vector (size n_observed) in the space of observed entries.
    If the full dense matrix P_omega(W) is passed to the Jacobian adjoint, the dense
    matrices (dU, dV) can be easily reconstructed through matrix multiplication.
    However, if a vector of observed entries is passed, the dense matrix must be
    reconstructed or otherwise iteratively accumulated. This is extremely slow.

    I see this as the greatest potential benefit for GPU acceleration with numba.
    If the reconstruction of the dense matrix can be sped up, only the vector of
    observed entries can be stored, drastically reducing memory requirements
    and reducing the total number of operations. If this can be accomplished, each
    iteration would only compute
    A -> s -> m_hat -> r -> (dU, dV)
    rather than
    A -> s -> S -> USV^T -> R -> (dU, dV)
    The matrix A must be computed regardless, so if we can reuse it rather than
    recomputing USV^T, it could be highly beneficial.
    """


def jacobian_action(U, V, S, dU, dV, observed_mask):
    """Compute J(dU, dV).

    The Jacobian action is a mapping from the tangent space of (U, V) to
    the space of observed entries. It computes how changes in U and V (dU, dV)
    affect the observed reconstruction error.

    J(dU, dV) = dU S V^T + U S dV^T"""

    rows, cols = np.where(observed_mask)
    dU -= U @ U.T @ dU
    dV -= V @ V.T @ dV
    W = dU @ S @ V.T
    W += U @ S @ dV.T
    w = W[rows, cols]

    return w


def jacobian_adjoint(U, V, S, w, observed_mask):
    """Compute J*(W).

    The Jacobian adjoint is defined with respect to the inner product by

    <J(dU, dV), W> = <(dU, dV), J*(W)>

    It defines a mapping from the space of observed entries back to the
    tangent space of (U, V). With J known,

    J*(W) = (W V S^T, W^T U S)

    This is projected back to the tangent space of (U, V):

    J*(W)_U = (I - U U^T) W V S^T
    J*(W)_V = (I - V V^T) W^T U S"""

    W = np.zeros(observed_mask.shape)
    W[observed_mask] = w

    dU = W @ V @ S.T
    dV = W.T @ U @ S

    dU -= U @ (U.T @ dU)
    dV -= V @ (V.T @ dV)

    return dU, dV


def pack(dU, dV):
    """Pack dU and dV to a single vector dx"""
    return np.concatenate([dU.ravel(), dV.ravel()])


def unpack(x, U_shape, V_shape):
    """Unpack the vector dx back to dU and dV."""
    nu = np.prod(U_shape)

    dU = x[:nu].reshape(U_shape)
    dV = x[nu:].reshape(V_shape)

    return dU, dV


def solve_gauss_newton_step(U, V, S, observed_mask, R, damping=1e-1):
    """Solve (J*J)dx = -J*R.

    The Gauss-Newton step is the vector dx = (dU, dV), where dU and dV are
    tangent vectors in their respective Grassmann manifolds. The step is the
    least-squares solution of the system Jdx = -R, and it is computed using
    the LSMR algorithm."""

    nvars = U.size + V.size

    def matvec(x):
        dU, dV = unpack(x, U.shape, V.shape)
        return jacobian_action(U, V, S, dU, dV, observed_mask)

    def rmatvec(y):
        dU, dV = jacobian_adjoint(U, V, S, y, observed_mask)
        return pack(dU, dV)

    J = LinearOperator(
        shape=(np.sum(observed_mask), nvars),
        matvec=matvec,
        rmatvec=rmatvec,
        dtype=U.dtype,
    )

    result = lsmr(J, -R.ravel(), atol=1e-5, btol=1e-5)
    step = result[0]
    iter = result[2]
    print(f"Iterations: {iter}")

    return unpack(step, U.shape, V.shape)


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
       b. Update U, V with the Gauss-Newton step dU, dV
       c. Project U, V back to Grassmann manifold

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

    def __init__(self, n_components=3, max_iterations=100, tol=1e-5):
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
        X_trimmed = _trim(X, observed_mask, m, n, n_observed)

        # Compute density for rescaling
        density = n_observed / (n * m)

        # Rescale observed values for sparse initialization
        X_trimmed *= density

        """

        # Note:
        # The original OptSpace paper gives a method for estimating the rank
        # of a matrix, but I couldn't get this working accurately for all
        # matrices. Depending on the singular value structure of the matrix,
        # it seemed to drastically underestimate the rank in some
        # cases, which in turn gives a very inaccurate reconstruction.
        # This section seems optional, so it can safely be ignored for now.

        # Estimate rank
        U, s, Vt = svd(X_trimmed, full_matrices=False)
        #U, s, Vt = svds(X_trimmed, k=min(m, n) * 0.2) # Assume rank r < 0.2 min(m,n)
        V = Vt.T
        U, s, V = _svd_sort(U, s, V)

        epsilon = n_observed / np.sqrt(m * n)
        rhat = _estimate_rank(s, epsilon)

        # Compute the rank-rhat projection of the trimmed matrix
        U = U[:, :rhat]
        V = V[:, :rhat]
        """

        # Initialize with truncated SVD
        # Note that we do not need to keep S in this step, since it is immediately
        # recomputed based on the observed entries only.

        if r < min(n, m) - 1:
            U, _, Vt = svds(X_trimmed, k=r)
            V = Vt.T
        else:
            U, _, Vt = svd(X_trimmed, full_matrices=False)
            U = U[:, :r]
            V = Vt[:r, :].T

        prev_obj = np.inf
        tol = self.tol

        R = np.empty(n_observed)
        S = np.empty(r**2)
        A = np.empty((n_observed, r**2))

        rows, cols = np.where(observed_mask)
        b = X[rows, cols]

        for iteration in range(self.max_iterations):
            # Matrix A such that (USV^T)_ij = A_(i,j)(k,l) vec(S)_(k,l)
            # A = np.einsum("ni,nj->nij", U[rows], V[cols]).reshape(n_observed, -1)

            """A = A.reshape(n_observed, r, r)
            np.einsum("ni,nj->nij", U[rows], V[cols], out=A)
            A = A.reshape(n_observed, r**2)
            AtA = A.T @ A
            AtAinv = np.linalg.inv(AtA)
            s = AtAinv @ (A.T @ b)"""

            # Solve least-squares problem As = b
            # s = np.linalg.solve(A.T @ A, A.T @ b)

            def mat_A(s):
                As = U @ s.reshape(r, r) @ V.T
                return As[rows, cols]

            def mat_A_T(w):
                # Atw = U[rows].T @ (w[:, None] * V[cols]).reshape(-1, r)
                # return Atw.ravel()
                # return np.einsum('ni,nj,n->ij', U[rows], V[cols], w).ravel()
                W = np.zeros_like(X)
                W[observed_mask] = w
                Atw = U.T @ W @ V
                return Atw.ravel()

            A_operator = LinearOperator(
                shape=(n_observed, r**2),
                matvec=mat_A,
                rmatvec=mat_A_T,
                dtype=U.dtype,
            )

            result = lsmr(A_operator, b, atol=1e-5, btol=1e-5)
            S = result[0]
            iter = result[2]
            print(f"S iterations: {iter}")
            # Compute optimal S given current U, V
            # S = _solve_S(A, U, V, b, rows, cols)

            # _update_residual(R, U, S, V, X)

            # Compute current error
            R = mat_A(S) - b

            # Current objective (Frobenius norm of error)
            obj = np.sum(R**2)
            print(f"Iteration {iteration}, objective: {obj:.6f}")  # Delete

            # Check convergence
            if np.abs(prev_obj - obj) < tol:
                self.converged = True
                break

            prev_obj = obj

            # Compute Gauss-Newton step
            dU, dV = solve_gauss_newton_step(U, V, S.reshape(r, r), observed_mask, R)

            # Retract updates back to Grassmann manifold
            U = retract_grassmann(U, dU)
            V = retract_grassmann(V, dV)

        self.X_hat = U @ S.reshape(r, r) @ V.T

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

        """
        Note:
        I've changed this from the original implementation.

        Currently, in fit(), U and V are continually updated as mxr and nxr orthogonal
        matrices, and S is computed as a rank 5 matrix (but NOT necessarily
        a diagonal one).

        So, we can form the reconstruction simply from USV^T. To recover a true
        diagonal matrix S, we would have to recompute the SVD. We can safely leave
        this to the RPCA function, since the RPCA function is essentially a wrapper
        for OptSpace followed by SVD anyway.

        OptSpace may be able to be rewritten as a function rather than a whole class,
        but features of the Gemelli OptSpace may not be compatible with this.

        Currently, though, much of this architecture is redundant outside of fit().
        """

        # if self.U is None:
        #     raise ValueError("Model has not been fitted. Call fit() first.")

        return self.X_hat  # self.U.dot(self.S).dot(self.V.T)

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
