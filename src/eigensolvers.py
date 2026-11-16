"""
Lowest eigenpairs of (sparse) Hermitian matrices.

Our Hamiltonians split into many disconnected blocks (invariant subspaces:
e.g. 97 blocks of <= 96 states for a 10-qubit hs Hamiltonian, 577 blocks of
<= 400 states for a 14-qubit one; the 3-local one splits much less).
Degenerate levels are common, both between blocks and inside them. That is where Krylov methods (ARPACK/eigsh) fail: a
single-vector Krylov space only sees one vector per distinct eigenvalue, so
copies of degenerate levels are missed (the original eigsh returned wrong
levels in almost every step for k = 9). Diagonalizing block by block with
dense eigh is exact and also much faster.

Methods
-------
"auto" / "blocks"
    Split H into its connected blocks (computed once per evolution, see
    find_blocks) and diagonalize each one: dense eigh if it has at most
    BLOCK_DENSE_MAX states (exact), otherwise the single-matrix "auto" rules
    below (ARPACK only for k = 1, block LOBPCG for k > 1). The k lowest
    levels of all blocks are merged; blocks whose Gershgorin lower bound is
    above the current k-th level are skipped. If H is a single block, "auto"
    uses the single-matrix rules below.
"dense"
    np.linalg.eigh on the full matrix. Exact; only for small N.
"shift-invert"
    Sparse LU of H - sigma*I, sigma a Gershgorin lower bound of the spectrum
    (so the eigenvalues closest to sigma are exactly the lowest ones).
    k = 1: ARPACK in shift-invert mode. k > 1: block LOBPCG preconditioned
    with that LU (resolves degeneracies, but slow for large N).
"lobpcg"
    Block LOBPCG with a Jacobi preconditioner: matrix-free, memory O(N k).

Single-matrix "auto" (H is one block): "dense" if N <= DENSE_MAX_DIM,
otherwise "shift-invert", or "lobpcg" if the LU exceeds lu_budget_mb.
"""

import warnings

import numpy as np
from scipy.linalg import LinAlgWarning
from scipy.sparse import csr_matrix, diags, identity, issparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import LinearOperator, eigsh, lobpcg, splu

# Below this dimension dense eigh is the fastest method for a whole matrix
# (measured crossover with shift-invert at N ~ 64-128).
DENSE_MAX_DIM = 64

# Blocks up to this size are diagonalized with dense eigh: exact, and at
# 1024 states (~0.5 s) as fast as block LOBPCG, the robust alternative.
BLOCK_DENSE_MAX = 1024

# Maximum memory (MB) allowed for the LU factors in shift-invert mode.
LU_BUDGET_MB = 500

# Extra vectors computed by the block iterative methods (LOBPCG block size,
# ARPACK k inside a large block): they speed up convergence when level k is
# (nearly) degenerate with the next ones.
EXTRA_VECTORS = 4

METHODS = ("auto", "blocks", "dense", "shift-invert", "lobpcg")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def gershgorin_lower_bound(H):
    """Lower bound of the spectrum of a Hermitian matrix (Gershgorin):
    every eigenvalue is >= min_i (H_ii - sum_{j != i} |H_ij|). Cost O(nnz)."""
    d = H.diagonal().real
    radii = np.asarray(abs(H).sum(axis=1)).ravel() - np.abs(d)
    return float((d - radii).min())


def spectral_shift(H):
    """Shift sigma strictly below the spectrum of H (so H - sigma*I is
    positive definite and never singular)."""
    bound = gershgorin_lower_bound(H)
    return bound - 1e-3 * max(1.0, abs(bound))


def _as_complex_sparse(H):
    H = H if issparse(H) else csr_matrix(H)
    if not np.iscomplexobj(H):
        H = H.astype(complex)
    return H.tocsr()


def _factorize(H, sigma):
    N = H.shape[0]
    return splu((H - sigma * identity(N, dtype=complex, format="csc")).tocsc())


def _lu_memory_mb(lu):
    return (lu.L.nnz + lu.U.nnz) * np.dtype(complex).itemsize / 1e6


def _starting_vector(N, v0):
    """Seeded random vector (reproducible: ARPACK's own random start is not
    seeded), mixed with v0 if given. An eigenvector (e.g. the one tracked at
    the previous step) lives inside a single block, and a Krylov space
    started there never leaves it; the random part reaches every block."""
    r = np.random.default_rng(1).standard_normal((N, 2)) @ np.array([1, 1j])
    r /= np.linalg.norm(r)
    if v0 is None:
        return r
    v0 = np.asarray(v0, dtype=complex).reshape(-1)
    norm = np.linalg.norm(v0)
    return r if norm == 0 else v0 / norm + r


# ---------------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------------


def find_blocks(*matrices):
    """
    Connected blocks of the union of the sparsity patterns of `matrices`.

    Returns (order, bounds): H[order][:, order] is block diagonal and block b
    is made of the basis states order[bounds[b]:bounds[b + 1]]. Every H(s)
    of an evolution between H0 and H1 has its pattern inside that of
    |H0| + |H1|, so find_blocks(H0, H1) is valid for the whole evolution.
    """
    pattern = abs(_as_complex_sparse(matrices[0]))
    for M in matrices[1:]:
        pattern = pattern + abs(_as_complex_sparse(M))
    n_blocks, labels = connected_components(pattern, directed=False)
    order = np.argsort(labels, kind="stable")
    bounds = np.concatenate([[0], np.cumsum(np.bincount(labels, minlength=n_blocks))])
    return order, bounds


def n_blocks(blocks):
    return len(blocks[1]) - 1


def _lowest_by_blocks(H, k, blocks, v0, tol, lu_budget_mb):
    order, bounds = blocks
    N = H.shape[0]
    Hp = H[order][:, order].tocsr()  # block diagonal, contiguous blocks
    v0p = None if v0 is None else np.asarray(v0, dtype=complex).reshape(-1)[order]

    # Gershgorin lower bound of every block: visit the most promising first
    d = Hp.diagonal().real
    radii = np.asarray(abs(Hp).sum(axis=1)).ravel() - np.abs(d)
    lower = np.minimum.reduceat(d - radii, bounds[:-1])

    values, pieces = [], []  # candidate levels and (start, stop, vector)
    kth = np.inf
    for b in np.argsort(lower, kind="stable"):
        if len(values) >= k and lower[b] > kth:
            break  # every remaining block lies above the current k-th level
        a, c = bounds[b], bounds[b + 1]
        kb = min(k, c - a)
        block = Hp[a:c, a:c]
        if c - a <= BLOCK_DENSE_MAX or kb >= c - a - 2:
            w, V = np.linalg.eigh(block.toarray())
            w, V = w[:kb], V[:, :kb]
        else:
            # Large block: degeneracies inside a block do occur (e.g. H0 is
            # diagonal), so no single-vector ARPACK for kb > 1 (see
            # _lowest_single: ARPACK only for k = 1, block LOBPCG otherwise)
            w, V = _lowest_single(
                block, kb, "auto", None if v0p is None else v0p[a:c], tol, None,
                lu_budget_mb,
            )
        values.extend(w)
        pieces.extend((a, c, V[:, j]) for j in range(kb))
        if len(values) >= k:
            kth = np.partition(values, k - 1)[k - 1]

    selected = np.argsort(values, kind="stable")[:k]
    eigenvectors = np.zeros((N, k), dtype=complex)
    for col, j in enumerate(selected):
        a, c, vec = pieces[j]
        eigenvectors[order[a:c], col] = vec
    return np.asarray(values)[selected], eigenvectors


# ---------------------------------------------------------------------------
# Single matrix
# ---------------------------------------------------------------------------


def select_eig_method(H, lu_budget_mb=LU_BUDGET_MB):
    """Single-matrix method that "auto" would use for H (H seen as one
    block). For "shift-invert" it factorizes H once to measure the LU.

    All H(s) = (1 - s) H0 + s H1 share the same sparsity pattern, so the
    choice made for one generic H(s) holds for the whole evolution."""
    H = _as_complex_sparse(H)
    if H.shape[0] <= DENSE_MAX_DIM:
        return "dense"
    lu = _factorize(H, spectral_shift(H))
    return "shift-invert" if _lu_memory_mb(lu) <= lu_budget_mb else "lobpcg"


def _lowest_single(H, k, method, v0, tol, ncv, lu_budget_mb, lu=None, sigma=None):
    N = H.shape[0]

    if method == "auto":
        if N <= DENSE_MAX_DIM:
            method = "dense"
        else:
            sigma = spectral_shift(H)
            lu = _factorize(H, sigma)
            if _lu_memory_mb(lu) <= lu_budget_mb:
                method = "shift-invert"
            else:
                method, lu = "lobpcg", None

    # Iterative methods need k well below N (ARPACK: k < N - 1)
    if method == "dense" or k >= N - 2:
        eigenvalues, eigenvectors = np.linalg.eigh(H.toarray())
        return eigenvalues[:k], eigenvectors[:, :k]

    if sigma is None:
        sigma = spectral_shift(H)
    v0 = _starting_vector(N, v0)

    if method == "shift-invert" and lu is None:
        lu = _factorize(H, sigma)

    if method == "shift-invert" and k == 1:
        # ARPACK (single-vector Krylov) in shift-invert mode: the fastest
        # option, and multiplicities do not matter for a single eigenpair.
        OPinv = LinearOperator((N, N), matvec=lu.solve, dtype=complex)
        eigenvalues, eigenvectors = eigsh(
            H, k=1, sigma=sigma, which="LM", OPinv=OPinv, v0=v0, tol=tol, ncv=ncv
        )
        return eigenvalues, eigenvectors / np.linalg.norm(eigenvectors)

    # k > 1 (or "lobpcg"): block LOBPCG, which resolves multiplicities.
    # Preconditioner: the shift-invert LU if available, otherwise Jacobi
    # (H_ii - sigma > 0 by construction of sigma).
    if lu is not None:
        M = LinearOperator((N, N), matvec=lu.solve, matmat=lu.solve, dtype=complex)
    else:
        M = diags(1.0 / (H.diagonal().real - sigma))

    m = min(k + EXTRA_VECTORS, N)
    rng = np.random.default_rng(0)  # reproducible starting block
    X = rng.standard_normal((N, m)) + 1j * rng.standard_normal((N, m))
    X[:, 0] = v0
    with warnings.catch_warnings():
        # lobpcg warns when it stops slightly above tol, falls back to eigh
        # for tiny N, or (LinAlgWarning) when its Gram matrix becomes
        # ill-conditioned because the vectors have already converged.
        # Convergence is checked below with the actual residuals instead.
        warnings.simplefilter("ignore", UserWarning)
        warnings.simplefilter("ignore", LinAlgWarning)
        eigenvalues, eigenvectors = lobpcg(
            H, X, M=M, largest=False, tol=max(tol, 1e-9), maxiter=1000
        )

    order = np.argsort(eigenvalues)[:k]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]

    residual = np.linalg.norm(H @ eigenvectors - eigenvectors * eigenvalues, axis=0)
    if residual.max() > 1e-6:
        warnings.warn(
            f"lobpcg did not converge (max residual {residual.max():.1e}).",
            RuntimeWarning,
        )

    return eigenvalues, eigenvectors


def block_ground_state(
    H, start, dense=True, seed=None, tol=1e-12, maxiter=500, atol=None
):
    """
    Not used in this version of the code.

    Ground state of the block H (one connected block). `start` are the
    positions of the basis states that start LOBPCG (those expected to
    dominate the low levels).

    Small blocks (if `dense`): dense eigh. Otherwise block LOBPCG started
    from the `start` basis states with a Jacobi preconditioner: when the low
    levels are those basis states dressed perturbatively and the rest of
    the spectrum lies well above, it converges in a few iterations, also
    where the LU of shift-invert does not fit in memory (its fill is
    almost dense). The tolerance is relative to the
    largest entry of H; `atol`, if given, caps it as an absolute one:
    LOBPCG uses min(tol * max|H|, atol) (for large blocks tol * max|H| can
    exceed the gap of the history sector). `seed` adds a small random part
    to the starting block, to get a second solution with independent
    rounding errors.

    Returns (E0, psi0, gap inside the block, residual |H psi0 - E0 psi0|).
    """
    N = H.shape[0]
    if dense and N <= BLOCK_DENSE_MAX:
        energies, vectors = np.linalg.eigh(H.toarray())
    else:
        d = H.diagonal().real
        X = np.zeros((N, len(start)), dtype=complex)
        X[start, np.arange(len(start))] = 1.0
        if seed is not None:
            X += 1e-3 * np.random.default_rng(seed).standard_normal(X.shape)
        # shift below the history-sector diagonal: d - sigma > 0 everywhere
        scale = abs(H).max()
        sigma = d.min() - scale
        abs_tol = tol * scale if atol is None else min(tol * scale, atol)
        with warnings.catch_warnings():
            # lobpcg warns when it stops at the rounding level, above tol,
            # or falls back to eigh for tiny N; the residual is returned
            warnings.simplefilter("ignore", UserWarning)
            energies, vectors = lobpcg(
                H,
                X,
                M=diags(1.0 / (d - sigma)),
                largest=False,
                tol=abs_tol,
                maxiter=maxiter,
            )
        order = np.argsort(energies)
        energies, vectors = energies[order], vectors[:, order]
    psi = vectors[:, 0]
    residual = np.linalg.norm(H @ psi - energies[0] * psi)
    return energies[0], psi, energies[1] - energies[0], residual


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def lowest_eigenpairs(
    H,
    k=1,
    method="auto",
    v0=None,
    tol=1e-10,
    ncv=None,
    lu_budget_mb=LU_BUDGET_MB,
    blocks=None,
):
    """
    k lowest eigenpairs of the Hermitian matrix H.

    Parameters
    ----------
    H : sparse or dense (N, N) Hermitian matrix
    k : int
        Number of eigenpairs.
    method : "auto", "blocks", "dense", "shift-invert" or "lobpcg"
        (see module doc).
    v0 : array or None
        Starting vector (e.g. the eigenvector of the previous step). Only
        used by the iterative methods.
    tol : float
        Tolerance of the iterative methods ("lobpcg" uses max(tol, 1e-9) on
        the residual, enough for eigenvalue errors ~1e-16).
    ncv : int or None
        Krylov subspace size for single-matrix "shift-invert" with k = 1.
    lu_budget_mb : float
        Max memory of a shift-invert LU before switching to LOBPCG.
    blocks : (order, bounds) or None
        Block structure from find_blocks (e.g. computed once for a whole
        evolution). If None, "auto"/"blocks" compute it from H.

    Returns
    -------
    eigenvalues : (k,) array, ascending
    eigenvectors : (N, k) array, eigenvectors[:, j] <-> eigenvalues[j]
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")

    H = _as_complex_sparse(H)

    if method in ("auto", "blocks"):
        if blocks is None:
            blocks = find_blocks(H)
        if n_blocks(blocks) > 1 or method == "blocks":
            return _lowest_by_blocks(H, k, blocks, v0, tol, lu_budget_mb)

    return _lowest_single(H, k, method, v0, tol, ncv, lu_budget_mb)
