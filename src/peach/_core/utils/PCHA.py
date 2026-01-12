# NB: from Krishnaswamy lab AAnet implementation: https://github.com/KrishnaswamyLab/AAnet/blob/master/example_notebooks/PCHA.py
## included here as a utility function for Archetypal_Matrices_VAE to compare results against
### depends on furthest_sum.py from AAnet
"""Principal Convex Hull Analysis (PCHA) / Archetypal Analysis.

=== MODULE API INVENTORY ===

MAIN FUNCTIONS:
├── furthest_sum(K, noc, i, exclude=[]) -> List[int]
│   └── Purpose: Furthest sum algorithm for efficient initial archetype seed generation
│   └── Inputs: K(numpy.2darray, data/kernel matrix), noc(int, number of candidate archetypes), i(int, initial observation), exclude(list, excluded indices)
│   └── Outputs: List[int] extracted candidate archetype indices
│   └── Side Effects: Distance calculations, kernel computations, iterative selection process
│
├── PCHA(X, noc, I=None, U=None, delta=0, verbose=False, conv_crit=1E-6, maxiter=500) -> Tuple[numpy.2darray, numpy.2darray, numpy.2darray, float, float]
│   └── Purpose: Core PCHA algorithm for finding archetypes through alternating least squares optimization
│   └── Inputs: X(numpy.2darray, [dimensions, examples] data matrix), noc(int, number of archetypes), I(1darray, dictionary entries), U(1darray, modeling entries), delta(float, regularization), verbose(bool, progress output), conv_crit(float, convergence criterion), maxiter(int, maximum iterations)
│   └── Outputs: Tuple(XC archetypes, S archetype weights, C construction weights, SSE sum squared error, varexpl variance explained)
│   └── Side Effects: Iterative optimization, matrix updates, convergence monitoring, optional progress printing
│
└── run_pcha_analysis(data: np.ndarray, n_archetypes: int, verbose: bool = False) -> Dict[str, np.ndarray]
    └── Purpose: Adapter function to run PCHA on data in standard format with consistent output
    └── Inputs: data(np.ndarray, [n_samples, n_dimensions] data matrix), n_archetypes(int, number of archetypes), verbose(bool, progress output)
    └── Outputs: Dict[str, np.ndarray] with 'archetypes', 'S_weights', 'C_weights', 'archetype_r2', 'SSE', 'XC_raw'
    └── Side Effects: Data transposition, PCHA execution, result formatting, performance reporting

INTERNAL FUNCTIONS:
├── S_update(S, XCtX, CtXtXC, muS, SST, SSE, niter) -> Tuple [INTERNAL]
│   └── Purpose: Update S matrix (archetype weights) for one iteration with gradient descent
│   └── Inputs: S(matrix, current weights), XCtX(matrix, cross terms), CtXtXC(matrix, construction terms), muS(float, step size), SST(float, total sum squares), SSE(float, current error), niter(int, iterations)
│   └── Outputs: Tuple(updated S, SSE, muS, SSt)
│   └── Side Effects: Gradient computation, step size adaptation, constraint enforcement
│
└── C_update(X, XSt, XC, SSt, C, delta, muC, mualpha, SST, SSE, niter=1) -> Tuple [INTERNAL]
    └── Purpose: Update C matrix (construction weights) and alpha parameters for one iteration
    └── Inputs: X(data matrix), XSt(transformed data), XC(current archetypes), SSt(S transpose product), C(construction matrix), delta(regularization), muC(step size), mualpha(alpha step), SST(total sum squares), SSE(current error), niter(int, iterations)
    └── Outputs: Tuple(updated C, SSE, muC, mualpha, CtXtXC, XC)
    └── Side Effects: Construction matrix updates, alpha parameter optimization, constraint enforcement

ALGORITHM COMPONENTS:
├── Alternating Least Squares: Iterative optimization between S (archetype weights) and C (construction weights)
├── Furthest Sum Initialization: Intelligent seed selection for stable convergence
├── Constraint Enforcement: Non-negativity and sum-to-one constraints for archetypal properties
├── Convergence Monitoring: SSE tracking with configurable tolerance and maximum iterations
└── Variance Explained: R² calculation for model quality assessment

DATA FORMAT REQUIREMENTS:
├── Input Format: X as [dimensions, examples] (transposed from typical ML format)
├── Output Format: Archetypes as [n_archetypes, dimensions] (standard format)
├── Matrix Conventions: S[noc, examples], C[examples, noc] for archetypal decomposition
└── Adapter Compatibility: run_pcha_analysis handles format conversion for integration

EXTERNAL DEPENDENCIES:
├── From numpy: Array operations, linear algebra, matrix manipulations
├── From scipy.sparse: csr_matrix for efficient sparse matrix operations
├── From scipy.spatial.distance: cdist for distance calculations (unused in current implementation)
├── From scipy.optimize: linear_sum_assignment for optimal matching (unused in current implementation)
├── From datetime: Performance timing and iteration monitoring
└── From numpy.matlib: repmat for matrix replication operations

DATA FLOW PATTERNS:
├── Input: Data matrix → Transposition → PCHA algorithm → Archetype extraction → Result formatting
├── Optimization: Initial seeds → Alternating S/C updates → Convergence check → Final archetypes
├── Constraint Enforcement: Gradient updates → Projection to constraints → Feasibility maintenance
└── Quality Assessment: SSE calculation → Variance explained → Performance metrics

ERROR HANDLING:
├── Initialization failures → InitializationException with descriptive message
├── Convergence issues → Maximum iteration limit with final state return
├── Matrix dimension mismatches → Handled by numpy array operations
├── Numerical instability → Step size adaptation and constraint projection
└── Data format compatibility → Adapter function with automatic transposition
"""

import time
from datetime import datetime as dt

import numpy as np
from numpy.matlib import repmat
from scipy.sparse import csr_matrix


def furthest_sum(K, noc, i, exclude=[]):
    """Furthest sum algorithm, to efficiently generat initial seed/archetypes.

    Note: Commonly data is formatted to have shape (examples, dimensions).
    This function takes input and returns output of the transposed shape,
    (dimensions, examples).

    Parameters
    ----------
    K : numpy 2d-array
        Either a data matrix or a kernel matrix.

    noc : int
        Number of candidate archetypes to extract.

    i : int
        inital observation used for to generate the FurthestSum.

    exclude : numpy.1darray
        Entries in K that can not be used as candidates.

    Output
    ------
    i : int
        The extracted candidate archetypes
    """

    def max_ind_val(l):
        return max(zip(range(len(l)), l, strict=False), key=lambda x: x[1])

    I, J = K.shape
    index = np.array(range(J))
    index[exclude] = 0
    index[i] = -1
    ind_t = i
    sum_dist = np.zeros((1, J), np.complex128)

    if J > noc * I:
        Kt = K
        Kt2 = np.sum(Kt**2, axis=0)
        for k in range(1, noc + 11):
            if k > noc - 1:
                Kq = np.dot(Kt[:, i[0]], Kt)
                sum_dist -= np.lib.scimath.sqrt(Kt2 - 2 * Kq + Kt2[i[0]])
                index[i[0]] = i[0]
                i = i[1:]
            t = np.where(index != -1)[0]
            Kq = np.dot(Kt[:, ind_t].T, Kt)
            sum_dist += np.lib.scimath.sqrt(Kt2 - 2 * Kq + Kt2[ind_t])
            ind, val = max_ind_val(sum_dist[:, t][0].real)
            ind_t = t[ind]
            i.append(ind_t)
            index[ind_t] = -1
    else:
        if I != J or np.sum(K - K.T) != 0:  # Generate kernel if K not one
            Kt = K
            K = np.dot(Kt.T, Kt)
            K = np.lib.scimath.sqrt(repmat(np.diag(K), J, 1) - 2 * K + repmat(np.asmatrix(np.diag(K)).T, 1, J))

        Kt2 = np.diag(K)  # Horizontal
        for k in range(1, noc + 11):
            if k > noc - 1:
                sum_dist -= np.lib.scimath.sqrt(Kt2 - 2 * K[i[0], :] + Kt2[i[0]])
                index[i[0]] = i[0]
                i = i[1:]
            t = np.where(index != -1)[0]
            sum_dist += np.lib.scimath.sqrt(Kt2 - 2 * K[ind_t, :] + Kt2[ind_t])
            ind, val = max_ind_val(sum_dist[:, t][0].real)
            ind_t = t[ind]
            i.append(ind_t)
            index[ind_t] = -1

    return i


def PCHA(X, noc, I=None, U=None, delta=0, verbose=False, conv_crit=1e-6, maxiter=500):
    """Return archetypes of dataset.

    Note: Commonly data is formatted to have shape (examples, dimensions).
    This function takes input and returns output of the transposed shape,
    (dimensions, examples).

    Parameters
    ----------
    X : numpy.2darray
        Data matrix in which to find archetypes

    noc : int
        Number of archetypes to find

    I : 1d-array
        Entries of X to use for dictionary in C (optional)

    U : 1d-array
        Entries of X to model in S (optional)


    Output
    ------
    XC : numpy.2darray
        I x noc feature matrix (i.e. XC=X[:,I]*C forming the archetypes)

    S : numpy.2darray
        noc x length(U) matrix, S>=0 |S_j|_1=1

    C : numpy.2darray
        noc x length(U) matrix, S>=0 |S_j|_1=1

    SSE : float
        Sum of Squared Errors

    varexlp : float
        Percent variation explained by the model
    """

    def S_update(S, XCtX, CtXtXC, muS, SST, SSE, niter):
        """Update S for one iteration of the algorithm."""
        noc, J = S.shape
        e = np.ones((noc, 1))
        for k in range(niter):
            SSE_old = SSE
            g = (np.dot(CtXtXC, S) - XCtX) / (SST / J)
            g = g - e * np.sum(g.A * S.A, axis=0)

            S_old = S
            while True:
                S = (S_old - g * muS).clip(min=0)
                S = S / np.dot(e, np.sum(S, axis=0))
                SSt = S * S.T
                SSE = SST - 2 * np.sum(XCtX.A * S.A) + np.sum(CtXtXC.A * SSt.A)
                if SSE <= SSE_old * (1 + 1e-9):
                    muS = muS * 1.2
                    break
                else:
                    muS = muS / 2

        return S, SSE, muS, SSt

    def C_update(X, XSt, XC, SSt, C, delta, muC, mualpha, SST, SSE, niter=1):
        """Update C for one iteration of the algorithm."""
        J, nos = C.shape

        if delta != 0:
            alphaC = np.sum(C, axis=0).A[0]
            C = np.dot(C, np.diag(1 / alphaC))

        e = np.ones((J, 1))
        XtXSt = np.dot(X.T, XSt)

        for k in range(niter):
            # Update C
            SSE_old = SSE
            g = (np.dot(X.T, np.dot(XC, SSt)) - XtXSt) / SST

            if delta != 0:
                g = np.dot(g, np.diag(alphaC))
            g = g.A - e * np.sum(g.A * C.A, axis=0)

            C_old = C
            while True:
                C = (C_old - muC * g).clip(min=0)
                nC = np.sum(C, axis=0) + np.finfo(float).eps
                C = np.dot(C, np.diag(1 / nC.A[0]))

                if delta != 0:
                    Ct = C * np.diag(alphaC)
                else:
                    Ct = C

                XC = np.dot(X, Ct)
                CtXtXC = np.dot(XC.T, XC)
                SSE = SST - 2 * np.sum(XC.A * XSt.A) + np.sum(CtXtXC.A * SSt.A)

                if SSE <= SSE_old * (1 + 1e-9):
                    muC = muC * 1.2
                    break
                else:
                    muC = muC / 2

            # Update alphaC
            SSE_old = SSE
            if delta != 0:
                g = (np.diag(CtXtXC * SSt).T / alphaC - np.sum(C.A * XtXSt.A)) / (SST * J)
                alphaC_old = alphaC
                while True:
                    alphaC = alphaC_old - mualpha * g
                    alphaC[alphaC < 1 - delta] = 1 - delta
                    alphaC[alphaC > 1 + delta] = 1 + delta

                    XCt = np.dot(XC, np.diag(alphaC / alphaC_old))
                    CtXtXC = np.dot(XCt.T, XCt)
                    SSE = SST - 2 * np.sum(XCt.A * XSt.A) + np.sum(CtXtXC.A * SSt.A)

                    if SSE <= SSE_old * (1 + 1e-9):
                        mualpha = mualpha * 1.2
                        XC = XCt
                        break
                    else:
                        mualpha = mualpha / 2

        if delta != 0:
            C = C * np.diag(alphaC)

        return C, SSE, muC, mualpha, CtXtXC, XC

    N, M = X.shape

    if I is None:
        I = range(M)
    if U is None:
        U = range(M)

    SST = np.sum(X[:, U] * X[:, U])

    # Initialize C
    try:
        i = furthest_sum(X[:, I], noc, [int(np.ceil(len(I) * np.random.rand()))])
    except IndexError:

        class InitializationException(Exception):
            pass

        raise InitializationException("Initialization does not converge. Too few examples in dataset.")

    j = range(noc)
    C = csr_matrix((np.ones(len(i)), (i, j)), shape=(len(I), noc)).todense()

    XC = np.dot(X[:, I], C)

    muS, muC, mualpha = 1, 1, 1

    # Initialise S
    XCtX = np.dot(XC.T, X[:, U])
    CtXtXC = np.dot(XC.T, XC)
    S = -np.log(np.random.random((noc, len(U))))
    S = S / np.dot(np.ones((noc, 1)), np.asmatrix(np.sum(S, axis=0)))
    SSt = np.dot(S, S.T)
    SSE = SST - 2 * np.sum(XCtX.A * S.A) + np.sum(CtXtXC.A * SSt.A)
    S, SSE, muS, SSt = S_update(S, XCtX, CtXtXC, muS, SST, SSE, 25)

    # Set PCHA parameters
    iter_ = 0
    dSSE = np.inf
    t1 = dt.now()
    varexpl = (SST - SSE) / SST

    if verbose:
        print("\nPrincipal Convex Hull Analysis / Archetypal Analysis")
        print("A " + str(noc) + " component model will be fitted")
        print("To stop algorithm press control C\n")

    dheader = "%10s | %10s | %10s | %10s | %10s | %10s | %10s | %10s" % (
        "Iteration",
        "Expl. var.",
        "Cost func.",
        "Delta SSEf.",
        "muC",
        "mualpha",
        "muS",
        " Time(s)   ",
    )
    dline = "-----------+------------+------------+-------------+------------+------------+------------+------------+"

    while np.abs(dSSE) >= conv_crit * np.abs(SSE) and iter_ < maxiter and varexpl < 0.9999:
        if verbose and iter_ % 100 == 0:
            print(dline)
            print(dheader)
            print(dline)
        told = t1
        iter_ += 1
        SSE_old = SSE

        # C (and alpha) update
        XSt = np.dot(X[:, U], S.T)
        C, SSE, muC, mualpha, CtXtXC, XC = C_update(X[:, I], XSt, XC, SSt, C, delta, muC, mualpha, SST, SSE, 10)

        # S update
        XCtX = np.dot(XC.T, X[:, U])
        S, SSE, muS, SSt = S_update(S, XCtX, CtXtXC, muS, SST, SSE, 10)

        # Evaluate and display iteration
        dSSE = SSE_old - SSE
        t1 = dt.now()
        if iter_ % 1 == 0:
            time.sleep(0.000001)
            varexpl = (SST - SSE) / SST
            if verbose:
                print(
                    "%10.0f | %10.4f | %10.4e | %10.4e | %10.4e | %10.4e | %10.4e | %10.4f \n"
                    % (iter_, varexpl, SSE, dSSE / np.abs(SSE), muC, mualpha, muS, (t1 - told).seconds)
                )

    # Display final iteration
    varexpl = (SST - SSE) / SST
    if verbose:
        print(dline)
        print(dline)
        print(
            "%10.0f | %10.4f | %10.4e | %10.4e | %10.4e | %10.4e | %10.4e | %10.4f \n"
            % (iter_, varexpl, SSE, dSSE / np.abs(SSE), muC, mualpha, muS, (t1 - told).seconds)
        )

    # Sort components according to importance
    ind, vals = zip(*sorted(enumerate(np.sum(S, axis=1)), key=lambda x: x[0], reverse=1), strict=False)
    S = S[ind, :]
    C = C[:, ind]
    XC = XC[:, ind]

    return XC, S, C, SSE, varexpl


# adapter function to run it on samples I already have set up in a compatible format
def run_pcha_analysis(data: np.ndarray, n_archetypes: int, verbose: bool = False) -> dict[str, np.ndarray]:
    """
    Run PCHA on data and return results in consistent format.

    Args:
        data: Data matrix [n_samples, n_dimensions]
        n_archetypes: Number of archetypes to find
        verbose: Whether to print PCHA progress

    Returns
    -------
        Dictionary with PCHA results
    """
    # PCHA expects [dimensions, examples] format
    X_pcha = data.T  # [20, 1000]

    print(f"Running PCHA with {n_archetypes} archetypes...")
    print(f"Data shape for PCHA: {X_pcha.shape}")

    # Run PCHA
    XC, S, C, SSE, varexpl = PCHA(X=X_pcha, noc=n_archetypes, verbose=verbose, maxiter=1000, conv_crit=1e-8)

    # Convert results to consistent format
    # XC is [dimensions, n_archetypes] = [20, 4]
    # We want [n_archetypes, dimensions] = [4, 20]
    archetypes_pcha = XC.T  # [4, 20]

    print("PCHA Results:")
    print(f"  Archetypes shape: {archetypes_pcha.shape}")
    print(f"  Archetype R²: {varexpl:.4f}")
    print(f"  SSE: {SSE:.4f}")

    return {
        "archetypes": archetypes_pcha,
        "S_weights": S,  # [n_archetypes, n_samples]
        "C_weights": C,  # [n_samples, n_archetypes]
        "archetype_r2": varexpl,
        "SSE": SSE,
        "XC_raw": XC,  # Original format
    }
