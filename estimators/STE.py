import numpy as np
import numpy.linalg as LA

# Subspace-constrained Tyler, translated from the matlab code at
# https://github.com/alexfengg/STE
def STE(X, d, gam):
    D, N = X.shape
    initcov = np.eye(D)
    eps = 1e-10

    oldcov = initcov - 1
    cov = initcov.copy()
    iter = 0
    res = []

    while LA.norm(oldcov - cov, ord='fro') / LA.norm(cov)  > 1e-6 and iter < 500:
        oldcov = cov.copy()
        temp = (X.T @ np.linalg.inv(cov + eps * np.eye(D))) * X.T
        w = 1.0 / (np.sum(temp, axis=1) + eps)
        cov = X @ (w[:, np.newaxis] * X.T) / (N * D)

        U, S, _ = np.linalg.svd(cov)
        S1 = np.real(S)

        # Adjust the spectrum
        S1[d:] = np.mean(S1[d:]) / gam

        cov = U @ np.diag(S1) @ U.T
        cov = cov / np.trace(cov)

        res.append(cov)
        iter += 1

    return cov, res

# Subspace-constrained Tyler, adapted for complex data
def STE_complex(X, d, gam):
    D, N = X.shape
    initcov = np.eye(D)
    eps = 1e-10

    oldcov = initcov - 1
    cov = initcov.copy()
    iter = 0
    res = []

    while LA.norm(oldcov - cov, ord='fro') / LA.norm(cov)  > 1e-6 and iter < 500:
        oldcov = cov.copy()
        temp = (X.conj() * LA.solve(cov + eps * np.eye(D), X)).T
        w = 1.0 / (np.sum(temp, axis=1) + eps)
        cov = (w * X ) @ X.conj().T
        
        U, S, _ = np.linalg.svd(cov)
        S1 = np.real(S)

        # Adjust the spectrum
        S1[d:] = np.mean(S1[d:]) / gam

        cov = U @ np.diag(S1) @ U.conj().T
        cov = cov / np.trace(cov)

        res.append(cov)
        iter += 1

    return cov, res

