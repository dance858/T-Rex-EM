import numpy as np 
import numpy.linalg as LA 

def solve_LS_via_SVD_optimized(beta, X, rank, delta):
    n, N = X.shape
    U, sigma, VT = LA.svd(np.sqrt(beta) * X, full_matrices=False)
    lmbda = np.power(sigma, 2)
    N = min(n, N)
    v = np.zeros((N, ))

    if np.abs(lmbda[rank]) < 1e-14:
        v[0:rank] = 1
    else:
        temp_sum = np.sum(1/lmbda[0:rank])
        for ii in range(rank+1, N+1):
            if ii != n:
                temp_sum += 1/lmbda[ii-1]
                theta = (ii - rank) / temp_sum

            if lmbda[ii-1] > theta and ii != n and theta >= lmbda[ii]:
                break 
        
        for i in range(0, N):
            if (lmbda[i] > theta):
                v[i] = 1 - theta/lmbda[i]

    new_beta = 1 / np.maximum(delta, LA.norm(X - (v * U) @ (U.T @ X), axis=0))
    return v, U, new_beta

# X is n x N (N samples of dimension n)
def Reaper_optimized(X, rank, delta=1e-10, max_iter= 500):
    N = X.shape[1]
    beta = np.ones((N, ))
   
    for iter in range(max_iter):
        v, U, beta = solve_LS_via_SVD_optimized(beta, X, rank, delta)
        
    obj = np.sum(LA.norm(X - (v * U) @ (U.T @ X), axis=0))
    return v, U, obj

def Euclidean_median_via_Weiszfield(X, max_iter, delta=1e-10):
    c = np.mean(X, axis=1, keepdims=True)

    for ii in range(max_iter):
        w = 1/np.maximum(LA.norm(X - c, axis=0), delta)
        c = np.sum(w*X, axis=1, keepdims=True)/np.sum(w)

    return c

# X is n x m (m samples of dimension n)
# they use mu = m*n/np.sum(LA.svd(X, compute_uv=False))
def outlier_pursuit_via_ADMM(X, lmbda, max_iter=100):
    n, m = X.shape
    mu =  m*n/np.sum(LA.svd(X, compute_uv=False))

    P = np.zeros((n, m))
    Q = np.zeros((n, m))

    for ii in range(max_iter):
        C = ColShrink(X - P + (1/mu) * Q, mu*lmbda) 
        P = SpecShrink(X - C + (1/mu) * Q, mu)
        Q = Q + mu * (X - P - C)

    obj = np.sum(LA.svd(P, compute_uv=False)) + np.sum(LA.norm(C, axis=0))  
    pri_rel_res = LA.norm(X - P - C, 'fro') / LA.norm(X, 'fro') 
    return obj, pri_rel_res, P

def ColShrink(C, threshold):
    return C * np.maximum(1 - threshold/LA.norm(C, axis=0), 0)
     
def SpecShrink(P, threshold):
    U, S, VT = LA.svd(P, full_matrices=False)
    S = np.maximum(S-threshold, 0)
    return (S * U) @ VT

def fit_PCA_subspace(X_original, rank, centering_mean=False, centering_median=False,
                     spherization=False):
    assert(not (centering_mean and centering_median))

    X = X_original.copy()
    c = np.zeros((X.shape[0], 1))

    if centering_mean:
        c = np.mean(X, axis=1, keepdims=True)
    elif centering_median:
        c = np.median(X, axis=1, keepdims=True)

    X = X - c 
    
    if spherization:
        X = X / LA.norm(X, axis=0)

    U, _, _ = np.linalg.svd(X, full_matrices=False)
    Ur = U[:, 0:rank]
    return Ur