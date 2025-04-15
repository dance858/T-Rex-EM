import numpy as np
import numpy.linalg as LA
from numba import njit
from estimators.est_utils import PCFA_via_corr, PCFA_via_corr_complex

# X is n x m (n is the dimension, m is the number of samples) 
def sample_mean_and_cov(X):
    mu = np.mean(X, axis=1, keepdims=True)
    zero_mean_X = X - mu 
    cov = (1 / X.shape[1]) * (zero_mean_X @ zero_mean_X.T)
    return mu, cov

def sample_mean_and_cov_complex(X):
    mu = np.mean(X, axis=1, keepdims=True)
    zero_mean_X = X - mu 
    cov = (1 / X.shape[1]) * (zero_mean_X @ zero_mean_X.conj().T)
    return mu, cov

def GFA_EM_simple(S, rank, F, d, max_iter=200, eta=0):   
    if F is None or d is None:
        F, d = PCFA_via_corr(S, rank)
    
    s = np.diag(S)   
    for _ in range(1, max_iter + 1):
        # E-step
        G = np.linalg.inv((F.T * (1 / d.reshape(1, -1))) @ F + np.eye(rank))
        B = G @ F.T * (1 / d.reshape(1, -1))
        Cxz = S @ B.T
        Czz = B @ Cxz + G

        # M-step
        F = Cxz @ np.linalg.inv(Czz)
        d = np.maximum(s - 2*np.sum(Cxz*F, axis=1) + np.sum(F * (F @ Czz), axis=1), eta)
       

    return F, d

@njit
def GFA_EM_simple_numba(S, rank, F, d, max_iter=200, eta=0): 
    s = np.diag(S)   
    for _ in range(1, max_iter + 1):
        # E-step
        G = np.linalg.inv((F.T * (1 / d.reshape(1, -1))) @ F + np.eye(rank))
        B = G @ F.T * (1 / d.reshape(1, -1))
        Cxz = S @ B.T
        Czz = B @ Cxz + G

        # M-step
        F = Cxz @ np.linalg.inv(Czz)
        d = np.maximum(s - 2*np.sum(Cxz*F, axis=1) + np.sum(F * (F @ Czz), axis=1), eta)

    return F, d

def GFA_EM_simple_via_data(a, s, X, rank, F, d, max_iter=1000, eta=0):    
    aX = a * X
    for _ in range(1, max_iter + 1):
        # E-step
        G = np.linalg.inv((F.T * (1 / d.reshape(1, -1))) @ F + np.eye(rank))
        B = G @ F.T * (1 / d.reshape(1, -1))
        Cxz = aX @ (X.T @ B.T)
        Czz = B @ Cxz + G

        # M-step
        F = Cxz @ np.linalg.inv(Czz)
        d = np.maximum(s - 2*np.sum(Cxz*F, axis=1) + np.sum(F * (F @ Czz), axis=1), eta)
        
    return F, d

def GFA_EM_complex(S, k, F0=None, d0=None, iter=100):

    # initialization
    if F0 is None or d0 is None:
        F, d = PCFA_via_corr_complex(S, k)
    else:
        F = F0
        d = d0

    for _ in range(iter):
        # E-step
        G = np.linalg.inv((F.conj().T * (1 / d.reshape(1, -1))) @ F + np.eye(k))
        B = G @ F.conj().T * (1 / d.reshape(1, -1))
        Cxz = S @ B.conj().T
        Czz = B @ Cxz + G

        # M-step
        F = Cxz @ np.linalg.inv(Czz)
        d = np.diag(S - 2 * Cxz @ F.conj().T + F @ Czz @ F.conj().T)
        
    return F, d