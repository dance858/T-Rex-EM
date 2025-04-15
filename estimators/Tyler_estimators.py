
import numpy as np 
import numpy.linalg as LA
from estimators.N_estimators import (sample_mean_and_cov, GFA_EM_simple_numba, 
                                     GFA_EM_simple_via_data)

from estimators.est_utils import PCFA_via_corr, TERMINATE_OBJ, TERMINATE_MAX_ITER
from numba import njit

def obj_Ty(F, d, X):
     # evaluate logdet(F @ F.T + D)
     L = LA.cholesky(np.eye(F.shape[1]) + F.T @ (F / d.reshape(-1, 1)))
     term1 = np.sum(np.log(d)) + 2*np.sum(np.log(np.diag(L)))
     
     # evaluate second term - could use low-rank structure here
     #temp1 = LA.solve(F @ F.T + np.diag(d), X) 
     dInvF = (F / d.reshape(-1, 1))
     temp1 = X / d.reshape(-1, 1) - dInvF @ LA.solve(np.eye(F.shape[1]) + F.T @ dInvF, dInvF.T @ X)  
     term2 = (F.shape[0] / X.shape[1]) * np.sum(np.log(np.sum(X * temp1, axis=0)))

     obj = term1 + term2 
     return obj

# EM algorithm for fitting a statistical factor model based on Tyler's estimator
def TRex_simple(X, rank, outer_max_iter=20, inner_max_iter=200, 
                           eps_rel_obj=1e-6):
     S = sample_mean_and_cov(X)[1]
     F, d = PCFA_via_corr(S, rank)

     status = TERMINATE_MAX_ITER
     objs = [obj_Ty(F, d, X)] 
    
     for ii in range(1, outer_max_iter + 1):
          #print(f"iteration / obj: ", ii, objs[-1])
          # evaluate Sk (E-step)
          dInvF = F / d.reshape(-1, 1)
          temp1 = X / d.reshape(-1, 1) - dInvF @ LA.solve(np.eye(F.shape[1]) + F.T @ dInvF, dInvF.T @ X)  
          a = 1 / np.sum(X * temp1, axis=0)
          Sk = (a * X ) @ X.T
          Sk /= np.trace(Sk)
          
          # evaluate M-step
          F, d = GFA_EM_simple_numba(Sk, rank, F, d, max_iter=inner_max_iter, eta=0)
          
          # track progress and check termination criteria
          objs.append(obj_Ty(F, d, X))
          new_obj = objs[-1]
          old_obj = objs[-2]

          if (abs(new_obj - old_obj) / abs(old_obj) < eps_rel_obj):
            status = TERMINATE_OBJ
            break

     stats = {"iterations": ii,  "objs": objs, "status": status}
     
     return F, d, stats

# EM algorithm for fitting a statistical factor model based on Tyler's estimator
def TRex_simple_via_data(X, rank, outer_max_iter=20, inner_max_iter=200, 
                                    eps_rel_obj=1e-6):
     print("Initializing T-Rex via PCA...")
     # initialize using PCA
     N = X.shape[1]
     U, sigma, _ = LA.svd(X, full_matrices=False)
     sigma_k = sigma[0:rank]
     Uk = U[:, 0:rank]
     F = (Uk @ np.diag(sigma_k))/np.sqrt(N)
     d = 1/N*np.sum(np.power(X, 2), axis=1) - np.sum(np.power(F, 2), axis=1) 
     print("Initialization complete")
    
     #S = sample_mean_and_cov(X)[1]
     #F, d = PCFA_via_corr(S, rank)
     status = TERMINATE_MAX_ITER
     objs = [obj_Ty(F, d, X)] 
    
     for ii in range(1, outer_max_iter + 1):
          print(f"iteration / obj: ", ii, objs[-1])
          # evaluate Sk (E-step)
          dInvF = F / d.reshape(-1, 1)
          temp1 = X / d.reshape(-1, 1) - dInvF @ LA.solve(np.eye(F.shape[1]) + F.T @ dInvF, dInvF.T @ X)  
          a = 1 / np.sum(X * temp1, axis=0)

          s =  np.sum(a * (np.power(X, 2)), axis=1)      # diag((a * X) @ X.T)
          trace_S = np.sum(s)
          a = a / trace_S
          s = s / trace_S
          
          # evaluate M-step
          F, d = GFA_EM_simple_via_data(a, s, X, rank, F, d,
                                              max_iter=inner_max_iter, eta=0)
          
          # track progress and check termination criteria
          objs.append(obj_Ty(F, d, X))
          new_obj = objs[-1]
          old_obj = objs[-2]

          if (abs(new_obj - old_obj) / abs(old_obj) < eps_rel_obj):
            status = TERMINATE_OBJ
            break

     stats = {"iterations": ii,  "objs": objs, "status": status}
     
     return F, d, stats

# EM algorithm for fitting a statistical factor model based on Tyler's estimator
@njit
def TRex_random_init(X, rank, F0, d0, outer_max_iter=20, inner_max_iter=200):
     F = F0
     d = d0

     for _ in range(outer_max_iter):
          dInvF = F / d.reshape(-1, 1)
          temp1 = X / d.reshape(-1, 1) - dInvF @ LA.solve(np.eye(F.shape[1]) + F.T @ dInvF, dInvF.T @ X)  
          a = 1 / np.sum(X * temp1, axis=0)
          Sk = (a * X ) @ X.T
          Sk /= np.trace(Sk)
          
          # evaluate M-step
          F, d = GFA_EM_simple_numba(Sk, rank, F, d, max_iter=inner_max_iter, eta=0)
              
     return F, d

def Tyler_standard_FP(X, max_iter):
    n, m = X.shape 
    Sigma = np.eye(n)
    for _ in range(max_iter):
        SigmainvX = LA.solve(Sigma, X)
        Sigma = np.zeros((n, n))
        for i in range(m):
            xi = X[:, i]
            denom = xi.T @ SigmainvX[:, i]
            Sigma += np.outer(xi, xi) / denom
       
        Sigma /= np.trace(Sigma)

    return Sigma