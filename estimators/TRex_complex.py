from estimators.N_estimators import (sample_mean_and_cov_complex, 
                                     GFA_EM_complex)
from estimators.est_utils import PCFA_via_corr_complex
import numpy as np 
import numpy.linalg as LA


# EM algorithm for fitting low-rank plus diagonal Tyler

def TRex_complex(X, rank, outer_max_iter=20, inner_max_iter=200):
     S = sample_mean_and_cov_complex(X)[1]
     F, d = PCFA_via_corr_complex(S, rank)
     n, m = X.shape
     I = np.eye(rank)

     for ii in range(1, outer_max_iter + 1):
          # evaluate Sk (E-step)
          # since the MUSIC example has small dimensions, there is no need 
          # to take low-rank structure into account when solving the equation
          temp1 = LA.solve(F @ F.conj().T + np.diag(d), X)
          a = 1 / np.sum(X.conj() * temp1, axis=0)
          Sk = (a * X ) @ X.conj().T
          Sk = (n / m) * Sk
          
          F, d = GFA_EM_complex(Sk, rank, F0=F, d0=d, iter=inner_max_iter)
          
     stats = {"iterations": ii}
     
     return F, d, stats