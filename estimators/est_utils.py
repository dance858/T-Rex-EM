import numpy as np 
import numpy.linalg as LA 

TERMINATE_MAX_ITER = "Terminated because of MAX_ITER"
TERMINATE_OBJ = "Terminated because of OBJ"

# used for initialization
def PCFA_via_corr(sigma, k):
    vola = np.sqrt(np.diag(sigma)).reshape(-1, 1)
    R = (1 / vola) * sigma * (1 / vola).T
    lmbda, Q = LA.eigh(R)
    lmbda = lmbda[::-1][0:k]
    Q = Q[:, ::-1][:, 0:k]

    # low-rank approximation of correlation matrix
    F = (Q @ np.diag(np.sqrt(lmbda)))
    d = np.diag(R - F @ F.T)
   
    # scale so it becomes low rank approximation of covariance matrix
    F = vola * F 
    d = (np.squeeze(vola)**2) * d
    
    return F, d

def PCFA_via_corr_complex(sigma, k):
    vola = np.sqrt(np.diag(sigma)).reshape(-1, 1)
    vola = np.real(vola)
    R = (1 / vola) * sigma * (1 / vola).T
    lmbda, Q = LA.eigh(R)
    lmbda = lmbda[::-1][0:k]
    Q = Q[:, ::-1][:, 0:k]

    # low-rank approximation of correlation matrix
    F = (Q @ np.diag(np.sqrt(lmbda)))
    d = np.diag(R - F @ F.conj().T)
    d = np.real(d)
    
    # scale so it becomes low rank approximation of covariance matrix
    F = vola * F 
    d = (np.squeeze(vola)**2) * d
    
    return F, d



