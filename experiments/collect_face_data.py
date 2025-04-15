import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.io
import numpy.linalg as LA
import cv2
import sys 
import pdb
sys.path.append("../")
from estimators.Tyler_estimators import TRex_simple_via_data
from estimators.subspace_recovery import (Reaper_optimized, outlier_pursuit_via_ADMM,
                                          Euclidean_median_via_Weiszfield, fit_PCA_subspace)

methods = ["PCA", "S-PCA", "S-Reaper", "Tyler-FA-EM", "OP"]
RANK = 9
MAX_ITER_OP = 50
MAX_ITER_REAPER = 50
MAX_ITER_TYLER = 5

# ------------------------------------------------------------------------------
#                              Load data 
# ------------------------------------------------------------------------------ 
print("Processing data ....")
plt.rcParams['figure.figsize'] = [7, 7]
plt.rcParams.update({'font.size': 18})
mat = scipy.io.loadmat('data/allFaces.mat')
faces = mat['faces']
nfaces = mat['nfaces'].reshape(-1)
X_original = faces[:,:nfaces[0]]
dim1_downsampled = 168       
dim2_downsampled = 192

# split into training and test set
X_train = X_original[:, :32]
X_test = X_original[:, 32:]

# add some random pictures to the training set
ADD_NOISY_PICTURES = True
if ADD_NOISY_PICTURES:
    folder = f'data/noise_dim1={dim1_downsampled}_dim2={dim2_downsampled}/'
    MAX_NOISE_PICTURES = 467
    counter = 0

    for filename in os.listdir(folder):
        if counter == MAX_NOISE_PICTURES:
            break
        if filename.endswith(('.jpg')):  
            counter += 1
            image_path = os.path.join(folder, filename)
            image = cv2.imread(image_path)[:, :, 0]
            X_train = np.hstack((X_train, image.reshape(-1, 1)))

print("Processing data finished!")
# ------------------------------------------------------------------------------
#             Center the data using he Euclidean median
# ------------------------------------------------------------------------------
print("Centering data ....")
center_train = Euclidean_median_via_Weiszfield(X_train, max_iter=500, delta=1e-8)
X_train_centered = X_train - center_train 
X_test_centered = X_test - center_train 
print("Centering finished!")

# ------------------------------------------------------------------------------
#                       Apply PCA and SPCA
# ------------------------------------------------------------------------------
if "PCA" in methods:
    print("Applying PCA and S-PCA....")
    Ur_PCA = fit_PCA_subspace(X_train_centered, RANK, spherization=False)
    PCA_proj_train = Ur_PCA @ (Ur_PCA.T @ X_train_centered) + center_train 
    PCA_proj_test = Ur_PCA @ (Ur_PCA.T @ X_test_centered) + center_train 

    Ur_SPCA = fit_PCA_subspace(X_train_centered, RANK, spherization=True)
    SPCA_proj_train = Ur_SPCA @ (Ur_SPCA.T @ X_train_centered) + center_train 
    SPCA_proj_test = Ur_SPCA @ (Ur_SPCA.T @ X_test_centered) + center_train
    print("PCA and S-PCA finished!")



# ------------------------------------------------------------------------------
#                       Apply S-Reaper
# ------------------------------------------------------------------------------
if "S-Reaper" in methods:
    print("Applying S-Reaper ....")
    X_train_centered_spherized = X_train_centered / LA.norm(X_train_centered, axis=0)
    v, U, obj = Reaper_optimized(X_train_centered_spherized, RANK, max_iter=MAX_ITER_REAPER)
    
    U_SReaper = U[:, 0:RANK]
    SReaper_proj_train = U_SReaper @ (U_SReaper.T @ X_train_centered) + center_train  
    SReaper_proj_test = U_SReaper @ (U_SReaper.T @ X_test_centered) + center_train  
    print("S-Reaper finished!")

# ------------------------------------------------------------------------------
#                         Apply outlier pursuit
# ------------------------------------------------------------------------------
if "OP" in methods:
    print("Applying outlier pursuit ....")
    lmbda = 0.8 * np.sqrt(X_train_centered.shape[0]/X_train_centered.shape[1])
    obj, rel_pri_res, P_ADMM = outlier_pursuit_via_ADMM(X_train_centered, lmbda, max_iter=MAX_ITER_OP) 
    U, S, VT = LA.svd(P_ADMM, full_matrices=False)
    U_OP = U[:, 0:RANK]
    OP_proj_train = U_OP @ (U_OP.T @ X_train_centered) + center_train 
    OP_proj_test = U_OP @ (U_OP.T @ X_test_centered) + center_train    
    print("OP finished! ")
# ------------------------------------------------------------------------------
#                       Apply Tyler-FA-EM
# ------------------------------------------------------------------------------
if "Tyler-FA-EM" in methods:
    print("Applying Tyler-FA-EM ....")
    F_Ty, d_Ty, stats_Ty = TRex_simple_via_data(X_train_centered, RANK,
                                    outer_max_iter=MAX_ITER_TYLER, inner_max_iter=200)

    U_Ty, _ = np.linalg.qr(F_Ty, mode='reduced')
    Tyler_proj_train = U_Ty @ (U_Ty.T @ X_train_centered) + center_train 
    Tyler_proj_test = U_Ty @ (U_Ty.T @ X_test_centered) + center_train  
    print("Tyler-FA-EM finished!")


# ------------------------------------------------------------------------------
#                       Store all data
# ------------------------------------------------------------------------------
np.savez(f'results/noisy_pics={MAX_NOISE_PICTURES}_TRexIter={MAX_ITER_TYLER}.npz', X_train=X_train, 
         X_test=X_test,
        Ur_PCA=Ur_PCA, PCA_proj_train=PCA_proj_train, PCA_proj_test=PCA_proj_test,
        Ur_SPCA=Ur_SPCA, SPCA_proj_train=SPCA_proj_train, SPCA_proj_test=SPCA_proj_test,
        U_SReaper=U_SReaper, SReaper_proj_train=SReaper_proj_train, SReaper_proj_test=SReaper_proj_test,
        U_OP=U_OP, OP_proj_train=OP_proj_train, OP_proj_test=OP_proj_test,
        U_Ty=U_Ty, Tyler_proj_train=Tyler_proj_train,
        Tyler_proj_test=Tyler_proj_test, F_Ty=F_Ty, d_Ty=d_Ty, stats_Ty=stats_Ty)
