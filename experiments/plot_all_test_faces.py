import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt

NOISY_PICS = 467
data = np.load(f'results/noisy_pics={NOISY_PICS}_TRexIter=5.npz')

dim1 = 168
dim2 = 192

X_test = data['X_test']
PCA_proj_test= data['PCA_proj_test']
SPCA_proj_test= data['SPCA_proj_test']
OP_proj_test = data['OP_proj_test']
SReaper_proj_test = data['SReaper_proj_test']
Tyler_proj_test = data['Tyler_proj_test']

column_titles = ["Original", "PCA", "S-PCA", "OP", "S-Reaper", "T-Rex"]

# -----------------------------------------------------------------------------
#                         Plot face images
# -----------------------------------------------------------------------------
for counter in range(4):
    fig,axs = plt.subplots(8, 6, figsize=(10, 14))
    for k in range(counter*8, (counter+1)*8):
        row = k - counter * 8
        axs[row, 0].imshow(np.reshape(X_test[:,k],(dim1, dim2)).T,cmap='gray')
        axs[row, 1].imshow(np.reshape(PCA_proj_test[:,k],(dim1,dim2)).T,cmap='gray')
        axs[row, 2].imshow(np.reshape(SPCA_proj_test[:,k],(dim1,dim2)).T,cmap='gray')
        axs[row, 3].imshow(np.reshape(OP_proj_test[:,k],(dim1,dim2)).T,cmap='gray')
        axs[row, 4].imshow(np.reshape(SReaper_proj_test[:,k],(dim1,dim2)).T,cmap='gray')
        axs[row, 5].imshow(np.reshape(Tyler_proj_test[:,k],(dim1,dim2)).T,cmap='gray')

        axs[row, 0].axis('off')
        axs[row, 1].axis('off')
        axs[row, 2].axis('off')
        axs[row, 3].axis('off')
        axs[row, 4].axis('off')
        axs[row, 5].axis('off')

    #if counter == 0:
    for col, title in enumerate(column_titles):
        axs[0, col].set_title(title, fontsize=20)

    plt.tight_layout()
    plt.savefig(f"figures/test_faces{counter}_noisy_pics={NOISY_PICS}.pdf")

# -----------------------------------------------------------------------------
#           Plot distance bewteen images and subspaces
# -----------------------------------------------------------------------------
PCA_dist =     LA.norm(PCA_proj_test - X_test, axis=0)
SPCA_dist =    LA.norm(SPCA_proj_test - X_test, axis=0)
OP_proj_dist = LA.norm(OP_proj_test - X_test, axis=0)
SReaper_dist = LA.norm(SReaper_proj_test - X_test, axis=0)
Ty_dist =      LA.norm(Tyler_proj_test - X_test, axis=0)


sorted_indices = np.argsort(PCA_dist)
PCA_dist = PCA_dist[sorted_indices]
SPCA_dist = SPCA_dist[sorted_indices]
OP_proj_dist = OP_proj_dist[sorted_indices]
SReaper_dist = SReaper_dist[sorted_indices]
Ty_dist = Ty_dist[sorted_indices]

smallest_distances_count = np.minimum(np.minimum(SPCA_dist, PCA_dist), np.minimum(OP_proj_dist, SReaper_dist))
smallest_distances_count = np.sum(Ty_dist == np.minimum(smallest_distances_count, Ty_dist))

print("smallest_distances_count: ", smallest_distances_count)

# Create the plot
plt.figure(figsize=(8, 6)) 
plt.plot(PCA_dist, PCA_dist, linestyle='-', color='k')
plt.plot(PCA_dist, SPCA_dist, marker='o', linestyle='-', color='b', label="S-PCA")
plt.plot(PCA_dist, OP_proj_dist, marker='v', linestyle='-', color='c', label="OP")
plt.plot(PCA_dist, SReaper_dist, marker='s', linestyle='-', color='g', label="S-Reaper")
plt.plot(PCA_dist, Ty_dist, marker='D', linestyle='-', color='r', label="T-Rex")

plt.xlabel('Distance to PCA subspace', fontsize=22)
plt.ylabel('Distance to robust subspace', fontsize=22)

# Add grid and legend
plt.grid(visible=True, linestyle='--', alpha=0.7)
plt.legend(fontsize=16)

# Customize ticks
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.savefig(f"figures/distance_plot_noisy_pics={NOISY_PICS}.pdf")