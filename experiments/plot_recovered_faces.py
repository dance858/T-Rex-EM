import numpy as np
import matplotlib.pyplot as plt

data = np.load(f'results/noisy_pics=467_TRexIter=5.npz')

dim1 = 168
dim2 = 192

X_train = data['X_train']
X_test = data['X_test']
PCA_proj_train= data['PCA_proj_train']
SPCA_proj_train= data['SPCA_proj_train']
OP_proj_train = data['OP_proj_train']
SReaper_proj_train = data['SReaper_proj_train']
Tyler_proj_train = data['Tyler_proj_train']
PCA_proj_test= data['PCA_proj_test']
SPCA_proj_test= data['SPCA_proj_test']
OP_proj_test = data['OP_proj_test']
SReaper_proj_test = data['SReaper_proj_test']
Tyler_proj_test = data['Tyler_proj_test']


train_k = [18, 22]
test_k = [10, 17]
fig,axs = plt.subplots(len(train_k) + len(test_k), 6, figsize=(10, 7))

column_titles = ["Original", "PCA", "S-PCA", "OP", "S-Reaper", "T-Rex"]

# plot two train images
counter = 0
for k in train_k:
    print("counter: ", counter)
    axs[counter, 0].imshow(np.reshape(X_train[:,k],(dim1, dim2)).T,cmap='gray')
    axs[counter, 1].imshow(np.reshape(PCA_proj_train[:,k],(dim1,dim2)).T,cmap='gray')
    axs[counter, 2].imshow(np.reshape(SPCA_proj_train[:,k],(dim1,dim2)).T,cmap='gray')
    axs[counter, 3].imshow(np.reshape(OP_proj_train[:,k],(dim1,dim2)).T,cmap='gray')
    axs[counter, 4].imshow(np.reshape(SReaper_proj_train[:,k],(dim1,dim2)).T,cmap='gray')
    axs[counter, 5].imshow(np.reshape(Tyler_proj_train[:,k],(dim1,dim2)).T,cmap='gray')

    axs[counter, 0].spines['top'].set_visible(False)
    axs[counter, 0].spines['right'].set_visible(False)
    axs[counter, 0].spines['left'].set_visible(False)
    axs[counter, 0].spines['bottom'].set_visible(False)
    axs[counter, 0].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    axs[counter, 1].axis('off')
    axs[counter, 2].axis('off')
    axs[counter, 3].axis('off')
    axs[counter, 4].axis('off')
    axs[counter, 5].axis('off')

    if counter == 0:
        for col, title in enumerate(column_titles):
            axs[0, col].set_title(title, fontsize=18)
    
    axs[counter, 0].set_ylabel("In-sample", fontsize=14)

    counter += 1



# plot two test images
for k in test_k:
    print("counter: ", counter)
    axs[counter, 0].imshow(np.reshape(X_test[:,k],(dim1, dim2)).T,cmap='gray')
    axs[counter, 1].imshow(np.reshape(PCA_proj_test[:,k],(dim1,dim2)).T,cmap='gray')
    axs[counter, 2].imshow(np.reshape(SPCA_proj_test[:,k],(dim1,dim2)).T,cmap='gray')
    axs[counter, 3].imshow(np.reshape(OP_proj_test[:,k],(dim1,dim2)).T,cmap='gray')
    axs[counter, 4].imshow(np.reshape(SReaper_proj_test[:,k],(dim1,dim2)).T,cmap='gray')
    axs[counter, 5].imshow(np.reshape(Tyler_proj_test[:,k],(dim1,dim2)).T,cmap='gray')

    axs[counter, 0].spines['top'].set_visible(False)
    axs[counter, 0].spines['right'].set_visible(False)
    axs[counter, 0].spines['left'].set_visible(False)
    axs[counter, 0].spines['bottom'].set_visible(False)
    axs[counter, 0].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    axs[counter, 1].axis('off')
    axs[counter, 2].axis('off')
    axs[counter, 3].axis('off')
    axs[counter, 4].axis('off')
    axs[counter, 5].axis('off')

    axs[counter, 0].set_ylabel("Out-of-sample", fontsize=14)

    counter += 1



plt.tight_layout()
plt.savefig(f"figures/recovered_faces.pdf", bbox_inches='tight')