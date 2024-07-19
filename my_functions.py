### LIBRARIES ###

# Python libraries
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D  # Importing the 3D plotting module
import matplotlib.colors as mcolors
from matplotlib.colors import TwoSlopeNorm
from sklearn.decomposition import PCA
from numpy import linalg as la
from scipy.spatial import distance_matrix
import torch

# Neighborhood analysis
from sklearn.manifold import TSNE

# Surface reconstruction
from scipy.spatial import Delaunay



### FUNCTIONS ###

# ---------------------------------------------------------------------------------------------------------------------------
def pca(matrix, name = 'matrix', n_components = 3, cmap = True):
    if type(matrix) == torch.Tensor:
        matrix = matrix.detach().numpy()
        
    elif type(matrix) == torch.nn.parameter.Parameter:
        matrix = matrix.detach().numpy()
        
    pca = PCA(n_components = 3)

    principal_components = pca.fit_transform(matrix)

    indices = np.arange(0,matrix.shape[0])
   
    x = principal_components[:,0]
    y = principal_components[:,1]
    z = principal_components[:,2]

    #ax.plot_surface(x, y, z, cmap = 'viridis')
    fig = plt.figure(figsize=(16,8))
    axes = fig.add_subplot(121, projection='3d')

    axes.scatter(x, y, z, c=indices)
    axes.set_xlabel('PCA 1')
    axes.set_ylabel('PCA 2')
    axes.set_zlabel('PCA 3')
    axes.set_title(f'{name} PCA')

    if cmap == True:
       ax2 = fig.add_subplot(122)
       norm = TwoSlopeNorm(vmin=-np.max(np.abs(matrix)), vmax=np.max(np.abs(matrix)), vcenter=0)
       heatmap = ax2.imshow(matrix, cmap='seismic', norm=norm)
       fig.colorbar(heatmap, ax=ax2, shrink = 0.4)
       ax2.set_title(f'{name} weight')

    plt.show()
# ---------------------------------------------------------------------------------------------------------------------------



# ---------------------------------------------------------------------------------------------------------------------------
def svd(vectors, name = 'vectors', text = False, values = False):
    
    if type(vectors) == torch.Tensor:
        vectors = vectors.detach().numpy()
    elif type(vectors) == torch.nn.parameter.Parameter:
        vectors = vectors.detach().numpy()
    
    N = vectors.shape[0]
    U, spectrum, Vt = la.svd(vectors)
    l_svd = (spectrum ** 2)/(N-1)
    V_svd = U

    fig, axes = plt.subplots(1, 2, figsize=(12,5))
    axes[0].plot(l_svd/sum(l_svd)*100)
    axes[0].set_xlim(0,10)
    axes[0].set_xlabel('Component')
    axes[0].set_ylabel('% over the total variance')
    axes[0].grid()

    axes[1].plot(np.cumsum(l_svd/sum(l_svd)*100))
    axes[1].set_xlabel('Component')
    axes[1].set_ylabel('Cumulative sum over total variance [%]')
    axes[1].grid()
    axes[1].plot()


    fig.suptitle(f'SVD for the {name} matrix', fontsize=16)

    plt.show()

    if text == True:
       print(f'Eigenvalues ({name} matrix): \n', Vt.transpose())
       print('\n')
       print(f'Eigenvectors ({name} matrix): \n', U)

    elif values == True:
       return U, spectrum, Vt # remove the comment to obtain the eigenvectors and eigenvalues
# ---------------------------------------------------------------------------------------------------------------------------



# ---------------------------------------------------------------------------------------------------------------------------
def svd_and_pca(vectors, name = 'vectors', text = False, values = False, cmap = True):
    
    if type(vectors) == torch.Tensor:
        vectors = vectors.detach().numpy()
    elif type(vectors) == torch.nn.parameter.Parameter:
        vectors = vectors.detach().numpy()
    
    N = vectors.shape[0]
    U, spectrum, Vt = la.svd(vectors)
    l_svd = (spectrum ** 2)/(N-1)
    V_svd = U

    fig, axes = plt.subplots(1, 2, figsize=(12,5))
    axes[0].plot(l_svd/sum(l_svd)*100)
    axes[0].set_xlim(0,10)
    axes[0].set_xlabel('Component')
    axes[0].set_ylabel('% over the total variance')
    axes[0].grid()

    axes[1].plot(np.cumsum(l_svd/sum(l_svd)*100))
    axes[1].set_xlabel('Component')
    axes[1].set_ylabel('Cumulative sum over total variance [%]')
    axes[1].grid()
    axes[1].plot()

    fig.suptitle(f'SVD for the {name} matrix', fontsize=16)

    if text == True:
       print(f'Eigenvalues ({name} matrix): \n', Vt.transpose())
       print('\n')
       print(f'Eigenvectors ({name} matrix): \n', U)


    pca = PCA(n_components = 3)

    principal_components = pca.fit_transform(vectors)

    indices = np.arange(0,vectors.shape[0])
   
    x = principal_components[:,0]
    y = principal_components[:,1]
    z = principal_components[:,2]

    #ax.plot_surface(x, y, z, cmap = 'viridis')
    fig = plt.figure(figsize=(16,8))
    axes = fig.add_subplot(121, projection='3d')

    axes.scatter(x, y, z, c=indices)
    axes.set_xlabel('PCA 1')
    axes.set_ylabel('PCA 2')
    axes.set_zlabel('PCA 3')
    axes.set_title(f'{name} PCA')

    if cmap == True:
       ax2 = fig.add_subplot(122)
       norm = TwoSlopeNorm(vmin=-np.max(np.abs(vectors)), vmax=np.max(np.abs(vectors)), vcenter=0)
       heatmap = ax2.imshow(vectors, cmap='seismic', norm=norm)
       fig.colorbar(heatmap, ax=ax2, shrink = 0.4)
       ax2.set_title(f'{name} weight')

    plt.show()

    if values == True:
       return U, spectrum, Vt # remove the comment to obtain the eigenvectors and eigenvalues
# ---------------------------------------------------------------------------------------------------------------------------



# ---------------------------------------------------------------------------------------------------------------------------
def apply_TSNE(df, vectors, perplexity_value = 10, n_components = 3, random_state = 42):    
    tsne   = TSNE(n_components = n_components, random_state = random_state, perplexity = perplexity_value)
    tsne2D = TSNE(n_components = n_components, random_state = random_state, perplexity = perplexity_value)
    
    embedded_vectors   = tsne.fit_transform(vectors)
    embedded_vectors2D = tsne2D.fit_transform(vectors)
    
    label_colors = {}
    colors = []
    
    string_values = df[df['words'].apply(lambda x: isinstance(x, str))]['words'].tolist()
    all_colors = list(mcolors.CSS4_COLORS)
    
    fig = plt.figure(figsize=(200, 200))
    ax = fig.add_subplot(211, projection='3d')
    indices = np.arange(0,np.shape(embedded_vectors)[0])
    ax.scatter(embedded_vectors[:, 0], embedded_vectors[:, 1], embedded_vectors[:, 2], s=200, label = df['words'], c=indices[:])
    
    counter = 0
    for txt in string_values:
        ax.text(embedded_vectors[counter, 0], embedded_vectors[counter, 1], embedded_vectors[counter, 2], txt)
        counter += 1
    
    ax.set_title(f't-SNE Visualization in 3D with Perplexity={perplexity_value}')
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_zlabel('Component 3')
    plt.show()
# ---------------------------------------------------------------------------------------------------------------------------



# ---------------------------------------------------------------------------------------------------------------------------
def surface_reconstruction(df, vectors, perplexity_value=10, n_components=3, random_state=42, surface_alpha = 0.5):
    # Perform t-SNE embedding
    tsne = TSNE(n_components=n_components, random_state=random_state, perplexity=perplexity_value)
    embedded_vectors = tsne.fit_transform(vectors)

    # Perform Delaunay triangulation
    tri = Delaunay(embedded_vectors)

     # Create subplots
    fig = plt.figure(figsize=(15, 10))

    # Subplot 1
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.plot_trisurf(embedded_vectors[:, 0], embedded_vectors[:, 1], embedded_vectors[:, 2], triangles=tri.simplices, color='r', alpha=surface_alpha)
    ax1.set_title('View 1')
    ax1.view_init(elev=30, azim=45)

    # Subplot 2
    ax2 = fig.add_subplot(222, projection='3d')
    ax2.plot_trisurf(embedded_vectors[:, 0], embedded_vectors[:, 1], embedded_vectors[:, 2], triangles=tri.simplices, color='g', alpha=surface_alpha)
    ax2.set_title('View 2')
    ax2.view_init(elev=20, azim=135)

    # Subplot 3
    ax3 = fig.add_subplot(223, projection='3d')
    ax3.plot_trisurf(embedded_vectors[:, 0], embedded_vectors[:, 1], embedded_vectors[:, 2], triangles=tri.simplices, color='b', alpha=surface_alpha)
    ax3.set_title('View 3')
    ax3.view_init(elev=10, azim=225)

    # Subplot 4
    ax4 = fig.add_subplot(224, projection='3d')
    ax4.plot_trisurf(embedded_vectors[:, 0], embedded_vectors[:, 1], embedded_vectors[:, 2], triangles=tri.simplices, color='y', alpha=surface_alpha)
    ax4.set_title('View 4')
    ax4.view_init(elev=40, azim=315)

    plt.show()
# ---------------------------------------------------------------------------------------------------------------------------


def metric_comparisons(results):
    from dadapy import Data
    from sklearn.metrics.pairwise import pairwise_distances
    
    metrics = ['sqeuclidean', 'chebyshev', 'braycurtis', 'cosine', 'euclidean', 'manhattan', 'correlation', 'minkowski', 'canberra']
    
    cols = 3
    rows = 3
    fig, axs = plt.subplots(rows, cols, figsize=(15,15))
    fig.suptitle('Normalized distance matrices for different real-valued metrics')
    metric_ID = []
    list_2nn = []
    list_gride = []
    
    for i in range(rows):
        for j in range(cols):            
            dist_mat = pairwise_distances(results, results, metric=metrics[cols*i + j])
            norm_dist_mat = np.max(dist_mat)
            title_string = metrics[cols*i + j]
            
            im = axs[i, j].imshow(dist_mat/norm_dist_mat, cmap = 'inferno')
            fig.colorbar(im, ax=axs[i, j], shrink=0.75)
            axs[i, j].set_title(title_string)
            
            d_distances = Data(results)
            d_distances.compute_distances(metric=metrics[cols*i + j])
            metric_ID.append(d_distances.compute_id_2NN())
    
            ids_2nn, errs_2nn, scales_2nn = d_distances.return_id_scaling_2NN()
            ids_gride, errs_gride, scales_gride = d_distances.return_id_scaling_gride()
            list_2nn.append([ids_2nn, errs_2nn, scales_2nn])
            list_gride.append([ids_gride, errs_gride, scales_gride])
            
    plt.tight_layout()
    plt.show()

    #####################################################################################################
    
    fig, axs = plt.subplots(rows, cols, figsize=(15,10))
    fig.suptitle('Estimated ID for different scale values and different metrics')
    
    for i in range(rows):
        for j in range(cols):
            col = "darkorange"
            title_string = metrics[cols*i + j]
            
            axs[i, j].plot(list_2nn[cols*i+j][2], list_2nn[cols*i+j][0], alpha=0.85)
            axs[i, j].errorbar(list_2nn[cols*i+j][2], list_2nn[cols*i+j][0], list_2nn[cols*i+j][1], fmt="None")
            axs[i, j].scatter(list_2nn[cols*i+j][2], list_2nn[cols*i+j][0], edgecolors="k", s=50, label="2nn decimation")
            axs[i, j].plot(list_gride[cols*i+j][2], list_gride[cols*i+j][0], alpha=0.85, color=col)
            axs[i, j].errorbar(list_gride[cols*i+j][2], list_gride[cols*i+j][0], list_gride[cols*i+j][1], fmt="None", color=col)
            axs[i, j].scatter(list_gride[cols*i+j][2], list_gride[cols*i+j][0], edgecolors="k", color=col, s=50, label="2nn gride")
            axs[i, j].set_xlabel(r"Scale", size=15)
            axs[i, j].set_ylabel("Estimated ID", size=15)
            axs[i, j].legend(frameon=False, fontsize=14)
            axs[i, j].set_title(title_string)
            
    plt.tight_layout()
    plt.show()

    #####################################################################################################
    
    colormap = plt.get_cmap('tab10')
    fig = plt.figure(figsize=(8, 4))
    
    for i in range(rows):
        for j in range(cols):
            color_index = cols * i + j
            col = colormap(color_index % 20)  # Ottenere un colore dalla colormap
            title_string = metrics[cols * i + j]
            label = f"Plot {color_index + 1}: {title_string}"  # Creare un'etichetta per la legenda
    
            plt.plot(list_gride[cols * i + j][0], alpha=0.85, color=col, label=label, marker='o')
            #plt.errorbar(list_gride[cols * i + j][2], list_gride[cols * i + j][0], list_gride[cols * i + j][1], fmt="None", color=col)
            #plt.scatter(list_gride[cols * i + j][2], list_gride[cols * i + j][0], edgecolors="k", color=col, s=50)
    
    plt.xlabel(r"Scale", size=15)
    
    plt.grid()
    plt.ylabel("Estimated ID", size=15)
    plt.legend()  # Aggiungere la legenda
    plt.title('Gride for different metrics')
    plt.show()