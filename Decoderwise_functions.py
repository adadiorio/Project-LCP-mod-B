###########################################################################################################################################
#                                  FUNCTIONS FOR THE PROCESSING AND ANALYSIS OF A PROMPT THROUGH A DECODER                                #
###########################################################################################################################################

# IMPORTS

# Python libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.colors as mcolors
import random
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import TwoSlopeNorm
from numpy import linalg as la
import matplotlib.gridspec as gridspec

# Transformers libraries
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

# Neighborhood and intrinsic dimension analysis
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from dadapy import Data
from sklearn.metrics.pairwise import pairwise_distances



# FUNCTIONS

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
    
    fig = plt.figure(figsize=(10, 5))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])
    ax_a = fig.add_subplot(gs[0, 0])  # a is on the top left
    ax_b = fig.add_subplot(gs[1, 0])  # b is on the bottom left
    ax_c = fig.add_subplot(gs[:, 1], projection='3d')  # c is on the right, spanning both rows

    ax_a.set_title(f'SVD for the {name} matrix')
    ax_a.plot(l_svd/sum(l_svd)*100)
    ax_a.set_xlim(-0.5,20.5)
    ax_a.set_xlabel('Component')
    ax_a.set_ylabel('% over the\ntotal variance')
    ax_a.grid()

    ax_b.plot(np.cumsum(l_svd/sum(l_svd)*100))
    ax_a.set_xlim(-0.5,20.5)
    ax_b.set_xlabel('Component')
    ax_b.set_ylabel('Cumulative sum\nover total variance [%]')
    ax_b.grid()
    ax_b.plot()

    pca = PCA(n_components = 3)

    principal_components = pca.fit_transform(vectors)

    indices = np.arange(0,vectors.shape[0])
   
    x = principal_components[:,0]
    y = principal_components[:,1]
    z = principal_components[:,2]

    ax_c.scatter(x, y, z, c=indices)
    ax_c.set_xlabel('PCA 1')
    ax_c.set_ylabel('PCA 2')
    ax_c.set_zlabel('PCA 3')
    ax_c.set_title(f'{name} PCA')
    
    plt.tight_layout()
    plt.show()
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
def metric_comparisons(results):
    
    metrics = ['euclidean', 'cosine']

    if np.shape(results)[0] < 100:
        metrics = ['euclidean']
    
    for i in range(len(metrics)):
        title_string = metrics[i]
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Real-valued metrics comparison: ' + title_string)
        
        dist_mat = pairwise_distances(results, results, metric=metrics[i])
        norm_dist_mat = np.max(dist_mat)
        
        im = axs[0].imshow(dist_mat/norm_dist_mat, cmap = 'inferno')
        fig.colorbar(im, ax=axs[0], shrink=0.75)
        axs[0].set_title("Distance matrix")
        
        d_distances = Data(results)
        d_distances.compute_distances(metric=metrics[i])
            
        ids_gride, errs_gride, scales_gride = d_distances.return_id_scaling_gride()
        
        axs[1].errorbar(scales_gride, ids_gride, errs_gride)
        axs[1].scatter(scales_gride, ids_gride, edgecolors="k", s=50)
        
        axs[1].set_xlabel(r"Scale", size=15)
        
        axs[1].grid()
        axs[1].set_ylabel("Estimated ID", size=15)
        axs[1].set_title('Estimated ID with DADApy Gride method')
        
        plt.tight_layout()
        plt.show()
# ---------------------------------------------------------------------------------------------------------------------------



# ---------------------------------------------------------------------------------------------------------------------------
def DecoderAnalysis(df_prompts_row, df_prompts_col, n_decoder, include_TSNE):

    read_prompts = pd.read_json('/home/ubuntu/Project/prompts.json')

    column_names = ['l', 'p', 'i', 't950', 't300', 't50']
    row_names = ['nat', 'ast', 'mat', 'psy', 'mus']
    prompt_name = str(column_names[df_prompts_col]) + '_' + str(row_names[df_prompts_row])

    prompt = read_prompts.iloc[df_prompts_row, df_prompts_col]

    if n_decoder == 1:
        PromptEmbedding(prompt, prompt_name, include_TSNE)
    
    df = pd.read_pickle("/mnt/DATA/output_" + prompt_name + "/word_to_vector.pkl")
    
    Attention(df, prompt_name, n_decoder, include_TSNE)
    SkipConnection(df, prompt_name, n_decoder, include_TSNE)
    FFNN(df, prompt_name, n_decoder, include_TSNE)    
# ---------------------------------------------------------------------------------------------------------------------------



# ---------------------------------------------------------------------------------------------------------------------------
def PromptEmbedding(prompt, prompt_name, include_TSNE):

    print("\nPROMPT EMBEDDING\n")

    model_id = 'openai-community/gpt2'
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, output_attentions=True)

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    gen_tokens = model.generate(
        input_ids,
        do_sample=True,
        temperature=3.,
        max_new_tokens=1,
    )
    
    gen_text = tokenizer.batch_decode(gen_tokens)[0]

    
    # VECTOR CONVERSIONS
    
    def get_code(code):
        return list(transformer_vocabulary.keys())[list(transformer_vocabulary.values()).index(code)]

    transformer_vocabulary = AutoTokenizer.from_pretrained(model_id).get_vocab()
    
    with torch.no_grad():
        embeddings = model.get_input_embeddings()(input_ids)

    words   = []
    tokens  = []
    vectors = []
    
    for i, (token, vector) in enumerate(zip(input_ids[0], embeddings[0])):
        words.append(get_code(token.item())[1:])
        tokens.append(token.item())
        vectors.append(vector.to(torch.float32).numpy())
    
    mapping = {
        'words':   words,
        'tokens':  tokens,
        'vectors': vectors
    }
    
    df = pd.DataFrame(mapping)
    
    words   = np.array(words)
    tokens  = np.array(tokens)
    vectors = np.array(vectors)
    
    # Saving in the output directory
    df.to_pickle("/mnt/DATA/output_" + prompt_name + "/word_to_vector.pkl")

    if include_TSNE == True:
        apply_TSNE(df, vectors)
    svd_and_pca(vectors)


    # POSITIONAL ENCODING

    # Accessing the positional embedding
    positional_embeddings = model.transformer.wpe.weight  # Tensor of positions
    print("Shape of positional embeddings matrix:", positional_embeddings.shape)
    
    # Sum of positional embedding matrix and input vectors matrix
    vectors_tensor = torch.from_numpy(vectors)  # Conversion of vectors to a torch object
    results = vectors_tensor + positional_embeddings[:tokens.shape[0], :]  # Sum
    results = results.detach().numpy()  # Conversion of the result in a numpy object

    if include_TSNE == True:
        apply_TSNE(df, results)
    svd_and_pca(positional_embeddings, 'positional embeddings')
    svd_and_pca(results, 'results')    
# ---------------------------------------------------------------------------------------------------------------------------



# ---------------------------------------------------------------------------------------------------------------------------
def Attention(df, prompt_name, n_decoder, include_TSNE):

    print("\nDECODER " + str(n_decoder) + " - ATTENTION LAYER\n")

    model_id = 'openai-community/gpt2'
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, output_attentions=True)
    
    attention = torch.load("/mnt/DATA/output_" + prompt_name + "/decoder/decoder_" + str(n_decoder) + "/AttentionHeads.pt")
    
    c_attn = model.transformer.h[0].attn.c_attn
    attention_weights = c_attn.weight
    attention_bias = c_attn.bias
    
    print("Attention weights' shape: ", attention_weights.shape)
    Wq = attention_weights[:, :768].detach().numpy()
    Wk = attention_weights[:, 768:-768].detach().numpy()
    Wv = attention_weights[:, -768:].detach().numpy()
    
    QKV_representation = torch.load("/mnt/DATA/output_" + prompt_name + "/decoder/decoder_" + str(n_decoder) + "/QKV_representation.pt")
    QKV_representation_matrix = QKV_representation.detach().numpy()[0,:,:]
    print(QKV_representation_matrix.shape)
    
    plt.figure(1, (40,20))
    norm = TwoSlopeNorm(vmin=-np.abs(QKV_representation_matrix).max(), vmax=np.abs(QKV_representation_matrix).max(), vcenter=0)
    plt.imshow(QKV_representation_matrix, cmap='seismic', norm=norm)
    plt.colorbar(shrink=0.25, pad=0.02)
    plt.title('QKV representation matrix')
    plt.show()

    # SPLIT IN 12 ATTENTION HEADS
    
    Attention_heads = torch.load("/mnt/DATA/output_" + prompt_name + "/decoder/decoder_" + str(n_decoder) + "/AttentionHeads.pt").detach().numpy()[0,]
    
    fig, axs = plt.subplots(2, 6, figsize=(20,6))
    
    norm = TwoSlopeNorm(vmin=0, vmax=1, vcenter=0.5)
    
    for i in range(2):
        for j in range(6):
            cax = axs[i,j].imshow(Attention_heads[2*j+i,]**0.25, cmap='gist_heat_r', norm=norm)
            fig.colorbar(cax, ax=axs[i,j], shrink=0.5, pad=0.02)
            axs[i,j].set_title('layer:'+str(2*j+i))
        
    plt.show()
    
    # CONCATENATE AND NORMALIZE
    
    attn_c_proj_weight = model.transformer.h[0].attn.c_proj.weight.detach().numpy()
    attn_c_proj_bias = model.transformer.h[0].attn.c_proj.bias.detach().numpy().flatten()
    Concat = torch.load("/mnt/DATA/output_" + prompt_name + "/decoder/decoder_" + str(n_decoder) + "/AttentionProj.pt").detach().numpy()[0,]
    
    svd_and_pca(Concat, 'Concat')
    if include_TSNE == True:
        apply_TSNE(df, Concat)
    metric_comparisons(Concat)
    
    # SECOND NORMALIZATION
    
    second_normalization = torch.load("/mnt/DATA/output_" + prompt_name + "/decoder/decoder_" + str(n_decoder) + "/SecondNormalization.pt").detach().numpy()[0]
    svd_and_pca(second_normalization, 'Second Normalization')
    if include_TSNE == True:
        apply_TSNE(df, second_normalization)
    metric_comparisons(second_normalization)    
# ---------------------------------------------------------------------------------------------------------------------------



# ---------------------------------------------------------------------------------------------------------------------------
def SkipConnection(df, prompt_name, n_decoder, include_TSNE):

    print("\nDECODER " + str(n_decoder) + " - SKIP CONNECTION\n")

    model_id = 'openai-community/gpt2'
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, output_attentions=True)
    
    output_sk1 = np.asarray(torch.load("/mnt/DATA/output_" + prompt_name + "/decoder/decoder_" + str(n_decoder) + "/AttentionPlusResidual.pt")[0][0])
    svd_and_pca(output_sk1, 'Attention + Residual Output')
    if include_TSNE == True:
        apply_TSNE(df, output_sk1)
    metric_comparisons(output_sk1)

    # DECODER OUTPUT

    output_sk2 = np.asarray(torch.load("/mnt/DATA/output_" + prompt_name + "/decoder/decoder_" + str(n_decoder) + "/Decoder_Final_Output.pt")[0][0])
    svd_and_pca(output_sk2, 'Decoder Output')
    if include_TSNE == True:
        apply_TSNE(df, output_sk2)
    metric_comparisons(output_sk2)    
# ---------------------------------------------------------------------------------------------------------------------------



# ---------------------------------------------------------------------------------------------------------------------------
def FFNN(df, prompt_name, n_decoder, include_TSNE):

    print("\nDECODER " + str(n_decoder) + " - FFNN LAYER\n")

    model_id = 'openai-community/gpt2'
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, output_attentions=True)
    
    FFNN = decoder = model.transformer.h[0].mlp

    first_layer_weight   = FFNN.c_fc.weight.detach().numpy()
    first_layer_bias     = FFNN.c_fc.bias.detach().numpy()
    second_layer_weight  = FFNN.c_proj.weight.detach().numpy()
    second_layer_bias    = FFNN.c_proj.bias.detach().numpy()
    
    print(FFNN)
    print("First  layer shape: ", first_layer_weight.shape)
    print("Second layer shape: ", second_layer_weight.shape)
        
    first_hidden_layer  = torch.load("/mnt/DATA/output_" + prompt_name + "/decoder/decoder_" + str(n_decoder) + "/FirstLayerNN.pt").detach().numpy()[0]
    second_hidden_layer = torch.load("/mnt/DATA/output_" + prompt_name + "/decoder/decoder_" + str(n_decoder) + "/SecondLayerNN.pt").detach().numpy()[0]
    
    svd_and_pca(first_hidden_layer, 'first hidden layer')
    metric_comparisons(first_hidden_layer)
    
    svd_and_pca(second_hidden_layer, 'second hidden layer')
    if include_TSNE == True:
        apply_TSNE(df, second_hidden_layer)
    metric_comparisons(second_hidden_layer)
# ---------------------------------------------------------------------------------------------------------------------------