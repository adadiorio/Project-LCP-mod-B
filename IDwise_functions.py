###########################################################################################################################################
#                                      FUNCTIONS FOR THE DIMENSIONALITY ANALYSIS OF MULTIPLE PROMPTS                                      #
###########################################################################################################################################

# IMPORTS

# Python libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
from numpy import linalg as la
import imageio
import mpmath

# Transformers libraries
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2LMHeadModel, GPT2Model
import torch.nn.functional as F

# Neighborhood and intrinsic dimension analysis
import scipy
from scipy.spatial.distance import cosine
from scipy.stats import entropy
from functools import reduce
from sklearn.metrics.pairwise import cosine_similarity

import os
import sys
sys.path.append('/home/ubuntu/Project')



# FUNCTIONS

# ---------------------------------------------------------------------------------------------------------------------------
def Prompt_whole_processing(df_prompts_row, df_prompts_col, output_gif, compute_kl):
    
    read_prompts = pd.read_json('prompts.json')

    column_names = ['l', 'p', 'i', 't950', 't300', 't50']
    row_names = ['nat', 'ast', 'mat', 'psy', 'mus']
    prompt_name = str(column_names[df_prompts_col]) + '_' + str(row_names[df_prompts_row])

    prompt = read_prompts.iloc[df_prompts_row, df_prompts_col]
    
    PreRun(prompt, prompt_name)
    Last_Token_Analysis(prompt, prompt_name, output_gif, compute_kl)
    Dim_evolution(prompt, prompt_name)
# ---------------------------------------------------------------------------------------------------------------------------

    
# ANALYSIS ON MULTIPLE PROMPTS FUNCTIONS

# ---------------------------------------------------------------------------------------------------------------------------
def Prompt_last_token_comparison(list_prompt_locations, percentage):
    
    read_prompts = pd.read_json('prompts.json')

    column_names = ['l', 'p', 'i', 't950', 't300', 't50']
    row_names = ['nat', 'ast', 'mat', 'psy', 'mus']

    list_prompts = [read_prompts.iloc[list_prompt_location[0], list_prompt_location[1]] for list_prompt_location in list_prompt_locations]
    list_prompt_names = [str(column_names[list_prompt_location[1]]) + '_' + str(row_names[list_prompt_location[0]]) for list_prompt_location in list_prompt_locations]
    n_prompts = len(list_prompts)

    # Define the directory containing the .pt files
    directory = "/mnt/DATA/output_" + prompt_name + "/last_token_pdf/"

    # Define the directory containing the .pt files
    directory = "/mnt/DATA/output_" + prompt_name + "/last_token_pdf/"
    
    # Load the probability distribution for decoder_12
    decoder_12 = load_decoder_softmax(12, directory)
    
    # Initialize lists to store results
    cos_similarities = []
    decoder_indices = list(range(1, 12))
    
    # Loop through decoders 1 to 11 and calculate distances
    for i in decoder_indices:
        attention_i = load_attention_softmax(i, directory)
        cos_sim = cosine_similarity(attention_i.detach().numpy().reshape(1, -1), decoder_12.detach().numpy().reshape(1, -1))
        cos_similarities.append(cos_sim)
        decoder_i = load_decoder_softmax(i, directory)
        cos_sim = cosine_similarity(decoder_i.detach().numpy().reshape(1, -1), decoder_12.detach().numpy().reshape(1, -1))
        cos_similarities.append(cos_sim)
    
    # Plot the results in a single plot
    plt.figure(figsize=(15, 10))
    
    x_plot = np.arange(1, 12, 0.5)
    plt.plot(x_plot, cos_similarities, marker='s', label='Cosine Similarity', color='r')
    
    plt.title('Statistical Distances Between Decoders and Decoder 12')
    plt.xlabel('Decoder Index')
    plt.ylabel('Value')
    plt.ylim(0, 0.1)
    #plt.yscale('log')
    plt.legend()
    plt.grid(True)
    
    for i in np.arange(0, 12):
        plt.axvline(x=i+1, color='black', linestyle='--', linewidth=1)
    
    plt.tight_layout()
    plt.show()
    
    print("\nPrompt_" + prompt_name + ": Last Token Analysis finished.")
# ---------------------------------------------------------------------------------------------------------------------------



# ---------------------------------------------------------------------------------------------------------------------------
def Prompt_dimensionality_comparison(list_prompt_locations, percentage):

    read_prompts = pd.read_json('prompts.json')

    column_names = ['l', 'p', 'i', 't950', 't300', 't50']
    row_names = ['nat', 'ast', 'mat', 'psy', 'mus']

    list_prompts = [read_prompts.iloc[list_prompt_location[0], list_prompt_location[1]] for list_prompt_location in list_prompt_locations]
    list_prompt_names = [str(column_names[list_prompt_location[1]]) + '_' + str(row_names[list_prompt_location[0]]) for list_prompt_location in list_prompt_locations]
    n_prompts = len(list_prompts)
    
    # BEFORE SKIP CONNECTION
    
    module_name = ["AttentionProj", "SecondLayerNN"]

    components = [[] for n in range(n_prompts)]
    grams      = [[] for n in range(n_prompts)]
    norms      = [[] for n in range(n_prompts)]
    cos_sims   = [[] for n in range(n_prompts)]
    list_prompt_lengths = []
    
    for i in range(1,13):
        for mn in module_name:
            for prompt_idx, (single_prompt_components, single_prompt_gram, single_prompt_norm, single_prompt_cos_sim) in enumerate(zip(components, grams, norms, cos_sims)):
                v = torch.load("/mnt/DATA/output_" + list_prompt_names[prompt_idx] + "/decoder/decoder_" + str(i) + "/" + mn + ".pt").detach().numpy()[0,]
                single_prompt_components.append(n_component(v, percent = percentage))
                single_prompt_gram.append(log_gram_det(v))
                single_prompt_norm.append(np.mean(np.linalg.norm(v - np.mean(v), axis=1)))
                single_prompt_cos_sim.append(np.mean(cosine_similarity_matrix(v)))
                list_prompt_lengths.append(len(v))
        
    list_prompt_lengths = list_prompt_lengths[:n_prompts]
    print('Prompt lengths:', list_prompt_lengths)
    
    # PLOTS

    # Colors
    colormap = plt.get_cmap('viridis')
    colors = [colormap(n / n_prompts) for n in range(n_prompts)]
    
    norm_vol = [[g/prompt_length for g in gram] for (gram, prompt_length) in zip(grams, list_prompt_lengths)]
    
    after_skip = False
    dimensionality_evolution_plot(components, list_prompt_names, colors, percentage, after_skip)
    volume_plot(norm_vol, list_prompt_names, colors, after_skip)
    mean_norm_plot(norms, list_prompt_names, colors, after_skip)
    mean_cosine_similarity_plot(cos_sims, list_prompt_names, colors, after_skip)

    # AFTER SKIP CONNECTION

    module_name = ["AttentionPlusResidual", "Decoder_Final_Output"]

    components = [[] for n in range(n_prompts)]
    grams      = [[] for n in range(n_prompts)]
    norms      = [[] for n in range(n_prompts)]
    cos_sims   = [[] for n in range(n_prompts)]
    
    for i in range(1,13):
        for mn in module_name:
            for prompt_idx, (single_prompt_components, single_prompt_gram, single_prompt_norm, single_prompt_cos_sim) in enumerate(zip(components, grams, norms, cos_sims)):
                v = torch.load("/mnt/DATA/output_" + list_prompt_names[prompt_idx] + "/decoder/decoder_" + str(i) + "/" + mn + ".pt")[0].detach().numpy()[0,]
                single_prompt_components.append(n_component(v, percent = percentage))
                single_prompt_gram.append(log_gram_det(v))
                single_prompt_norm.append(np.mean(np.linalg.norm(v - np.mean(v), axis=1)))
                single_prompt_cos_sim.append(np.mean(cosine_similarity_matrix(v)))

    norm_vol = [[g/prompt_length for g in gram] for (gram, prompt_length) in zip(grams, list_prompt_lengths)]
    
    # PLOTS
    after_skip = True
    dimensionality_evolution_plot(components, list_prompt_names, colors, percentage, after_skip)
    volume_plot(norm_vol, list_prompt_names, colors, after_skip)
    mean_norm_plot(norms, list_prompt_names, colors, after_skip)
    mean_cosine_similarity_plot(cos_sims, list_prompt_names, colors, after_skip)

    print("\nComparative Dimensionality Evolution Analysis finished.")
# ---------------------------------------------------------------------------------------------------------------------------



# ---------------------------------------------------------------------------------------------------------------------------
def PreRun(prompt, prompt_name):
    
    directory = "/mnt/DATA/output_" + prompt_name
    
    model_id = 'openai-community/gpt2'
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, output_attentions=True)
    gpt2_model = model.transformer
    
    # Function to be called by the hook
    output_list, module_list = [], []

    def hook_fn(module, input, output):
        output_list.append(output)
        module_list.append(module)
    
    # Attaching hook to all layers
    for layer in model.modules():
        layer.register_forward_hook(hook_fn)

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    with torch.no_grad():
        outputs = model(input_ids)

    Token2Vec          = output_list[0][0]
    PositionalEncoding = output_list[1][0]
    PositionPlusVect   = output_list[2][0]
    
    torch.save(Token2Vec, directory + "/word2vec/Token2Vec.pt")
    torch.save(PositionalEncoding, directory + "/word2vec/PositionalEncoding.pt")
    torch.save(PositionPlusVect, directory + "/word2vec/PositionPlusVect.pt")

    index_list = np.array([3, 4, 5, 7, 9, 11, 13, 15])

    module_name = ["FirstNormalization", "QKV_representation", "AttentionHeads", "AttentionProj", "SecondNormalization", "FirstLayerNN", "SecondLayerNN", "Decoder_Final_Output"]
    # Create Decoder_mask and flatten it
    Decoder_mask = np.concatenate([index_list + i*13 for i in range(12)])
    
    # Assuming output_list is already defined, we can proceed
    # Extract elements for Decoder_list
    Decoder_list = [output_list[mask] for mask in Decoder_mask]
    
    PositionalEmbedding = output_list[2]
    
    Decoder_list = [output_list[mask] for mask in Decoder_mask]
    
    for i in range(12):
        for j in range(8):
            torch.save(Decoder_list[j+i*8], directory + "/decoder/decoder_"+str(i+1) +"/"+module_name[j]+".pt")

    # Extract also Output Attention + Residual connection

    SecondLayerNN_list         =  [torch.load(directory + f"/decoder/decoder_{i+1}/SecondLayerNN.pt")[0] for i in range(12)]
    Decoder_Final_Output_list  =  [torch.load(directory + f"/decoder/decoder_{i+1}/Decoder_Final_Output.pt")[0][0] for i in range(12)]
    
    AttentionPlusResidual_list =  [DecOut - SecLayer  for DecOut, SecLayer in zip(Decoder_Final_Output_list, SecondLayerNN_list)]
    
    for i, AttentionPlusResidual in enumerate(AttentionPlusResidual_list):
        torch.save((AttentionPlusResidual.unsqueeze(0),), directory + f"/decoder/decoder_{i+1}/AttentionPlusResidual.pt")

    # Extract the final normalization layer and the 'inverse matrix'
    ln_f = gpt2_model.ln_f
    lm_head = model.lm_head
    
    # Let's save the 'projection' on the vocabulary before and after the softmax
    for i, (out_attention, out_decoder) in enumerate(zip(AttentionPlusResidual_list, Decoder_Final_Output_list)):        
        final_norm = ln_f(out_attention)
        projection = lm_head(final_norm[-1])
        softmax = F.softmax(projection, dim=-1)
        
        top_values, top_indices = torch.topk(softmax, 10, dim=-1)
    
        torch.save(projection, directory + f"/last_token_pdf/decoder_{i+1}/attention_projection.pt")
        torch.save(softmax, directory + f"/last_token_pdf/decoder_{i+1}/attention_softmax.pt")
    
        final_norm = ln_f(out_decoder)
        projection = lm_head(final_norm[-1])
        softmax = F.softmax(projection, dim=-1)
    
        torch.save(projection, directory + f"/last_token_pdf/decoder_{i+1}/out_decoder_projection.pt")
        torch.save(softmax, directory + f"/last_token_pdf/decoder_{i+1}/out_decoder_softmax.pt")
        
        # Extract the top 10 entries with highest softmax values
        top_values, top_indices = torch.topk(softmax, 10, dim=-1)

    print("\nPrompt_" + prompt_name + ": PreRun finished.")
# ---------------------------------------------------------------------------------------------------------------------------



# ---------------------------------------------------------------------------------------------------------------------------
def Last_Token_Analysis(prompt, prompt_name, output_gif, compute_kl):

    # GIF ON THE EVOLUTION OF PROBABILITY
    
    # Define the directory containing the .pt files
    directory = "/mnt/DATA/output_" + prompt_name + "/last_token_pdf/"
    df_tokens = pd.read_pickle("/mnt/DATA/output_" + prompt_name + "/word_to_vector.pkl")

    if output_gif == True:
        
        # Create figure and axes
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
        # ---------------------------------------------------------------------------------------------------------------------------
        # Function to update the bar plots for each frame in the GIF
        def update(frame, df_tokens):
            ax1.clear()
            ax2.clear()
            create_bar_plot(frame, ax1, ax2, df_tokens)
        # ---------------------------------------------------------------------------------------------------------------------------
        
        # ---------------------------------------------------------------------------------------------------------------------------
        def create_bar_plot(decoder_index, ax1, ax2, df_tokens):
            # Load attention probabilities
            attention_softmax = load_attention_softmax(decoder_index, directory)
            probs_attention = attention_softmax.detach().numpy()
            top_indices_attention, top_probs_attention = get_top_10_entries(probs_attention)

            words_xticklabels_attn = [arr[0] if arr.size > 0 else arr for token in top_indices_attention for arr in [df_tokens.loc[df_tokens['tokens'] == token, 'words'].values]]
            ax1.bar(range(10), top_probs_attention, tick_label=top_indices_attention, color='red')
            ax1.set_ylim(0, 1)
            ax1.set_title(f"Attention Softmax - Decoder {decoder_index}")
            ax1.set_xticks(np.arange(10))
            ax1.set_xticklabels(words_xticklabels_attn, rotation=60)
            ax1.set_xlabel("Word")
            ax1.set_ylabel("Probabilities")
            
            # Load softmax probabilities
            decoder_softmax = load_decoder_softmax(decoder_index, directory)
            probs_softmax = decoder_softmax.detach().numpy()
            top_indices_softmax, top_probs_softmax = get_top_10_entries(probs_softmax)

            words_xticklabels_soft = [arr[0] if arr.size > 0 else arr for token in top_indices_softmax for arr in [df_tokens.loc[df_tokens['tokens'] == token, 'words'].values]]
            ax2.bar(range(10), top_probs_softmax, tick_label=top_indices_softmax)
            ax2.set_ylim(0, 1)
            ax2.set_title(f"Out Decoder Softmax - Decoder {decoder_index}")
            ax2.set_xticks(np.arange(10))
            ax2.set_xticklabels(words_xticklabels_soft, rotation=60)
            ax2.set_xlabel("Word")
            ax2.set_ylabel("Probabilities")
        # ---------------------------------------------------------------------------------------------------------------------------
        
            
        # Create animation
        decoder_indices = list(range(1, 13))  # Including decoder 12
        ani = animation.FuncAnimation(fig, update, fargs=(df_tokens,), frames=decoder_indices, repeat=False)
        
        # Save the animation as a GIF
        ani.save('probability_evolution/' + prompt_name + '.gif', writer='imagemagick', fps=0.8)


    # DISTANCE BETWEEN DECODERS
    
    # Load the probability distribution for decoder_12
    decoder_12 = load_decoder_softmax(12, directory)
    
    # Initialize lists to store results
    kl_divergences = []
    cos_similarities = []
    decoder_indices = list(range(1, 12))
    
    # Loop through decoders 1 to 11 and calculate distances
    for i in decoder_indices:
        attention_i = load_attention_softmax(i, directory)
        cos_sim = cosine_similarity(attention_i.detach().numpy().reshape(1, -1), decoder_12.detach().numpy().reshape(1, -1))
        cos_similarities.append(cos_sim)
        if compute_kl == True:
            kl_div = kl_divergence(attention_i.detach().numpy(), decoder_12.detach().numpy())
            kl_divergences.append(kl_div)

        decoder_i = load_decoder_softmax(i, directory)
        cos_sim = cosine_similarity(decoder_i.detach().numpy().reshape(1, -1), decoder_12.detach().numpy().reshape(1, -1))
        cos_similarities.append(cos_sim)
        if compute_kl == True:
            kl_div = kl_divergence(decoder_i.detach().numpy(), decoder_12.detach().numpy())
            kl_divergences.append(kl_div)
    
    # Plot the results in a single plot
    plt.figure(figsize=(10, 6))
    
    x_plot = np.arange(1, 12, 0.5)
    plt.plot(x_plot, np.asarray(cos_similarities).reshape(np.shape(x_plot)), marker='s', label='Cosine Similarity', color='r')
    if compute_kl == True:
        plt.plot(x_plot, kl_divergences, marker='o', label='Kullback-Leibler Divergence', color='b')
    
    plt.title('Distance indicators between decoder outputs and the predicted token')
    plt.xlabel('Decoder Index')
    plt.xticks(np.arange(1, 12))
    plt.ylabel('Value')
    plt.yscale('log')
    plt.legend(loc='upper right')
    plt.grid(True)
    
    for i in np.arange(0, 12):
        plt.axvline(x=i+1, color='black', linestyle='--', linewidth=1)
    
    plt.show()
    
    print("\nPrompt_" + prompt_name + ": Last Token Analysis finished.")
# ---------------------------------------------------------------------------------------------------------------------------



# ---------------------------------------------------------------------------------------------------------------------------
def Dim_evolution(prompt, prompt_name):

    # WHOLE TRANSFORMER
    percentage = 90
    module_name = ["AttentionProj", "AttentionPlusResidual", "SecondLayerNN", "Decoder_Final_Output"]
    component = []
    
    for i in range(1,13):
        for idx, mn in enumerate(module_name):
            if idx % 2 == 0:
                v = torch.load("/mnt/DATA/output_" + prompt_name + "/decoder/decoder_" + str(i) + "/" + mn + ".pt").detach().numpy()[0,]
            elif idx % 2 == 1: 
                v = torch.load("/mnt/DATA/output_" + prompt_name + "/decoder/decoder_" + str(i) + "/" + mn + ".pt")[0].detach().numpy()[0,]
            component.append(n_component(v, percent = percentage))

    # Create the dimensionality evolution graph
    fig, ax = plt.subplots(1,1,figsize=(20, 5))
    ax.plot(component, color='red')
    
    x_tick_labels = []
    for i in range(1, 13, 1):
        x_tick_labels.append(f"attn_cproj_{i}")
        x_tick_labels.append(f"attn+res_{i}")
        x_tick_labels.append(f"FFNN_{i}")
        x_tick_labels.append(f"FFNN+res_{i}")
    
    for i in np.arange(0, 48, 4):
        ax.axvline(x=i+3, color='black', linestyle='--', linewidth=1)
    
    ax.set_xticks([i for i in range(0, 48)], x_tick_labels, rotation=90)
    
    ax.set_title("Prompt " + prompt_name + ": Evolution of the dimensionality throughout the whole transformer")
    ax.set_xlabel("Stages of the model")
    ax.set_ylabel(f"Number of principal components\nexplaining {percentage}% of the total variance")
    plt.show()
    
    # BEFORE SKIP CONNECTION
    
    percentages = [75, 80, 85, 90, 95]
    
    module_name = ["AttentionProj", "SecondLayerNN"]
    components = [[] for n in range(len(percentages))]
    
    for i in range(1,13):
        for mn in module_name:
            v = torch.load("/mnt/DATA/output_" + prompt_name + "/decoder/decoder_" + str(i) + "/" + mn + ".pt").detach().numpy()[0,]
            for percentage, component in zip(percentages, components):
                component.append(n_component(v, percent = percentage))

    # Create the dimensionality evolution graph
    fig, ax = plt.subplots(1,1,figsize=(10, 6))
    for percentage, component in zip(percentages, components):
        ax.plot(component, label = f"{percentage}%")
    
    x_tick_labels = []
    for i in range(1, 13, 1):
        x_tick_labels.append(f"attn_cproj_{i}")
        x_tick_labels.append(f"FFNN_{i}")
    
    for i in np.arange(0, 24, 2):
        ax.axvline(x=i+1, color='black', linestyle='--', linewidth=1)
    
    ax.set_xticks([i for i in range(0,24)], x_tick_labels, rotation=60)
    
    ax.set_title("Prompt " + prompt_name + ": Evolution of the dimensionality of the hidden states")
    ax.set_xlabel("Stages of the model")
    ax.set_ylabel("Number of principal components")
    ax.legend()
    plt.show()

    
    # AFTER SKIP CONNECTION

    module_name = ["AttentionPlusResidual", "Decoder_Final_Output"]
    components = [[] for n in range(len(percentages))]
    
    for i in range(1,13):
        for mn in module_name:
            v = torch.load("/mnt/DATA/output_" + prompt_name + "/decoder/decoder_" + str(i) + "/" + mn + ".pt")[0].detach().numpy()[0,]
            for percentage, component in zip(percentages, components):
                component.append(n_component(v, percent = percentage))

    # Create the dimensionality evolution graph
    fig, ax = plt.subplots(1,1,figsize=(10, 6))
    for percentage, component in zip(percentages, components):
        ax.plot(component, label = f"{percentage}%")
    
    x_tick_labels = []
    for i in range(1, 13, 1):
        x_tick_labels.append(f"attn+res_{i}")
        x_tick_labels.append(f"FFNN+res_{i}")
    ax.set_xticks([i for i in range(0,24)], x_tick_labels, rotation=45)
    
    for i in np.arange(0, 24, 2):
        ax.axvline(x=i+1, color='black', linestyle='--', linewidth=1)
    
    ax.set_title("Prompt " + prompt_name + ": Evolution of the dimensionality after the skip connections")
    ax.set_xlabel("Stages of the model")
    ax.set_ylabel("Number of principal components")
    ax.legend()
    plt.show()
    
    print("\nPrompt_" + prompt_name + ": Dimensionality Evolution Analysis finished.")
# ---------------------------------------------------------------------------------------------------------------------------

# AUXILIARY FUNCTIONS

# ---------------------------------------------------------------------------------------------------------------------------
def log_gram_det(prompt):
    A = prompt.dot(prompt.T)

    P, L, U = scipy.linalg.lu(A)
    
    # Calculate log determinant using mpmath for higher precision (log_det on the diagonal is always 1)
    log_det_U = sum(mpmath.log(mpmath.mpf(str(abs(value)))) for value in np.diag(U))
    
    return log_det_U
# ---------------------------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------------------------------------
def cosine_similarity_matrix(v, last_vector=None):
    if last_vector is None:
        last_vector = v[-1, :]
    norms_v = np.linalg.norm(v, axis=1)
    norm_last_vector = np.linalg.norm(last_vector)
    
    dot_products = v.dot(last_vector)
    cosine_similarities = dot_products / (norms_v * norm_last_vector)
    
    return cosine_similarities
# ---------------------------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------------------------------------
def n_component(v, percent = 80):
    
    N = v.shape[0]
    U, spectrum, Vt = la.svd(v)
    l_svd = (spectrum ** 2)/(N-1)
    V_svd = U

    values = np.cumsum(l_svd/sum(l_svd)*100)

    diff = np.abs(values - percent)

    return np.argmin(diff)
# ---------------------------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------------------------------------
# Function to load the .pt file for a given decoder index
def load_decoder_softmax(decoder_index, directory):
    file_path = os.path.join(directory, f"decoder_{decoder_index}/out_decoder_softmax.pt")
    return torch.load(file_path)
# ---------------------------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------------------------------------
def load_attention_softmax(decoder_index, directory):
    file_path = os.path.join(directory, f"decoder_{decoder_index}/attention_softmax.pt")
    return torch.load(file_path)
# ---------------------------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------------------------------------
# Function to get the top 10 entries with higher probabilities
def get_top_10_entries(prob_dist):
    sorted_indices = np.argsort(prob_dist)
    top_10_indices = sorted_indices[-10:][::-1]
    top_10_probs = prob_dist[top_10_indices]
    return top_10_indices, top_10_probs
# ---------------------------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------------------------------------
# Function to calculate Kullback-Leibler divergence
def kl_divergence(p, q):
    return entropy(p, q)
# ---------------------------------------------------------------------------------------------------------------------------


# PLOTTING FUNCTIONS

# ---------------------------------------------------------------------------------------------------------------------------
def mean_norm_plot(norms, list_prompt_names, colors, after_skip):
    fig = plt.figure(figsize=(12,6))

    for prompt_idx, single_prompt_norm in enumerate(norms):
        plt.plot(single_prompt_norm, label = "prompt " + list_prompt_names[prompt_idx], color=colors[prompt_idx])
    
    for i in np.arange(0, 24, 2):
        plt.axvline(x=i+1, color='black', linestyle='--', linewidth=1)
        
    x_tick_labels = []
    for i in range(1, 13, 1):
        x_tick_labels.append(f"attn+res_{i}")
        x_tick_labels.append(f"FFNN+res_{i}")
    
    plt.xticks([i for i in range(0,24)], x_tick_labels, rotation=60)

    if after_skip == False:
        skip_position = 'before'
    elif after_skip == True:
        skip_position = 'after'
        
    plt.title("Prompt comparison: Evolution of the mean norm of the hidden states\n" + skip_position + ' the skip connection')
    plt.xlabel("Stages of the model")
    plt.ylabel("Mean Norm")
    plt.yscale('log')
    plt.subplots_adjust(right=0.75)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()
# ---------------------------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------------------------------------
def volume_plot(grams, list_prompt_names, colors, after_skip):
    fig = plt.figure(figsize=(12,6))

    for prompt_idx, single_prompt_gram in enumerate(grams):
        plt.plot(single_prompt_gram, label = "prompt " + list_prompt_names[prompt_idx], color=colors[prompt_idx])
    
    for i in np.arange(0, 24, 2):
        plt.axvline(x=i+1, color='black', linestyle='--', linewidth=1)

    x_tick_labels = []
    for i in range(1, 13, 1):
        x_tick_labels.append(f"attn+res_{i}")
        x_tick_labels.append(f"FFNN+res_{i}")
    
    plt.xticks([i for i in range(0,24)], x_tick_labels, rotation=60)

    if after_skip == False:
        skip_position = 'before'
    elif after_skip == True:
        skip_position = 'after'
        
    plt.title("Prompt comparison: Evolution of the normalized volume of the hidden states\n" + skip_position + ' the skip connection')
    plt.xlabel("Stages of the model")
    plt.ylabel("Log Volume")
    plt.subplots_adjust(right=0.75)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()
# ---------------------------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------------------------------------
def dimensionality_evolution_plot(components, list_prompt_names, colors, percentage, after_skip):
    fig, ax = plt.subplots(1,1,figsize=(12, 6))
    
    for prompt_idx, single_prompt_components in enumerate(components):
        ax.plot(single_prompt_components, label = "prompt " + list_prompt_names[prompt_idx], color=colors[prompt_idx])
    
    for i in np.arange(0, 24, 2):
        ax.axvline(x=i+1, color='black', linestyle='--', linewidth=1)

    x_tick_labels = []
    for i in range(1, 13, 1):
        x_tick_labels.append(f"attn+res_{i}")
        x_tick_labels.append(f"FFNN+res_{i}")
    
    ax.set_xticks([i for i in range(0,24)], x_tick_labels, rotation=45)

    if after_skip == False:
        skip_position = 'before'
    elif after_skip == True:
        skip_position = 'after'
        
    ax.set_title("Prompt comparison: Evolution of the dimensionality\n" + skip_position + ' the skip connection, ' + str(percentage) + "%")
    ax.set_xlabel("Stages of the model")
    ax.set_ylabel("Number of principal components")
    plt.subplots_adjust(right=0.75)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()
# ---------------------------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------------------------------------
def mean_cosine_similarity_plot(cos_sims, list_prompt_names, colors, after_skip):
    fig = plt.figure(figsize=(12,6))

    for prompt_idx, single_prompt_cos_sim in enumerate(cos_sims):
        plt.plot(single_prompt_cos_sim, label = "prompt " + list_prompt_names[prompt_idx], color=colors[prompt_idx])
    
    for i in np.arange(0, 24, 2):
        plt.axvline(x=i+1, color='black', linestyle='--', linewidth=1)

    x_tick_labels = []
    for i in range(1, 13, 1):
        x_tick_labels.append(f"attn+res_{i}")
        x_tick_labels.append(f"FFNN+res_{i}")
    
    plt.xticks([i for i in range(0,24)], x_tick_labels, rotation=60)
    
    if after_skip == False:
        skip_position = 'before'
    elif after_skip == True:
        skip_position = 'after'
        
    plt.title("Prompt comparison: Evolution of the mean cosine similarity\n" + skip_position + ' the skip connection')
    plt.xlabel("Stages of the model")
    plt.ylabel("Normalized Mean Cosine Similarity")
    plt.subplots_adjust(right=0.75)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()
# ---------------------------------------------------------------------------------------------------------------------------
