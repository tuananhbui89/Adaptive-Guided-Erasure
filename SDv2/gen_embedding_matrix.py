import numpy as np
import torch 
from transformers import CLIPTokenizer, CLIPTextModel
import os 
from pathlib import Path
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
import torch.nn.functional as F
from einops import rearrange
from PIL import Image
import clip
import argparse

from utils_alg import sample_model, get_models, load_model_from_config, load_model_from_config_compvis
from constants import *
from ldm.models.diffusion.ddim import DDIMSampler


# Util Functions
# def load_model_from_config(config, ckpt, device="cpu", verbose=False):
#     """Loads a model from config and a ckpt
#     if config is a path will use omegaconf to load
#     """
#     if isinstance(config, (str, Path)):
#         config = OmegaConf.load(config)

#     pl_sd = torch.load(ckpt, map_location="cpu")
#     global_step = pl_sd["global_step"]
#     sd = pl_sd["state_dict"]
#     model = instantiate_from_config(config.model)
#     m, u = model.load_state_dict(sd, strict=False)
#     model.to(device)
#     model.eval()
#     model.cond_stage_model.device = device
#     return model

@torch.no_grad()
def create_embedding_matrix(model, start=0, end=LEN_TOKENIZER_VOCAB, model_name='SD-v1-4', save_mode='array'):
    if model_name == 'SD-v1-4':
        tokenizer_vocab = model.cond_stage_model.tokenizer.get_vocab()
    elif model_name == 'SD-v2-1':
        tokenizer_vocab = model.cond_stage_model.tokenizer.encoder
    else:
        raise ValueError("model_name should be either 'SD-v1-4' or 'SD-v2-1'")

    if save_mode == 'array':
        all_embeddings = []
        for token in tokenizer_vocab:
            if tokenizer_vocab[token] < start or tokenizer_vocab[token] >= end:
                continue
            print(token, tokenizer_vocab[token])
            token_ = token.replace('</w>','')
            emb_ = model.get_learned_conditioning([token_])
            all_embeddings.append(emb_)
        return torch.cat(all_embeddings, dim=0) # shape (49408, 77, 768)
    elif save_mode == 'dict':
        all_embeddings = {}
        for token in tokenizer_vocab:
            if tokenizer_vocab[token] < start or tokenizer_vocab[token] >= end:
                continue
            print(token, tokenizer_vocab[token])
            token_ = token.replace('</w>','')
            emb_ = model.get_learned_conditioning([token_])
            all_embeddings[token] = emb_
        return all_embeddings
    else:
        raise ValueError("save_mode should be either 'array' or 'dict'")

@torch.no_grad()
def save_embedding_matrix(model, model_name='SD-v1-4', save_mode='array'):
    for start in range(0, LEN_TOKENIZER_VOCAB, 5000):
        print(f"start: {start} / {LEN_TOKENIZER_VOCAB}")
        end = min(LEN_TOKENIZER_VOCAB, start+5000)
        embedding_matrix = create_embedding_matrix(model, start=start, end=end, model_name=model_name, save_mode=save_mode)
        if model_name == 'SD-v1-4':
            torch.save(embedding_matrix, f'models/embedding_matrix_{start}_{end}_{save_mode}.pt')
        elif model_name == 'SD-v2-1':
            torch.save(embedding_matrix, f'models/embedding_matrix_{start}_{end}_{save_mode}_v2-1.pt')

# save_embedding_matrix(model)

def retrieve_embedding_token(model_name, query_token):
    for start in range(0, LEN_TOKENIZER_VOCAB, 5000):
        # print(f"start: {start} / {LEN_TOKENIZER_VOCAB}")
        end = min(LEN_TOKENIZER_VOCAB, start+5000)
        if model_name == 'SD-v1-4':
            embedding_matrix = torch.load(f'models/embedding_matrix_{start}_{end}_dict.pt')
        elif model_name == 'SD-v2-1':
            embedding_matrix = torch.load(f'models/embedding_matrix_{start}_{end}_dict_v2-1.pt')
        else:
            raise ValueError("model_name should be either 'SD-v1-4' or 'SD-v2-1'")
        if query_token in embedding_matrix:
            return embedding_matrix[query_token]

@torch.no_grad()
def search_closest_tokens(concept, model, k=5, reshape=True, sim='cosine', model_name='SD-v1-4'):
    """
    Given a concept, i.e., "nudity", search for top-k closest tokens in the embedding space
    """
    if model_name == 'SD-v1-4':
        tokenizer_vocab = model.cond_stage_model.tokenizer.get_vocab()
    elif model_name == 'SD-v2-1':
        tokenizer_vocab = model.cond_stage_model.tokenizer.encoder

    # inverse the dictionary
    tokenizer_vocab_indexing = {v: k for k, v in tokenizer_vocab.items()}

    # Get the embedding of the concept
    concept_embedding = model.get_learned_conditioning([concept]) # shape (1, 77, 768)

    # Calculate the cosine similarity between the concept and all tokens
    # load the embedding matrix 
    all_similarities = []

    for start in range(0, LEN_TOKENIZER_VOCAB, 5000):
        # print(f"start: {start} / {LEN_TOKENIZER_VOCAB}")
        end = min(LEN_TOKENIZER_VOCAB, start+5000)
        if model_name == 'SD-v1-4':
            embedding_matrix = torch.load(f'models/embedding_matrix_{start}_{end}.pt')
        elif model_name == 'SD-v2-1':
            embedding_matrix = torch.load(f'models/embedding_matrix_{start}_{end}_v2-1.pt')
        else:
            raise ValueError("model_name should be either 'SD-v1-4' or 'SD-v2-1'")
        
        if reshape == True:
            concept_embedding = concept_embedding.view(concept_embedding.size(0), -1)
            embedding_matrix = embedding_matrix.view(embedding_matrix.size(0), -1)
        if sim == 'cosine':
            similarities = F.cosine_similarity(concept_embedding, embedding_matrix, dim=-1)
        elif sim == 'l2':
            similarities = - F.pairwise_distance(concept_embedding, embedding_matrix, p=2)
        all_similarities.append(similarities)
    similarities = torch.cat(all_similarities, dim=0)
    # sorting the similarities
    sorted_similarities, indices = torch.sort(similarities, descending=True)
    print(f"sorted_similarities: {sorted_similarities[:10]}")
    print(f"indices: {indices[:10]}")

    sim_dict = {}
    for im, i in enumerate(indices):
        token = tokenizer_vocab_indexing[i.item()]
        sim_dict[token] = sorted_similarities[im]
    
    top_k_tokens = list(sim_dict.keys())[:k]
    print(f"Top-{k} closest tokens to the concept {concept} are: {top_k_tokens}")
    return top_k_tokens, sim_dict

@torch.no_grad()
def search_closest_output(concept, model, sampler, tokens_embedding, start_guidance, time_step, start_concept, k=5, sim='l2'):
    """
    given a concept, i.e., "nudity", search for top-k closest tokens in the token vocabulary that produces the closest output with a given model
    TODO: using CLIP score as a similarity metric 
    Args: 
        concept: str, the concept to search for
        model: the model to use for inference
        tokens_embedding: dictionary containing the token embeddings that are used to search for the closest tokens
            example: {'nudity': tensor([0.1, 0.2, ..., 0.3]), 'sexy': tensor([0.2, 0.3, ..., 0.4]), ...}
            token embeddings are of shape (1, 77, 768)
        k: int, the number of top-k tokens to return
    """

    # create a lambda function for cleaner use of sampling code (only denoising till time step t)
    # till_T = 0 means the model will sample till the end of the diffusion process 
    # till_T = T means the starting point of the diffusion process
    # the smaller the value of till_T, the more denoising is applied to the image
    quick_sample_till_t = lambda cond, s, code, t: sample_model(model, sampler,
                                                                 cond, IMAGE_SIZE, IMAGE_SIZE, DDIM_STEPS, s, DDIM_ETA,
                                                                 start_code=code, till_T=t, verbose=False)
    device = model.cond_stage_model.device
    # fix start code 
    start_code = torch.randn(1, 4, IMAGE_SIZE // 8, IMAGE_SIZE // 8).to(device)

    # get the condition 
    if start_concept is None:
        start_cond = model.get_learned_conditioning([STARTING_CONCEPT])
    else:
        start_cond = model.get_learned_conditioning([start_concept])

    # if start from a specific time step t 
    if time_step is not None:
        z = quick_sample_till_t(start_cond, start_guidance, start_code, time_step)
    else: 
        # start with noisy 
        z = start_code
    
    # get the output of the model w.r.t. the input concept 
    og_num = round((int(time_step)/DDIM_STEPS)*1000)
    og_num_lim = round((int(time_step + 1)/DDIM_STEPS)*1000)
    t_enc_ddpm = torch.randint(og_num, og_num_lim, (1,))
    
    @torch.no_grad()
    def get_output(cond, z, model, t_enc_ddpm, device):
        e = model.apply_model(z.to(device), t_enc_ddpm.to(device), cond.to(device))
        return e
    
    # get the output of the model w.r.t. the input concept
    if type(concept) is str:
        concept_cond = model.get_learned_conditioning([concept])
    else:
        raise ValueError("concept should be a string")
    
    output_concept = get_output(concept_cond, z, model, t_enc_ddpm, device)
    sim_dict = {}

    with torch.no_grad():
        for token in tokens_embedding:
            token_cond = model.get_learned_conditioning([token])
            output_token = get_output(token_cond, z, model, t_enc_ddpm, device)
            if sim == 'cosine':
                similarity = F.cosine_similarity(output_concept.flatten(), 
                                                output_token.flatten(), dim=-1)
            elif sim == 'l2':
                similarity = - F.pairwise_distance(output_concept.flatten(), 
                                                output_token.flatten(), p=2)
            sim_dict[token] = similarity
            print(f"similarity between concept and token {token}: {similarity}")
        
    sorted_sim_dict = {k: v for k, v in sorted(sim_dict.items(), key=lambda item: item[1], reverse=True)}
    top_k_tokens = list(sorted_sim_dict.keys())[:k]
    print(f"Top-{k} closest tokens to the concept {concept} are: {top_k_tokens}")

    # # write similarity scores to a csv file 
    # with open(f'evaluation_folder/similarity/similarity_scores_{concept}_{time_step}_{start_concept}.csv', 'w') as f:
    #     for key in sorted_sim_dict.keys():
    #         f.write("%s,%s\n"%(key, sorted_sim_dict[key].item()))

    return top_k_tokens, sorted_sim_dict


@torch.no_grad()
def search_closest_output_multi_steps(concept, model, sampler, tokens_embedding, start_guidance, time_step, start_concept, k=5, sim='l2', multi_steps=2):
    """
    given a concept, i.e., "nudity", search for top-k closest tokens in the token vocabulary that produces the closest output with a given model
    applying multi-steps inference instead of just one-step as before. 
    Args: 
        concept: str, the concept to search for
        model: the model to use for inference
        tokens_embedding: dictionary containing the token embeddings that are used to search for the closest tokens
            example: {'nudity': tensor([0.1, 0.2, ..., 0.3]), 'sexy': tensor([0.2, 0.3, ..., 0.4]), ...}
            token embeddings are of shape (1, 77, 768)
        k: int, the number of top-k tokens to return
    """

    # create a lambda function for cleaner use of sampling code (only denoising till time step t)
    # till_T = 0 means the model will sample till the end of the diffusion process 
    # till_T = T means the starting point of the diffusion process
    # the smaller the value of till_T, the more denoising is applied to the image
    quick_sample_till_t = lambda cond, s, code, t: sample_model(model, sampler,
                                                                 cond, IMAGE_SIZE, IMAGE_SIZE, DDIM_STEPS, s, DDIM_ETA,
                                                                 start_code=code, till_T=t, verbose=False)
    device = model.cond_stage_model.device
    # fix start code 
    start_code = torch.randn(1, 4, IMAGE_SIZE // 8, IMAGE_SIZE // 8).to(device)

    # get the condition 
    if start_concept is None:
        start_cond = model.get_learned_conditioning([STARTING_CONCEPT])
    else:
        start_cond = model.get_learned_conditioning([start_concept])

    # if start from a specific time step t 
    if time_step is not None:
        z = quick_sample_till_t(start_cond, start_guidance, start_code, time_step)
    else: 
        # start with noisy 
        z = start_code
    
    # get the output of the model w.r.t. the input concept 
    og_num = round((int(time_step)/DDIM_STEPS)*1000)
    og_num_lim = round((int(time_step + 1)/DDIM_STEPS)*1000)
    t_enc_ddpm = torch.randint(og_num, og_num_lim, (1,))
    
    # assert time_step - multi_steps >= 0, "time_step - multi_steps should be greater than or equal to 0"


    @torch.no_grad()
    def get_output(cond, z, model, sampler):
        # CHANGE HERE: start from intermediate time step time_step (z is the start code) and sample 2 steps, i.e., till time_step-multi_steps
        if time_step - multi_steps >= 0:
            output = sample_model(model, sampler, cond, IMAGE_SIZE, IMAGE_SIZE, DDIM_STEPS, start_guidance, DDIM_ETA, start_code=z, till_T = time_step - multi_steps, t_start = time_step)
        else: 
            assert time_step == 0
            output = sample_model(model, sampler, cond, IMAGE_SIZE, IMAGE_SIZE, DDIM_STEPS, start_guidance, DDIM_ETA, start_code=z, till_T = 0, t_start = multi_steps)

        return output
    
    # get the output of the model w.r.t. the input concept
    if type(concept) is str:
        concept_cond = model.get_learned_conditioning([concept])
    else:
        raise ValueError("concept should be a string")
    
    output_concept = get_output(concept_cond, z, model, sampler)
    sim_dict = {}
    output_dict = {}
    output_dict[concept] = output_concept

    with torch.no_grad():
        for token in tokens_embedding:
            token_cond = model.get_learned_conditioning([token])
            output_token = get_output(token_cond, z, model, sampler)
            output_dict[token] = output_token
            if sim == 'cosine':
                similarity = F.cosine_similarity(output_concept.flatten(), 
                                                output_token.flatten(), dim=-1)
            elif sim == 'l2':
                similarity = - F.pairwise_distance(output_concept.flatten(), 
                                                output_token.flatten(), p=2)
            sim_dict[token] = similarity
            print(f"similarity between concept and token {token}: {similarity}")
        
    sorted_sim_dict = {k: v for k, v in sorted(sim_dict.items(), key=lambda item: item[1], reverse=True)}
    top_k_tokens = list(sorted_sim_dict.keys())[:k]
    print(f"Top-{k} closest tokens to the concept {concept} are: {top_k_tokens}")

    # # write similarity scores to a csv file 
    # with open(f'evaluation_folder/similarity/similarity_scores_{concept}_{time_step}_{start_concept}.csv', 'w') as f:
    #     for key in sorted_sim_dict.keys():
    #         f.write("%s,%s\n"%(key, sorted_sim_dict[key].item()))

    return top_k_tokens, sorted_sim_dict, output_dict

@torch.no_grad()
def compare_two_models(model, model_org, sampler, tokens_embedding, start_guidance, time_step, start_concept, k=5, sim='l2'):
    """
    given a concept, i.e., "nudity", search for top-k closest tokens in the token vocabulary that produces the closest output with a given model
    TODO: using CLIP score as a similarity metric 
    Args: 
        model: the model to use for inference
        model_org: the original model to use for inference
        tokens_embedding: dictionary containing the token embeddings that are used to search for the closest tokens
            example: {'nudity': tensor([0.1, 0.2, ..., 0.3]), 'sexy': tensor([0.2, 0.3, ..., 0.4]), ...}
            token embeddings are of shape (1, 77, 768)
        k: int, the number of top-k tokens to return
    """

    # create a lambda function for cleaner use of sampling code (only denoising till time step t)
    # till_T = 0 means the model will sample till the end of the diffusion process 
    # till_T = T means the starting point of the diffusion process
    # the smaller the value of till_T, the more denoising is applied to the image
    quick_sample_till_t = lambda cond, s, code, t: sample_model(model, sampler,
                                                                 cond, IMAGE_SIZE, IMAGE_SIZE, DDIM_STEPS, s, DDIM_ETA,
                                                                 start_code=code, till_T=t, verbose=False)
    device = model.cond_stage_model.device
    # fix start code 
    start_code = torch.randn(1, 4, IMAGE_SIZE // 8, IMAGE_SIZE // 8).to(device)

    # get the condition 
    if start_concept is None:
        start_cond = model.get_learned_conditioning([STARTING_CONCEPT])
    else:
        start_cond = model.get_learned_conditioning([start_concept])

    # if start from a specific time step t 
    if time_step is not None:
        z = quick_sample_till_t(start_cond, start_guidance, start_code, time_step)
    else: 
        # start with noisy 
        z = start_code
    
    # get the output of the model w.r.t. the input concept 
    og_num = round((int(time_step)/DDIM_STEPS)*1000)
    og_num_lim = round((int(time_step + 1)/DDIM_STEPS)*1000)
    t_enc_ddpm = torch.randint(og_num, og_num_lim, (1,))
    
    @torch.no_grad()
    def get_output(cond, z, model, t_enc_ddpm, device):
        e = model.apply_model(z.to(device), t_enc_ddpm.to(device), cond.to(device))
        return e
    sim_dict = {}
    
    with torch.no_grad():
        for token in tokens_embedding:
            token_cond = model.get_learned_conditioning([token])
            output_1 = get_output(token_cond, z, model, t_enc_ddpm, device)
            output_2 = get_output(token_cond, z, model_org, t_enc_ddpm, device)
            if sim == 'cosine':
                similarity = F.cosine_similarity(output_1.flatten(), 
                                                output_2.flatten(), dim=-1)
            elif sim == 'l2':
                similarity = - F.pairwise_distance(output_1.flatten(), 
                                                output_2.flatten(), p=2)
            sim_dict[token] = similarity
            print(f"similarity between two models for token {token}: {similarity}")
    
    sorted_sim_dict = {k: v for k, v in sorted(sim_dict.items(), key=lambda item: item[1], reverse=True)}
    top_k_tokens = list(sorted_sim_dict.keys())[:k]
    print(f"Top-{k} closest tokens to the concept are: {top_k_tokens}")

    return top_k_tokens, sorted_sim_dict


@torch.no_grad()
def compare_two_models_multi_steps(model, model_org, sampler, tokens_embedding, start_guidance, time_step, start_concept, k=5, sim='l2', multi_steps=2):
    """
    given a concept, i.e., "nudity", search for top-k closest tokens in the token vocabulary that produces the closest output with a given model
    applying multi-steps inference instead of just one-step as before.
    Args: 
        model: the model to use for inference
        model_org: the original model to use for inference
        tokens_embedding: dictionary containing the token embeddings that are used to search for the closest tokens
            example: {'nudity': tensor([0.1, 0.2, ..., 0.3]), 'sexy': tensor([0.2, 0.3, ..., 0.4]), ...}
            token embeddings are of shape (1, 77, 768)
        k: int, the number of top-k tokens to return
    """

    # create a lambda function for cleaner use of sampling code (only denoising till time step t)
    # till_T = 0 means the model will sample till the end of the diffusion process 
    # till_T = T means the starting point of the diffusion process
    # the smaller the value of till_T, the more denoising is applied to the image
    quick_sample_till_t = lambda cond, s, code, t: sample_model(model, sampler,
                                                                 cond, IMAGE_SIZE, IMAGE_SIZE, DDIM_STEPS, s, DDIM_ETA,
                                                                 start_code=code, till_T=t, verbose=False)
    device = model.cond_stage_model.device
    # fix start code 
    start_code = torch.randn(1, 4, IMAGE_SIZE // 8, IMAGE_SIZE // 8).to(device)

    # get the condition 
    if start_concept is None:
        start_cond = model.get_learned_conditioning([STARTING_CONCEPT])
    else:
        start_cond = model.get_learned_conditioning([start_concept])

    # if start from a specific time step t 
    if time_step is not None:
        z = quick_sample_till_t(start_cond, start_guidance, start_code, time_step)
    else: 
        # start with noisy 
        z = start_code
    
    # get the output of the model w.r.t. the input concept 
    og_num = round((int(time_step)/DDIM_STEPS)*1000)
    og_num_lim = round((int(time_step + 1)/DDIM_STEPS)*1000)
    t_enc_ddpm = torch.randint(og_num, og_num_lim, (1,))
    
    def get_output(cond, z, model, sampler):
        # CHANGE HERE: start from intermediate time step time_step (z is the start code) and sample 2 steps, i.e., till time_step-multi_steps
        if time_step - multi_steps >= 0:
            output = sample_model(model, sampler, cond, IMAGE_SIZE, IMAGE_SIZE, DDIM_STEPS, start_guidance, DDIM_ETA, start_code=z, till_T = time_step - multi_steps, t_start = time_step)
        else: 
            assert time_step == 0
            output = sample_model(model, sampler, cond, IMAGE_SIZE, IMAGE_SIZE, DDIM_STEPS, start_guidance, DDIM_ETA, start_code=z, till_T = 0, t_start = multi_steps)

        return output
    
    sim_dict = {}
    
    with torch.no_grad():
        for token in tokens_embedding:
            token_cond = model.get_learned_conditioning([token])
            output_1 = get_output(token_cond, z, model, sampler)
            output_2 = get_output(token_cond, z, model_org, sampler)
            if sim == 'cosine':
                similarity = F.cosine_similarity(output_1.flatten(), 
                                                output_2.flatten(), dim=-1)
            elif sim == 'l2':
                similarity = - F.pairwise_distance(output_1.flatten(), 
                                                output_2.flatten(), p=2)
            sim_dict[token] = similarity
            print(f"similarity between two models for token {token}: {similarity}")
    
    sorted_sim_dict = {k: v for k, v in sorted(sim_dict.items(), key=lambda item: item[1], reverse=True)}
    top_k_tokens = list(sorted_sim_dict.keys())[:k]
    print(f"Top-{k} closest tokens to the concept are: {top_k_tokens}")

    return top_k_tokens, sorted_sim_dict

@torch.no_grad()
def decode_and_extract_visual_feature(model, z, device):
    clip_model, clip_preprocess = clip.load("ViT-B/32", device)

    x = model.decode_first_stage(z)
    x = torch.clamp((x + 1.0)/2.0, min=0.0, max=1.0)
    x = rearrange(x, 'b c h w -> b (c h) w')
    image = clip_preprocess(Image.fromarray((x[0].cpu().numpy()*255).astype(np.uint8)))
    with torch.no_grad():
        image_features = clip_model.encode_image(image.unsqueeze(0).to(device))
    return image_features
    
@torch.no_grad()
def search_closest_output_CLIP(concept, model, sampler, tokens_embedding, start_guidance, time_step, start_concept, k=5, sim='l2'):
    """
    given a concept, i.e., "nudity", search for top-k closest tokens in the token vocabulary that produces the closest output with a given model
    TODO: using CLIP score as a similarity metric 
    Args: 
        concept: str, the concept to search for
        model: the model to use for inference
        tokens_embedding: dictionary containing the token embeddings that are used to search for the closest tokens
            example: {'nudity': tensor([0.1, 0.2, ..., 0.3]), 'sexy': tensor([0.2, 0.3, ..., 0.4]), ...}
            token embeddings are of shape (1, 77, 768)
        k: int, the number of top-k tokens to return
    """

    # create a lambda function for cleaner use of sampling code (only denoising till time step t)
    # till_T = 0 means the model will sample till the end of the diffusion process 
    # till_T = T means the starting point of the diffusion process
    # the smaller the value of till_T, the more denoising is applied to the image
    quick_sample_till_t = lambda cond, s, code, t: sample_model(model, sampler,
                                                                 cond, IMAGE_SIZE, IMAGE_SIZE, DDIM_STEPS, s, DDIM_ETA,
                                                                 start_code=code, till_T=t, verbose=False)
    device = model.cond_stage_model.device
    # fix start code 
    start_code = torch.randn(1, 4, IMAGE_SIZE // 8, IMAGE_SIZE // 8).to(device)

    # get the condition 
    if start_concept is None:
        start_cond = model.get_learned_conditioning([STARTING_CONCEPT])
    else:
        start_cond = model.get_learned_conditioning([start_concept])

    # if start from a specific time step t 
    if time_step is not None:
        z = quick_sample_till_t(start_cond, start_guidance, start_code, time_step)
    else: 
        # start with noisy 
        z = start_code
    
    # get the output of the model w.r.t. the input concept 
    og_num = round((int(time_step)/DDIM_STEPS)*1000)
    og_num_lim = round((int(time_step + 1)/DDIM_STEPS)*1000)
    t_enc_ddpm = torch.randint(og_num, og_num_lim, (1,))
    
    @torch.no_grad()
    def get_output(cond, z, model, t_enc_ddpm, device):
        e = model.apply_model(z.to(device), t_enc_ddpm.to(device), cond.to(device))
        return e
    
    # get the output of the model w.r.t. the input concept
    if type(concept) is str:
        concept_cond = model.get_learned_conditioning([concept])
    else:
        raise ValueError("concept should be a string")
    
    output_concept = get_output(concept_cond, z, model, t_enc_ddpm, device)

    # decode using decoder
    concept_image_features = decode_and_extract_visual_feature(model, output_concept, device)


    sim_dict = {}

    with torch.no_grad():
        for token in tokens_embedding:
            token_cond = model.get_learned_conditioning([token])
            output_token = get_output(token_cond, z, model, t_enc_ddpm, device)
            # decode using decoder
            token_image_features = decode_and_extract_visual_feature(model, output_token, device)
            if sim == 'cosine':
                similarity = F.cosine_similarity(concept_image_features.flatten(), 
                                                token_image_features.flatten(), dim=-1)
            elif sim == 'l2':
                similarity = - F.pairwise_distance(concept_image_features.flatten(),
                                                token_image_features.flatten(), p=2)
                
            sim_dict[token] = similarity
            print(f"similarity between concept and token {token}: {similarity}")

        
    sorted_sim_dict = {k: v for k, v in sorted(sim_dict.items(), key=lambda item: item[1], reverse=True)}
    top_k_tokens = list(sorted_sim_dict.keys())[:k]
    print(f"Top-{k} closest tokens to the concept {concept} are: {top_k_tokens}")

    # # write similarity scores to a csv file 
    # with open(f'evaluation_folder/similarity/similarity_scores_clip_{concept}_{time_step}_{start_concept}.csv', 'w') as f:
    #     for key in sorted_sim_dict.keys():
    #         f.write("%s,%s\n"%(key, sorted_sim_dict[key].item()))

    return top_k_tokens, sorted_sim_dict

def test_load_model(args, save_mode='dict'):

    config_path = args.config_path
    ckpt_path = args.ckpt_path
    model_name = args.model_name
    device = 'cuda'

    model = load_model_from_config(config_path, ckpt_path, device=device)

    save_embedding_matrix(model, model_name=model_name, save_mode=save_mode)

    return model

def test_search_closest_output(args):
    # config_path = 'configs/stable-diffusion/v1-inference.yaml'
    config_path = args.config_path
    model_name = args.model_name
    ckpt_path = args.ckpt_path

    # concepts = ["nudity", "sexy", "cassette player", "garbage truck", "person", "a photo"]

    # concepts = [args.concept]

    concepts = ["nudity"]

    # time_steps = [0, 10, 20, 30, 40, 50]

    time_steps = [args.time_step]

    start_concept = "a photo"

    sim = 'l2'

    collections_tokens = None

    for concept in concepts:
        devices = ['cuda', 'cuda']

        for time_step in time_steps:


            config = OmegaConf.load(config_path)

            if model_name == 'SD-v1-4':
                model = load_model_from_config(config_path, ckpt_path, device=devices[0])
            elif model_name == 'SD-v2-1':
                model = load_model_from_config(config_path, ckpt_path, device=devices[0])
            else:
                model = load_model_from_config_compvis(config, ckpt_path, verbose=True)
            
            sampler = DDIMSampler(model) 

            if model_name == 'SD-v1-4':
                tokenizer_vocab = model.cond_stage_model.tokenizer.get_vocab()
                assert len(tokenizer_vocab) == LEN_TOKENIZER_VOCAB
                # inverse the dictionary
                tokenizer_vocab_indexing = {v: k for k, v in tokenizer_vocab.items()}
                token_embedding = model.cond_stage_model.transformer.text_model.embeddings.token_embedding.weight # shape (49408, 768)
                assert token_embedding.shape == (LEN_TOKENIZER_VOCAB, EMBEDDING_DIM)
                tokens_embedding = {tokenizer_vocab_indexing[i]: token_embedding[i] for i in range(LEN_TOKENIZER_VOCAB)}
                assert len(tokens_embedding) == LEN_TOKENIZER_VOCAB

            elif model_name == 'SD-v2-1':
                tokenizer_vocab = model.cond_stage_model.tokenizer.encoder
                assert len(tokenizer_vocab) == LEN_TOKENIZER_VOCAB
                tokenizer_vocab_indexing = {v: k for k, v in tokenizer_vocab.items()}
            
            else:
                tokenizer_vocab = model.cond_stage_model.tokenizer.encoder
                assert len(tokenizer_vocab) == LEN_TOKENIZER_VOCAB
                tokenizer_vocab_indexing = {v: k for k, v in tokenizer_vocab.items()}                

            # if collections_tokens is None: 
            #     collections_tokens = []
            #     for _concept in ["person", "nudity", "sexy", "cassette player", "garbage truck", ""]:
            #         top_k_tokens, _ = search_closest_tokens(_concept, model, k=20)
            #         collections_tokens.extend(top_k_tokens)
            #     print(collections_tokens)

            collections_tokens = list(tokenizer_vocab.keys())
            
            top_k_tokens, sorted_sim_dict, _ = search_closest_output_multi_steps(concept, model, sampler, collections_tokens, start_guidance=3.0, time_step=time_step, start_concept=start_concept, k=10, sim=sim, multi_steps=args.multi_steps)
            
            os.makedirs('evaluation_folder/similarity', exist_ok=True)

            with open(f'evaluation_folder/similarity/all_similarity_scores_{concept}_{time_step}_{start_concept}_{model_name}_{sim}_multi_steps_{args.multi_steps}.csv', 'w') as f:
                for key in sorted_sim_dict.keys():
                    f.write("%s,%s\n"%(key, sorted_sim_dict[key].item()))
            
            # top_k_tokens, sorted_sim_dict = search_closest_output_CLIP(concept, model, sampler, collections_tokens, start_guidance=3.0, time_step=time_step, start_concept=start_concept, k=10, sim=sim)
            # with open(f'evaluation_folder/similarity/similarity_scores_clip_{concept}_{time_step}_{start_concept}_{model_name}_{sim}.csv', 'w') as f:
            #     for key in sorted_sim_dict.keys():
            #         f.write("%s,%s\n"%(key, sorted_sim_dict[key].item()))


def test_search_closest_output_customized(args):
    # config_path = 'configs/stable-diffusion/v1-inference.yaml'
    config_path = args.config_path
    model_name = args.model_name
    ckpt_path = args.ckpt_path

    # concepts = ["nudity", "sexy", "cassette player", "garbage truck", "person", "a photo"]

    # concepts = [args.concept]

    concepts = ["nudity"]

    # concepts = ["nudity", "cassette player", "person"]

    # time_steps = [0, 10, 20, 30, 40, 50]

    time_steps = [args.time_step]

    start_concept = "a photo"

    sim = 'l2'

    collections_tokens = ['women', 'men', 'person', 'hat', 'apple', 'bamboo', 
                          'notebooks', 'australians', 'president', 'boat', 'lexus', 'money', 'banana']


    num_repeat = 100 

    for concept in concepts:
        devices = ['cuda', 'cuda']

        for time_step in time_steps:

            config = OmegaConf.load(config_path)

            if model_name == 'SD-v1-4':
                model = load_model_from_config(config_path, ckpt_path, device=devices[0])
            elif model_name == 'SD-v2-1':
                model = load_model_from_config(config_path, ckpt_path, device=devices[0])
            else:
                model = load_model_from_config_compvis(config, ckpt_path, verbose=True)
            
            sampler = DDIMSampler(model) 

            os.makedirs('evaluation_folder/similarity', exist_ok=True)

            with open(f'evaluation_folder/similarity/customized_similarity_scores_{concept}_{time_step}_{start_concept}_{model_name}_{sim}_multi_steps_{args.multi_steps}.csv', 'w') as f:
                for i in range(num_repeat):
                    _, sorted_sim_dict, _ = search_closest_output_multi_steps(concept, model, sampler, collections_tokens, start_guidance=3.0, time_step=time_step, start_concept=start_concept, k=10, sim=sim, multi_steps=args.multi_steps)
                
                    for key in sorted_sim_dict.keys():
                        f.write("%i,%s,%s\n"%(i,key, sorted_sim_dict[key].item()))
                

def test_search_closest_tokens(args):
    # config_path = 'configs/stable-diffusion/v1-inference.yaml'
    config_path = args.config_path
    model_name = args.model_name
    ckpt_path = args.ckpt_path

    # concepts = ["nudity", "sexy", "cassette player", "garbage truck", "person", "a photo"]

    concepts = ["nudity"]

    start_concept = "a photo"

    sim = 'cosine'

    # model_name = 'SD-v1-4'
    # ckpt_paths = '../Better_Erasing/models/erase/sd-v1-4-full-ema.ckpt'
    # model_name = 'SD-v2-1'
    # ckpt_paths = '/home/tbui/pb90_scratch/bta/workspace/GenerativeAI/Adversarial_Erasure_SDv2/checkpoints/v2-1_512-ema-pruned.ckpt'

    model = load_model_from_config(config_path, ckpt_path, device='cuda')

    for sim in ['cosine', 'l2']:
        for concept in concepts:
            _, sorted_sim_dict = search_closest_tokens(concept, model, k=20, sim=sim, model_name=model_name)

            os.makedirs('evaluation_folder/similarity', exist_ok=True)

            with open(f'evaluation_folder/similarity/all_closest_tokens_{concept}_{start_concept}_{model_name}_{sim}.csv', 'w') as f:
                for key in sorted_sim_dict.keys():
                    f.write("%s,%s\n"%(key, sorted_sim_dict[key].item()))
                


def test_search_closest_tokens_in_set(args, embedding_space='text'):
    # config_path = 'configs/stable-diffusion/v1-inference.yaml'
    config_path = args.config_path
    model_name = args.model_name
    ckpt_path = args.ckpt_path

    # concepts = ["nudity", "sexy", "cassette player", "garbage truck", "person", "a photo"]

    concept = "nudity"

    set_of_tokens = ['women', 'men', 'person', 'hat', 'apple', 'bamboo', 'notebooks</w>', 'australians</w>', 'president', 'boat', 'lexus</w>', 'money', 'banana</w>', 'naked']

    start_concept = "a photo"

    sim = 'l2'

    time_step = args.time_step

    num_repeat = 100

    # model_name = 'SD-v1-4'
    # ckpt_paths = '../Better_Erasing/models/erase/sd-v1-4-full-ema.ckpt'
    # model_name = 'SD-v2-1'
    # ckpt_paths = '/home/tbui/pb90_scratch/bta/workspace/GenerativeAI/Adversarial_Erasure_SDv2/checkpoints/v2-1_512-ema-pruned.ckpt'

    model = load_model_from_config(config_path, ckpt_path, device='cuda')

    if model_name == 'SD-v1-4':
        tokenizer_vocab = model.cond_stage_model.tokenizer.get_vocab()
    elif model_name == 'SD-v2-1':
        tokenizer_vocab = model.cond_stage_model.tokenizer.encoder

    # next, compute the similarity between the concept and each token in the set of tokens
    def similarity(concept_embedding, token_embedding, sim='cosine'):
        if sim == 'cosine':
            res = F.cosine_similarity(concept_embedding.flatten(), token_embedding.flatten(), dim=-1)
        elif sim == 'l2':
            res = - F.pairwise_distance(concept_embedding.flatten(), token_embedding.flatten(), p=2)
        return res

    os.makedirs('evaluation_folder/similarity', exist_ok=True)

    if embedding_space == 'text':
        file_name = f'evaluation_folder/similarity/all_closest_tokens_in_set_{concept}_{start_concept}_{model_name}_{sim}_{embedding_space}.csv'
    elif embedding_space == 'image':
        file_name = f'evaluation_folder/similarity/all_closest_tokens_in_set_{concept}_{start_concept}_{model_name}_{sim}_{embedding_space}_{time_step}_multi_steps_{args.multi_steps}.csv'
    
    with open(file_name, 'w') as f:
        for i in range(num_repeat):
            # ---------------
            # compute in embedding space 
            # first, retrieve all the token embeddings 
            if embedding_space == 'text':
                concept_embedding = model.get_learned_conditioning([concept]) # shape (1, 77, 768)
                set_of_tokens_embedding = {}
                for token in set_of_tokens:
                    token_embedding = retrieve_embedding_token(model_name, token)
                    set_of_tokens_embedding[token] = token_embedding
            
            elif embedding_space == 'image':
                sampler = DDIMSampler(model)
                _, _, output_embeddings = search_closest_output_multi_steps(concept, model, sampler, set_of_tokens, start_guidance=3.0, time_step=time_step, start_concept=start_concept, k=len(set_of_tokens), sim=sim, multi_steps=args.multi_steps)
                concept_embedding = output_embeddings[concept]
                set_of_tokens_embedding = {k: v for k, v in output_embeddings.items() if k in set_of_tokens}

            all_sim_concept_vs_tokens = []
            for token in set_of_tokens:
                res = similarity(concept_embedding, set_of_tokens_embedding[token], sim=sim)
                all_sim_concept_vs_tokens.append(res)
            
            all_sim_concept_vs_tokens = torch.stack(all_sim_concept_vs_tokens) # shape (14,)
            assert all_sim_concept_vs_tokens.shape == (len(set_of_tokens),)

            # next, compute the similarity between each token in the set of tokens and all other tokens in the set of tokens
            all_tokens_embedding = torch.stack(list(set_of_tokens_embedding.values())) # shape (14, 77, 768)
            # assert all_tokens_embedding.shape == (len(set_of_tokens), 77, 768)
            all_tokens_1 = all_tokens_embedding.unsqueeze(1) # shape (14, 1, 77, 768)
            all_tokens_2 = all_tokens_embedding.unsqueeze(0) # shape (1, 14, 77, 768)
            
            if sim == 'cosine':
                all_sim_tokens_vs_tokens = F.cosine_similarity(all_tokens_1.flatten(2), all_tokens_2.flatten(2), dim=-1) # shape (14, 14)
            elif sim == 'l2':
                all_sim_tokens_vs_tokens = - F.pairwise_distance(all_tokens_1.flatten(2), all_tokens_2.flatten(2), p=2) # shape (14, 14)
            
            # final similarity: sim(concept, token_i) = sum (all_sim_tokens_vs_tokens(i,j) * all_sim_concept_vs_tokens(j)) for all token j in the set of tokens
            # normalize so that all similarity score is positive
            all_sim_tokens_vs_tokens = all_sim_tokens_vs_tokens + torch.abs(torch.min(all_sim_tokens_vs_tokens))
            all_sim_concept_vs_tokens = all_sim_concept_vs_tokens + torch.abs(torch.min(all_sim_concept_vs_tokens))
            final_sim_concept_vs_tokens = torch.matmul(all_sim_tokens_vs_tokens, all_sim_concept_vs_tokens) # shape (14,)


            assert final_sim_concept_vs_tokens.shape == (len(set_of_tokens),)
            # if torch.sum(all_sim_tokens_vs_tokens[0,:] * all_sim_concept_vs_tokens) != final_sim_concept_vs_tokens[0]:
            #     print(f"{torch.sum(all_sim_tokens_vs_tokens[0,:] * all_sim_concept_vs_tokens)} != {final_sim_concept_vs_tokens[0]}")
            #     print(all_sim_tokens_vs_tokens)
            #     print(all_sim_concept_vs_tokens)
            #     print(final_sim_concept_vs_tokens)
            #     print('at iteration', i)
            #     raise ValueError("Error in computing the final similarity")


            # normalization with the size of the set of tokens
            final_sim_concept_vs_tokens = final_sim_concept_vs_tokens / len(set_of_tokens)**2

            sorted_sim_dict = {k: v for k, v in sorted(zip(set_of_tokens, final_sim_concept_vs_tokens), key=lambda item: item[1], reverse=True)}
            print(f"Top-{len(set_of_tokens)} closest tokens to the concept {concept} are: {sorted_sim_dict}") 
            # ---------------

            for key in sorted_sim_dict.keys():
                f.write("%i,%s,%s\n"%(i,key, sorted_sim_dict[key].item()))



                

def test_compare_two_models(args):
    # raise NotImplementedError("This function is not implemented yet")
    # config_path = 'configs/stable-diffusion/v1-inference.yaml'
    config_path = args.config_path

    # time_steps = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

    time_steps = [args.time_step]

    start_concept = "a photo"

    sim = args.sim

    collections_tokens = None

    concepts = ["nudity"]

    devices = ['cuda', 'cuda']

    ckpt_path = '/home/tbui/pb90_scratch/bta/workspace/GenerativeAI/Adversarial_Erasure_SDv2/checkpoints/v2-1_512-ema-pruned.ckpt'
    model_org = load_model_from_config(config_path, ckpt_path, device=devices[0])

    if args.model_name == 'SD-v1-4':
        tokenizer_vocab = model_org.cond_stage_model.tokenizer.get_vocab()
        assert len(tokenizer_vocab) == LEN_TOKENIZER_VOCAB
        # inverse the dictionary
        tokenizer_vocab_indexing = {v: k for k, v in tokenizer_vocab.items()}
        token_embedding = model_org.cond_stage_model.transformer.text_model.embeddings.token_embedding.weight # shape (49408, 768)
        assert token_embedding.shape == (LEN_TOKENIZER_VOCAB, EMBEDDING_DIM)
        tokens_embedding = {tokenizer_vocab_indexing[i]: token_embedding[i] for i in range(LEN_TOKENIZER_VOCAB)}
        assert len(tokens_embedding) == LEN_TOKENIZER_VOCAB
    elif args.model_name == 'SD-v2-1':
        tokenizer_vocab = model_org.cond_stage_model.tokenizer.encoder
        assert len(tokenizer_vocab) == LEN_TOKENIZER_VOCAB
        tokenizer_vocab_indexing = {v: k for k, v in tokenizer_vocab.items()}
    else:
        tokenizer_vocab = model_org.cond_stage_model.tokenizer.encoder
        assert len(tokenizer_vocab) == LEN_TOKENIZER_VOCAB
        tokenizer_vocab_indexing = {v: k for k, v in tokenizer_vocab.items()}

    collections_tokens = list(tokenizer_vocab.keys())   

    for concept in concepts:
        if concept in ["nudity"]:
            ckpt_paths = {
                'esd-sdv21-nudity': 'models/compvis-word_nudity-method_noxattn-sg_3-ng_1-iter_1000-lr_1e-05-info_SDv21/compvis-word_nudity-method_noxattn-sg_3-ng_1-iter_1000-lr_1e-05-info_SDv21.pt',
            }
        else:
            ckpt_paths = {
            }

        for time_step in time_steps:

            for model_name in ckpt_paths:

                config = OmegaConf.load(config_path)

                if model_name == 'SD-v1-4':
                    model = load_model_from_config(config_path, ckpt_paths[model_name], device=devices[0])
                else:
                    model = load_model_from_config_compvis(config, ckpt_paths[model_name], verbose=True)
                
                sampler = DDIMSampler(model) 

                _, sorted_sim_dict = compare_two_models_multi_steps(model, model_org, sampler, collections_tokens, start_guidance=3.0, time_step=time_step, start_concept=start_concept, k=10, sim=sim, multi_steps=args.multi_steps)
                
                with open(f'evaluation_folder/similarity/compare_2models_{concept}_{time_step}_{start_concept}_{model_name}_{sim}_multi_steps_{args.multi_steps}.csv', 'w') as f:
                    for key in sorted_sim_dict.keys():
                        f.write("%s,%s\n"%(key, sorted_sim_dict[key].item()))
                
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Compare two models')
    parser.add_argument('--concept', type=str, default="nudity", help='concept')
    parser.add_argument('--time_step', type=int, default=0, help='time step to to generate the base image')
    parser.add_argument('--sim', type=str, default='l2', help='similarity metric to use')
    parser.add_argument('--config_path', type=str, default='configs/stable-diffusion/v2-inference-fp32.yaml', help='path to the config file')
    parser.add_argument('--model_name', type=str, default='SD-v2-1', help='model name')
    parser.add_argument('--ckpt_path', type=str, default='/home/tbui/pb90_scratch/bta/workspace/GenerativeAI/Adversarial_Erasure_SDv2/checkpoints/v2-1_512-ema-pruned.ckpt', help='path to the checkpoint')
    parser.add_argument('--multi_steps', type=int, default=2, help='number of steps to sample')
    
    args = parser.parse_args()
    # test_load_model(args, save_mode='dict') # just run one time for all model base SD-v1-4 or SD-v2-1
    # test_search_closest_tokens(args) # just run one time for all model base SD-v1-4 or SD-v2-1
    # test_search_closest_output(args)
    # test_search_closest_output_customized(args)
    # test_compare_two_models(args)
    # test_search_closest_tokens_in_set(args, embedding_space='text')
    test_search_closest_tokens_in_set(args, embedding_space='image')
