from omegaconf import OmegaConf
import torch
from PIL import Image
from torchvision import transforms
import os
from tqdm import tqdm
from einops import rearrange
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import einops

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config
import random
import glob
import re
import shutil
import pdb
import argparse
from convertModels import savemodelDiffusers
from PIL import Image
from torch.autograd import Variable
from utils_exp import get_prompt, sanitize_filename, str2bool
from gen_embedding_matrix import learn_k_means_from_input_embedding, learn_k_means_from_output, save_embedding_matrix, search_closest_tokens, retrieve_embedding_token

import set_threads

# Util Functions
def load_model_from_config(config, ckpt, device="cpu", verbose=False):
    """Loads a model from config and a ckpt
    if config is a path will use omegaconf to load
    """
    if isinstance(config, (str, Path)):
        config = OmegaConf.load(config)

    pl_sd = torch.load(ckpt, map_location="cpu")
    global_step = pl_sd["global_step"]
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()
    model.cond_stage_model.device = device
    return model


@torch.no_grad()
def sample_model(model, sampler, c, h, w, ddim_steps, scale, ddim_eta, start_code=None, n_samples=1,t_start=-1,log_every_t=None,till_T=None,verbose=True):
    """Sample the model"""
    uc = None
    if scale != 1.0:
        uc = model.get_learned_conditioning(n_samples * [""])
    log_t = 100
    if log_every_t is not None:
        log_t = log_every_t
    shape = [4, h // 8, w // 8]
    samples_ddim, inters = sampler.sample(S=ddim_steps,
                                     conditioning=c,
                                     batch_size=n_samples,
                                     shape=shape,
                                     verbose=False,
                                     x_T=start_code,
                                     unconditional_guidance_scale=scale,
                                     unconditional_conditioning=uc,
                                     eta=ddim_eta,
                                     verbose_iter = verbose,
                                     t_start=t_start,
                                     log_every_t = log_t,
                                     till_T = till_T
                                    )
    if log_every_t is not None:
        return samples_ddim, inters
    return samples_ddim

def load_img(path, target_size=512):
    """Load an image, resize and output -1..1"""
    image = Image.open(path).convert("RGB")


    tform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.CenterCrop(target_size),
            transforms.ToTensor(),
        ])
    image = tform(image)
    return 2.*image - 1.


def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def plot_loss(losses, path,word, n=100):
    v = moving_average(losses, n)
    plt.plot(v, label=f'{word}_loss')
    plt.legend(loc="upper left")
    plt.title('Average loss in trainings', fontsize=20)
    plt.xlabel('Data point', fontsize=16)
    plt.ylabel('Loss value', fontsize=16)
    plt.savefig(path)
    plt.close()


def get_models(config_path, ckpt_path, devices):
    model_orig = load_model_from_config(config_path, ckpt_path, devices[1])
    sampler_orig = DDIMSampler(model_orig)

    model = load_model_from_config(config_path, ckpt_path, devices[0])
    sampler = DDIMSampler(model)

    return model_orig, sampler_orig, model, sampler

def save_to_dict(var, name, dict):
    if var is not None:
        if isinstance(var, torch.Tensor):
            var = var.cpu().detach().numpy()
        if isinstance(var, list):
            var = [v.cpu().detach().numpy() if isinstance(v, torch.Tensor) else v for v in var]
    else:
        return dict
    
    if name not in dict:
        dict[name] = []
    
    dict[name].append(var)
    return dict



def train(prompt, train_method, start_guidance, negative_guidance, iterations, lr, config_path, ckpt_path, diffusers_config_path, devices, seperator=None, image_size=512, ddim_steps=50, args=None):
    '''
    Function to train diffusion models to erase concepts from model weights

    Parameters
    ----------
    prompt : str
        The concept to erase from diffusion model (Eg: "Van Gogh").
    train_method : str
        The parameters to train for erasure (noxattn, xattan).
    start_guidance : float
        Guidance to generate images for training.
    negative_guidance : float
        Guidance to erase the concepts from diffusion model.
    iterations : int
        Number of iterations to train.
    lr : float
        learning rate for fine tuning.
    config_path : str
        config path for compvis diffusion format.
    ckpt_path : str
        checkpoint path for pre-trained compvis diffusion weights.
    diffusers_config_path : str
        Config path for diffusers unet in json format.
    devices : str
        2 devices used to load the models (Eg: '0,1' will load in cuda:0 and cuda:1).
    seperator : str, optional
        If the prompt has commas can use this to seperate the prompt for individual simulataneous erasures. The default is None.
    image_size : int, optional
        Image size for generated images. The default is 512.
    ddim_steps : int, optional
        Number of diffusion time steps. The default is 50.

    Returns
    -------
    None

    '''
    # PROMPT CLEANING
    word_print = prompt.replace(' ','')

    prompt, preserved = get_prompt(prompt)

    if seperator is not None:
        erased_words = prompt.split(seperator)
        erased_words = [word.strip() for word in erased_words]
        preserved_words = preserved.split(seperator)
        preserved_words = [word.strip() for word in preserved_words]
    else:
        erased_words = [prompt]
        preserved_words = [preserved]
    
    print('to be erased:', erased_words)
    print('to be preserved:', preserved_words)
    preserved_words.append('')

    ddim_eta = 0

    model_orig, _, model, sampler = get_models(config_path, ckpt_path, devices)

    # choose parameters to train based on train_method
    parameters = []
    for name, param in model.model.diffusion_model.named_parameters():
        # train all layers except x-attns and time_embed layers
        if train_method == 'noxattn':
            if name.startswith('out.') or 'attn2' in name or 'time_embed' in name:
                pass
            else:
                print(name)
                parameters.append(param)
        # train only self attention layers
        if train_method == 'selfattn':
            if 'attn1' in name:
                print(name)
                parameters.append(param)
        # train only x attention layers
        if train_method == 'xattn':
            if 'attn2' in name:
                print(name)
                parameters.append(param)
        # train only qkv layers in x attention layers
        if train_method == 'xattn_matching':
            if 'attn2' in name and ('to_q' in name or 'to_k' in name or 'to_v' in name):
                print(name)
                parameters.append(param)
                # return_nodes[name] = name
        # train all layers
        if train_method == 'full':
            print(name)
            parameters.append(param)
        # train all layers except time embed layers
        if train_method == 'notime':
            if not (name.startswith('out.') or 'time_embed' in name):
                print(name)
                parameters.append(param)
        if train_method == 'xlayer':
            if 'attn2' in name:
                if 'output_blocks.6.' in name or 'output_blocks.8.' in name:
                    print(name)
                    parameters.append(param)
        if train_method == 'selflayer':
            if 'attn1' in name:
                if 'input_blocks.4.' in name or 'input_blocks.7.' in name:
                    print(name)
                    parameters.append(param)
    
    def decode_and_save_image(model_orig, z, path):
        x = model_orig.decode_first_stage(z)
        x = torch.clamp((x + 1.0)/2.0, min=0.0, max=1.0)
        x = rearrange(x, 'b c h w -> b h w c')
        image = Image.fromarray((x[0].cpu().numpy()*255).astype(np.uint8))
        plt.imshow(image)
        plt.xticks([])
        plt.yticks([])
        plt.savefig(path)
        plt.close()

    model.train()
    # create a lambda function for cleaner use of sampling code (only denoising till time step t)
    quick_sample_till_t = lambda cond, s, code, t: sample_model(model, sampler,
                                                                 cond, image_size, image_size, ddim_steps, s, ddim_eta,
                                                                 start_code=code, till_T=t, verbose=False)

    losses = []
    opt = torch.optim.Adam(parameters, lr=lr)
    criteria = torch.nn.MSELoss()
    history_dict = {}

    name = f'compvis-adversarial-gumbel-word_{word_print}-method_{train_method}-sg_{start_guidance}-ng_{negative_guidance}-iter_{iterations}-lr_{lr}-info_{args.info}'
    models_path = args.models_path
    os.makedirs(f'evaluation_folder/{name}', exist_ok=True)
    os.makedirs(f'invest_folder/{name}', exist_ok=True)
    os.makedirs(f'{models_path}/{name}', exist_ok=True)

    pbar = tqdm(range(args.pgd_num_steps*iterations))

    def create_prompt(word):
        prompt = f'{word}'
        emb = model.get_learned_conditioning([prompt])
        init = emb
        return init

    fixed_start_code = torch.randn((1, 4, 64, 64)).to(devices[0])    


    # create a matrix of embeddings for the entire vocabulary
    if not os.path.exists('models/embedding_matrix_dict_EN3K.pt'):
        save_embedding_matrix(model, model_name='SD-v1-4', save_mode='dict', vocab='EN3K')

    if not os.path.exists('models/embedding_matrix_array_EN3K.pt'):
        save_embedding_matrix(model, model_name='SD-v1-4', save_mode='array', vocab='EN3K')
    
    if not os.path.exists('models/embedding_matrix_array_Imagenet.pt'):
        save_embedding_matrix(model, model_name='SD-v1-4', save_mode='array', vocab='Imagenet')


    # Search the closest tokens in the vocabulary for each erased word, using the similarity matrix
    # if vocab in ['EN3K', 'Imagenet', 'CLIP'], then use the pre-defined vocabulary
    # if vocab in concept_dict.all_concepts, then use the custom concepts, i.e., 'nudity', 'artistic', 'human_body'
    # if vocab is 'keyword', then use the keywords in the erased words, defined in utils_concept.py

    from utils_concept import ConceptDict
    concept_dict = ConceptDict()
    concept_dict.load_all_concepts()

    print('ignore_special_tokens:', args.ignore_special_tokens)
    
    all_sim_dict = dict()
    for word in erased_words:
        if args.vocab in ['EN3K', 'Imagenet', 'CLIP']:
            vocab = args.vocab
        elif args.vocab in concept_dict.all_concepts:
            # i.e., nudity, artistic, human_body
            vocab = concept_dict.get_concepts_as_dict(args.vocab)
        elif args.vocab == 'keyword':
            # i.e., 'Cassette Player', 'Chain Saw', 'Church', 'Gas Pump', 'Tench', 'Garbage Truck', 'English Springer', 'Golf Ball', 'Parachute', 'French Horn'
            vocab = concept_dict.get_concepts_as_dict(word)
        else:
            raise ValueError(f'Word {word} not found in concept dictionary, it should be either in EN3K, Imagenet, CLIP, or in the concept dictionary')
        
        top_k_tokens, sorted_sim_dict = search_closest_tokens(word, model, k=args.gumbel_k_closest, sim='l2', model_name='SD-v1-4', ignore_special_tokens=args.ignore_special_tokens, vocab=vocab)
        all_sim_dict[word] = {key:sorted_sim_dict[key] for key in top_k_tokens}

    if args.gumbel_num_centers > 0:
        assert args.gumbel_num_centers % len(erased_words) == 0, 'Number of centers should be divisible by number of erased words'
    preserved_dict = dict()

    for word in erased_words:
        temp = learn_k_means_from_input_embedding(sim_dict=all_sim_dict[word], num_centers=args.gumbel_num_centers)
        preserved_dict[word] = temp

    history_dict = save_to_dict(preserved_dict, f'preserved_set_0', history_dict)


    # create a matrix of embeddings for the preserved set
    print('Creating preserved matrix')
    weight_pi_dict = dict()
    preserved_matrix_dict = dict()
    for erase_word in erased_words:
        preserved_set = preserved_dict[erase_word]
        for i, word in enumerate(preserved_set):
            if i == 0:
                preserved_matrix = create_prompt(word)
            else:
                preserved_matrix = torch.cat((preserved_matrix, create_prompt(word)), dim=0)
            print(i, word, preserved_matrix.shape)    
        preserved_matrix = preserved_matrix.flatten(start_dim=1) # [n, 77*768]
        weight_pi = torch.zeros((1, preserved_matrix.shape[0]), device=devices[0], dtype=preserved_matrix.dtype) # [1, n]
        weight_pi = weight_pi + 1 / preserved_matrix.shape[0]
        weight_pi = Variable(weight_pi, requires_grad=True)
        weight_pi_dict[erase_word] = weight_pi
        preserved_matrix_dict[erase_word] = preserved_matrix
    
    print('weight_pi_dict:', weight_pi_dict)
    history_dict = save_to_dict(weight_pi_dict, f'one_hot_dict_0', history_dict)

    # optimizer for all pi vectors
    opt_weight_pi = torch.optim.Adam([weight_pi for weight_pi in weight_pi_dict.values()], lr=args.gumbel_lr)

    """
    Gumbel-Softmax function
        if `hard` is 1, then it is one-hot, if `hard` is 0, then it is a new soft version, which takes the top-k highest values and normalize them to 1
    """
    def gumbel_softmax(logits, temperature=args.gumbel_temp, hard=args.gumbel_hard, eps=1e-10, k=args.gumbel_topk):
        u = torch.rand_like(logits)
        gumbel = -torch.log(-torch.log(u + eps) + eps)
        y = logits + gumbel
        y = torch.nn.functional.softmax(y / temperature, dim=-1)
        if hard == 1:
            y_hard = torch.zeros_like(logits)
            y_hard.scatter_(-1, torch.argmax(y, dim=-1, keepdim=True), 1.0)
            y = (y_hard - y).detach() + y
        elif hard == 0:
            top_k_values, _ = torch.topk(y, k, dim=-1)
            top_k_mask = y >= top_k_values[..., -1].unsqueeze(-1)
            y = y * top_k_mask.float()
            y = y / y.sum(dim=-1, keepdim=True)
        return y

    for i in pbar:
        word = random.sample(erased_words,1)[0]

        opt.zero_grad()
        model.zero_grad()
        model_orig.zero_grad()
        opt_weight_pi.zero_grad()

        c_e = f'{word}'

        emb_c_e = model.get_learned_conditioning([c_e])

        emb_c_t = torch.reshape(torch.matmul(gumbel_softmax(weight_pi_dict[word]), preserved_matrix_dict[word]).unsqueeze(0), (1, 77, 768))
        assert emb_c_t.shape == emb_c_e.shape

        # clone the emb_c_t for the time step
        emb_0 = emb_c_t.clone().detach()

        t_enc = torch.randint(ddim_steps, (1,), device=devices[0])
        # time step from 1000 to 0 (0 being good)
        og_num = round((int(t_enc)/ddim_steps)*1000)
        og_num_lim = round((int(t_enc+1)/ddim_steps)*1000)

        t_enc_ddpm = torch.randint(og_num, og_num_lim, (1,), device=devices[0])

        start_code = torch.randn((1, 4, 64, 64)).to(devices[0])

        with torch.no_grad():
            # generate an image with the concept
            z_c_e = quick_sample_till_t(emb_c_e.to(devices[0]), start_guidance, start_code, int(t_enc))
            z_c_t = quick_sample_till_t(emb_c_t.to(devices[0]), start_guidance, start_code, int(t_enc))

            # get conditional and unconditional scores from frozen model at time step t and image z_c_e
            eps_0_org = model_orig.apply_model(z_c_e.to(devices[1]), t_enc_ddpm.to(devices[1]), emb_0.to(devices[1]))
            eps_e_org = model_orig.apply_model(z_c_e.to(devices[1]), t_enc_ddpm.to(devices[1]), emb_c_e.to(devices[1]))
            eps_t_org = model_orig.apply_model(z_c_t.to(devices[1]), t_enc_ddpm.to(devices[1]), emb_c_t.to(devices[1]))

        # breakpoint()
        # get conditional score
        eps_e = model.apply_model(z_c_e.to(devices[0]), t_enc_ddpm.to(devices[0]), emb_c_e.to(devices[0]))
        eps_t = model.apply_model(z_c_t.to(devices[0]), t_enc_ddpm.to(devices[0]), emb_c_t.to(devices[0]))

        eps_0_org.requires_grad = False
        eps_e_org.requires_grad = False
        eps_t_org.requires_grad = False

        # using DDIM inversion to project the x_t to x_0
        # check that the alphas is in descending order
        assert torch.all(sampler.ddim_alphas[:-1] >= sampler.ddim_alphas[1:])
        alpha_bar_t = sampler.ddim_alphas[int(t_enc)]
        eps_e_pred = (z_c_e - torch.sqrt(1 - alpha_bar_t) * eps_e) / torch.sqrt(alpha_bar_t)
        eps_t_pred = (z_c_t - torch.sqrt(1 - alpha_bar_t) * eps_t) / torch.sqrt(alpha_bar_t)

        eps_e_org_pred = (z_c_e - torch.sqrt(1 - alpha_bar_t) * eps_e_org) / torch.sqrt(alpha_bar_t)
        eps_0_org_pred = (z_c_e - torch.sqrt(1 - alpha_bar_t) * eps_0_org) / torch.sqrt(alpha_bar_t)
        eps_t_org_pred = (z_c_t - torch.sqrt(1 - alpha_bar_t) * eps_t_org) / torch.sqrt(alpha_bar_t)

        if i % args.pgd_num_steps == 0:
            # optimize the model
            loss = 0
            loss += criteria(eps_e_pred.to(devices[0]), eps_0_org_pred.to(devices[0]) - (negative_guidance * (eps_e_org_pred.to(devices[0]) - eps_0_org_pred.to(devices[0]))))
            loss += args.lamda * criteria(eps_t_pred.to(devices[0]), eps_t_org_pred.to(devices[0])) # preserved concepts, output are the same without prompt

            # update weights to erase the concept
            loss.backward()
            losses.append(loss.item())
            pbar.set_postfix({"loss": loss.item()})
            history_dict = save_to_dict(loss.item(), 'loss', history_dict)
            opt.step()
        else:
            # update the weight_pi vector
            opt.zero_grad()
            opt_weight_pi.zero_grad()
            model.zero_grad()
            model_orig.zero_grad()
            # weight_pi.grad = None
            loss = 0 
            loss -= criteria(eps_e_pred.to(devices[0]), eps_0_org_pred.to(devices[0]))
            loss -= args.lamda * criteria(eps_t_pred.to(devices[0]), eps_t_org_pred.to(devices[0])) # maximize the preserved loss
            loss.backward()
            preserved_set = preserved_dict[word]
            opt_weight_pi.step()
            history_dict = save_to_dict([weight_pi_dict[word].cpu().detach().numpy(), i, preserved_set[torch.argmax(weight_pi_dict[word], dim=1)], word], 'weight_pi', history_dict)
            history_dict = save_to_dict(weight_pi_dict, f'one_hot_dict_{i}', history_dict)

        if i % (args.save_freq) == 0:
            with torch.no_grad():
                for word in erased_words:
                    preserved_set = preserved_dict[word]
                    word_r = preserved_set[torch.argmax(weight_pi_dict[word], dim=1)]
                    emb_r_eval = torch.reshape(torch.matmul(gumbel_softmax(weight_pi_dict[word]), preserved_matrix_dict[word]).unsqueeze(0), (1, 77, 768))
                    emb_n_eval = model.get_learned_conditioning([word])
                    z_r_till_T = quick_sample_till_t(emb_r_eval.to(devices[0]), start_guidance, fixed_start_code, int(ddim_steps))
                    decode_and_save_image(model_orig, z_r_till_T, path=sanitize_filename(f'evaluation_folder/{name}/im_r_till_T_{i}_{word}_{word_r}.png'))
                    z_n_till_T = quick_sample_till_t(emb_n_eval.to(devices[0]), start_guidance, fixed_start_code, int(ddim_steps))
                    decode_and_save_image(model_orig, z_n_till_T, path=sanitize_filename(f'evaluation_folder/{name}/im_n_till_T_{i}_{word}.png'))

        if i % 100 == 0:
            save_history(losses, name, word_print, models_path=models_path)
            torch.save(history_dict, f'invest_folder/{name}/history_dict_{i}.pt')

    model.eval()

    save_model(model, name, None, models_path=models_path, save_compvis=True, save_diffusers=True, compvis_config_file=config_path, diffusers_config_file=diffusers_config_path)
    save_history(losses, name, word_print, models_path=models_path)
    
def save_model(model, name, num, models_path, compvis_config_file=None, diffusers_config_file=None, device='cpu', save_compvis=True, save_diffusers=True):

    folder_path = f'{models_path}/{name}'
    os.makedirs(folder_path, exist_ok=True)
    if num is not None:
        path = f'{folder_path}/{name}-epoch_{num}.pt'
    else:
        path = f'{folder_path}/{name}.pt'

    if save_compvis:
        torch.save(model.state_dict(), path)

    if save_diffusers:
        print('Saving Model in Diffusers Format')
        savemodelDiffusers(name, compvis_config_file, diffusers_config_file, device=device )

def save_history(losses, name, word_print, models_path):
    folder_path = f'{models_path}/{name}'
    os.makedirs(folder_path, exist_ok=True)
    with open(f'{folder_path}/loss.txt', 'w') as f:
        f.writelines([str(i) for i in losses])
    plot_loss(losses,f'{folder_path}/loss.png' , word_print, n=3)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Finetuning stable diffusion model to erase concepts')
    parser.add_argument('--prompt', help='prompt corresponding to concept to erase', type=str, required=True)
    parser.add_argument('--train_method', help='method of training', type=str, required=True)
    parser.add_argument('--start_guidance', help='guidance of start image used to train', type=float, required=False, default=3)
    parser.add_argument('--negative_guidance', help='guidance of negative training used to train', type=float, required=False, default=1)
    parser.add_argument('--iterations', help='iterations used to train', type=int, required=False, default=1000)
    parser.add_argument('--lr', help='learning rate used to train', type=float, required=False, default=1e-5)
    parser.add_argument('--config_path', help='config path for stable diffusion v1-4 inference', type=str, required=False, default='configs/stable-diffusion/v1-inference.yaml')
    parser.add_argument('--ckpt_path', help='ckpt path for stable diffusion v1-4', type=str, required=False, default='models/ldm/stable-diffusion-v1/sd-v1-4-full-ema.ckpt')
    parser.add_argument('--diffusers_config_path', help='diffusers unet config json path', type=str, required=False, default='diffusers_unet_config.json')
    parser.add_argument('--devices', help='cuda devices to train on', type=str, required=False, default='0,0')
    parser.add_argument('--seperator', help='separator if you want to train bunch of erased_words separately', type=str, required=False, default=None)
    parser.add_argument('--image_size', help='image size used to train', type=int, required=False, default=512)
    parser.add_argument('--ddim_steps', help='ddim steps of inference used to train', type=int, required=False, default=50)
    parser.add_argument('--info', help='info to add to model name', type=str, required=False, default='')
    parser.add_argument('--save_freq', help='frequency to save data, per iteration', type=int, required=False, default=10)
    parser.add_argument('--models_path', help='method of prompting', type=str, required=True, default='models')

    parser.add_argument('--gumbel_lr', help='learning rate for prompt', type=float, required=False, default=1e-3)
    parser.add_argument('--gumbel_temp', help='temperature for gumbel softmax', type=float, required=False, default=2)
    parser.add_argument('--gumbel_hard', help='hard for gumbel softmax, 0: soft, 1: hard', type=int, required=False, default=0, choices=[0,1])
    parser.add_argument('--gumbel_num_centers', help='number of centers for kmeans, if <= 0 then do not apply kmeans', type=int, required=False, default=100)
    parser.add_argument('--gumbel_update', help='update frequency for preserved set, if <= 0 then do not update', type=int, required=False, default=100)
    parser.add_argument('--gumbel_time_step', help='time step for the starting point to estimate epsilon', type=int, required=False, default=0)
    parser.add_argument('--gumbel_multi_steps', help='multi steps for calculating the output', type=int, required=False, default=2)
    parser.add_argument('--gumbel_k_closest', help='number of closest tokens to consider', type=int, required=False, default=1000)
    parser.add_argument('--gumbel_topk', help='number of top-k values in the soft gumbel softmax to be considered', type=int, required=False, default=5)
    parser.add_argument('--ignore_special_tokens', help='ignore special tokens in the embedding matrix', type=str2bool, required=False, default=True)
    parser.add_argument('--vocab', help='vocab', type=str, required=False, default='EN3K')
    parser.add_argument('--pgd_num_steps', help='number of step to optimize adversarial concepts', type=int, required=False, default=2)
    parser.add_argument('--lamda', help='lambda for the loss function', type=float, required=False, default=1)

    args = parser.parse_args()
    
    prompt = args.prompt
    train_method = args.train_method
    start_guidance = args.start_guidance
    negative_guidance = args.negative_guidance
    iterations = args.iterations
    lr = args.lr
    config_path = args.config_path
    ckpt_path = args.ckpt_path
    diffusers_config_path = args.diffusers_config_path
    devices = [f'cuda:{int(d.strip())}' for d in args.devices.split(',')]
    seperator = args.seperator
    image_size = args.image_size
    ddim_steps = args.ddim_steps

    train(prompt=prompt, train_method=train_method, start_guidance=start_guidance, negative_guidance=negative_guidance, iterations=iterations, lr=lr, config_path=config_path, ckpt_path=ckpt_path, diffusers_config_path=diffusers_config_path, devices=devices, seperator=seperator, image_size=image_size, ddim_steps=ddim_steps, args=args)
