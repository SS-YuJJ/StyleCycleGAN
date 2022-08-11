from random import random
import torchvision

from models.stylegan2.stylegan2_pytorch import ModelLoader, Trainer
from models.stylegan2.stylegan2_pytorch import image_noise, mixed_list, noise_list, latent_to_w, styles_def_to_tensor

###########################################
model_args = dict(
        name = 'styleGAN2_celebA',
        results_dir = './results',
        models_dir = './models',
        batch_size = 16,
        base_dir = './models/stylegan_models',
        gradient_accumulate_every = 1,
        image_size = 256,
        network_capacity = 16,
        fmap_max = 512,
        transparent = False,
        lr = 2e-4,
        lr_mlp = 0.1,
        ttur_mult = 1.5,
        rel_disc_loss = 1.5,
        num_workers = None,
        save_every = 1000,
        evaluate_every = 1000,
        num_image_tiles = 4,
        trunc_psi = 0.75,
        fp16 = False,
        no_pl_reg = False,
        cl_reg = False,
        fq_layers = [],
        fq_dict_size = 256,
        attn_layers = [],
        no_const = False,
        aug_prob = 0.,
        aug_types = ['translation', 'cutout'],
        top_k_training = False,
        generator_top_k_gamma = 0.99,
        generator_top_k_frac = 0.5,
        dual_contrast_loss = False,
        dataset_aug_prob = 0.,
        calculate_fid_every = None,
        calculate_fid_num_images = 12800,
        clear_fid_cache = False,
        mixed_prob = 0.9,
        log = False
    )

###########################################

model = Trainer(**model_args)
load_from = -1
model.load(load_from)

# =========== Dummy var ===============
save_path = './imgs/test_stylegan.jpg'
batch_size = 16
num_rows = 4
######################################

stylegan_S = model.GAN.S
stylegan_G = model.GAN.G

image_size = stylegan_G.image_size
latent_dim = stylegan_G.latent_dim
num_layers = stylegan_G.num_layers
mixed_prob = 0.9
get_latents_fn = mixed_list if random() < mixed_prob else noise_list
print(get_latents_fn)
style = get_latents_fn(batch_size, num_layers, latent_dim, device='cuda:0')
print(f"****** style shape ******* {(style)}")

# noise = image_noise(batch_size, image_size, device=0)
# print(f"****** noise ******* {noise.shape}")

# w_space = latent_to_w(stylegan_S, style)
# print(f"****** w_space 1******* {len(w_space)}")
# print(f"****** w_space 2******* {(w_space[0][0].shape)}")
# print(f"****** w_space 3******* {(w_space[1][0].shape)}")
# print(f"****** w_space 4******* {(w_space[0][1])}")
# print(f"****** w_space 5******* {(w_space[1][1])}")

# w_styles = styles_def_to_tensor(w_space)
# print(f"****** w_styles ******* {w_styles.shape}")

# generated_images = stylegan_G(w_styles, noise)
# print(f"****** generated_images ******* {generated_images.shape}")


# trunc_psi = 0.75
# latents = noise_list(num_rows ** 2, num_layers, latent_dim, device=0)
# n = image_noise(num_rows ** 2, image_size, device=0)
# generated_images = model.generate_truncated(stylegan_S, stylegan_G, latents, n, trunc_psi = trunc_psi)
# torchvision.utils.save_image(generated_images, fp=save_path, nrow=num_rows)