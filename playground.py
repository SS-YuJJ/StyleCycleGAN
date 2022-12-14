from data import unaligned_dataset
from models.networks import CLIPWithLinearHead, StyleGenerator, CLIPInnerEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
import os
import numpy as np
from util.util import save_image
from loguru import logger
import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from tqdm import tqdm
import torchvision
from torchvision.transforms import Resize, ToTensor
from PIL import Image


try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')



def checkPath(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print("Making new folder: ", path)

# ================================
TOTAL_iter = 3000
DEVICE = "cuda:0"
IMG_SIZE=128
# ================================



def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor.cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        # print("======= ========= ======", image_numpy.shape)
        image_numpy = (np.transpose(image_numpy, (1, 2, 0))) * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


if __name__ == '__main__':

    image_filepaths = []

    for subdir, dir, files in os.walk(
        "./small_test_dataset"
    ):
        image_filepaths.extend(
            os.path.join(subdir, file) for file in files if file.endswith(".jpg")
        )

    x_ref = torch.stack(
        [
            Resize(size=(IMG_SIZE, IMG_SIZE))(
                ToTensor()(Image.open(filepath).convert("RGB"))
            )
            for filepath in image_filepaths
        ]
    ).to(DEVICE)

    

    ######################################################################
    # Loop for sequntially training
    ######################################################################
    # for layer_num in range(1, 13):
    for layer_num in range(1, 0, -1):      # 12->1

        # =============================================================

        img_dir = f'./playground_imgs/autoEnc_lr=5e-6_fullSeqClipSytleG_innerClip_layer_{layer_num}'
        loss_log_path = f"./playground_losses/autoEnc_lr=5e-6_fullSeqClipSytleG_innerClip_loss_layer_{layer_num}.txt"
        checkPath(img_dir)
        checkPath("./playground_losses/")

        run_name = f"innerClip_layer_{layer_num}"
        wandb_run = wandb.init(
            project='autoEnc_lr=5e-6_fullSeqClipSytleG_innerClip_diffLayers', 
            name=run_name,
            reinit=True
        ) if not wandb.run else wandb.run
        
        # ============================================================
        print("**"*50)
        print(f"Curreny number of CLIP inner feature layers used in losses = [{layer_num}]")
        print("**"*50)
        logger.info(f"Images saved at: {img_dir}")
        logger.info(f"Loss txt file saved at: {loss_log_path}")
        
        # =============================================================
        
        generator = StyleGenerator(
            image_size = IMG_SIZE, 
            network_capacity = 8,
            load_from=2225,
        )
        generator.clip_encoder.build((16,3,224,224))
        generator.to(DEVICE)

        optim = AdamW(
            lr=5e-6,
            params=generator.get_training_parameters(),
            weight_decay=0.0,
        )   

        
        clip_encoder = CLIPInnerEncoder(layer_num=layer_num).to(DEVICE)
        
        with open(loss_log_path,"a") as f:
            now = time.strftime("%c")
            f.write("==== Time: [%s] ===== aligned small dataset ==== StyleGenerator uses FULL sequential CLIP embedding at the beginning (not 512 tokens) ==============\n"% now)


        with tqdm(total=TOTAL_iter) as pbar:
            
            for idx_step in range(TOTAL_iter):
                pred = generator.forward(x_ref)

                pred_clip = clip_encoder(pred)
                real_clip = clip_encoder(x_ref)

                loss = F.l1_loss(pred_clip, real_clip)

                clip_encoder.zero_grad()
                generator.clip_encoder.zero_grad()
                generator.stylegan_S.zero_grad()
                generator.stylegan_G.zero_grad()
                optim.zero_grad()
                
                loss.backward()
                optim.step()

                loss_np=loss.detach().cpu().numpy()
                wandb.log({'Train loss':loss_np})
                pbar.set_postfix(train_loss=loss_np, lr=optim.param_groups[0]["lr"])
                pbar.update(1)

                if idx_step % 50 == 0 and idx_step != 0:
                    save_tensor = torch.concat([x_ref, pred], dim=0)
                    save_tensor = torchvision.utils.make_grid(
                        save_tensor, 
                        len(x_ref), 
                        normalize=True,
                        value_range=None,
                        scale_each=True,
                        pad_value=0
                    )
                    
                    image_numpy = tensor2im(save_tensor)
                    img_path = os.path.join(img_dir, 'autoEnc_lr=5e-6_fullSeqClipSytleG_innerClip_layer_%d_[%d_iters].png' % (layer_num, idx_step))
                    
                    wandb.log({"Per 50-epoch result":wandb.Image(image_numpy)})
                    save_image(image_numpy, img_path)
                
                with open(loss_log_path,"a") as f:
                    item = loss.detach().cpu().numpy()
                    f.write('%s\n' % item)
        wandb_run.finish()


    ######################################################################
    # Use clip's final feature output
    ######################################################################


    # =============================================================

    # img_dir = f'./playground_imgs/autoEnc_lr=5e-6_fullSeqClipSytleG_linearClip512'
    # checkPath(img_dir)
    # checkPath("./playground_losses/")
    # loss_log_path = f"./playground_losses/autoEnc_lr=5e-6_fullSeqClipSytleG_linearClip512_loss.txt"

    # logger.info(f"Images saved at: {img_dir}")
    # logger.info(f"Loss txt file saved at: {loss_log_path}")

    # run_name = f"linearClip_512token"
    # wandb_run = wandb.init(
    #         project='autoEnc_lr=5e-6_fullSeqClipSytleG_innerClip_diffLayers', 
    #         name=run_name,
    #         reinit=True
    #     ) if not wandb.run else wandb.run
    # # =============================================================
    
    # generator = StyleGenerator(
    #     image_size = IMG_SIZE, 
    #     network_capacity = 8,
    #     load_from = 2225,
    # )
    # generator.clip_encoder.build((16,3,224,224))
    # generator.to(DEVICE)
    
    # optim = AdamW(
    #     lr=5e-6,
    #     params=generator.get_training_parameters(),
    #     weight_decay=0.0,
    # )   

    
    # clip_encoder = CLIPWithLinearHead().to(DEVICE)

    # with open(loss_log_path,"a") as f:
    #     now = time.strftime("%c")
    #     f.write("==== Time: [%s] ===== aligned small dataset == 512 linear clip loss == StyleGenerator uses FULL sequential CLIP embedding at the beginning (512 tokens) ==============\n"% now)

    # with tqdm(total=TOTAL_iter) as pbar:
        
    #     for idx_step in range(TOTAL_iter):
    #         pred = generator.forward(x_ref)

    #         pred_clip = clip_encoder(pred)
    #         real_clip = clip_encoder(x_ref)

    #         loss = F.l1_loss(pred_clip, real_clip)

    #         clip_encoder.zero_grad()

    #         generator.clip_encoder.zero_grad()
    #         generator.stylegan_S.zero_grad()
    #         generator.stylegan_G.zero_grad()

    #         optim.zero_grad()
            
    #         loss.backward()
    #         optim.step()

    #         loss_np=loss.detach().cpu().numpy()
    #         wandb.log({'Train loss':loss_np})
    #         pbar.set_postfix(train_loss=loss_np, lr=optim.param_groups[0]["lr"])
    #         pbar.update(1)

    #         if idx_step % 50 == 0 and idx_step != 0:
    #             save_tensor = torch.concat([x_ref, pred], dim=0)
    #             save_tensor = torchvision.utils.make_grid(
    #                 save_tensor, 
    #                 len(x_ref), 
    #                 normalize=True,
    #                 value_range=None,
    #                 scale_each=True,
    #                 pad_value=0
    #             )
                
    #             image_numpy = tensor2im(save_tensor)
    #             img_path = os.path.join(img_dir, 'autoEnc_lr=5e-6_fullSeqClipSytleG_linearClip512_[%d_iters].png' % (idx_step))
    #             wandb.log({"Per 50-epoch result":wandb.Image(image_numpy)})
    #             save_image(image_numpy, img_path)
            
    #         with open(loss_log_path,"a") as f:
    #             item = loss.detach().cpu().numpy()
    #             f.write('%s\n' % item)
    # wandb_run.finish()