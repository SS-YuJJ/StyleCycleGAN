from tkinter import N
from PIL import Image
import os
import torchvision.transforms as transforms
import torchvision
from util.util import save_image
import numpy as np
import torch
from loguru import logger

IMG_EXTENSIONS = [
    ".jpg",
    ".JPG",
    ".jpeg",
    ".JPEG",
    ".png",
    ".PNG",
    ".ppm",
    ".PPM",
    ".bmp",
    ".BMP",
    ".tif",
    ".TIF",
    ".tiff",
    ".TIFF",
]

def checkPath(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print("Making new folder: ", path)

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

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


folder_dir = "./imgs/"

#################################################################
#                   Integrate images for each layer
#################################################################

for layer_num in range(1,13):
    img_folder_dir = os.path.join(folder_dir, f"innerCLIP_seqClipSytleG_lr=5e-6_layer_{layer_num}")
    logger.info(f"Now in image folder: {img_folder_dir}")

    image_paths = []
    num_list = list(np.arange(250, 3000,250))
    # print(num_list)

    for root, dir, fnames in sorted(os.walk(img_folder_dir)):
        # if len(dir)!=0:
        for fname in fnames:
            if is_image_file(fname):
                # print(fname)
                splits = fname.split('_')       # innerCLIP_loss_layer_1_[50_iters]
                iter_num = int(splits[4][1:])
                
                if iter_num % 250 == 0:
                    path = os.path.join(root, fname)
                    idx = num_list.index(iter_num)
                    image_paths.insert(idx,path)

    # image_paths = sorted(map(lambda x: int(x.split('/')[-1].split('_')[4][1:]), image_paths))
    
    target_folder = "./imgs/innerCLIP_seqClipSytleG_lr=5e-6[all]/"
    checkPath(target_folder)
    target_img_path = os.path.join(target_folder, f"innerCLIP_seqClipSytleG_lr=5e-6_layer_{layer_num}.jpg")
    trans = transforms.ToTensor()


    image_tensors = []

    image = Image.open(image_paths[0])
    image = trans(image)
    image = image[:, :int(image.shape[1]/2), :]
    image_tensors.append(image)

    for i in range(len(image_paths)):
        image = Image.open(image_paths[i])
        image = trans(image)
        image = image[:, int(image.shape[1]/2):, :]
        image_tensors.append(image)

    grids = torchvision.utils.make_grid(image_tensors, 1)
    image_numpy = tensor2im(grids)
    save_image(image_numpy, target_img_path)


#################################################################
#               Integrate last epoch images of all layers
#################################################################
# image_paths = []
# for layer_num in range(1,13):
#     img_folder_dir = os.path.join(folder_dir, f"innerCLIP_seqClipSytleG_lr=5e-6_layer_{layer_num}")
#     logger.info(f"Now in image folder: {img_folder_dir}")

#     # num_list = list(np.arange(250, 3000,250))       # 2750
#     # print(num_list)

#     for root, dir, fnames in sorted(os.walk(img_folder_dir)):
#         # if len(dir)!=0:
#         for fname in fnames:
#             if is_image_file(fname):
#                 # print(fname)
#                 splits = fname.split('_')       # innerCLIP_loss_layer_1_[50_iters]
#                 iter_num = int(splits[4][1:])
                
#                 if iter_num == 2750:
#                     path = os.path.join(root, fname)
#                     image_paths.append(path)
#                     print(path)

#     # image_paths = sorted(map(lambda x: int(x.split('/')[-1].split('_')[4][1:]), image_paths))
    
# target_folder = "./imgs/innerCLIP_seqClipSytleG_lr=5e-6[all]/"
# checkPath(target_folder)
# target_img_path = os.path.join(target_folder, f"innerCLIP_seqClipSytleG_lr=5e-6_LastEpoch_1-12Layer.jpg")
# trans = transforms.ToTensor()


# image_tensors = []
# # Get ground truth, draw in first row
# image = Image.open(image_paths[0])
# image = trans(image)
# image = image[:, :int(image.shape[1]/2), :]
# image_tensors.append(image)

# for i in range(len(image_paths)):
#     image = Image.open(image_paths[i])
#     image = trans(image)
#     image = image[:, int(image.shape[1]/2):, :]
#     image_tensors.append(image)

# grids = torchvision.utils.make_grid(image_tensors, 1)
# image_numpy = tensor2im(grids)
# save_image(image_numpy, target_img_path)

#################################################################