import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torch.optim import Adam
import os
import random
import numpy as np
import cv2
from PIL import Image

from evaluation.classifier import ClassifierDatasetUTK, Classifier

from rich.traceback import install
from tqdm import tqdm
from loguru import logger

IMG_EXTENSIONS = [
    ".jpg"
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def get_pathNname(dir):
    image_paths = []       # male

    assert os.path.isdir(dir), "%s is not a valid directory" % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname) and fname[0]!='.':
                path = os.path.join(root, fname)
                image_paths.append(path)


    return image_paths


def checkPath(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print("Making new folder: ", path)

def getImgFromPath(img_paths, target_folder_path):
    checkPath(target_folder_path)

    for i in tqdm(range(len(img_paths))):
        splits = img_paths[i].split("\\")
        file_name = splits[-1]
        
        img = Image.open(img_paths[i]).convert("RGB")
        target_img_path = os.path.join(target_folder_path, file_name)
        img.save(target_img_path)


    return 



if __name__ == "__main__":

    label_names = ["male", "female"]
    dataset_name = "UTKFace_male2female"

    root = os.getcwd()
    
    dataroot = '../../UTKFace'                             # total num = 24106
    logger.info(f"Dataset read from: {dataroot}")
    
    img_paths = get_pathNname(dataroot)
    random.shuffle(img_paths)

    dataset_size = len(img_paths)
    train_num = int(dataset_size * 0.7)
    val_num = int(dataset_size * 0.1)

    train_imgpaths = img_paths[:train_num]
    val_imgpaths = img_paths[train_num:train_num+val_num]
    test_imgpaths = img_paths[train_num+val_num:]

    target_root = "../datasets/utkface_male2female"
    train_dir = os.path.join(target_root, "train")
    val_dir = os.path.join(target_root, "val")
    test_dir = os.path.join(target_root, "test")

    getImgFromPath(train_imgpaths, train_dir)
    getImgFromPath(val_imgpaths, val_dir)
    getImgFromPath(test_imgpaths, test_dir)