import json
import os

from PIL import Image
from rich.traceback import install
from loguru import logger
from tqdm import tqdm

import random

install(show_locals=False, extra_lines=1, word_wrap=True, width=350)

current_path = os.getcwd()

def checkPath(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print("Making new folder: ", path)

def getImgFromPath(img_names, target_folder_path, celebA_folder_path, ):
    checkPath(target_folder_path)

    for i in tqdm(range(len(img_names))):
        src_img_path = os.path.join(celebA_folder_path, img_names[i])
        img = Image.open(src_img_path).convert("RGB")
        target_img_path = os.path.join(target_folder_path, img_names[i])
        img.save(target_img_path)

    return 

if __name__ == "__main__":
    attr_file_name = 'list_attr_celeba.txt'
    attr_file_path = os.path.join(current_path, attr_file_name)
    
    ################################
    attr_name = 'Male'
    dataset_name = "male2female_align"        # domain a --> domain b
    ################################

    celebA_folder_path = os.path.join(current_path, "img_align_celeba")
    dataset_path = os.path.join(os.path.join(current_path, "datasets"), dataset_name)
    checkPath(dataset_path)
    logger.info(f"The path of dataset: {dataset_path}")

    train_a_folder_path = os.path.join(dataset_path, "trainA")
    train_b_folder_path = os.path.join(dataset_path, "trainB")
    test_a_folder_path = os.path.join(dataset_path, "testA")
    test_b_folder_path = os.path.join(dataset_path, "testB")

    test_ratio = 0.05

    with open(attr_file_path, 'r') as f:
        
        lines = f.readlines()   # First line: sample count
                                # Second line: attribute names
                                # File details start from line idx=2

        attributes = lines[1].split()       # position 0-39: attribute values
        
        attr_idx = attributes.index(attr_name)
        logger.info(f"The index of attribute [ {attr_name} ] = {attr_idx}")

        domain_a_imgnames = []
        domain_b_imgnames = []

        # file_counter = 0
        for idx in range(2, len(lines)):
            # print(lines[idx])
            line_splits = lines[idx].split()    # position 0: name of img file
                                                # position 1-40: attribute values
            # print(line_splits[attr_idx+1])
            this_attr = int(line_splits[attr_idx+1])
            
            if this_attr == 1:
                domain_a_imgnames.append(line_splits[0]) 
            else:
                domain_b_imgnames.append(line_splits[0])

            # file_counter+=1
            # if file_counter > 20 :
            #     break
        domain_a_num = len(domain_a_imgnames)
        domain_b_num = len(domain_b_imgnames)
        test_num_a = int(domain_a_num * test_ratio)
        test_num_b = int(domain_b_num * test_ratio)
        random.shuffle(domain_a_imgnames)
        random.shuffle(domain_b_imgnames)
        train_a_imgnames = domain_a_imgnames[:-test_num_a]
        test_a_imgnames = domain_a_imgnames[-test_num_a:]
        train_b_imgnames = domain_b_imgnames[:-test_num_b]
        test_b_imgnames = domain_b_imgnames[-test_num_b:]

        logger.info(f"Domain A has {domain_a_num} images ({len(train_a_imgnames)} train images; {len(test_a_imgnames)} test images)")
        logger.info(f"Domain A has {domain_b_num} images ({len(train_b_imgnames)} train images; {len(test_b_imgnames)} test images)")

        getImgFromPath(train_a_imgnames, train_a_folder_path, celebA_folder_path)
        getImgFromPath(test_a_imgnames, test_a_folder_path, celebA_folder_path)
        getImgFromPath(train_b_imgnames, train_b_folder_path, celebA_folder_path)
        getImgFromPath(test_b_imgnames, test_b_folder_path, celebA_folder_path)
        



        



