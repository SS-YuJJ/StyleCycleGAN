

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torch.optim import Adam
import os
import random
import numpy as np


from evaluation.classifier import ClassifierDatasetUTK, Classifier

from rich.traceback import install
import tqdm
from loguru import logger

install(show_locals=False, extra_lines=1, word_wrap=True, width=350)

########################
LR = 1e-4
BATCH_SIZE = 64
NUM_WORKERS = 1
TOTAL_EPOCH = 30
DEVICE="cuda:0"
SAVE_FREQ=5
########################

def save_model(the_model, save_path, move_back=True):
    torch.save(the_model.cpu().state_dict(), save_path)
    if move_back:
        the_model.cuda()

IMG_EXTENSIONS = [
    ".jpg"
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
    image_paths = []
    image_labels = []
    assert os.path.isdir(dir), "%s is not a valid directory" % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname) and fname[0]!='.':
                path = os.path.join(root, fname)
                image_paths.append(path)

                splits = fname.split('_')
                image_labels.append(int(splits[1]))

    return image_paths, image_labels



###############################################################
#                          Main
###############################################################

if __name__ == "__main__":

    logger.info(f"Learning rate = {LR}")
    logger.info(f"Batch size = {BATCH_SIZE}")
    logger.info(f"Total number of trained epochs = {TOTAL_EPOCH}")
    logger.info(f"Model saving frequence = {SAVE_FREQ} epochs")

    label_names = ["male", "female"]
    dataset_name = "UTKFace_male2female"

    root = os.getcwd()
    
    dataroot = "./datasets/utkface_male2female"             # total num = 24106
    logger.info(f"Dataset read from: {dataroot}")
    
    train_dir = os.path.join(dataroot, "train")
    val_dir = os.path.join(dataroot, "val")
    test_dir = os.path.join(dataroot, "test")

    train_imgpaths, train_imglabels = make_dataset(train_dir)
    val_imgpaths, val_imglabels = make_dataset(val_dir)
    test_imgpaths, test_imglabels = make_dataset(test_dir)

    
    train_dataset = ClassifierDatasetUTK(
        img_paths=train_imgpaths, 
        img_labels=train_imglabels
    )
    val_dataset = ClassifierDatasetUTK(
        img_paths=val_imgpaths, 
        img_labels=val_imglabels
    )
    test_dataset = ClassifierDatasetUTK(
        img_paths=test_imgpaths, 
        img_labels=test_imglabels
    )

    train_dataloader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
    )
    val_dataloader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
    )
    test_dataloader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
    )

    # get the number of images in the dataset.
    train_num = len(train_dataset)    
    val_num = len(val_dataset)
    test_num = len(test_dataset) 
    logger.info(f"Created training dataset of size {train_num}")
    logger.info(f"Created validation dataset of size {val_num}")
    logger.info(f"Created testing dataset of size {test_num}")

    model_save_dir = os.path.join(root, "evaluation")
    logger.info(f"Model save dir: {model_save_dir}")

    # # =====================================================
    model = Classifier(
        model_name='ViT-B/16', 
        num_hidden_layers=3,
        hidden_size=128,
        output_dim=1,
    ) 

    model.build((BATCH_SIZE, 3, 256, 256))
    model.to(DEVICE)    

    optim = Adam(
        params=model.get_trainable_parameters(),
        lr=LR,
    )

    criterion  = nn.BCEWithLogitsLoss()
    # # # =====================================================
    
    val_acc_list = []

    for epoch_idx in range(TOTAL_EPOCH):
        train_losses = []
        train_correct_count = 0
        
        with tqdm.tqdm(total=len(train_dataloader)) as pbar:
            for i, data in enumerate(train_dataloader):
                img = data['image'].to(DEVICE)
                label = data['label'].unsqueeze(1).to(DEVICE)

                pred = model.forward(img)
                loss = criterion(pred, label.float())
                loss_np = loss.detach().cpu().numpy()
                train_losses.append(loss_np)

                optim.zero_grad()
                loss.backward()
                optim.step()

                pred_np = pred.detach().cpu().numpy()
                label_np = label.detach().cpu().numpy().astype(np.float32)
                pred_label = np.where(pred_np<0.5, 0, 1)
                train_correct_count += np.equal(pred_label, label_np).astype(np.int16).sum()

                pbar.set_postfix(train_loss=loss_np)
                pbar.update(1)

        val_losses = []
        val_correct_count = 0
        # Finished one-epoch training, try validation
        with torch.no_grad():
            with tqdm.tqdm(total=len(val_dataloader)) as pbar:
                for val_i, val_data in enumerate(val_dataloader):
                    img = val_data['image'].to(DEVICE)
                    label = val_data['label'].unsqueeze(1).to(DEVICE)
                    
                    pred = model.forward(img)
                    val_loss = criterion(pred, label.float())
                    val_loss_numpy = val_loss.detach().cpu().numpy()
                    val_losses.append(val_loss_numpy)

                    label = label.detach().cpu().numpy().astype(np.float32)
                    pred = pred.detach().cpu().numpy()
                    pred_label = np.where(pred<0.5, 0, 1)
                    val_correct_count += np.equal(pred_label, label).astype(np.int16).sum()
                    
                    # paths = val_data['img_path']
                    # for j in range(paths.shape[0]):
                    #     print(f"path: {paths[j]}; pred_label:{pred_label[j]}")

                    pbar.set_postfix(val_loss=val_loss_numpy)
                    pbar.update(1)
            
        model_save_path = os.path.join(model_save_dir, "UTKFace_male2female_epoch_%d.pth" % epoch_idx)
        save_model(model, model_save_path, move_back=True)
        logger.info("Saved the model [Epoch %d]" % epoch_idx)
        
        train_loss_mean = np.array(train_losses).mean()
        val_loss_mean = np.array(val_losses).mean()
        train_acc = (train_correct_count / train_num) * 100
        val_acc = (val_correct_count / val_num) * 100
        val_acc_list.append(val_acc)

        print("**" * 50)
        print(f"[Epoch {epoch_idx}]")
        print("Train loss mean = {:.5f};\t Val loss mean = {:.5f}".format(train_loss_mean, val_loss_mean))
        print("Train accuracy = {:.3f}%;\t Val accuracy = {:.3f}%".format(train_acc, val_acc))
        print("**" * 50)

    print()
    print("=="* 50)
    # print(val_acc_list)
    val_acc_max = max(val_acc_list)
    val_acc_max_idx = val_acc_list.index(val_acc_max)
    print("The model with max validation loss: [Epoch {:d}]".format(val_acc_max_idx))
    print("=="* 50)
    
    best_model_path = os.path.join(model_save_dir, "UTKFace_male2female_epoch_%d.pth" % val_acc_max_idx)
    state_dict = torch.load(best_model_path)
    best_model = Classifier(
        model_name='ViT-B/16', 
        num_hidden_layers=3,
        hidden_size=128,
        output_dim=1,
    ) 

    best_model.build((BATCH_SIZE, 3, 224, 224))
    best_model.load_state_dict(state_dict)
    best_model.to(DEVICE)
    logger.info(f"Successfully loaded the classifier from path:[{best_model_path}]")


    # Test the model
    test_losses = []
    test_correct_count = 0
    with torch.no_grad():
        with tqdm.tqdm(total=len(test_dataloader)) as pbar:
            for test_i, test_data in enumerate(test_dataloader):
                img = test_data['image'].to(DEVICE)
                label = test_data['label'].unsqueeze(1).to(DEVICE)
                
                pred = best_model.forward(img)
                loss = criterion(pred, label.float())
                test_losses.append(loss)

                label = label.detach().cpu().numpy().astype(np.float32)
                pred = pred.detach().cpu().numpy()
                pred_label = np.where(pred<0.5, 0, 1)
                test_correct_count += np.equal(pred_label, label).astype(np.int16).sum()

                pbar.set_postfix(test_loss=loss)
                pbar.update(1)
    
    test_loss_mean = torch.mean(torch.tensor(test_losses))
    test_acc = (test_correct_count / test_num) * 100
    print("==" * 50)
    print("Test loss mean = {:.5f}".format(test_loss_mean))
    print("Test accuracy = {:.3f}%".format(test_acc))
    print("==" * 50)

