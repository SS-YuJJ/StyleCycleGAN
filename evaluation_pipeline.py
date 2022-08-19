from loguru import logger
from rich.traceback import install

import os
import argparse
import torch
from torch import nn
import time
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data
from PIL import Image

from torchvision.models.inception import inception_v3
from data.image_folder import is_image_file

import numpy as np
from scipy.stats import entropy
import tqdm
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms
from evaluation.classifier import ClassifierDatasetUTK, Classifier

install(show_locals=False, extra_lines=1, word_wrap=True, width=350)

###############################################################
#                       Inception Score
###############################################################
def inception_score(
        imgs, 
        cuda=True, 
        batch_size=32, 
        resize=False, 
        splits=1
    ):
    """Computes the inception score of the generated images imgs
    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(imgs)

    assert batch_size > 0, "Batch size should be positive!"
    assert N > batch_size, "The size of dataset should be larger than batch size"

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = DataLoader(imgs, batch_size=batch_size)

    # Load the InceptionV3 model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval()

    # Unsample to fit input shape of InceptionV3 model
    # up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)

    def get_pred(x):
        # if resize:
        #     x = up(x)
        x = inception_model(x)

        return F.softmax(x, dim=1).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader, 0):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)


###############################################################
#                       Build Dataset
###############################################################

class ISEvaluateDataset(Dataset):
    def __init__(
        self, 
        imgpaths
    ):
        
        self.dataset_size = len(imgpaths)
        self.imgpaths = imgpaths

        self.transforms = transforms.Compose(
            [
                transforms.Resize((299, 299)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
        
        img_path = self.imgpaths[index % self.dataset_size]
        img = Image.open(img_path).convert("RGB")
        img = self.transforms(img)

        return img

def make_eval_dataset(dir):
    image_paths = []
    image_labels = []
    assert os.path.isdir(dir), "%s is not a valid directory" % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            
            if is_image_file(fname) and "fake" in fname:
                this_path = os.path.join(root, fname)
                image_paths.append(this_path)

                splits = fname.split('_')       # 000723_fake_A.png
                if splits[2] == 'A':
                    image_labels.append(0)      # male
                else:
                    image_labels.append(1)      # female

    return image_paths, image_labels


###############################################################
#                          Functions
###############################################################

def checkPath(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print("Making new folder: ", path)

def inception_score_evaluate(
    is_dataset,
):
    print("=" * 40)    
    print ("Calculating Inception Score...")
    
    IS_mean, IS_std = inception_score(
        is_dataset, 
        cuda=True, 
        batch_size=32, 
        resize=True, 
        splits=10
    )
    
    print("**" * 50)
    print("The mean of inception scores = {:.5f}".format(IS_mean))
    print("The standard deviation of inception scores = {:.5f}".format(IS_std))
    print("**" * 50)
    
    return IS_mean, IS_std


###############################################################
#                          Main
###############################################################


parser = argparse.ArgumentParser(description='Evaluation pipeline parser.')

parser.add_argument("--model_name", type=str, help="Name of the model.")
parser.add_argument("--load_iter", type=int, help="The number of trained iters.")

args = parser.parse_args()

########### classification #############
BATCH_SIZE = 16
NUM_WORKERS = 1
TOTAL_EPOCH = 1
DEVICE="cuda:0"
MODEL_NAME = args.model_name
LOAD_ITER = args.load_iter
########################################

if __name__ == "__main__":

    root = os.getcwd()
    label_names = ["male", "female"]
    classifier_file_name = "UTKFace_male2female_epoch_26.pth"           # Classifier index 26 has best val accuracy
    classifier_path = os.path.join(os.path.join(root, "evaluation"), classifier_file_name)
    result_folder = os.path.join(os.path.join(root, "results"), MODEL_NAME, f"test_latest_iter{LOAD_ITER}")
    evaluation_result_folder= "./evaluation/pipeline_results"
    checkPath(evaluation_result_folder)
    evaluation_result_path = os.path.join(evaluation_result_folder, f"{MODEL_NAME}_eval_results.txt")

    logger.info(f"Evaluating model: {MODEL_NAME}")
    logger.info(f"Model results taken from path: {result_folder}")

    img_paths, img_labels = make_eval_dataset(dir=result_folder)        # 0 for male, 1 for female

    # ============= Evaluate Inception Score =============
    IS_eval_dataset = ISEvaluateDataset(img_paths)
    logger.info(f"IS evaluation on [{len(IS_eval_dataset)}] images.")

    IS_mean, IS_std = inception_score_evaluate(IS_eval_dataset)

    # ================== Classification ===================
    classify_dataset = ClassifierDatasetUTK(
        img_paths=img_paths, 
        img_labels=img_labels
    )
    classify_dataloader = DataLoader(
            classify_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
    )
    logger.info(f"Classification on [{len(classify_dataset)}] images.")

    # Load classifier
    state_dict = torch.load(classifier_path)
    classifier = Classifier(
        model_name='ViT-B/16', 
        num_hidden_layers=3,
        hidden_size=128,
        output_dim=1,
    ) 

    classifier.build((BATCH_SIZE, 3, 224, 224))
    classifier.load_state_dict(state_dict)
    classifier.to(DEVICE)
    logger.info(f"Successfully loaded the classifier from path:[{classifier_path}]")

    # Test the model

    test_correct_count = 0
    probs_A = []
    probs_B = []
    with torch.no_grad():
        with tqdm.tqdm(total=len(classify_dataloader)) as pbar:
            for i, classi_data in enumerate(classify_dataloader):
                img = classi_data['image'].to(DEVICE)
                label = classi_data['label'].unsqueeze(1).to(DEVICE)
                
                pred = classifier.forward(img)

                label = label.detach().cpu().numpy().astype(np.float32)
                pred = pred.detach().cpu().numpy()
                
                pred_label = np.where(pred<0.5, 0, 1)
                test_correct_count += np.equal(pred_label, label).astype(np.int16).sum()
                for i in range(pred.shape[0]):
                    if pred_label[i] == 0: # 0 for male
                        probs_A.append(pred[i])
                    else:
                        probs_B.append(pred[i])
                pbar.update(1)
    

    test_acc = (test_correct_count / len(classify_dataset)) * 100
    print("**" * 50)
    print("Classification accuracy = {:.3f}%".format(test_acc))
    print("**" * 50)
    print("25 Male probabilities:\n", probs_A[:25])
    print("**" * 50)
    print("25 Female probabilities:\n", probs_B[:25])
    print("**" * 50)
    prob_A_mean = np.array(probs_A).mean()
    prob_A_mean = np.array(probs_B).mean()
    print(f"Male prob mean = {prob_A_mean}")
    print(f"Female prob mean = {prob_B_mean}")


    with open(evaluation_result_path, 'a') as f:
        now = time.strftime("%c")
        f.write("===================== Time: [%s] ========================\n"% now)
        f.write(f"Evaluated model: {MODEL_NAME}\n")
        f.write(f"Model results taken from path: {result_folder}\n")
        f.write(f"IS evaluation on [{len(IS_eval_dataset)}] images.\n")
        f.write(f"Classification on [{len(classify_dataset)}] images.\n")
        f.write("The mean of inception scores = {:.5f}\n".format(IS_mean))
        f.write("The standard deviation of inception scores = {:.5f}\n".format(IS_std))
        f.write("Classification accuracy = {:.3f}%".format(test_acc))

