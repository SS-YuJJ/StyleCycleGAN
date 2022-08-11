import clip
from torchvision.transforms.functional import normalize
from torch.utils.data import Dataset
from typing import Tuple
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tr
from loguru import logger
import numpy as np

class Classifier(nn.Module):
    def __init__(
        self, 
        model_name, 
        num_hidden_layers=3,
        hidden_size=128,
        output_dim=1,
        ):
        """
        @ model_name: the CLIP model names: 'RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64',
                                            'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
        """
        super().__init__()
        self.model_name = model_name
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.output_dim = output_dim
        self.is_built = False

        self.channel_mean = [0.48145466, 0.4578275, 0.40821073]
        self.channel_std = [0.26862954, 0.26130258, 0.27577711]

    def preprocess_img(self, x):
        if x.shape[2] != 224:
            x = F.interpolate(x, size=(224, 224), mode='bilinear')
        return normalize(x, mean=self.channel_mean, std=self.channel_std)
    
    def get_clip_encoding(self, x):
        return self.clip_model_visual(x)

    def get_trainable_parameters(self):
        return self.model.parameters()

    def build(self, input_shape: Tuple[int, int, int, int]):
        clip_model, _ = clip.load(self.model_name, device="cpu")
        self.clip_model_visual = clip_model.visual
        logger.info(f"CLIP visualizer loaded.")

        x_dummy = torch.zeros(input_shape)
        out = self.preprocess_img(x_dummy)      # [b, 3, 224, 224]
        logger.info(f"Shape of out after normalization: {out.shape}")

        out = self.get_clip_encoding(out)       # [b, 512]
        logger.info(f"Shape of out after clip's encoding: {out.shape}")

        self.model = nn.ModuleDict()

        for i in range(self.num_hidden_layers):
            self.model[f"hidden_linear_{i}"] = nn.Linear(
                out.shape[1],
                self.hidden_size,
                bias=True,
            )

            out = self.model[f"hidden_linear_{i}"](out)
            out = F.leaky_relu(out)

        logger.info(f"Shape of out after hidden layers: {out.shape}")

        self.model[f"final_linear"] = nn.Linear(
            out.shape[1],
            self.output_dim,
            bias=True,
        )
        out = self.model[f"final_linear"](out)
        out = nn.Sigmoid()(out)

        self.is_built = True
        logger.info(f"Shape of input: {x_dummy.shape}",
                    f"Shape of output: {out.shape}")

    def forward(self, x):
        if self.is_built == False:
            self.build(x.shape)

        out = self.preprocess_img(x)            # [b, 3, 224, 224]
        out = self.get_clip_encoding(out)       # [b, 512]

        for i in range(self.num_hidden_layers):
            out = self.model[f"hidden_linear_{i}"](out)
            out = F.leaky_relu(out)

        out = self.model[f"final_linear"](out)
        out = nn.Sigmoid()(out)

        return out


class ClassifierDataset(Dataset):
    def __init__(
        self, 
        label_names:list, 
        domain_a_imgpaths, 
        domain_b_imgpaths
    ):
        
        self.label_names = label_names       # "0" stands for domain a
                                                # "1" stands for domain b

        domain_a_imgpaths = domain_a_imgpaths
        domain_b_imgpaths = domain_b_imgpaths
        domain_a_num = len(domain_a_imgpaths)
        domain_b_num = len(domain_b_imgpaths)
        self.dataset_size = domain_a_num + domain_b_num

        self.labels = np.hstack((np.zeros(domain_a_num), np.ones(domain_b_num)))
        self.imgpaths = domain_a_imgpaths + domain_b_imgpaths

        self.transforms = tr.Compose(
            [
                tr.Resize(size=224, max_size=None, antialias=None),
                tr.CenterCrop(size=(224, 224)),
                tr.ToTensor(),
            ]
        )

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
        
        img_path = self.imgpaths[index % self.dataset_size]
        label = int(self.labels[index % self.dataset_size])
        text_label = self.label_names[int(label)]

        img = Image.open(img_path).convert("RGB")
        img = self.transforms(img)

        return {'image': img,
                'img_path': img_path,
                'label': label,
                'text_label': text_label}

class ClassifierDatasetUTK(Dataset):
    def __init__(
        self, 
        img_paths, 
        img_labels
    ):

        self.dataset_size = len(img_paths)

        self.labels = img_labels
        self.imgpaths = img_paths

        self.transforms = tr.Compose(
            [
                tr.Resize(size=224, max_size=None, antialias=None),
                tr.CenterCrop(size=(224, 224)),
                tr.ToTensor(),
            ]
        )

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
        
        img_path = self.imgpaths[index % self.dataset_size]
        label = int(self.labels[index % self.dataset_size])

        img = Image.open(img_path).convert("RGB")
        img = self.transforms(img)

        return {'image': img,
                'img_path': img_path,
                'label': label,
                }