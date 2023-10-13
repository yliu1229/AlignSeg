import os
import sys
import argparse
import cv2
import random
import colorsys

import skimage.io
from skimage.measure import find_contours
from matplotlib.patches import Polygon
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as pth_transforms
from torchvision.transforms import GaussianBlur
import torch.nn.functional as F
import numpy as np
from PIL import Image
from skimage.measure import label
from matplotlib import pyplot as plt

from model.align_model import AlignSegmentor
from utils import neq_load_external


def norm(t):
    return F.normalize(t, dim=-1, eps=1e-10)


def process_attentions(attentions: torch.Tensor, spatial_res: int, threshold: float = 0.6, blur_sigma: float = 0.6) \
        -> torch.Tensor:
    """
    Process [0,1] attentions to binary 0-1 mask. Applies a Guassian filter, keeps threshold % of mass and removes
    components smaller than 3 pixels.
    The code is adapted from https://github.com/facebookresearch/dino/blob/main/visualize_attention.py but removes the
    need for using ground-truth data to find the best performing head. Instead we simply average all head's attentions
    so that we can use the foreground mask during training time.
    :param attentions: torch 4D-Tensor containing the averaged attentions
    :param spatial_res: spatial resolution of the attention map
    :param threshold: the percentage of mass to keep as foreground.
    :param blur_sigma: standard deviation to be used for creating kernel to perform blurring.
    :return: the foreground mask obtained from the ViT's attention.
    """
    # Blur attentions
    attentions = GaussianBlur(7, sigma=(blur_sigma))(attentions)
    attentions = attentions.reshape(attentions.size(0), 1, spatial_res ** 2)
    # Keep threshold% of mass
    val, idx = torch.sort(attentions)
    val /= torch.sum(val, dim=-1, keepdim=True)
    cumval = torch.cumsum(val, dim=-1)
    th_attn = cumval > (1 - threshold)
    idx2 = torch.argsort(idx)
    th_attn[:, 0] = torch.gather(th_attn[:, 0], dim=1, index=idx2[:, 0])
    th_attn = th_attn.reshape(attentions.size(0), 1, spatial_res, spatial_res).float()
    # Remove components with less than 3 pixels
    for j, th_att in enumerate(th_attn):
        labelled = label(th_att.cpu().numpy())
        for k in range(1, np.max(labelled) + 1):
            mask = labelled == k
            if np.sum(mask) <= 2:
                th_attn[j, 0][mask] = 0
    return th_attn


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluate segmentation on pretrained model')
    parser.add_argument('--pretrained_weights', default='./epoch10.pth',
                        type=str, help="Path to pretrained weights to load.")
    parser.add_argument("--image_path", default='',
                        type=str, help="Path of the image to load.")
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument("--image_size", default=(480, 480), type=int, nargs="+", help="Resize image.")
    parser.add_argument('--output_dir', default='./outputs/', help='Path where to save visualizations.')
    parser.add_argument("--threshold", type=float, default=0.6, help="""We visualize masks
        obtained by thresholding the self-attention maps to keep xx% of the mass.""")
    args = parser.parse_args()

    device = torch.device("cpu")
    # build model
    model = AlignSegmentor(arch='vit_small',
                           patch_size=16,
                           embed_dim=384,
                           hidden_dim=384,
                           num_heads=2,
                           num_queries=5,
                           nmb_crops=[1, 0],
                           num_decode_layers=1,
                           last_self_attention=True)

    # set model to eval mode
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.to(device)

    # load pretrained weights
    if os.path.isfile(args.pretrained_weights):
        pratrained_model = torch.load(args.pretrained_weights, map_location="cpu")
        msg = model.load_state_dict(pratrained_model['state_dict'], strict=False)
        print(msg)
    else:
        print('no pretrained pth found!')

    queries = model.clsQueries.weight

    prototypes = torch.load('../log_tmp/prototypes21.pth')
    prototypes = prototypes.to(device)
    # calculate query assignment score
    sim_query_proto = norm(queries) @ norm(prototypes).T
    sim_query_proto = sim_query_proto.clamp(min=0.0)

    for i in range(sim_query_proto.size(0)):
        print('Proto', i, '=', sim_query_proto[i]*10)
