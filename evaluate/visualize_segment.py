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


def apply_mask(image, mask, color, alpha=0.5):
    for c in range(3):
        image[:, :, c] = image[:, :, c] * (1 - alpha * mask) + alpha * mask * color[c] * 255
    return image


def random_colors(N, bright=True):
    """
    Generate random colors.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def display_instances(image, mask, fname="test", figsize=(5, 5), blur=False, contour=True, alpha=0.5):
    fig = plt.figure(figsize=figsize, frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax = plt.gca()

    N = 1
    mask = mask[None, :, :]
    # Generate random colors
    colors = random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    margin = 0
    ax.set_ylim(height + margin, -margin)
    ax.set_xlim(-margin, width + margin)
    ax.axis('off')
    masked_image = image.astype(np.uint32).copy()

    for i in range(N):
        color = colors[i]
        _mask = mask[i]
        if blur:
            _mask = cv2.blur(_mask, (10, 10))
        # Mask
        masked_image = apply_mask(masked_image, _mask, color, alpha)
        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        if contour:
            padded_mask = np.zeros((_mask.shape[0] + 2, _mask.shape[1] + 2))
            padded_mask[1:-1, 1:-1] = _mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                p = Polygon(verts, facecolor="none", edgecolor=color)
                ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8), aspect='auto')
    fig.savefig(fname)
    print(f"{fname} saved.")


def display_segments(image, masks, fname="test", figsize=(5, 5), alpha=0.7):
    N = 5
    # Generate random colors
    # colors = random_colors(N)
    colors = [(128, 0, 0), (30, 144, 255), (75, 0, 130), (184, 134, 11), (0, 128, 0)]
    colors = [(x / 255, y / 255, z / 255) for (x, y, z) in colors]

    fig = plt.figure(figsize=figsize, frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax = plt.gca()

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    margin = 0
    ax.set_ylim(height + margin, -margin)
    ax.set_xlim(-margin, width + margin)
    ax.axis('off')

    for i in range(N):
        color = colors[i]
        _mask = masks[i]
        # Mask
        masked_image = image.astype(np.uint32).copy()
        masked_image = apply_mask(masked_image, _mask, color, alpha)

        ax.imshow(masked_image.astype(np.uint8), aspect='auto')
        file = os.path.join(fname, "cls" + str(i) + ".png")
        fig.savefig(file)
        print(f"{file} saved.")


def display_allsegments(image, masks, n=5, fname="test", figsize=(5, 5), alpha=0.6):
    N = n   # num of colors
    # colors = [(128,0,0), (184,134,11), (0,128,0), (62,78,94), (0,0,0)]  # last two backgrounds
    colors = [(128,0,0), (30,144,255), (75,0,130), (184,134,11), (0,128,0)]  # for coco
    colors = [(x/255, y/255, z/255) for (x, y, z) in colors]
    print(colors)

    # Generate random colors
    # colors = random_colors(N)

    fig = plt.figure(figsize=figsize, frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax = plt.gca()

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    margin = 0
    ax.set_ylim(height + margin, -margin)
    ax.set_xlim(-margin, width + margin)
    ax.axis('off')

    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]
        _mask = masks[i]
        # Mask
        masked_image = apply_mask(masked_image, _mask, color, alpha)

        ax.imshow(masked_image.astype(np.uint8), aspect='auto')
        file = os.path.join(fname, "cls" + str(i) + ".png")
        fig.savefig(file)
        print(f"{file} saved.")


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
    parser.add_argument("--image_path", default='', type=str, help="Path of the image to load.")
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument("--image_size", default=(480, 480), type=int, nargs="+", help="Resize image.")
    parser.add_argument('--output_dir', default='./outputs/', help='Path where to save visualizations.')
    parser.add_argument("--threshold", type=float, default=0.6, help="""We visualize masks
        obtained by thresholding the self-attention maps to keep xx% of the mass.""")
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # build model
    # '''
    model = AlignSegmentor(arch='vit_small',
                           patch_size=16,
                           embed_dim=384,
                           hidden_dim=768,
                           num_heads=3,
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

    if os.path.isfile(args.image_path):
        with open(args.image_path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
    else:
        print(f"Provided image path {args.image_path} is non valid.")
        sys.exit(1)
    transform = pth_transforms.Compose([
        pth_transforms.Resize(args.image_size),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    img = transform(img)

    # make the image divisible by the patch size
    # img = c, w, h (3, 480, 480); unsqueeze -> (1, 3, 480, 480)
    w, h = img.shape[1] - img.shape[1] % args.patch_size, img.shape[2] - img.shape[2] % args.patch_size
    img = img[:, :w, :h].unsqueeze(0)

    w_spatial_res = img.shape[-2] // args.patch_size

    # get aligned_queries, spatial_token_output and attention_map
    all_queries, gc_output, _, attn_hard, _, _ = model([img.to(device)], threshold=args.threshold)

    os.makedirs(args.output_dir, exist_ok=True)
    torchvision.utils.save_image(torchvision.utils.make_grid(img, normalize=True, scale_each=True),
                                 os.path.join(args.output_dir, "img.png"))

    # interpolate binary mask
    attn_hard = nn.functional.interpolate(attn_hard.unsqueeze(1), scale_factor=args.patch_size, mode="nearest")[0].cpu().numpy()
    image = skimage.io.imread(os.path.join(args.output_dir, "img.png"))
    display_instances(image, attn_hard[0], fname=os.path.join(args.output_dir, "mask_" + str(args.threshold) + ".png"), blur=False)

    # calculate query assignment score
    gc_token_sim = torch.einsum("bnc,bqc->bnq", norm(gc_output), norm(all_queries[0]))
    gc_token_cls = torch.softmax(gc_token_sim, dim=-1)
    gc_token_cls = gc_token_cls.reshape(1, w_spatial_res, w_spatial_res, -1).permute(0, 3, 1, 2)

    # Smooth interpolation
    masks_prob = F.interpolate(gc_token_cls, size=w, mode='bilinear')
    masks_oh = masks_prob.argmax(dim=1)
    masks_oh = torch.nn.functional.one_hot(masks_oh, masks_prob.shape[1])
    masks_oh = masks_oh.squeeze(dim=0).permute(2, 0, 1)

    masks = []
    for i in range(masks_prob.shape[1]):
        mask = masks_oh[i].cpu().numpy()
        # print('mask = ', mask.shape, mask)
        masks.append(mask)
    display_allsegments(image, masks, n=5, fname=args.output_dir)
