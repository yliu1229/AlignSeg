import json
import os
import torch
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode


label_to_color = {
    0: (245, 245, 220),     # accessory
    1: (0, 100, 0),         # animal
    2: (178, 34, 34),       # appliance
    3: (0, 0, 139),         # building
    4: (148, 0, 211),       # ceiling
    5: (105, 105, 105),     # electronic
    6: (205, 92, 92),       # floor
    7: (244, 164, 96),      # food
    8: (245, 222, 179),     # food-stuff
    9: (75, 0, 130),        # furniture
    10: (138, 43, 226),     # furniture-stuff
    11: (72, 61, 139),      # ground
    12: (25, 25, 112),      # indoor
    13: (253, 245, 230),    # kitchen
    14: (47, 79, 79),       # outdoor
    15: (139, 0, 0),        # person
    16: (124, 252, 0),      # plant
    17: (210, 180, 140),    # raw-material
    18: (135, 206, 235),    # sky
    19: (85, 107, 47),      # solid
    20: (255, 105, 180),    # sports
    21: (210, 105, 30),     # structural
    22: (211, 211, 211),    # textile
    23: (184, 134, 11),     # vehicle
    24: (128, 128, 128),    # wall
    25: (32, 178, 170),     # water
    26: (189, 183, 107),    # window
    27: (255, 250, 250),    # other
}

super_cat_to_id = {
    'accessory': 0, 'animal': 1, 'appliance': 2, 'building': 3,
    'ceiling': 4, 'electronic': 5, 'floor': 6, 'food': 7,
    'food-stuff': 8, 'furniture': 9, 'furniture-stuff': 10, 'ground': 11,
    'indoor': 12, 'kitchen': 13, 'outdoor': 14, 'person': 15,
    'plant': 16, 'raw-material': 17, 'sky': 18, 'solid': 19,
    'sports': 20, 'structural': 21, 'textile': 22, 'vehicle': 23,
    'wall': 24, 'water': 25, 'window': 26,
    'other': 27
}


def visual_mask(img_path, transforms, cat_id_map, RGB=False):

    mask = Image.open(img_path)
    mask = transforms(mask)
    save_path = img_path.replace('.png', '_RGB.jpg')

    # move 'id' labels from [0, 182] to [0,27] with 27=={182,255}
    # (182 is 'other' and 0 is things)
    mask *= 255
    assert torch.min(mask).item() >= 0
    mask[mask == 255] = 182
    assert torch.max(mask).item() <= 182
    for cat_id in torch.unique(mask):
        mask[mask == cat_id] = cat_id_map[cat_id.item()]

    assert torch.max(mask).item() <= 27
    assert torch.min(mask).item() >= 0

    mask = mask.squeeze(0).numpy().astype(int)
    mask = mask.astype(np.uint8)

    if not RGB:
        img = Image.fromarray(mask, 'L')
        plt.figure(figsize=(8, 8))
        plt.axis('off')
        plt.imshow(img)
        # plt.savefig('mask.png', bbox_inches='tight', pad_inches=0.0)
        plt.tight_layout(pad=0.0, h_pad=0.0, w_pad=0.0)
        plt.show()
    else:
        # visualize by configuring palette
        img = Image.fromarray(mask, 'L')
        img_p = img.convert('P')
        img_p.putpalette([rgb for pixel in label_to_color.values() for rgb in pixel])

        img_rgb = img_p.convert('RGB')
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1), plt.imshow(img)
        plt.subplot(1, 3, 2), plt.imshow(img_p)
        plt.subplot(1, 3, 3), plt.imshow(img_rgb)
        plt.tight_layout(), plt.show()
        img_rgb.save(save_path)


if __name__ == '__main__':

    root = './COCO/annotations/stuffthingmaps_trainval2017/'
    json_file = "stuffthing_2017.json"
    mask_name = 'val2017/000000512194.png'

    mask_transforms = T.Compose([T.Resize((448, 448), interpolation=InterpolationMode.NEAREST), T.ToTensor()])

    with open(os.path.join(root, json_file)) as f:
        an_json = json.load(f)
        all_cat = an_json['categories']

        super_cats = set([cat_dict['supercategory'] for cat_dict in all_cat])
        super_cats.remove("other")  # remove others from prediction targets as this is not semantic
        super_cat_to_id = {super_cat: i for i, super_cat in enumerate(sorted(super_cats))}
        super_cat_to_id["other"] = 27  # ignore_index
        # Align 'id' labels: PNG_label = GT_label - 1
        cat_id_map = {(cat_dict['id'] - 1): super_cat_to_id[cat_dict['supercategory']] for cat_dict in all_cat}

        visual_mask(os.path.join(root, mask_name), mask_transforms, cat_id_map, RGB=False)
