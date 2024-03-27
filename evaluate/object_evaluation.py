import argparse

import os
import sys

import torch
import torch.nn.functional as F
import torchvision.transforms as T

import yaml
from torchvision.transforms.functional import InterpolationMode

from data.movi_data import MOViDataModule
from model.align_model import AlignSegmentor
from eval_utils import ARIMetric, AverageBestOverlapMetric


def norm(t):
    return F.normalize(t, dim=-1, eps=1e-10)


def eval_objectmasks():
    with open(args.config_path) as file:
        config = yaml.safe_load(file.read())
    # print('Config: ', config)

    data_config = config['data']
    val_config = config['val']
    input_size = data_config["size_crops"]
    torch.manual_seed(val_config['seed'])

    # Init data and transforms
    val_image_transforms = T.Compose([T.Resize((input_size, input_size)),
                                      T.ToTensor(),
                                      T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    val_target_transforms = T.Compose([T.Resize((input_size, input_size), interpolation=InterpolationMode.NEAREST),
                                       T.ToTensor()])

    data_dir = data_config["data_dir"]
    dataset_name = data_config["dataset_name"]
    if "movi" in dataset_name:
        ignore_index = 0
        num_classes = 17
        data_module = MOViDataModule(data_dir=data_dir,
                                     dataset_name=data_config['dataset_name'],
                                     batch_size=val_config["batch_size"],
                                     return_masks=True,
                                     drop_last=True,
                                     num_workers=config["num_workers"],
                                     train_split="frames",
                                     val_split="images",
                                     train_image_transform=None,
                                     val_image_transform=val_image_transforms,
                                     val_target_transform=val_target_transforms)
    else:
        raise ValueError(f"{dataset_name} not supported")

    # Init method
    patch_size = val_config["patch_size"]
    spatial_res = input_size / patch_size
    num_proto = val_config['num_queries']
    assert spatial_res.is_integer()
    model = AlignSegmentor(arch=val_config['arch'],
                           patch_size=val_config['patch_size'],
                           embed_dim=val_config['embed_dim'],
                           hidden_dim=val_config['hidden_dim'],
                           num_heads=val_config['decoder_num_heads'],
                           num_queries=val_config['num_queries'],
                           num_decode_layers=val_config['num_decode_layers'],
                           last_self_attention=val_config['last_self_attention'])

    # set model to eval mode
    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    # load pretrained weights
    if val_config["checkpoint"] is not None:
        checkpoint = torch.load(val_config["checkpoint"])
        msg = model.load_state_dict(checkpoint["state_dict"], strict=True)
        print(msg)
    else:
        print('no pretrained pth found!')

    dataloader = data_module.val_dataloader()
    ARI_metric = ARIMetric()
    BO_metric = AverageBestOverlapMetric()

    # Calculate IoU for each image individually
    for idx, batch in enumerate(dataloader):
        imgs, masks = batch
        B = imgs.size(0)
        # assert B == 1  # image has to be evaluated individually
        all_queries, tokens, _, _, res, _ = model([imgs])  # tokens=(1,N,dim)

        # calculate token assignment
        token_cls = torch.einsum("bnc,bqc->bnq", norm(tokens), norm(all_queries[0]))
        token_cls = torch.softmax(token_cls, dim=-1)
        token_cls = token_cls.reshape(B, res, res, -1).permute(0, 3, 1, 2)  # (1,num_query,res,res)

        # downsample masks / upsample preds to masks_eval_size
        preds = F.interpolate(token_cls, size=(val_config['mask_eval_size'], val_config['mask_eval_size']),
                              mode='bilinear')
        masks *= 255
        if masks.size(3) != val_config['mask_eval_size']:
            masks = F.interpolate(masks, size=(val_config['mask_eval_size'], val_config['mask_eval_size']),
                                  mode='nearest')

        # turn masks to one-hot
        masks = masks.squeeze(dim=1).reshape(B, -1)
        masks = masks.long()
        num_classes = masks.max().item() + 1
        masks = torch.nn.functional.one_hot(masks, num_classes)
        masks = masks.permute(0, 2, 1).reshape(B, num_classes,
                                               val_config['mask_eval_size'],
                                               val_config['mask_eval_size'])    # to (B, K, H, W)

        ARI_metric.update(preds, masks)
        BO_metric.update(preds, masks)
        # sys.exit(1)

    ARI_metric.compute()
    BO_metric.compute()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default='../configs/eval_movi_config.yml', type=str)

    args = parser.parse_args()

    eval_objectmasks()
