import argparse

import os
import torch
import torch.nn.functional as F
import torchvision.transforms as T

import yaml
from torchvision.transforms.functional import InterpolationMode

from data.coco_data import CocoDataModule
from data.voc_data import VOCDataModule
from model.align_model import AlignSegmentor
from utils import PredsmIoU


def norm(t):
    return F.normalize(t, dim=-1, eps=1e-10)


def eval_overcluster():
    with open(args.config_path) as file:
        config = yaml.safe_load(file.read())
    # print('Config: ', config)

    data_config = config['data']
    val_config = config['val']
    input_size = data_config["size_crops"]
    torch.manual_seed(val_config['seed'])
    torch.cuda.manual_seed_all(val_config['seed'])

    # Init data and transforms
    val_image_transforms = T.Compose([T.Resize((input_size, input_size)),
                                      T.ToTensor(),
                                      T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    val_target_transforms = T.Compose([T.Resize((input_size, input_size), interpolation=InterpolationMode.NEAREST),
                                       T.ToTensor()])

    data_dir = data_config["data_dir"]
    dataset_name = data_config["dataset_name"]
    if dataset_name == "voc":
        ignore_index = 255
        num_classes = 21
        data_module = VOCDataModule(batch_size=val_config["batch_size"],
                                    return_masks=True,
                                    num_workers=config["num_workers"],
                                    train_split="trainaug",
                                    val_split="val",
                                    data_dir=data_dir,
                                    train_image_transform=None,
                                    drop_last=True,
                                    val_image_transform=val_image_transforms,
                                    val_target_transform=val_target_transforms)
    elif "coco" in dataset_name:
        assert len(dataset_name.split("-")) == 2
        mask_type = dataset_name.split("-")[-1]
        assert mask_type in ["all", "stuff", "thing"]
        if mask_type == "all":
            num_classes = 27
        elif mask_type == "stuff":
            num_classes = 15
        elif mask_type == "thing":
            num_classes = 12
        ignore_index = 255
        file_list = os.listdir(os.path.join(data_dir, "images", "train2017"))
        file_list_val = os.listdir(os.path.join(data_dir, "images", "val2017"))
        # random.shuffle(file_list_val)
        data_module = CocoDataModule(batch_size=val_config["batch_size"],
                                     num_workers=config["num_workers"],
                                     file_list=file_list,
                                     data_dir=data_dir,
                                     file_list_val=file_list_val,
                                     mask_type=mask_type,
                                     train_transforms=None,
                                     val_transforms=val_image_transforms,
                                     val_target_transforms=val_target_transforms)
    elif dataset_name == "ade20k":
        num_classes = 111
        ignore_index = 255
        val_target_transforms = T.Compose([T.Resize((input_size, input_size), interpolation=InterpolationMode.NEAREST)])
        data_module = None
    else:
        raise ValueError(f"{dataset_name} not supported")

    # Init method
    patch_size = val_config["patch_size"]
    spatial_res = input_size / patch_size
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
    model.to(cuda)

    # load pretrained weights
    if val_config["checkpoint"] is not None:
        checkpoint = torch.load(val_config["checkpoint"])
        msg = model.load_state_dict(checkpoint["state_dict"], strict=True)
        print(msg)
    else:
        print('no pretrained pth found!')

    dataloader = data_module.val_dataloader()
    metric = PredsmIoU(val_config['num_queries'], num_classes)

    # Calculate IoU for each image individually
    for idx, batch in enumerate(dataloader):
        imgs, masks = batch
        B = imgs.size(0)
        assert B == 1   # image has to be evaluated individually
        all_queries, tokens, _, _, res, _ = model([imgs.to(cuda)])     # tokens=(1,N,dim)

        # calculate token assignment
        token_cls = torch.einsum("bnc,bqc->bnq", norm(tokens), norm(all_queries[0]))
        token_cls = torch.softmax(token_cls, dim=-1)
        token_cls = token_cls.reshape(B, res, res, -1).permute(0, 3, 1, 2)  # (1,num_query,res,res)
        token_cls = token_cls.max(dim=1, keepdim=True)[1].float()       # (1,1,res,res)

        # downsample masks/upsample preds to masks_eval_size
        preds = F.interpolate(token_cls, size=(val_config['mask_eval_size'], val_config['mask_eval_size']), mode='nearest')
        masks *= 255
        if masks.size(3) != val_config['mask_eval_size']:
            masks = F.interpolate(masks, size=(val_config['mask_eval_size'], val_config['mask_eval_size']), mode='nearest')

        metric.update(masks[masks != ignore_index], preds[masks != ignore_index])
        # sys.exit(1)

    metric.compute()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default='../configs/eval_voc_config.yml', type=str)
    parser.add_argument('--gpu', default='0', type=str)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    cuda = torch.device('cuda')
    eval_overcluster()
