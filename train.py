import argparse
import os
import time

import torch
import yaml

from torchvision.transforms import ToTensor, Compose, Resize, Normalize
from torchvision.transforms.functional import InterpolationMode

from data.coco_data import CocoDataModule
from data.voc_data import VOCDataModule
from model.align_model import AlignSegmentor
from model.criterion import AlignCriterion
from model.transforms import TrainTransforms
from utils import AverageMeter, save_checkpoint, neq_load_external


def set_path(config):
    if config['train']['checkpoint']:
        model_path = os.path.dirname(config['train']['checkpoint'])
    else:
        model_path = './log_tmp/{0}-{1}-bs{2}/model'.format(config["data"]["dataset_name"],
                                                            config["train"]["arch"],
                                                            config["train"]["batch_size"])

    if not os.path.exists(model_path): os.makedirs(model_path)
    return model_path


def exclude_from_wt_decay(named_params, weight_decay: float, lr: float):
    params = []
    excluded_params = []
    query_param = []

    for name, param in named_params:
        if not param.requires_grad:
            continue
        # do not regularize biases nor Norm parameters
        if name.endswith(".bias") or len(param.shape) == 1:
            excluded_params.append(param)
        elif 'clsQueries' in name:
            query_param.append(param)
        else:
            params.append(param)
    return [{'params': params, 'weight_decay': weight_decay, 'lr': lr},
            {'params': excluded_params, 'weight_decay': 0., 'lr': lr},
            {'params': query_param, 'weight_decay': 0., 'lr': lr * 1}]


def configure_optimizers(model, train_config):
    # Separate Decoder params from ViT params
    # only train Decoder
    decoder_params_named = []
    for name, param in model.named_parameters():
        if name.startswith("backbone"):
            param.requires_grad = False
        elif train_config['fix_prototypes'] and 'clsQueries' in name:
            param.requires_grad = False
        else:
            decoder_params_named.append((name, param))

    # Prepare param groups. Exclude norm and bias from weight decay if flag set.
    if train_config['exclude_norm_bias']:
        params = exclude_from_wt_decay(decoder_params_named,
                                       weight_decay=train_config["weight_decay"],
                                       lr=train_config['lr_decoder'])
    else:
        decoder_params = [param for _, param in decoder_params_named]
        params = [{'params': decoder_params, 'lr': train_config['lr_decoder']}]

    # Init optimizer and lr schedule
    optimizer = torch.optim.AdamW(params, weight_decay=train_config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    return optimizer, scheduler


def start_train():
    with open(args.config_path) as file:
        config = yaml.safe_load(file.read())
    # print('Config: ', config)

    data_config = config['data']
    train_config = config['train']
    torch.manual_seed(train_config['seed'])
    torch.cuda.manual_seed_all(train_config['seed'])

    # Init data modules and tranforms
    dataset_name = data_config["dataset_name"]
    train_transforms = TrainTransforms(size_crops=data_config["size_crops"],
                                       nmb_crops=data_config["nmb_crops"],
                                       min_intersection=data_config["min_intersection_crops"],
                                       min_scale_crops=data_config["min_scale_crops"],
                                       max_scale_crops=data_config["max_scale_crops"],
                                       augment_image=data_config["augment_image"])

    # Setup voc dataset used for evaluation
    val_size = data_config["size_crops_val"]
    val_image_transforms = Compose([Resize((val_size, val_size)),
                                    ToTensor(),
                                    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    val_target_transforms = Compose([Resize((val_size, val_size), interpolation=InterpolationMode.NEAREST),
                                     ToTensor()])

    # Setup train data
    if dataset_name == "voc":
        train_data_module = VOCDataModule(batch_size=train_config["batch_size"],
                                          num_workers=config["num_workers"],
                                          train_split="trainaug",
                                          val_split="val",
                                          data_dir=data_config["voc_data_path"],
                                          train_image_transform=train_transforms,
                                          val_image_transform=val_image_transforms,
                                          val_target_transform=val_target_transforms)
    elif dataset_name == 'coco':
        file_list = os.listdir(os.path.join(data_config["data_dir"], "images/train2017"))
        train_data_module = CocoDataModule(batch_size=train_config["batch_size"],
                                           num_workers=config["num_workers"],
                                           file_list=file_list,
                                           data_dir=data_config["data_dir"],
                                           train_transforms=train_transforms,
                                           val_transforms=None)
    else:
        raise ValueError(f"Data set {dataset_name} not supported")

    model_path = set_path(config)

    model = AlignSegmentor(arch=train_config['arch'],
                           patch_size=train_config['patch_size'],
                           embed_dim=train_config['embed_dim'],
                           hidden_dim=train_config['hidden_dim'],
                           num_heads=train_config['decoder_num_heads'],
                           num_queries=train_config['num_queries'],
                           nmb_crops=data_config["nmb_crops"],
                           num_decode_layers=train_config['num_decode_layers'],
                           last_self_attention=train_config['last_self_attention'])
    model = model.to(cuda)

    criterion = AlignCriterion(patch_size=train_config['patch_size'],
                               num_queries=train_config['num_queries'],
                               nmb_crops=data_config["nmb_crops"],
                               roi_align_kernel_size=train_config['roi_align_kernel_size'],
                               ce_temperature=train_config['ce_temperature'],
                               negative_pressure=train_config['negative_pressure'],
                               last_self_attention=train_config['last_self_attention'])
    criterion = criterion.to(cuda)

    # Initialize model
    start_epoch = 0
    if train_config["checkpoint"] is not None:
        checkpoint = torch.load(train_config["checkpoint"])
        start_epoch = checkpoint['epoch']
        msg = model.load_state_dict(checkpoint["state_dict"], strict=True)
        print(msg)
    elif train_config["checkpoint"] is None \
            and train_config["pretrained_model"] is not None \
            and train_config["prototype_queries"] is not None:
        # initialize model with pre-trained ViT and prepared Prototypes
        pretrained_model = torch.load(train_config["pretrained_model"], map_location=torch.device('cpu'))
        neq_load_external(model, pretrained_model)
        protos = torch.load(train_config["prototype_queries"]).to(cuda)
        model.set_clsQuery(protos)
    elif train_config["checkpoint"] is None \
            and train_config["pretrained_model"] is not None \
            and train_config["prototype_queries"] is None:
        # only load pre-trained ViT
        pretrained_model = torch.load(train_config["pretrained_model"], map_location=torch.device('cpu'))
        neq_load_external(model, pretrained_model)

    # Optionally fix ViT, Queries
    optimizer, scheduler = configure_optimizers(model, train_config)
    dataloader = train_data_module.train_dataloader()

    for epoch in range(start_epoch, train_config['max_epochs']):

        train(dataloader, model, optimizer, criterion, epoch)

        scheduler.step()
        print('\t Epoch: ', epoch, 'with lr: ', scheduler.get_last_lr())

        if epoch % train_config['save_checkpoint_every_n_epochs'] == 0:
            # save check_point
            save_checkpoint({'epoch': epoch + 1,
                             'net': train_config['arch'],
                             'state_dict': model.state_dict(),
                             'optimizer': optimizer.state_dict(),
                             }, gap=train_config['save_checkpoint_every_n_epochs'],
                            filename=os.path.join(model_path, 'epoch%s.pth' % str(epoch + 1)), keep_all=False)

    print('Training %d epochs finished' % (train_config['max_epochs']))


def train(data_loader, model, optimizer, criterion, epoch):
    losses = AverageMeter()
    model.train()

    for idx, batch in enumerate(data_loader):
        inputs, bboxes = batch  # inputs = [sum(num_crops), (B, 3, w, h)]
        B = inputs[0].size(0)
        tic = time.time()
        for i in range(len(inputs)):
            inputs[i] = inputs[i].to(cuda, non_blocking=True)
        bboxes['gc'] = bboxes['gc'].to(cuda, non_blocking=True)
        bboxes['all'] = bboxes['all'].to(cuda, non_blocking=True)

        results = model(inputs)

        # Calculate loss
        loss = criterion(results, bboxes)
        losses.update(loss.item(), B, step=len(data_loader))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if idx % 1 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.local_avg:.4f}) Time:{3:.2f}\t'.
                  format(epoch, idx, len(data_loader), time.time() - tic, loss=losses))

    return losses.local_avg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default='./configs/train_voc_config.yml', type=str)
    parser.add_argument('--gpu', default='0', type=str)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    cuda = torch.device('cuda')
    start_train()
