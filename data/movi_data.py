import os

from typing import Optional, Callable
from PIL import Image
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
from typing import Tuple, Any


class MOViDataModule:

    def __init__(self,
                 data_dir: str,
                 dataset_name: str,
                 train_split: str,
                 val_split: str,
                 train_image_transform: Optional[Callable],
                 val_image_transform: Optional[Callable],
                 val_target_transform: Optional[Callable],
                 batch_size: int,
                 num_workers: int,
                 shuffle: bool = True,
                 return_masks: bool = False,
                 drop_last: bool = True):
        """
        Data module for MOVi data.
        If return_masks is set train_image_transform should be callable with imgs and masks or None.
        """
        super().__init__()
        self.root = os.path.join(data_dir, dataset_name)
        self.train_split = train_split
        self.val_split = val_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_image_transform = train_image_transform
        self.val_image_transform = val_image_transform
        self.val_target_transform = val_target_transform
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.return_masks = return_masks

        # Set up datasets in __init__ as we need to know the number of samples to init cosine lr schedules
        self.movi_train = MOViDataset(root=self.root, image_set=train_split, transforms=self.train_image_transform,
                                      return_masks=self.return_masks)
        self.movi_val = MOViDataset(root=self.root, image_set=val_split, transform=self.val_image_transform,
                                    target_transform=self.val_target_transform)
        print('--- Loaded ' + dataset_name + ' with Train %d, Val %d ---' % (len(self.movi_train), len(self.movi_val)))

    def __len__(self):
        return len(self.movi_train)

    def train_dataloader(self):
        return DataLoader(self.movi_train, batch_size=self.batch_size,
                          shuffle=self.shuffle, num_workers=self.num_workers,
                          drop_last=self.drop_last, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.movi_val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
                          drop_last=self.drop_last, pin_memory=True)


class MOViDataset(VisionDataset):

    def __init__(
            self,
            root: str,
            image_set: str = "frames",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            transforms: Optional[Callable] = None,
            return_masks: bool = False
    ):
        super(MOViDataset, self).__init__(root, transforms, transform, target_transform)
        self.image_set = image_set
        if self.image_set == "frames":      # set for training
            img_folder = "frames"
        elif self.image_set == "images":    # set for validation
            img_folder = "images"
        else:
            raise ValueError(f"No support for image set {self.image_set}")
        image_dir = os.path.join(root, img_folder)
        seg_dir = os.path.join(root, 'masks')
        if not os.path.isdir(seg_dir) or not os.path.isdir(image_dir) or not os.path.isdir(root):
            raise RuntimeError('Dataset not found or corrupted.')

        self.images = [os.path.join(image_dir, x) for x in os.listdir(image_dir)]
        self.masks = [os.path.join(seg_dir, x) for x in os.listdir(seg_dir)]
        self.return_masks = return_masks

        assert all([Path(f).is_file() for f in self.masks]) and all([Path(f).is_file() for f in self.images])

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img = Image.open(self.images[index]).convert('RGB')
        if self.image_set == "images":      # for validation
            # print('img = ', self.images[index])
            mask = Image.open(self.masks[index])
            print("image: ", self.images[index])
            if self.transforms:
                img, mask = self.transforms(img, mask)
            return img, mask
        elif self.image_set == "frames":    # for training
            if self.transforms:
                if self.return_masks:
                    mask = Image.open(self.masks[index])
                    res = self.transforms(img, mask)
                else:
                    res = self.transforms(img)
                return res
            return img

    def __len__(self) -> int:
        return len(self.images)
