## [AlignSeg] Rethinking Self-Supervised Semantic Segmentation: Achieving End-to-End Segmentation

This is the PyTorch implementation of AlignSeg.

<div>
  <img width="100%" alt="Self-supervised Semantic Segmentation Evaluation Protocols"
  src="images/End-to-end self-supervised semantic segmentation.png">
</div>

### Dataset Setup

Please download the data and organize as detailed in the next subsections.

##### Pascal VOC
Here's a [zipped version](https://www.dropbox.com/s/6gd4x0i9ewasymb/voc_data.zip?dl=0) for convenience.

The structure for training and evaluation should be as follows:
```
dataset root.
└───SegmentationClass
│   │   *.png
│   │   ...
└───SegmentationClassAug # contains segmentation masks from trainaug extension 
│   │   *.png
│   │   ...
└───images
│   │   *.jpg
│   │   ...
└───sets
│   │   train.txt
│   │   trainaug.txt
│   │   val.txt
```

#### COCO-Stuff-27
The structure for training and evaluation should be as follows:
```
dataset root.
└───annotations
│   └─── annotations
│       └─── stuffthingmaps_trainval2017
│           │    stuffthing_2017.json
│           └─── train2017
│               │   *.png
│               │   ...
│           └─── val2017
│               │   *.png
│               │   ...
└───coco
│   └─── images
│       └─── train2017
│           │   *.jpg
│           │   ...
│       └─── val2017
│           │   *.jpg
│           │   ...
```
The “curated” split introduced by IIC can be downloaded [here](https://www.robots.ox.ac.uk/~xuji/datasets/COCOStuff164kCurated.tar.gz).

### Self-supervised Training with Frozen ViT

We provide the training configuration files for PVOC and COCO-Stuff in ```/configs``` folder, fill in your own path to dataset and pre-trained ViT.

As the image encoder is frozen during training, the self-supervised training is quite efficient and can be implemented with only one GPU. 
To start training on PVOC, you can run the following exemplary command:
  ```
  python train.py --config_path ./configs/train_voc_config.yml
  ```

The pre-trained ViT by DINO can be found [here](https://github.com/facebookresearch/dino).

### End-to-End Semantic Segmentation Inference

AlignSeg can perform real-time and end-to-end segmentation inference.

To perform segmentation inference and visualization, you can run the following exemplary command:
  ```
  python evaluate/visualize_segment.py --pretrained_weights {model.pth} --image_path ./images/2007_000464.jpg
  ```
replace `{model.pth}` with the path to the pre-trained model.

### Evaluation

We provide the evaluation configuration files for PVOC and COCO-Stuff-27 in ```/configs``` folder, fill in your own path to dataset and pre-trained model.

To evaluate the pre-trained model on PVOC, you can run the following exemplary command:
```
python evaluate/sup_overcluster.py --config_path ../configs/eval_voc_config.yml
```

### Pre-trained Models

We provide our pre-trained models, they can be downloaded by links below.

<table>
  <tr>
    <th>Encoder</th>
    <th>Dataset</th>
    <th>mIoU</th>
    <th>Download</th>
  </tr>
  <tr align="center">
    <td>ViT-S/16</td>
    <td>PVOC</td>
    <td>69.5</td>
    <td><a href="https://pan.baidu.com/s/1XoNgG-1lFs07HZGaxBTaHw?pwd=nz7k">model</a></td>
  </tr>
  <tr align="center">
    <td>ViT-S/16</td>
    <td>COCO-Stuff-27</td>
    <td>35.1</td>
    <td><a href="https://pan.baidu.com/s/1A7GeSNDwwfQPsCH404jYAw?pwd=rsxd">model</a></td>
  </tr>
</table>

<table>
  <tr>
    <th>Encoder</th>
    <th>Dataset</th>
    <th>FG-ARI</th>
    <th>mBO</th>
    <th>Download</th>
  </tr>
  <tr align="center">
    <td>ViT-S/16</td>
    <td>MOVi-C</td>
    <td>48.0</td>
    <td>31.2</td>
    <td><a href="https://pan.baidu.com/s/1xca8dLAOO_m04Jrn3eP8UQ?pwd=aant">model</a></td>
  </tr>
  <tr align="center">
    <td>ViT-S/16</td>
    <td align="center">MOVi-E</td>
    <td>44.1</td>
    <td>20.4</td>
    <td><a href="https://pan.baidu.com/s/1_2rcs5Qr4bHiKH9tN8X4dw?pwd=yk4j">model</a></td>
  </tr>
</table>






