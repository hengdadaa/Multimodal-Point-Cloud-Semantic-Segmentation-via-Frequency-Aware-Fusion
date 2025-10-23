# Installation

## Requirements
- PyTorch >= 1.8
- PyYAML (`yaml`)
- `easydict`
- `pyquaternion`
- PyTorch Lightning  
  *(tested with `pytorch_lightning==1.3.8` and `torchmetrics==0.5`)*
- `torch-scatter`  
  *(install via wheel index)*
- `nuScenes-devkit` *(optional for nuScenes)*
- `spconv`  
  *(tested with `spconv==2.1.16` on CUDA 11.1, install package `spconv-cu111==2.1.16`)*
- `torchsparse` *(optional, for MinkowskiNet / SPVCNN)*

---



# Data Preparation

## SemanticKITTI
请从 SemanticKITTI 官网下载数据，并**另外**从 KITTI Odometry 官网下载彩色图像数据（image_2）。将所有压缩包解压到同一个目录（默认 `./dataset`）。

可选下载链接（如需可替换占位符为真实网址）：
- SemanticKITTI: [Official][semkitti_url]
- KITTI Odometry color: [Official][kitti_odom_url]

期望的目录结构：
```text
./dataset/
└── SemanticKitti/
    └── sequences/
        ├── 00/
        │   ├── velodyne/
        │   │   ├── 000000.bin
        │   │   ├── 000001.bin
        │   │   └── ...
        │   ├── labels/
        │   │   ├── 000000.label
        │   │   ├── 000001.label
        │   │   └── ...
        │   ├── image_2/
        │   │   ├── 000000.png
        │   │   ├── 000001.png
        │   │   └── ...
        │   └── calib.txt
        ├── 08/   # for validation
        ├── 11/   # 11-21 for testing
        └── 21/
            └── ...


./dataset/
└── nuscenes/
    ├── v1.0-trainval/
    ├── v1.0-test/
    ├── samples/
    ├── sweeps/
    ├── maps/
    └── lidarseg/

