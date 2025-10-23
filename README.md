# 🚀 Installation

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



# 🚀 Data Preparation

## SemanticKITTI
Please download the data from the SemanticKITTI official website, and additionally download the color image data (image_2) from the KITTI Odometry official website. Extract all the compressed files into the same directory (default ./dataset).

- SemanticKITTI: [[semkitti_url](https://semantic-kitti.org/)]
- KITTI Odometry color:  data_odometry_color.zip
https://pan.baidu.com/s/1sdJA0PLg9l2Y5IuiXixN6A   Extraction Code: u9wr 

Expected directory structure：
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
```

## nuScenes

Please download the Full dataset  from the nuScenes official website and make sure to include the lidarseg annotation package. Extract it to ./dataset/nuscenes/.
NuScenes:[[NuScenes_url](https://www.nuscenes.org/)]
```
./dataset/
└── nuscenes/
    ├── v1.0-trainval/
    ├── v1.0-test/
    ├── samples/
    ├── sweeps/
    ├── maps/
    └── lidarseg/
```


# 🚀 Train
### SemanticKITTI
Run training:
```bash
cd <root dir of this repo>
python main.py \
  --log_dir 2DPASS_semkitti \
  --config config/2DPASS-semantickitti.yaml \
  --gpu 0
  ```
### nuScenes
Run  training:
```bash
cd <root dir of this repo>
python main.py \
  --log_dir 2DPASS_nusc \
  --config config/2DPASS-nuscenese.yaml \
  --gpu 0 
```
### Testing / Inference
SemanticKITTI testing with test-time augmentation (TTA):
```bash
cd <root dir of this repo>
python main.py \
  --config config/2DPASS-semantickitti.yaml \
  --gpu 0 \
  --test \
  --num_vote  \
  --checkpoint <path-to-pytorch-checkpoint>
```
--num_vote controls the number of test-time views.
If your GPU memory is smaller, reduce this value.
Set --num_vote=1 to disable TTA (typically ~2% performance drop).
