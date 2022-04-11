
## Installation (Windows)
```shell
# Anaconda create a new environment
conda create -n UbihereDrone2021 python=3.8
conda activate UbihereDrone2021

# Install required libraries, pytorch needs to be installed independently
cd UAV Geolocalization
pip install -r requirements.txt

# pytorch 1.7.1 GPU version if cuda avaiable
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.2 -c pytorch
# Install pytorch 1.7.1 if CPU only
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cpuonly -c pytorch
```

We provide the [download link](https://drive.google.com/drive/folders/1w7e0qmWQpv5HeU5w9utWeC43E45Xh1I6?usp=sharing) and saved directories structure

```
UAV Geolocalization
├── assets (~3.7GB)
    ├── images
    └── videos
└── featurebase (~5GB)
    ├── GIS_mask.png
    ├── QuadTree_idx.pkl
    └── satmap_kpts.npz
└── models
    └── weights (each ~47MB)
        ├── superglue_indoor.pth
        └── superglue_indoor.pth
```

We use superglue as an advanced feature matching algorithm. However, due to its strict LICENSE requirements, we recommend downloading its pretrained models [here](https://github.com/magicleap/SuperGluePretrainedNetwork/tree/master/models/weights).

## Feature Extraction offline
[Satellite map generator](https://github.com/OSUPCVLab/UbihereDrone2021/tree/main/Satellite%20Map%20Generation) could generate target area satellite map combined with row by col patches with the size of 1280x720. We extract features from satellite map patch by patch and concatenate them in the same order of row by column. Here is the generated target area example feature map. Red points on the are extracted features containing postion and descriptor saved as `satmap_kpts.npz` into `featurebase` folder.

<img src="https://github.com/OSUPCVLab/UbihereDrone2021/blob/main/UAV%20Geolocalization/demo/feature_vis.png" width=90% height=90% />

### Run feature extractor offline
```shell
# Running on the CPU
python feature_extractor.py --input=./assets/satmaps/ --output_dir=./featurebase/ --map_row_col 3 3 --force_cpu

# Running on the GPU
python feature_extractor.py --input=./assets/satmaps/ --output_dir=./featurebase/ --map_row_col 3 3
```
- Use `--map_row_col` to indicate generated satellite map patches with rows by columns sequence


## UAV test flight
Our test flight is over the blue box area as the left image shows, which covers abundant contexture including road, buildings, greens, river, urban area and recreational facilities. The test assets include demo images and video taken by UAV around urban environment. Feel free to download them and run our demo.
The right image contains blue and red trajectory, which are UAV inferenced GPS and ground truth respectitively.

<p float="left">
  <img src="https://github.com/OSUPCVLab/UbihereDrone2021/blob/main/UAV%20Geolocalization/demo/map.png" width=45% height=45% />
  <img src="https://github.com/OSUPCVLab/UbihereDrone2021/blob/main/UAV%20Geolocalization/demo/demo result.png" width=52.5% height=52.5% /> 
</p>

### Run the demo on a dictionary of images
The `--input` flag accepts a path to a dictionary containing a batch of UAV taken images in time squeence. To run the demo, make sure the test images are saved into `./assets/images/` folder.
```shell
# Running on the CPU
python test.py --input=./assets/images/ --output=./output/images/ --range 900 900 --Init_height=140 --patience=20 --matching_vis --apply_GIS --force_cpu

# Running on the GPU
python test.py --input=./assets/images/ --output=./output/images/ --range 900 900 --Init_height=140 --patience=20 --matching_vis --apply_GIS
```

- Use `--range` to change google satellite image size (default: 900x900)
- Use `--Init_GPS` to preset UAV starting points, required for each flight
- Use `--Init_height` to preset UAV starting flight height, which is required to be >= 110 meters (default: 140)
- Use `--Orien` to preset UAV initial heading direction, which is required if not North-faced initially (default: 0)
- Use `--apply_GIS` to apply semantic building labeling extracted from OpenStreetMap to achieve more accurate height estimation and GPS prediction (default: False). For more details, please refer to [`GISMaskfromOSM`](https://github.com/OSUPCVLab/UbihereDrone2021/tree/main/GISMaskfromOSM).

### Run the demo on a dictionary of video
The `--input` flag accepts a path to a UAV taken video. To run the demo, make sure the test video are saved into `./assets/videos/` folder.
```shell
# Running on the CPU
python test.py --input=./assets/videos/demo.mp4 --output=./output/videos/ --range 900 900 --Init_height=140 --patience=20 --matching_vis --apply_GIS --force_cpu

# Running on the GPU
python test.py --input=./assets/videos/demo.mp4 --output=./output/videos/ --range 900 900 --Init_height=140 --patience=20 --matching_vis --apply_GIS
```

### Run the demo on a LIVE stream
The `--input` flag accepts a live stream URL (such as youtube live link). Please check [CamGear](https://abhitronix.github.io/vidgear/v0.2.5-stable/gears/camgear/overview/) for more details. GPU is required somehow to achieve real-time flight test.
```shell
# Running on the GPU
python test.py --input=<URL> --output=./output/LIVE/ --range 900 900 --Init_height=140 --patience=20 --matching_vis --apply_GIS
```
