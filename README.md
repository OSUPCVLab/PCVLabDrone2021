# UbihereDrone2021
## Project Description
### We launched this project to achieve UAV GeoLocalization in GPS-denied environment (GPS weak, unstable or unavailable).

![demo_vid](https://github.com/OSUPCVLab/UbihereDrone2021/blob/main/UAV%20Geolocalization/demo/Webp.net-gifmaker.gif)

## TODO List and ETA
- [x] Google satellite map download, please refer to Deniz's contributed [repo](https://github.com/OSUPCVLab/UAVGeolocalization/tree/main/dataset-generation-gmaps-osm) (2021-05)
- [x] Figure out feature matching and apply [SuperGlue](https://github.com/magicleap/SuperGluePretrainedNetwork) as the matching algorithm among google satellite map and UAV taken image (2021-07)
- [x] UAV flight rotation invirance (2022-03)
- [x] UAV flight scale invirance (2022-04)
- [x] OSU campus flight test using [Litchi] (https://flylitchi.com/hub)(2022-04)

:triangular_flag_on_post: **Updates**
- Check out [QuadTree](https://medium.com/@waleoyediran/spatial-indexing-with-quadtrees-b998ae49336), a spatial indexing algorithm that improves geo-queries in a 2D-space.

## Prerequisites
- Satellite map generation with google maps API
- GIS Mask generator from OpenStreetMap
- Feature extractor

Thanks to Deniz's [Repo](https://github.com/OSUPCVLab/UAVGeolocalization/tree/main/dataset-generation-gmaps-osm), we are able to generate target area google satellite images. This map image shows our selected area for UAV geolocalization without GPS support.

## Installation
```shell
# Anaconda create a new environment
conda create -n UbihereDrone2021 python=3.8
conda activate UbihereDrone2021

# Install required libraries
cd UAV Geolocalization
pip install -r requirements.txt
```

We provide the [download link](https://drive.google.com/drive/folders/1w7e0qmWQpv5HeU5w9utWeC43E45Xh1I6?usp=sharing) to 
  - the test assets (~3.7GB).
  - featurebase (~5GB).
  - 2 pretrained models of indoor and outdoor (each ~47MB).

We use superglue as an advanced feature matching algorithm. However, due to its strict LICENSE requirements, we recommend downloading its pretrained models [here](https://github.com/magicleap/SuperGluePretrainedNetwork/tree/master/models/weights).

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
The `--input` flag accepts a live stream URL (such as youtube live link), running on GPU as faster processing speed is required somehow.
```shell
# Running on the GPU
python test.py --input=<URL> --output=./output/LIVE/ --range 900 900 --Init_height=140 --patience=20 --matching_vis --apply_GIS
```

## Additional Notes
- Discussions or questions are welcomed. Please contact wei.909@osu.edu
