# Acknowledgement
Thanks to Deniz's [Repo](https://github.com/OSUPCVLab/UAVGeolocalization/tree/main/dataset-generation-gmaps-osm), we are able to generate target area google satellite images. This map image shows our selected area for UAV geolocalization without GPS support.


## Installation (Windows)
```shell
# Anaconda create a new environment
conda create -n mapgenerator2021 python=3.8
conda activate mapgenerator2021

# Install required libraries, pytorch needs to be installed independently
cd Satellite Map Generation
pip install -r requirements.txt
```
- Download the latest or suitable [version for](https://chromedriver.chromium.org/downloads) Chrome

We use superglue as an advanced feature matching algorithm. However, due to its strict LICENSE requirements, we recommend downloading its pretrained models [here](https://github.com/magicleap/SuperGluePretrainedNetwork/tree/master/models/weights).

## Feature Extraction offline

## UAV test flight
Our test flight is over the blue box area as the left image shows, which covers abundant contexture including road, buildings, greens, river, urban area and recreational facilities. The test assets include demo images and video taken by UAV around urban environment. Feel free to download them and run our demo.
The right image contains blue and red trajectory, which are UAV inferenced GPS and ground truth respectitively.

<p float="left">
  <img src="https://github.com/OSUPCVLab/UbihereDrone2021/blob/main/UAV%20Geolocalization/demo/map.png" width=45% height=45% />
  <img src="https://github.com/OSUPCVLab/UbihereDrone2021/blob/main/UAV%20Geolocalization/demo/demo result.png" width=52.5% height=52.5% /> 
</p>

### Run the demo on a dictionary of images
The `--input` flag accepts a path to a dictionary containing a batch of UAV taken images in time squeence. To run the demo, make sure the test images are saved into `./assets/images/` folder.

