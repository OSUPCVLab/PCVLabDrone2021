# UbihereDrone2021
## Project Description
### We launched this project to achieve UAV GeoLocalization in GPS-denied environment (GPS weak, unstable or unavailable).

## Demo
![demo_vid](assets/loftr-github-demo.gif)

## TODO List and ETA
- [x] Google satellite map download, please refer to Deniz's contributed [repo](https://github.com/OSUPCVLab/UAVGeolocalization/tree/main/dataset-generation-gmaps-osm) (2021-05)
- [x] Figure out feature matching and apply [SuperGlue](https://github.com/magicleap/SuperGluePretrainedNetwork) as the matching algorithm among google satellite map and UAV taken image (2021-07)
- [x] UAV flight rotation invirance (2022-03)
- [x] UAV flight scale invirance (2022-04)
- [x] OSU campus flight test using [Litchi](https://flylitchi.com/hub)(2022-04)

Discussions or questions are welcomed. Please contact wei.909@osu.edu

:triangular_flag_on_post: **Updates**
- Check out [QuadTree](https://medium.com/@waleoyediran/spatial-indexing-with-quadtrees-b998ae49336), a spatial indexing algorithm that improves geo-queries in a 2D-space.



## Installation
```shell
# Anaconda create a new environment
conda create -n UbihereDrone2021 python=3.7
conda activate UbihereDrone2021

# Install required libraries
pip install -r requirements.txt
```

## Run UAV flight demos

### Google satellite maps download
Please download the satellite maps extracted features from this [link] and save them into `./encoder' 
