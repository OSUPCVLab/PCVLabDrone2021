# UbihereDrone2021
## Project Description
We launched this project to achieve UAV GeoLocalization in GPS-denied environment (GPS weak, unstable or unavailable). We used [DJI Mavic air 2](https://www.dji.com/mavic-air-2?site=brandsite&from=nav) for data collection and test flight. Other drone brands may lead to incorrect results for unknown reason.

![demo_vid](https://github.com/OSUPCVLab/UbihereDrone2021/blob/main/UAV%20Geolocalization/demo/Webp.net-gifmaker.gif)

## TODO List and ETA
- [x] Google satellite map download, please refer to Deniz's contributed [repo](https://github.com/OSUPCVLab/UAVGeolocalization/tree/main/dataset-generation-gmaps-osm) (2021-05)
- [x] Figure out feature matching and apply [SuperGlue](https://github.com/magicleap/SuperGluePretrainedNetwork) as the matching algorithm among google satellite map and UAV taken image (2021-07)
- [x] UAV flight rotation estimation (2022-03)
- [x] UAV flight height estimation (2022-04)
- [x] OSU campus flight test using [Litchi](https://flylitchi.com/hub) (2022-04)

:triangular_flag_on_post: **Updates**
- Check out [QuadTree](https://medium.com/@waleoyediran/spatial-indexing-with-quadtrees-b998ae49336), a spatial indexing algorithm that improves geo-queries in a 2D-space.
- Generate GIS building mask in correspondence with target area satellite image. GIS building mask comes from [OSMnx](https://osmnx.readthedocs.io/en/stable/) could help improve geolocalization accuracy and UAV flight height estimation.

## Prerequisites
This is a vision-based project. We use images taken by UAV embedded camera as the only data source for geolocalization. Our completing method is to match features from UAV taken images with other data sources with similar contexture information such as [GoogleSatelliteMap](https://www.google.com/maps/@40.0014409,-83.0193795,1131m/data=!3m1!1e3) and [OpenStreetMap](https://www.openstreetmap.org/#map=16/40.0001/-83.0215). Therefore, we provide two sub-repos with respect to satellite map generation and corresponding GIS mask generation. Both repos require creating new conda environment due to specific libraries version dependencies.
- [Satellite Map Generation](https://github.com/OSUPCVLab/UbihereDrone2021/tree/main/Satellite%20Map%20Generation)
- [GIS Mask from OpenStreetMap](https://github.com/OSUPCVLab/UbihereDrone2021/tree/main/GISMaskfromOSM)

## Main part
See [UAV Geolocalization](https://github.com/OSUPCVLab/UbihereDrone2021/tree/main/UAV%20Geolocalization) for more details.

## Additional Notes
- Discussions or questions are welcomed. Please contact wei.909@osu.edu
- Our test flight is done around Ohio State University main campus. If you want test around other place, please recollect satellite image, GIS mask and rebuild featurebase.

## Cite
If you use our code or collected data in your project, please cite the paper:

```BibTeX
@Article{drones7090569,
AUTHOR = {Wei, Jianli and Yilmaz, Alper},
TITLE = {A Visual Odometry Pipeline for Real-Time UAS Geopositioning},
JOURNAL = {Drones},
VOLUME = {7},
YEAR = {2023},
NUMBER = {9},
ARTICLE-NUMBER = {569},
URL = {https://www.mdpi.com/2504-446X/7/9/569},
ISSN = {2504-446X},
ABSTRACT = {The state-of-the-art geopositioning is the Global Navigation Satellite System (GNSS), which operates based on the satellite constellation providing positioning, navigation, and timing services. While the Global Positioning System (GPS) is widely used to position an Unmanned Aerial System (UAS), it is not always available and can be jammed, introducing operational liabilities. When the GPS signal is degraded or denied, the UAS navigation solution cannot rely on incorrect positions GPS provides, resulting in potential loss of control. This paper presents a real-time pipeline for geopositioning functionality using a down-facing monocular camera. The proposed approach is deployable using only a few initialization parameters, the most important of which is the map of the area covered by the UAS flight plan. Our pipeline consists of an offline geospatial quad-tree generation for fast information retrieval, a choice from a selection of landmark detection and matching schemes, and an attitude control mechanism that improves reference to acquired image matching. To evaluate our method, we collected several image sequences using various flight patterns with seasonal changes. The experiments demonstrate high accuracy and robustness to seasonal changes.},
DOI = {10.3390/drones7090569}
}
```
