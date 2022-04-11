# UbihereDrone2021
## Project Description
### We launched this project to achieve UAV GeoLocalization in GPS-denied environment (GPS weak, unstable or unavailable).

![demo_vid](https://github.com/OSUPCVLab/UbihereDrone2021/blob/main/UAV%20Geolocalization/demo/Webp.net-gifmaker.gif)

## TODO List and ETA
- [x] Google satellite map download, please refer to Deniz's contributed [repo](https://github.com/OSUPCVLab/UAVGeolocalization/tree/main/dataset-generation-gmaps-osm) (2021-05)
- [x] Figure out feature matching and apply [SuperGlue](https://github.com/magicleap/SuperGluePretrainedNetwork) as the matching algorithm among google satellite map and UAV taken image (2021-07)
- [x] UAV flight rotation invirance (2022-03)
- [x] UAV flight scale invirance (2022-04)
- [x] OSU campus flight test using [Litchi](https://flylitchi.com/hub) (2022-04)

:triangular_flag_on_post: **Updates**
- Check out [QuadTree](https://medium.com/@waleoyediran/spatial-indexing-with-quadtrees-b998ae49336), a spatial indexing algorithm that improves geo-queries in a 2D-space.

## Prerequisites
This is a vision-based project. We use images taken by UAV embedded camera as the only data source for geolocalization. Our completing method is to match features from UAV taken images with other data sources with similar contexture information such as [GoogleSatelliteMap](https://www.google.com/maps/@40.0014409,-83.0193795,1131m/data=!3m1!1e3) and [OpenStreetMap](https://www.openstreetmap.org/#map=16/40.0001/-83.0215). Therefore, we provide two sub-repos with respect to satellite map generation and corresponding GIS mask generation. Both repos require creating new conda environment due to specific libraries version requirements.
- [Satellite Map Generation](https://github.com/OSUPCVLab/UbihereDrone2021/tree/main/Satellite%20Map%20Generation)
- [GIS Mask from OpenStreetMap](https://github.com/OSUPCVLab/UbihereDrone2021/tree/main/GISMaskfromOSM)

## Main part
See [UAV Geolocalization](https://github.com/OSUPCVLab/UbihereDrone2021/tree/main/UAV%20Geolocalization) for more details.

## Additional Notes
- Discussions or questions are welcomed. Please contact wei.909@osu.edu
- Our test flight is done around Ohio State University main campus. If you want test around other place, please recollect satellite image, GIS mask and rebuild featurebase.
