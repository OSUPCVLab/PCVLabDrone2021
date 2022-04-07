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
- Satellite map generation with google maps API
- GIS Mask generator from OpenStreetMap
- Feature extractor

Thanks to Deniz's [Repo](https://github.com/OSUPCVLab/UAVGeolocalization/tree/main/dataset-generation-gmaps-osm), we are able to generate target area google satellite images. This map image shows our selected area for UAV geolocalization without GPS support.
