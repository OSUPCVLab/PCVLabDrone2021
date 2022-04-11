## GIS data
In this project, we need distinguish keypoints over ground or higher construction such as buildings for more accurate flight height estimation and geolocalization. However, google maps does not provide such information but OpenStreeMap has such geospatial entities. We therefore extract building GIS mask from OSM with its assocaited python library [OSMnx](https://osmnx.readthedocs.io/en/stable/osmnx.html#module-osmnx.geometries).

## Installation (Windows)
Please refer to [OSMnx](https://github.com/gboeing/osmnx) for more details. Here we provide installment commands satisfying our Windows environment.
```shell
# Anaconda create a new environment as OSMnx installment required
conda create -n GISgenerator2021 --strict-channel-priority osmnx
conda activate GISgenerator2021

# Install required libraries
cd GISMaskfromOSM
pip install opencv-python jupyter
```

## Modify `GISgenerator.ipynb`
```python
# set up GIS map bounday, center point and distance
# Please refer to MapAnalysis for target map width and height
north, south = 40.00050846915017, 39.99829017146142
east, west = -83.0139618955573, -83.01911201346185
satmap_center = (39.99939933, -83.01653695)
dist = 300

create_square_from_osm(addr=satmap_center, bbox=(north, south, east, west), dist=dist)
```
### Parameters
- `north, south, east, west` OSM entities within a N, S, E, W bounding box, please refer to [MapAnalysis.ipynb](https://github.com/OSUPCVLab/UbihereDrone2021/blob/main/Satellite%20Map%20Generation/MapAnalysis.ipynb) for values.
- `satmap_center` Satellite map center GPS 
- `dist` Assume satellite map width and height are W and H respectively. `dist` satisfies dist>max(W/2, H/2)
 
We provided a target area located within OSU campus. Here is the example satellite map and GIS building mask visualization image. Buildings from the satellite map are semantically labeled as yellow.

<img src="https://github.com/OSUPCVLab/UbihereDrone2021/blob/main/GISMaskfromOSM/sat_mask.png" width=90% height=90% />


## Notes

