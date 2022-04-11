## GIS data
In this project, we need distinguish keypoints over ground or higher construction such as buildings for more accurate flight height estimation and geolocalization. However, google maps does not provide such information but OpenStreeMap has such geospatial entities. We therefore extract building GIS mask from OSM with its assocaited python library [OSMnx](https://osmnx.readthedocs.io/en/stable/osmnx.html#module-osmnx.geometries).

## Installation (Windows)
Please refer to [OSMnx](https://github.com/gboeing/osmnx) for more details. Here we provide installment commands satisfying our Windows environment. If you have not built Google satellite map generator environment, please refer to this [repo]() to set up the environment.
```shell
# Anaconda activate the built environment
conda create -n GISgenerator2021 --strict-channel-priority osmnx
conda activate GISgenerator2021

# Install required libraries, pytorch needs to be installed independently
cd GISMaskfromOSM
pip install opencv-python jupyter
conda create -n GISgenerator2021 --strict-channel-priority osmnx
```
- The Chromedriver inside the repo is for **Windows**. The version: _ChromeDriver 100.0.4896.60_
- Download the latest or suitable [version](https://chromedriver.chromium.org/downloads) for Chrome

## Modify `main.py`
```python
if __name__=='__main__':
    # Example: 5x5 -> 25 images
    Lat, Long = 40.01835966827935, -83.03297664244631  # For 30*17 Larger Map, 2.3km^2

    take_screenshot(
        lat=Lat,  # Top left corner latitude
        long=Long,
        row=30,  # 5 rows
        col=17,  # 5 columns
        file_name="image",  # Map image: "image-map-{number}.png"
        number=0
    )
```
### Parameters
- `Lat, Long` latitude and longitude of the center of first screenshot image (1280x720)
- `row` Row count along the way to south  
- `col` Column count along the way to east
 
 For generating our used satellite image map, which is 2.3km\^2, we set up row and col to be 30 and 17 respectively. If you need to generate your target satellite image area, please set up a new `Lat, Long` and refer to this ratio.

## Notes
- Estimating GPS could be viewed as a distance measurement problem. Therefore, after collecting the satellite image maps, we need to calculate ground sample distance ([GSD](https://en.wikipedia.org/wiki/Ground_sample_distance)). Please refer to `MapAnalysis.ipynb` for details.
- Target rea top-left GPS coordinates are referred to as `--satmap_init_gps` in the [test.py](https://github.com/OSUPCVLab/UbihereDrone2021/blob/main/UAV%20Geolocalization/test.py).
- You can also replace google satellite image maps with other sources but having similart texture. Remember to calculate GSD and acquire your collected area north, south, east and west GPS coordinates.
