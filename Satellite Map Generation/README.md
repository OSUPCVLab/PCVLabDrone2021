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
- Download the latest or suitable [version](https://chromedriver.chromium.org/downloads) for Chrome

We use superglue as an advanced feature matching algorithm. However, due to its strict LICENSE requirements, we recommend downloading its pretrained models [here](https://github.com/magicleap/SuperGluePretrainedNetwork/tree/master/models/weights).

## Modify `main.py`
<details>
  <summary>[code snippets]</summary>

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
  `Lat, Long` latitude and longitude of the center of first screenshot image (1280x720)
  
  `row` Row count in the way to the south
  
  `col` Column count in the way to the east
  
