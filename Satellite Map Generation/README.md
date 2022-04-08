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
- The Chromedriver inside the repo is for **Windows**. The version: _ChromeDriver 90.0.4430.24_
- Download the latest or suitable [version](https://chromedriver.chromium.org/downloads) for Chrome

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
- `Lat, Long` latitude and longitude of the center of first screenshot image (1280x720)
- `row` Row count in the way to the south  
- `col` Column count in the way to the east
 
 For generating our used satellite image map, which is 2.3km^2, we set up row and col to be 30 and 17 respectively.
</details>

## Notes
