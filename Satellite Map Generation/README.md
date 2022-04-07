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
  from src.loftr import LoFTR, default_cfg

  # Initialize LoFTR
  matcher = LoFTR(config=default_cfg)
  matcher.load_state_dict(torch.load("weights/indoor_ds.ckpt")['state_dict'])
  matcher = matcher.eval().cuda()
