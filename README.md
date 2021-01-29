# MultiStar: Instance Segmentation of Overlapping Objects with Star-Convex Polygons

[Link](https://arxiv.org/abs/2011.13228) to the paper, presented at [ISBI 2021](https://biomedicalimaging.org/2021/).

## How to use this repository:
- Install the required packages as a conda environment with *MultiStar.yml*. Additionally, you need [this](https://github.com/imagirom/ConfNets) U-Net implementation, which needs to be stored in a folder */confnets* in the same location as the other scripts (e.g. *dataset.py*, *evaluation.py*, ...).
- In the paper we evaluated our algorithm on two datasets: *DSBOV* and *OSC-ISBI*. They can be downloaded / recreated and stored in the proper format and location by running the scripts in */datagen*. *OSC-ISBI* is composed of two datasets (*ISBI14* and *ISBI15*), but this is taken care of by using the *DatasetPlus* class in *dataset.py* instead of the *Dataset* class. *DatasetPlus* accesses data from both datases.
- Example usage for the *DSBOV* dataset with gpu available is demonstrated in usage_example.py
- Results are saved in the directory *experiments*.
