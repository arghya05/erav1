# Assignment 14

### Group Members:
* Shashank Pathak
* Divyam Malay Shah


## Data Description:

### Data Source : [Link](https://drive.google.com/drive/folders/1_EW9AxnaZap_tZlIO7XSldNAkg_D0czV?usp=sharing)

### Directory layout

    .
    ├── Images                   # Images scraped from various sources belonging to the 4 classes
    ├── Labels                   # Bounding Box annotations 
    ├── planercnn_output         # Planar Regions for the images (Output after running PlaneRCNN on the data)
    ├── depthmap                 # Depth maps for the images (Output after running MiDaS on the data) 
    ├── train.txt                # List of images in train set
    ├── test.txt                 # List of images in test set
    └── classes.txt              # List of class labels

### References:

* PlaneRCNN : [Link](https://github.com/NVlabs/planercnn) (3D Plane Detection)
* MiDaS: [Link](https://github.com/intel-isl/MiDaS) (Monocular Depth Estimation)
