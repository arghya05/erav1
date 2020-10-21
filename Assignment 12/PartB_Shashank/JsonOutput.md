### Description of Output JSON File
# Describing the JSON File

The annotations json file consist of the annotations for the scraped Images for the following objects:
* Hardhats
* Vest
* Mask
* Boots

This annotations were generated using the VGG Image Annotation Tool [Link](http://www.robots.ox.ac.uk/~vgg/software/via/via_demo.html)


Here is a sample annotation file to understand the output from the too.

``` 
  {
    "filename": "african-american-man-in-hardhat-and-safety-vest-picture-id637499188.jpg",
    "file_size": 41248,
    "file_attributes": {
      "caption": "",
      "public_domain": "no",
      "image_url": ""
    },
    "region_count": 2,
    "region_id": 0,
    "region_shape_attributes": {
      "name": "rect",
      "x": 73,
      "y": 71,
      "width": 286,
      "height": 187
    },
    "region_attributes": {
      "": "",
      "class": "0"
    }
  },
  {
    "filename": "african-american-man-in-hardhat-and-safety-vest-picture-id637499188.jpg",
    "file_size": 41248,
    "file_attributes": {
      "caption": "",
      "public_domain": "no",
      "image_url": ""
    },
    "region_count": 2,
    "region_id": 1,
    "region_shape_attributes": {
      "name": "rect",
      "x": 96,
      "y": 337,
      "width": 310,
      "height": 274
    },
    "region_attributes": {
      "": "",
      "class": "1"
    }
  }
 ```

The tool creates regions which can be understood as uniques object anotation and can take many arbitary shapes depending upon what we choose like :
* Rectangle
* Circle
* Ellipse
* Polygon
* Points
* Polyline

I have used rectangle for all the annotations.

 A `rect` region has the following shape attributes:
* x : x coordinate of centre
* y : y coordinate of the centre
* width : width of the rectangle region
* height: height of the rectangle region


 
There are two main types of attribute we can define using the tool. 
* Region Attributes : These are the attribute specific to the annotations, more precisely for our case the object meta data like object tpye. I have modified the preset region attributes and remove things like lumination etc because those are not relevant to our usecase. For our case this consist of the one attribute which is the class of the object.
* Hardhats ---->0
* Vest ---->1
* Mask ---->2
* Boots ---->3

* File Attributes : Every Image file will have some attributes unique to that file like :
1. `filename`: Name of the file
2. `file_size`: Size of the image file,
Alos there are file attributes which corresponds to the metadata info of the file. it is also full configurable. For this I have taken the default attributes i.e. `caption`, `public_domain`, etc. We can modify these based on our use case.
