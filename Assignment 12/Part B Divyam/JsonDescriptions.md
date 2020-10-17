# Describing the JSON File

The annotations json(divyam_assignment-12_json.json) contains information related to the annotations on each image within the "Assignment12 images" directory.
Metadata for each image is indexed by the concatination of its filename and size in bytes. This json was generated using the [Oxford](http://www.robots.ox.ac.uk/~vgg/software/via/via_demo.html) annotation tool.

Lets take the image '2Q__ (1).jpg' as an example to understand how our json captures the annotation infromaton. The size of this image is 8968 bytes and therefore is indexed as
'2Q__ (1).jpg' + '8968' = `'2Q__ (1).jpg8968'`. 

Now lets take a look at the nested JSON object within this(`'2Q__ (1).jpg8968'`) parent key.

``` 
{'file_attributes': {'caption': '', 'image_url': '', 'public_domain': 'no'},
 'filename': '2Q__ (1).jpg',
 'regions': [{'region_attributes': {'image_quality': {'frontal': True,
     'good': True,
     'good_illumination': True},
    'name': 'hardhat1',
    'type': 'hardhat'},
   'shape_attributes': {'height': 63,
    'name': 'rect',
    'width': 74,
    'x': 56,
    'y': 14}},
  {'region_attributes': {'image_quality': {'frontal': True,
     'good': True,
     'good_illumination': True},
    'name': 'mask1',
    'type': 'mask'},
   'shape_attributes': {'height': 44,
    'name': 'rect',
    'width': 46,
    'x': 71,
    'y': 79}}],
 'size': 8968}
 ```
 
 1. The `file_attributes` field is used to capture metadata involving "captions", 'image_url' and flag wheter the domain of 'image_url' is public.
 
 2. `filename` is the name of the image file.
 
 3. `regions` is the list of bounding boxes(bbox) that wave been annotated. Each bbox element in this list comprises of the following attributes:
   -  `region_attributes` such as the annotated object `name`(Description), `type`(Class) and `image_quality`(is the object frontal? how is its quality? How is the illumination?).
   - `shape_attributes` that describe the shape `name`(rectangle, circle etc), `height`, `width`(for rectangles) and `x`, `y` coordinates of the centroid of annotation(rect bbox in our case).
 
 4. `size` of image in bytes.
