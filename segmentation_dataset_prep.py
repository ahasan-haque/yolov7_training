import os
import json
from pprint import pprint
import cv2
import numpy as np
from imantics import Polygons, Mask, Annotation, Image

def avg_img(image):
    image = np.asarray( image, dtype=np.uint8)
    h, w, _ = image.shape
    h= int(h/2)
    w = int(w/2)
    pol_0 = image[:h,:w].astype(np.uint8)
    pol_1 = image[:h,w:].astype(np.uint8)
    pol_2 = image[h:,:w].astype(np.uint8)
    pol_3 = image[h:,w:].astype(np.uint8)
    image = (pol_0+pol_1+pol_2+pol_3)/4
    return image

image_path = 'sequence_1/polarization/000000.png'
#mask_path = 'sequence_1/mask/000000.png'
mask_path = "/Users/ahasan/Downloads/1_000000.png"
path = "/Users/ahasan/Downloads/data-for-yolov7-training/images/test/3_000000.png"

"""
polarized_image = cv2.imread(image_path)
image = avg_img(polarized_image)
mask = cv2.imread(mask_path)
super_threshold_indices = mask == 4
mask[super_threshold_indices] = 255
mask[~super_threshold_indices] = 0
mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
polygons = Mask(mask).polygons()
mask = Mask(mask)
"""
mask = cv2.imread(path)
print(mask.shape)

segmentation = [
    [
        239.97,  260.24,  222.04,  270.49,  199.84,  253.41,  213.5,  227.79,  259.62,  200.46,  274.13,  202.17,  277.55,  210.71,  249.37,  253.41,  237.41,  264.51,  242.54,  261.95,  228.87,  271.34
    ]
]
print(sum([j for i in segmentation for j in i]))
"""
print(Annotation(mask=mask, polygons=polygons).area)
#print(polygons.points)
print(polygons.segmentation)
"""
print(Annotation.from_polygons(segmentation, image=Image(mask)).area)