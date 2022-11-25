import os
import json
from pprint import pprint
import cv2
import numpy as np

def avg_img(image):
    image = np.asarray( image, dtype=np.float32)
    h, w, _ = image.shape
    h= int(h/2)
    w = int(w/2)
    pol_0 = image[:h,:w].astype(np.float32)
    pol_1 = image[:h,w:].astype(np.float32)
    pol_2 = image[h:,:w].astype(np.float32)
    pol_3 = image[h:,w:].astype(np.float32)
    image = (pol_0+pol_1+pol_2+pol_3)/4
    return image

def draw_rectangle(image, lst):
    for item in lst:
      x, y, w, h = item
      image = cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
    return image

def get_bounding_boxes(objects, mask_path):
    lst = []    
    for i, object_info in enumerate(objects):
        mask = cv2.imread(mask_path)
        super_threshold_indices = mask == i + 1
        mask[super_threshold_indices] = 255
        mask[~super_threshold_indices] = 0
        img = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        x,y,w,h = cv2.boundingRect(img)
        class_id = object_info['class_id']
        yolo_bb = [str(i) for i in coco_to_yolo(x, y, w, h, 640, 480)]
        lst.append('{} {}'.format(class_id, ' '.join(yolo_bb))) 
    return lst

def coco_to_yolo(x1, y1, w, h, image_w, image_h):
    return [((2*x1 + w)/(2*image_w)) , ((2*y1 + h)/(2*image_h)), w/image_w, h/image_h]

def yolo_to_coco(val1, val2, val3, val4, image_w, image_h):
    return [((2 * val1 * image_w) - (val3 * image_w)) / 2, ((2 * val2 * image_h) - (val4 * image_h)) / 2, val3 * image_w, val4 * image_h]

((2 * val2 * img_h)- (val4 * img_h)) / 2 = y1

def create_original_image(existing_data_path, sequence, file_name, data_type):
    training_image_path = f'../data_for_yolov7_training/images/{data_type}'
    image_path = f'{existing_data_path}/polarization/{file_name}'
    polarized_image = cv2.imread(image_path)
    average_image = avg_img(polarized_image)
    cv2.imwrite(f'{training_image_path}/{sequence}_{file_name}', average_image)

def create_label_file(existing_data_path, sequence, mask_file_name, label_file_name, objects, datatype):
    mask_path = f'{existing_data_path}/mask/{mask_file_name}'
    training_label_path = f'../data_for_yolov7_training/labels/{datatype}'
    bounding_boxes = get_bounding_boxes(objects, mask_path)
    label_file_path = f'{training_label_path}/{sequence}_{label_file_name}'
    with open(label_file_path, 'w') as f:
        for bounding_box in bounding_boxes:
            f.write('{}\n'.format(bounding_box))



global_mapper = {
    'train': ['', 34],
    'test': ['test_', 4],
    'val': ['val_', 2]
}


for mode, (file_prefix, seq_count) in global_mapper.items():
    for i in range(1, seq_count + 1):
        existing_data_path = f'../data/{file_prefix}sequence_{i}'
        gt_pose_path = f'{existing_data_path}/gt_pose.json'
        with open(gt_pose_path, 'r') as f:
            img_to_obj_mapper = json.load(f)
            for key, value in img_to_obj_mapper.items():
                image_file_name = "{}.png".format(key.zfill(6))
                label_file_name = "{}.txt".format(key.zfill(6))

                create_original_image(existing_data_path, i, image_file_name, mode)
                create_label_file(existing_data_path, i, image_file_name, label_file_name, value, mode)
            