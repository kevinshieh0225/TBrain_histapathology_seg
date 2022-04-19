import os
import json
import numpy as np
import cv2

dataset_root = './SEG_Train_Datasets'
anno_path = os.path.join(dataset_root, 'Train_Annotations')

mask_path = os.path.join(dataset_root, 'Train_Masks')
os.makedirs(os.path.join(dataset_root, 'Train_Masks'), exist_ok=True)
for jsonfile in os.listdir(anno_path):
    f = open(os.path.join(anno_path, jsonfile))
    data = json.load(f)
    mask = np.zeros((data['imageHeight'], data['imageWidth'], 1), dtype=np.uint8)
    for polygan in data['shapes']:
        pts = np.array(polygan['points'], dtype=np.int32)
        cv2.fillPoly(mask, [pts], color=255)
    save_mask_path = jsonfile.split('.')[0] + '.png'
    save_mask_path = os.path.join(mask_path, save_mask_path)
    cv2.imwrite(save_mask_path, mask)