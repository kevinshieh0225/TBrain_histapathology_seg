import os, json, cv2
import numpy as np
from utils.config import load_setting

ds_cfg = load_setting()
dataset_root = ds_cfg['dataset_root']
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