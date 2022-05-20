import os, cv2, json
import numpy as np
from tqdm import tqdm
from utils.config import load_setting, searchnewname
from inference import connectTH
import copy

name = 'voting'
model_path = {
        'U+_nc_ef4ap_FTL_10fd0':1,
        'U+_nc_ef4ap_FTL_10fd5':1,
        'U+_nc_ef4ap_FTL_10fd4':1,
        'U+_nc_ef4ap_FTL_10fd6':1,
        'U+_nc_ef4ap_FTL_10fd7':1,
        }
height = 942
width = 1716

if __name__ == "__main__":
    ds_dict = load_setting()
    Public_Image = ds_dict['public_root']
    name = searchnewname(name, ds_dict['inference_root'])
    Public_save_path = os.path.join(ds_dict['inference_root'], name)
    os.makedirs(Public_save_path, exist_ok=True)
    with open(os.path.join(Public_save_path, 'model_path.json'), 'w') as fp:
        json.dump(model_path, fp)
    model_path = {os.path.join(ds_dict['inference_root'], pth):model_path[pth] for pth in model_path}
    
    for image_id in tqdm(os.listdir(Public_Image)):
        image_id = image_id.replace('jpg', 'png')
        mask = np.zeros((height,width))
        vote = 0
        for infpth in model_path:
            infmask = cv2.imread(os.path.join(infpth, image_id), 0)
            infmask = np.where(infmask != 0, 1, 0)
            if np.sum(infmask) > 350:
                mask += (infmask * model_path[infpth])
                vote += model_path[infpth]
        mask = np.where(mask > vote/2, 1, 0)
        mask_copy = copy.deepcopy(mask)
        connectTH(mask_copy, mask_copy, mode=1, threshold=400)
        if np.sum(mask_copy) > 400:
            mask = mask_copy
        connectTH(mask, mask^1, mode=0, threshold=50000)
        cv2.imwrite(os.path.join(Public_save_path, image_id), mask*255)
