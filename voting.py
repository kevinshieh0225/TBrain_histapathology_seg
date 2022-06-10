import os, cv2, json
import numpy as np
from tqdm import tqdm
from utils.config import load_setting, searchnewname
from inference import connectTH
import copy

name = 'voting'
hs_model_path = {
        # public best combinations
        # 'base_DL_plus_10fd3':1,
        # 'base_DL_plus_10fd8':1,
        # 'U+_nc_ef4ap_sDL_10fd6':1,
        # 'U+_nc_ef4ap_sDL_10fd8':1,
        # 'U+_nc_ef4ap_FTL_10fd4_soup5':1,
        }
ls_model_path = {
        'base_small_plus_10fd0':1,
        'base_small_plus_10fd3':1,
        'base_small_plus_10fd4':1,
        'base_small_plus_10fd6':1,
        'base_small_plus_10fd8':1,
        }
ms_list = json.load(open('./trainlist/lowscale.json'))["low"]
height = 942
width = 1716

if __name__ == "__main__":
    ds_dict = load_setting()
    Public_Image = ds_dict['public_root']
    name = searchnewname(name, ds_dict['inference_root'])
    print(f'Start {name}')
    Public_save_path = os.path.join(ds_dict['inference_root'], name)
    os.makedirs(Public_save_path, exist_ok=True)
    with open(os.path.join(Public_save_path, 'model_path.json'), 'w') as fp:
        json.dump(hs_model_path, fp)
        fp.write('\n')
        json.dump(ls_model_path, fp)        
    hs_model_path = {os.path.join(ds_dict['inference_root'], pth):hs_model_path[pth] for pth in hs_model_path}
    ls_model_path = {os.path.join(ds_dict['inference_root'], pth):ls_model_path[pth] for pth in ls_model_path}
    
    for image_id in tqdm(os.listdir(Public_Image)):
        model_path = ls_model_path if image_id.replace('.jpg', '') in ms_list else hs_model_path
        image_id = image_id.replace('jpg', 'png')
        mask = np.zeros((height,width))
        vote = 0
        for infpth in model_path:
            infmask = cv2.imread(os.path.join(infpth, image_id), 0)
            infmask = np.where(infmask != 0, 1, 0)
            rate = model_path[infpth] if np.sum(infmask) > 150 else 0 # model_path[infpth]/(len(model_path)-1)
            mask += (infmask * rate)
            vote += rate
        mask = np.where(mask > vote/2, 1, 0)
        mask_copy = copy.deepcopy(mask)
        connectTH(mask, mask^1, mode=0, threshold=50000)
        cv2.imwrite(os.path.join(Public_save_path, image_id), mask*255)
