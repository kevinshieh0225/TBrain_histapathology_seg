import os, torch, cv2, json
import numpy as np
from tqdm import tqdm
import copy
from utils.dataloader import get_preprocessing
from utils.config import loadmodel, load_setting, searchnewname
from inference import connectTH

name = 'embedding'
hs_model_path = {
        'base_DL_plus_10fd3':1,
        'base_DL_plus_10fd8':1,
        'U+_nc_ef4ap_sDL_10fd6':1,
        'U+_nc_ef4ap_sDL_10fd8':1,
        'U+_nc_ef4ap_FTL_10fd4_soup5':1
        }
ls_model_path = {
        'base_small_plus_10fd0':1,
        'base_small_plus_10fd3':1,
        'base_small_plus_10fd4':1,
        'base_small_plus_10fd6':1,
        'base_small_plus_10fd8':1,
        }
ls_list = json.load(open('./trainlist/lowscale.json'))["low"]

height = 800
width = 1600

if __name__ == "__main__":
    ds_dict = load_setting()
    Public_Image = ds_dict['public_root']
    name = searchnewname(name, ds_dict['inference_root'])
    Public_save_path = os.path.join(ds_dict['inference_root'], name)
    os.makedirs(Public_save_path, exist_ok=True)
    with open(os.path.join(Public_save_path, 'model_path.json'), 'w') as fp:
        json.dump(hs_model_path, fp)
        fp.write('\n')
        json.dump(ls_model_path, fp)  
    hs_model_path = {os.path.join('./result', pth):hs_model_path[pth] for pth in hs_model_path}
    ls_model_path = {os.path.join('./result', pth):ls_model_path[pth] for pth in ls_model_path}

    modeldict = {
        'hs':{},
        'ls':{},
    }
    for name, model_path in [['hs',hs_model_path],['ls', ls_model_path]]:
        for pth in model_path:
            _, model_ = loadmodel(pth)
            model_.eval()
            modeldict[name][pth] = model_.to('cuda')

    preprocess = get_preprocessing()

    for image_id in tqdm(os.listdir(Public_Image)):
        image = cv2.imread(os.path.join(Public_Image, image_id))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        origin_h, origin_w, _ = image.shape
        if image.shape != (height, width, 3):
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LANCZOS4)
        image = preprocess(image=image)['image'].unsqueeze(0).to('cuda')
        mask = np.zeros((height,width))
        with torch.no_grad():
            if image_id.replace('.jpg', '') in ls_list:
                threshold = np.sum([ls_model_path[pth] for pth in ls_model_path]) * 0.75
                for pth in modeldict['ls']:
                    mask += torch.sigmoid(modeldict['ls'][pth](image)).squeeze().cpu().numpy() \
                        * ls_model_path[pth]
            else:
                threshold = np.sum([hs_model_path[pth] for pth in hs_model_path]) * 0.75
                for pth in modeldict['hs']:
                    mask += torch.sigmoid(modeldict['hs'][pth](image)).squeeze().cpu().numpy() \
                        * hs_model_path[pth]

        mask = cv2.resize(mask, (origin_w, origin_h), interpolation=cv2.INTER_LANCZOS4)
        mask = np.where(mask > threshold, 1, 0)
        mask_copy = copy.deepcopy(mask)
        connectTH(mask_copy, mask_copy, mode=1, threshold=150)
        if np.sum(mask_copy) > 150:
            mask = mask_copy
        connectTH(mask, mask^1, mode=0, threshold=50000)
        image_id = image_id.replace('jpg', 'png')
        cv2.imwrite(os.path.join(Public_save_path, image_id), mask*255)
