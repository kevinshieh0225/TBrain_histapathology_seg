import os, torch, cv2
import numpy as np
from scipy import ndimage
from tqdm import tqdm
from utils.dataloader import get_preprocessing
from utils.config import load_setting, loadmodel

height = 800
width = 1600
THRESHOLD = 0.75
pretrain_path_list = [
        'base_DL_plus_10fd0',
        # 'base_DL_plus_10fd3',
        # 'U+_nc_ef4ap_sDL_10fd3',
        # 'U+_nc_ef4ap_sDL_10fd7',
        # 'U+_nc_ef4ap_FTL_10fd4_soup5',
        ]
pretrain_path_list = [os.path.join('./result', pth) for pth in pretrain_path_list]
device = 'cuda' # cpu cuda
load_last = False

def connectTH(mask, map, mode=1, threshold=150):
    # identify pixel connected size
    pgroup, Nlabels = ndimage.measurements.label(map)
    label_size = [(pgroup == label).sum() for label in range(Nlabels + 1)]
    # remove those above a threshold
    mode ^= 1
    for label,size in enumerate(label_size):
        if size < threshold:
            mask[pgroup == label] = mode

if __name__ == "__main__":
    
    for pretrain_path in pretrain_path_list:
        opts_dict, model = loadmodel(pretrain_path, load_last)
        model.eval()
        model.to(device)
        
        ds_dict = load_setting()
        Public_Image = ds_dict['public_root']
        if load_last == True:
            Public_save_path = os.path.join(ds_dict['inference_root'], opts_dict['expname']+'_last')
        else:
            Public_save_path = os.path.join(ds_dict['inference_root'], opts_dict['expname'])
        os.makedirs(Public_save_path, exist_ok=True)

        preprocess = get_preprocessing()
        
        for image_id in tqdm(os.listdir(Public_Image)):
            image = cv2.imread(os.path.join(Public_Image, image_id))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            origin_h, origin_w, _ = image.shape
            if image.shape != (height, width, 3):
                image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LANCZOS4)
            image = preprocess(image=image)['image']
            image = image.unsqueeze(0).to(device)
            with torch.no_grad():
                mask = torch.sigmoid(model(image)).squeeze().cpu().numpy()
            mask = cv2.resize(mask, (origin_w, origin_h), interpolation=cv2.INTER_LANCZOS4)
            mask = np.where(mask > THRESHOLD, 1, 0)
            connectTH(mask, mask^1, mode=0, threshold=50000)
            image_id = image_id.replace('jpg', 'png')
            cv2.imwrite(os.path.join(Public_save_path, image_id), mask*255)
