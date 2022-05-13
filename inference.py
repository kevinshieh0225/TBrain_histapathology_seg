import os, torch, cv2
import numpy as np
from scipy import ndimage
from tqdm import tqdm
from utils.dataloader import get_preprocessing
from utils.config import load_wdb_config, load_setting
from utils.network import Litsmp

THRESHOLD = 0.75

def modelsetting(pretrain_path):
    for pth in os.listdir(pretrain_path):
        if pth.find('.ckpt') != -1:
            weight = os.path.join(pretrain_path, pth)
            break
    checkpoint_dict = torch.load(weight)
    if 'hyper_parameters' in checkpoint_dict:
        opts_dict = checkpoint_dict['hyper_parameters']
        model = Litsmp.load_from_checkpoint(weight)
    else:
        cfgpath = os.path.join(pretrain_path, 'expconfig.yaml')
        opts_dict = load_wdb_config(cfgpath)
        model = Litsmp.load_from_checkpoint(weight, opts_dict=opts_dict)
    
    return opts_dict, model

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
    pretrain_path = './result/U+_nc_moreaug_FTL/'

    opts_dict, model = modelsetting(pretrain_path)
    model.eval()

    ds_dict = load_setting()
    Public_Image = ds_dict['public_root']
    Public_save_path = os.path.join(ds_dict['inference_root'], opts_dict['expname'])
    os.makedirs(Public_save_path, exist_ok=True)

    height = opts_dict['aug']['resize_height']
    width = height if opts_dict['iscrop'] else 2 * height
    preprocess = get_preprocessing()
    
    if opts_dict['iscrop'] == 0:
        for image_id in tqdm(os.listdir(Public_Image)):
            image = cv2.imread(os.path.join(Public_Image, image_id))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            origin_h, origin_w, _ = image.shape
            if image.shape != (height, width, 3):
                image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LANCZOS4)
            image = preprocess(image=image)['image']
            image = image.unsqueeze(0)
            with torch.no_grad():
                mask = torch.sigmoid(model(image)).squeeze().cpu().numpy()
            mask = cv2.resize(mask, (origin_w, origin_h), interpolation=cv2.INTER_LANCZOS4)
            mask = np.where(mask > THRESHOLD, 1, 0)
            connectTH(mask, mask, mode=1, threshold=420)
            connectTH(mask, mask^1, mode=0, threshold=10000)
            image_id = image_id.replace('jpg', 'png')
            cv2.imwrite(os.path.join(Public_save_path, image_id), mask*255)

    elif opts_dict['iscrop'] == 1:
        cropPath = ds_dict['crop_public_root']
        block = int(width/4)
        for image_id in tqdm(os.listdir(Public_Image)):
            for n in range(3):
                start = block * (1 if n > 0 else 0)
                end = block * (4 - (1 if n != 2 else 0))
                imgPath = image_id.split('.')[0]
                image = cv2.imread(os.path.join(cropPath, f'{imgPath}_{n}.png'))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                origin_image = cv2.imread(os.path.join(Public_Image, image_id))
                origin_h, origin_w, _ = origin_image.shape
                if image.shape != (height, width, 3):
                    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LANCZOS4)
                image = preprocess(image=image)['image']
                image = image.unsqueeze(0)
                with torch.no_grad():
                    mask = torch.sigmoid(model(image)).squeeze().cpu().numpy()
                if n == 0:
                    cat_mask = mask[:,start:end]
                else:
                    cat_mask = cv2.hconcat([cat_mask, mask[:,start:end]])

            mask = cv2.resize(cat_mask, (origin_w, origin_h), interpolation=cv2.INTER_LANCZOS4)
            mask = np.where(mask > THRESHOLD, 1, 0) * 255
            image_id = image_id.replace('jpg', 'png')
            cv2.imwrite(os.path.join(Public_save_path, image_id), mask)
