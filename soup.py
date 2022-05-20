from operator import truediv
import os, torch, cv2
import numpy as np
from scipy import ndimage
from tqdm import tqdm
from utils.dataloader import get_preprocessing
from utils.config import load_setting, loadmodel
import copy

THRESHOLD = 0.5
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
    # make sure all selected models configs are the same
    load_last = False # use last epoch or best f1 score
    device = 'cuda:0' # cpu
    path_list = ['./result/U+_nc_ef4ap_FTL_soup_0',
                './result/U+_nc_ef4ap_FTL_soup_1',
                # './result/U+_nc_ef4ap_FTL_10fd3',
                # './result/U+_nc_ef4ap_FTL_10fd4',
                # './result/U+_nc_ef4ap_FTL_10fd5',
                # './result/U+_nc_ef4ap_FTL_10fd6',
                ]


    model_weights = []
    for ckp_path in path_list:
        opts_dict, model = loadmodel(ckp_path, load_last)
        model_weights.append(model.state_dict())

    # Cooking the soup
    soup_model = model #use last model as container

    for key in model_weights[0]:
        for idx, state_dict in enumerate(model_weights):
            if idx == 0:
                average = state_dict[key]
            else:
                average += state_dict[key]
        average = average/ float(len(model_weights))
        model_weights[0][key] = average

    soup_model.load_state_dict(model_weights[0])
    soup_model.eval()
    soup_model.to(device)

    ds_dict = load_setting()
    Public_Image = ds_dict['public_root']
    Public_save_path = os.path.join(ds_dict['inference_root'], opts_dict['expname']+'_soup')
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
            image = image.to(device)
            with torch.no_grad():
                mask = torch.sigmoid(soup_model(image)).squeeze().cpu().numpy()
            mask = cv2.resize(mask, (origin_w, origin_h), interpolation=cv2.INTER_LANCZOS4)
            mask = np.where(mask > THRESHOLD, 1, 0)
            mask_copy = copy.deepcopy(mask)
            connectTH(mask_copy, mask_copy, mode=1, threshold=400)
            if np.sum(mask_copy) > 400:
                mask = mask_copy
            connectTH(mask, mask^1, mode=0, threshold=50000)
            image_id = image_id.replace('jpg', 'png')
            cv2.imwrite(os.path.join(Public_save_path, image_id), mask*255)
    else: 
        pass