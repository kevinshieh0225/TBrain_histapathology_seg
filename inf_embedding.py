import os, torch, cv2, csv
import numpy as np
from tqdm import tqdm
from utils.dataloader import get_preprocessing
from utils.config import loadmodel, load_setting
from inference import connectTH


THRESHOLD = 0.75
model_path = [
        'U+_nc_moreaug_FTL_fd0',
        'U+_nc_moreaug_FTL_fd1',
        'U+_nc_efb4_noisy_fd0',
        'base_noncrop_list_bigimg',
        'U+_nc_moreaug_FTL',
        ]
height = 800
width = 1600

if __name__ == "__main__":
    ds_dict = load_setting()
    Public_Image = ds_dict['public_root']
    Public_save_path = os.path.join(ds_dict['inference_root'], '5-fold')
    os.makedirs(Public_save_path, exist_ok=True)
    with open(os.path.join(Public_save_path, 'model_path.csv'), 'w') as f:
        write = csv.writer(f)
        write.writerow(model_path)
    model_path = ['./result/' + pth for pth in model_path]
    imagePaths = [image_id for image_id in os.listdir(Public_Image)]

    model = []
    for pth in model_path:
        _, model_ = loadmodel(pth)
        model_.eval()
        model.append(model_)

    preprocess = get_preprocessing()

    for image_id in tqdm(imagePaths):
        image = cv2.imread(os.path.join(Public_Image, image_id))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        origin_h, origin_w, _ = image.shape
        if image.shape != (height, width, 3):
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LANCZOS4)
        image = preprocess(image=image)['image'].unsqueeze(0)
        mask = np.zeros((800,1600))
        for model_ in model:
            with torch.no_grad():
                mask += torch.sigmoid(model_(image)).squeeze().cpu().numpy()
        mask /= len(model)
        mask = cv2.resize(mask, (origin_w, origin_h), interpolation=cv2.INTER_LANCZOS4)
        mask = np.where(mask > THRESHOLD, 1, 0) * 255
        connectTH(mask, mask, mode=1, threshold=420)
        connectTH(mask, mask^1, mode=0, threshold=50000)
        image_id = image_id.replace('jpg', 'png')
        cv2.imwrite(os.path.join(Public_save_path, image_id), mask)
