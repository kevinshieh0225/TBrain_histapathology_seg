import os, torch, cv2
from utils.dataloader import get_preprocessing
import numpy as np
from utils.config import load_wdb_config, load_setting
from utils.network import Litsmp
from tqdm import tqdm

THRESHOLD = 0.75

if __name__ == "__main__":
    pretrain_path = './result/U+_nc_moreaug_FTL_fd0/'
    cfgpath = os.path.join(pretrain_path, 'expconfig.yaml')
    weight = os.path.join(pretrain_path, 'epoch=68-step=4899.ckpt')
    ds_dict = load_setting()

    Public_Image = ds_dict['public_root']
    opts_dict = load_wdb_config(cfgpath)
    Public_save_path = os.path.join(ds_dict['inference_root'], opts_dict['expname'])
    os.makedirs(Public_save_path, exist_ok=True)

    preprocess = get_preprocessing()

    model = Litsmp.load_from_checkpoint(weight, opts_dict=opts_dict)
    model.eval()
    height = opts_dict['aug']['resize_height']
    width = height * 2
    imagePaths = [image_id for image_id in os.listdir(Public_Image)]
    for image_id in tqdm(imagePaths):
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
        mask = np.where(mask > THRESHOLD, 1, 0) * 255
        image_id = image_id.replace('jpg', 'png')
        cv2.imwrite(os.path.join(Public_save_path, image_id), mask)
