import os, torch, cv2
from plot import visualize
from dataloader import get_preprocessing
import segmentation_models_pytorch as smp

from config import load_wdb_config
from network import Litsmp
from tqdm import tqdm

def inference(model, image, device):
    x_tensor = torch.from_numpy(image).to(device).unsqueeze(0)
    mask = model.predict(x_tensor)
    return mask.squeeze().cpu().numpy().round()

def inference_dataloader(
        model,
        test_dataset,
        device,
        ):
    # for i in range(len(test_dataset)):
    for i in range(10):
        name = os.path.basename(test_dataset.masks_fps[i])
        print(name)
        image_vis = test_dataset[i][0].transpose(1, 2, 0)
        image, gt_mask = test_dataset[i]
        pr_mask = inference(model, image, device)
        visualize( 
            image=image_vis, 
            ground_truth_mask=gt_mask.transpose(1, 2, 0)[...,0], 
            predicted_mask=pr_mask
            )



if __name__ == "__main__":
    pretrain_path = './result/dlp_efn5_apex_b32_AdamW/'
    cfgpath = os.path.join(pretrain_path, 'expconfig.yaml')
    weight = os.path.join(pretrain_path, 'epoch=184-step=9805.ckpt')
    inference_path = '/work/u7085556/TBrain_histapathology_seg'

    Public_Image = os.path.join(inference_path, 'Public_Image')
    opts_dict = load_wdb_config(cfgpath, inference_path)
    Public_save_path = os.path.join(inference_path, opts_dict['expname'])
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
            mask = model(image).squeeze().cpu().numpy().round()
        mask = cv2.resize(mask, (origin_w, origin_h), interpolation=cv2.INTER_LANCZOS4)
        mask *= 255
        mask = mask.astype(int)
        image_id = image_id.replace('jpg', 'png')
        cv2.imwrite(os.path.join(Public_save_path, image_id), mask)
