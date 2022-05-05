import cv2
import os
from tqdm import tqdm

dataset_root = './data/SEG_Train_Datasets'
save_dataset_root = './data/SEG_Train_Datasets_Resize'
img_path = os.path.join(dataset_root, 'Train_Images')
mask_path = os.path.join(dataset_root, 'Train_Masks')
resize_path_images = os.path.join(save_dataset_root, 'Train_Images')
resize_path_masks = os.path.join(save_dataset_root, 'Train_Masks')

width = 720
height = 1440

os.makedirs(resize_path_images, exist_ok=True)
os.makedirs(resize_path_masks, exist_ok=True)

for image_id in tqdm(os.listdir(img_path)):
    image_id = image_id.split('.')[0]
    image = cv2.imread(os.path.join(img_path, f'{image_id}.jpg'))
    image = cv2.resize(image, (height, width), interpolation=cv2.INTER_LANCZOS4)

    mask = cv2.imread(os.path.join(mask_path, f'{image_id}.png'))
    mask = cv2.resize(mask, (height, width), interpolation=cv2.INTER_LANCZOS4)
    for n in range(3):
        cv2.imwrite(os.path.join(resize_path_images, f'{image_id}_{n}.png'), image[:,360*n:360*n+720,:])
        cv2.imwrite(os.path.join(resize_path_masks, f'{image_id}_{n}.png'), mask[:,360*n:360*n+720,:])