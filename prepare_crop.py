import cv2
import os
from tqdm import tqdm
from config import load_dataset_root

WIDTH = 800
HEIGHT = 1600

def trainset_crop(dataset_root, save_dataset_root):
    img_path = os.path.join(dataset_root, 'Train_Images')
    mask_path = os.path.join(dataset_root, 'Train_Masks')
    resize_path_images = os.path.join(save_dataset_root, 'Train_Images')
    resize_path_masks = os.path.join(save_dataset_root, 'Train_Masks')

    os.makedirs(resize_path_images, exist_ok=True)
    os.makedirs(resize_path_masks, exist_ok=True)

    for image_id in tqdm(os.listdir(img_path)):
        image_id = image_id.split('.')[0]
        image = cv2.imread(os.path.join(img_path, f'{image_id}.jpg'))
        image = cv2.resize(image, (HEIGHT, WIDTH), interpolation=cv2.INTER_LANCZOS4)

        mask = cv2.imread(os.path.join(mask_path, f'{image_id}.png'))
        mask = cv2.resize(mask, (HEIGHT, WIDTH), interpolation=cv2.INTER_LANCZOS4)
        block = int(HEIGHT/4)
        for n in range(3):
            start = block * n
            end = block * (n + 2)
            cv2.imwrite(os.path.join(resize_path_images, f'{image_id}_{n}.png'), image[:,start:end,:])
            cv2.imwrite(os.path.join(resize_path_masks, f'{image_id}_{n}.png'), mask[:,start:end,:])

if __name__ == '__main__':
    root = load_dataset_root()['root']
    dataset_root = os.path.join(root, 'SEG_Train_Datasets')
    save_dataset_root = os.path.join(root, 'SEG_Train_Datasets_Resize')
    trainset_crop(dataset_root, save_dataset_root)
