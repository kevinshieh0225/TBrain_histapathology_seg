from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import os, torch
import cv2, json
import numpy as np
import albumentations as albu
from albumentations.pytorch.transforms import ToTensorV2
import wandb
from config import unflatten_json, load_setting

class SegmentationDataset(Dataset):
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    """
    
    CLASSES = ['background', 'stas']
    
    def __init__(
            self, 
            imageslist, 
            maskslist, 
            classes=None, 
            width=1280,
            height=640,
            augmentation=None, 
            preprocessing=None,
    ):
        self.images_fps = imageslist
        self.masks_fps = maskslist
        
        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        
        self.w = width
        self.h = height

        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        
        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if image.shape != (self.h, self.w, 3):
            image = cv2.resize(image, (self.w, self.h), interpolation=cv2.INTER_LANCZOS4)

        mask = cv2.imread(self.masks_fps[i], 0)
        if mask.shape != (self.h, self.w, 3):
            mask = cv2.resize(mask, (self.w, self.h), interpolation=cv2.INTER_LANCZOS4)  
        mask = mask.astype('bool')
        
        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return image, mask
        
    def __len__(self):
        return len(self.masks_fps)

#   origin
#   albu.HorizontalFlip(p=0.5),
#   albu.VerticalFlip(p=0.5),        
#   albu.HueSaturationValue(p=0.6),
#   albu.Sharpen(p=0.5),
#   albu.RandomBrightnessContrast(p=0.4),
def get_training_augmentation():
    train_transform = [
        albu.Flip(p=0.7),
        albu.ColorJitter(brightness=0.1, hue=0.1, p=0.8),
        # albu.HueSaturationValue(val_shift_limit=15, p=0.8),
        albu.Sharpen(lightness=(0.9, 1.1), p=0.5),
        # albu.Affine()
    ]
    return albu.Compose(train_transform)

def get_preprocessing():
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    def norm_scale(x, **kwargs):
        return x.to(torch.float32) / 255.0

    _transform = [
        # Normalize(**norm),
        # Normalize(),
        # albu.CLAHE(always_apply=True),
        ToTensorV2(transpose_mask=True),
        albu.Lambda(image=norm_scale),
    ]
    return albu.Compose(_transform)

def splitdataset(img_path, mask_path, opts_dict, ds_cfg):
    classes = opts_dict['classes']
    height = opts_dict['aug']['resize_height']
    width = height * 2 if opts_dict['iscrop'] == 1 else height

    if(opts_dict['readlist'] == 1):
        train_test_list = json.load(open(ds_cfg['train_valid_list']))
        trainlist = train_test_list['train']
        validlist = train_test_list['valid']
        if(opts_dict['iscrop'] == 1):
            trainlist = [f'{image_id}_0' for image_id in trainlist] + \
                        [f'{image_id}_1' for image_id in trainlist] + \
                        [f'{image_id}_2' for image_id in trainlist]
            validlist = [f'{image_id}_0' for image_id in validlist] + \
                        [f'{image_id}_1' for image_id in validlist] + \
                        [f'{image_id}_2' for image_id in validlist]
            xtrain = [os.path.join(img_path, f'{image_id}.png') for image_id in trainlist]
            xvalid = [os.path.join(img_path, f'{image_id}.png') for image_id in validlist]
        else:
            xtrain = [os.path.join(img_path, f'{image_id}.jpg') for image_id in trainlist]
            xvalid = [os.path.join(img_path, f'{image_id}.jpg') for image_id in validlist]

        ytrain = [os.path.join(mask_path, f'{image_id}.png') for image_id in trainlist]
        yvalid = [os.path.join(mask_path, f'{image_id}.png') for image_id in validlist]

    elif(opts_dict['iscrop'] != 1):
        imagePaths = [os.path.join(img_path, image_id) for image_id in os.listdir(img_path)]
        maskPaths = [os.path.join(mask_path, image_id).replace("jpg", "png") for image_id in os.listdir(img_path)]
        xtrain, xvalid, ytrain, yvalid = train_test_split(imagePaths, maskPaths, test_size=0.2, random_state=42)
    else:
        IDPaths , xtrain, xvalid, ytrain, yvalid = [], [], [], [], []
        for image_id in os.listdir(img_path):
            id = image_id.split('_')[0]
            if id not in IDPaths:
                IDPaths.append(id)
        idtrain, idvalid = train_test_split(IDPaths, test_size=0.2, random_state=42)
        for id in idtrain:
            for n in range(3):
                xtrain.append(os.path.join(img_path, f'{id}_{n}.png'))
                ytrain.append(os.path.join(mask_path, f'{id}_{n}.png'))
        for id in idvalid:
            for n in range(3):
                xvalid.append(os.path.join(img_path, f'{id}_{n}.png'))
                yvalid.append(os.path.join(mask_path, f'{id}_{n}.png'))


    trainset = SegmentationDataset(xtrain, ytrain, classes, width, height, augmentation=get_training_augmentation(), preprocessing=get_preprocessing())
    validset = SegmentationDataset(xvalid, yvalid, classes, width, height, preprocessing=get_preprocessing())

    print(f'\ntrainset: {len(trainset)}\nvalidset: {len(validset)}\n')
    return trainset, validset

def create_trainloader(img_path, mask_path, opts_dict, ds_cfg):
    batch_size = opts_dict['batchsize']
    trainset, validset = splitdataset(img_path, mask_path, opts_dict, ds_cfg)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers = 4)
    validloader = DataLoader(validset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers = 4)
    return trainloader, validloader

if __name__ == "__main__":
    wandb.init(config='cfg/wandbcfg.yaml', mode="disabled")
    opts_dict = wandb.config.as_dict()
    unflatten_json(opts_dict)
    ds_cfg = load_setting()
    dataset_root = ds_cfg['crop_dataset_root'] if opts_dict['iscrop'] == 1 else ds_cfg['dataset_root']
    imagePaths = os.path.join(dataset_root, 'Train_Images')
    maskPaths = os.path.join(dataset_root, 'Train_Masks')
    trainloader, validloader = create_trainloader(
                                    imagePaths,
                                    maskPaths,
                                    opts_dict,
                                    ds_cfg
                                )
