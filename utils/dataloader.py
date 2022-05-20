from torch.utils.data import Dataset, DataLoader
import os, torch
import cv2, json
import numpy as np
import albumentations as albu
from albumentations.pytorch.transforms import ToTensorV2

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
            width=1600,
            height=800,
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

def get_training_augmentation():
    return albu.Compose([
        albu.Flip(p=0.75),
        albu.ColorJitter(brightness=0.1, hue=0.15, p=0.8),
        albu.OneOf([
            albu.ShiftScaleRotate(rotate_limit=25, scale_limit=0.05),
            albu.ShiftScaleRotate(scale_limit=(-0.5, 0.2), shift_limit=0 ,rotate_limit=0),
        ], p=0.8),
        albu.ElasticTransform(p=0.5),
        albu.OneOf([
            albu.RandomBrightnessContrast(),
            albu.RandomGamma(),
        ], p=0.5),

        albu.Sharpen(lightness=(0.8, 1.2), p=0.4),
    ])

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
        ToTensorV2(transpose_mask=True),
        albu.Lambda(image=norm_scale),
    ]
    return albu.Compose(_transform)

def splitdataset(img_path, mask_path, opts_dict, ds_cfg):
    classes = opts_dict['classes']
    height = opts_dict['aug']['resize_height']
    width = height if opts_dict['iscrop'] else height * 2
    opts_dict['aug']['resize_width'] = width

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

    trainset = SegmentationDataset(xtrain, ytrain, classes, width, height, \
        augmentation=get_training_augmentation(), preprocessing=get_preprocessing())
    validset = SegmentationDataset(xvalid, yvalid, classes, width, height, \
        preprocessing=get_preprocessing())

    print(f'\ntrainset: {len(trainset)}\nvalidset: {len(validset)}\n')
    return trainset, validset

def create_trainloader(img_path, mask_path, opts_dict, ds_cfg):
    batch_size = opts_dict['batchsize']
    trainset, validset = splitdataset(img_path, mask_path, opts_dict, ds_cfg)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers = 4)
    validloader = DataLoader(validset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers = 4)
    return trainloader, validloader

