from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
import os
import cv2
import numpy as np
import albumentations as albu
import segmentation_models_pytorch as smp
from config import receive_arg

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


def get_training_augmentation():
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),        
        albu.HueSaturationValue(p=0.6),
        albu.Sharpen(p=0.5),
        albu.RandomBrightnessContrast(p=0.4),
    ]
    return albu.Compose(train_transform)

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)

def splitdataset(img_path, mask_path, classes, width, height, preprocessing_fn):
    imagePaths = [os.path.join(img_path, image_id) for image_id in os.listdir(img_path)]
    maskPaths = [os.path.join(mask_path, image_id).replace("jpg", "png") for image_id in os.listdir(img_path)]

    xtrain, xvalid, ytrain, yvalid = train_test_split(imagePaths, maskPaths, test_size=0.2, random_state=42)

    trainset = SegmentationDataset(xtrain, ytrain, classes, width, height, augmentation=get_training_augmentation(), preprocessing=get_preprocessing(preprocessing_fn))
    validset = SegmentationDataset(xvalid, yvalid, classes, width, height, preprocessing=get_preprocessing(preprocessing_fn))

    print(f'trainset: {len(trainset)}\nvalidset: {len(validset)}\n')
    return trainset, validset

def create_trainloader(img_path, mask_path, preprocessing_fn, opts_dict):
    classes = opts_dict['classes']
    batch_size = opts_dict['batchsize']
    height = opts_dict['aug']['resize_height']
    width = height * 2
    trainset, validset = splitdataset(img_path, mask_path, classes, width, height, preprocessing_fn)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers = os.cpu_count())
    validloader = DataLoader(validset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers = os.cpu_count())
    return trainloader, validloader

if __name__ == "__main__":
    opts_dict = receive_arg()

    batchsize = 8
    dataset_root = './SEG_Train_Datasets'
    preprocessing_fn = smp.encoders.get_preprocessing_fn(
                        opts_dict['model']['encoder_name'],
                        opts_dict['model']['encoder_weights']
                    )
    imagePaths = os.path.join(dataset_root, 'Train_Images')
    maskPaths = os.path.join(dataset_root, 'Train_Masks')
    trainloader, validloader = create_trainloader(
                                    imagePaths,
                                    maskPaths,
                                    preprocessing_fn,
                                    opts_dict,
                                )