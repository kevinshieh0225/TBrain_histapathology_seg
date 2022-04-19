import matplotlib.pyplot as plt
import os
from dataloader import SegmentationDataset

def plot(name, savedir, trainhistory, validhistory):
    plt.figure(figsize=(10,5))
    plt.plot(trainhistory, label = 'train')
    plt.plot(validhistory,  label = 'valid')
    plt.title(name)
    plt.xlabel("epochs")
    plt.savefig(savedir)

def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(50, 50))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()




if __name__ == "__main__":
    ENCODER = 'resnet50'
    ENCODER_WEIGHTS = 'imagenet'
    CLASSES = ['stas']
    ACTIVATION = 'sigmoid' 
    DEVICE = 'cuda'

    batchsize = 8
    dataset_root = './SEG_Train_Datasets'
    imagePaths = os.path.join(dataset_root, 'Train_Images')
    maskPaths = os.path.join(dataset_root, 'Train_Masks')
    dataset = SegmentationDataset(imagePaths, maskPaths, classes=['stas'])
    image, mask = dataset[4] # get some sample
    visualize(
        image=image, 
        stas_mask=mask.squeeze(),
    )