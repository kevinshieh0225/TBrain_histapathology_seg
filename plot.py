import matplotlib.pyplot as plt
import os
from dataloader import splitdataset
from config import load_wdb_config

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
    opts_dict = load_wdb_config()
    dataset_root = './SEG_Train_Datasets'
    imagePaths = os.path.join(dataset_root, 'Train_Images')
    maskPaths = os.path.join(dataset_root, 'Train_Masks')
    trainset, validset = splitdataset(imagePaths, maskPaths, opts_dict)
    image, mask = validset[4] # get some sample
    image = image.numpy().transpose(1, 2, 0)
    visualize(
        image=image, 
        stas_mask=mask.squeeze(),
    )