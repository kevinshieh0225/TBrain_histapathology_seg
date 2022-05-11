import matplotlib.pyplot as plt

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