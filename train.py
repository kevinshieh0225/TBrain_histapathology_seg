import os
import torch
import segmentation_models_pytorch as smp

from dataloader import create_trainloader
from train_module import modeltrain

def main():
    ENCODER = 'resnet50'    # 'resnet50'
    ENCODER_WEIGHTS = 'imagenet'        # 'noisy-students' 'ssl'
    SAVEPATH = './result/unet_res50_imgnet'
    CLASSES = ['stas']
    ACTIVATION = 'sigmoid' 
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCHSIZE = 8
    EPOCHS = 40

    # model parameter
    model = smp.Unet(
                encoder_name=ENCODER,
                encoder_weights=ENCODER_WEIGHTS,
                classes=len(CLASSES),
                activation=ACTIVATION,
                )
    criterion = smp.utils.losses.DiceLoss()
    metrics = [
        smp.utils.metrics.Fscore(threshold=0.5),
    ]
    optimizer = torch.optim.Adam([ 
        dict(params=model.parameters(), lr=0.0001),
    ])

    # dataloader
    dataset_root = './SEG_Train_Datasets'
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    imagePaths = os.path.join(dataset_root, 'Train_Images')
    maskPaths = os.path.join(dataset_root, 'Train_Masks')
    trainloader, validloader = create_trainloader(
                                    imagePaths,
                                    maskPaths,
                                    CLASSES,
                                    preprocessing_fn,
                                    BATCHSIZE
                                )
    # training
    history = modeltrain(
                model=model,
                trainloader=trainloader,
                validloader=validloader,
                optimizer=optimizer,
                criterion=criterion,
                metrics=metrics,
                epochs=EPOCHS,
                save_path=SAVEPATH,
                device=DEVICE,
                )

if __name__ == "__main__":
    main()