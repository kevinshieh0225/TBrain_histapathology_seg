import os
import segmentation_models_pytorch as smp

from dataloader import create_trainloader
from trainmodule import modeltrain
from network import Litsmp

def main():
    ENCODER = 'resnet50'    # 'resnet50'
    ENCODER_WEIGHTS = 'imagenet'        # 'noisy-students' 'ssl'
    EXPNAME = 'AICUP_dlv3plus_res50'
    SAVEPATH = './result/dlv3plus_res50'
    CLASSES = ['stas']
    ACTIVATION = 'sigmoid' 
    BATCHSIZE = 8
    EPOCHS = 5

    # model parameter
    model = Litsmp(
            smp.DeepLabV3Plus(
                encoder_name=ENCODER,
                encoder_weights=ENCODER_WEIGHTS,
                classes=len(CLASSES),
                activation=ACTIVATION,
            )
        )

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
    modeltrain(
        model=model,
        trainloader=trainloader,
        validloader=validloader,
        expname=EXPNAME,
        epochs=EPOCHS,
        save_path=SAVEPATH,
        )

if __name__ == "__main__":
    main()