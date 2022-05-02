import os
import segmentation_models_pytorch as smp

from dataloader import create_trainloader
from trainmodule import modeltrain

from config import wandb_config

def main():
    project = 'TBrain_histapathology_segmentation'
    name = 'Unet_efnb4'
    opts_dict, wandb_logger = wandb_config(project, name, cfg='cfg/wandbcfg.yaml')

    # dataloader
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
    # training
    modeltrain(
        trainloader=trainloader,
        validloader=validloader,
        wandb_logger=wandb_logger,
        opts_dict=opts_dict,
        )

if __name__ == "__main__":
    main()