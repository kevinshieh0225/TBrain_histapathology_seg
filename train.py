import os

from dataloader import create_trainloader
from trainmodule import modeltrain

from config import wandb_config

def main():
    project = 'TBrain_histapathology_segmentation'
    name = 'dlp_efn5_apex_b32_AdamW'
    opts_dict, wandb_logger = wandb_config(project, name, cfg='cfg/wandbcfg.yaml')

    # dataloader
    dataset_root = './SEG_Train_Datasets'
    imagePaths = os.path.join(dataset_root, 'Train_Images')
    maskPaths = os.path.join(dataset_root, 'Train_Masks')
    trainloader, validloader = create_trainloader(
                                imagePaths,
                                maskPaths,
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