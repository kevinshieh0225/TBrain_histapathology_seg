import os
import segmentation_models_pytorch as smp

from dataloader import create_trainloader
from trainmodule import modeltrain
from network import Litsmp

from config import receive_arg

def main():
    opts_dict = receive_arg()

    # model parameter
    model = Litsmp(opts_dict)

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
        model=model,
        trainloader=trainloader,
        validloader=validloader,
        opts_dict=opts_dict,
        )

if __name__ == "__main__":
    main()