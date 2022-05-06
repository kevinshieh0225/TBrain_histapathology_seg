import os
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from network import Litsmp
from pytorch_lightning.strategies import DDPStrategy
from dataloader import create_trainloader
from config import wandb_config, load_setting

def main():
    cfg = load_setting()
    project, name = cfg['project'], cfg['name']
    opts_dict, wandb_logger = wandb_config(project, name, cfg='cfg/wandbcfg.yaml')

    # dataloader
    dataset_root = cfg['dataset_root']
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

def modeltrain(
        trainloader,
        validloader,
        wandb_logger,
        opts_dict
        ):
    epochs = opts_dict['epochs']
    save_path = opts_dict['savepath']

    # checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=save_path,
        save_top_k=1,
        monitor="valid loss",
        mode="min",
        )
    lr_monitor = LearningRateMonitor()
    
    # model parameter
    model = Litsmp(opts_dict)
    trainer = Trainer(
        logger=wandb_logger,
        max_epochs=epochs,
        callbacks=[checkpoint_callback, lr_monitor],
        gpus=-1,
        accumulate_grad_batches=opts_dict['accumulate_grad_batches'],
        # amp_backend="apex",
        # amp_level='01',
        strategy=DDPStrategy(find_unused_parameters=True),
        )
    
    trainer.fit(
        model=model,
        train_dataloaders=trainloader,
        val_dataloaders=validloader,
        )

if __name__ == "__main__":
    main()