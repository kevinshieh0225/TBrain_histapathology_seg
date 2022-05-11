import os, wandb
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning import seed_everything

from utils.network import Litsmp
from utils.dataloader import create_trainloader
from utils.config import wandb_config, load_setting

# Set seed
seed = 42
seed_everything(seed)


def main():
    ds_cfg = load_setting()
    project, name = ds_cfg['project'], ds_cfg['name']
    if(ds_cfg['iscvl'] == 1):
        fold_list_root = ds_cfg['train_valid_list'].replace('.json', '')
        for n_fold in range(5):
            name = ds_cfg['name'] + f'_fd{n_fold}'
            ds_cfg['train_valid_list'] = f'{fold_list_root}_{n_fold}.json'
            print(f'\nStart fold {n_fold} experiment\n')
            trainprocess(project, name, ds_cfg)
            wandb.finish()
    else:
        trainprocess(project, name, ds_cfg)

def trainprocess(project, name, ds_cfg):
    opts_dict, wandb_logger = wandb_config(project, name, cfg='cfg/wandbcfg.yaml')

    # dataloader
    dataset_root = ds_cfg['crop_dataset_root'] \
        if opts_dict['iscrop'] == 1 else ds_cfg['dataset_root']
    imagePaths = os.path.join(dataset_root, 'Train_Images')
    maskPaths = os.path.join(dataset_root, 'Train_Masks')
    trainloader, validloader = create_trainloader(
                                imagePaths,
                                maskPaths,
                                opts_dict,
                                ds_cfg
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
