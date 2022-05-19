import os, wandb
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning import seed_everything

from utils.network import Litsmp
from utils.dataloader import create_trainloader
from utils.config import wandb_config, load_setting

# Set seed
# seed = 42
# seed_everything(seed)


def main():
    ds_cfg = load_setting()
    project, name = ds_cfg['project'], ds_cfg['name']
    fold_list_root = ds_cfg['train_valid_list']
    n_fold = 5 if '5' in fold_list_root else 10
    if ds_cfg['dev'] == 1:
        project = project + '_dev'
    if(ds_cfg['iscvl'] == 1):
        for n in range(1, n_fold):
            name = ds_cfg['name'] + f'_{n_fold}fd{n}'
            ds_cfg['train_valid_list'] = f'{fold_list_root}_{n}.json'
            print(f'\nStart fold {n}/{n_fold} experiment\n')
            trainprocess(project, name, ds_cfg)
            wandb.finish()
    else:
        ds_cfg['train_valid_list'] = fold_list_root + '_0.json'
        name = ds_cfg['name'] + f'_{n_fold}fd0'
        trainprocess(project, name, ds_cfg)

def trainprocess(project, name, ds_cfg):
    opts_dict, wandb_logger = wandb_config(project, name, cfg='cfg/wandbcfg.yaml')
    
    # model parameter
    if opts_dict['isckpt'] != 'None':
        for pth in os.listdir(opts_dict['isckpt']):
            if '.ckpt' in pth:
                weight = os.path.join(opts_dict['isckpt'], pth)
                break
        print(f'\nLoad weight from: {weight}\n')
        model = Litsmp.load_from_checkpoint(weight, opts_dict=opts_dict)
    else:
        model = Litsmp(opts_dict)

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
        model=model,
        opts_dict=opts_dict,
        trainloader=trainloader,
        validloader=validloader,
        wandb_logger=wandb_logger,
        )


def modeltrain(
        model,
        opts_dict,
        trainloader,
        validloader,
        wandb_logger,
        ):
    epochs = opts_dict['epochs']
    save_path = opts_dict['savepath']

    # checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=save_path,
        save_top_k=1,
        monitor="valid fscore",
        mode="max",
        )
    save_last_checkpoint = ModelCheckpoint(
        save_top_k=1,
        monitor="epoch",
        mode="max",
        dirpath=save_path,
        filename="last_{epoch:02d}-{global_step}",
    )
    lr_monitor = LearningRateMonitor()
    
    trainer = Trainer(
        logger=wandb_logger,
        max_epochs=epochs,
        callbacks=[checkpoint_callback, save_last_checkpoint, lr_monitor],
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
