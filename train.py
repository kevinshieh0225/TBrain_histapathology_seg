import os, wandb
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning import seed_everything

from utils.network import Litsmp
from utils.dataloader import create_trainloader
from utils.config import wandb_config, load_setting, loadmodel

# Set seed
# seed = 42
# seed_everything(seed)


def main():
    ds_cfg = load_setting()
    project, name = ds_cfg['project'], ds_cfg['name']
    fold_list_root = ds_cfg['train_valid_list']
    n_fold = 5 if '5' in fold_list_root else 10
    #soup
    n_ingreadients = ds_cfg['ningredients'] 
    fold_number = ds_cfg['base_fold']
    if(ds_cfg['soup'] == 1):
        for n in range(0, n_ingreadients):
            name = ds_cfg['name'] + f'_soup_{n}'
            ds_cfg['train_valid_list'] = f'{fold_list_root}_{fold_number}.json'
            print(f'\nCooking soup {n}/{n_ingreadients}\n')
            trainprocess(project, name, ds_cfg)
            wandb.finish()
    elif(ds_cfg['iscvl'] == 1):
        for n in [0, 3, 4, 5, 6, 7, 8]:
            name = ds_cfg['name'] + f'_{n_fold}fd{n}'
            ds_cfg['train_valid_list'] = f'{fold_list_root}_{n}.json'
            print(f'\nStart fold {n}/{n_fold} experiment\n')
            trainprocess(project, name, ds_cfg)
            wandb.finish()
    else:
        ds_cfg['train_valid_list'] = fold_list_root + '_4.json'
        name = ds_cfg['name'] + f'_{n_fold}fd4'
        trainprocess(project, name, ds_cfg)

def trainprocess(project, name, ds_cfg):
    opts_dict, wandb_logger = wandb_config(project, name, cfg='cfg/wandbcfg.yaml')
    
    # Load model from checkpoint
    if opts_dict['isckpt'] != 'None': 
        pretrain_path = opts_dict['isckpt']
        load_last = opts_dict['loadlast']
        _, model = loadmodel(pretrain_path, load_last)
    else: # new model
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
