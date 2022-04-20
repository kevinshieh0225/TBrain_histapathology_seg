from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import os, yaml

def modeltrain(
        model,
        trainloader,
        validloader,
        opts_dict
        ):
    expname = opts_dict['expname']
    epochs = opts_dict['epochs']
    save_path = opts_dict['savepath']

    # wandb
    wandb_logger = WandbLogger(
        project='TBrain_histapathology_segmentation',
        name=expname,
        log_model="all"
        )
    wandb_logger.experiment.config.update(opts_dict)

    # checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=save_path,
        save_top_k=2,
        monitor="valid loss"
        )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    trainer = Trainer(
        logger=wandb_logger,
        max_epochs=epochs,
        callbacks=[checkpoint_callback, lr_monitor],
        gpus=-1,
        amp_backend="native",
        )
    
    os.makedirs(save_path, exist_ok=True)
    ymlsavepath = os.path.join(save_path, 'expconfig.yml')
    with open(ymlsavepath, 'w') as yaml_file:
        yaml.dump(opts_dict, yaml_file, default_flow_style=False)

    trainer.fit(
        model=model,
        train_dataloaders=trainloader,
        val_dataloaders=validloader,
        )


