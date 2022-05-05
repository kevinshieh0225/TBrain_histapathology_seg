from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import os, yaml
from network import Litsmp
from pytorch_lightning.strategies import DDPStrategy

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
        # accumulate_grad_batches=4,
        # amp_backend="apex",
        # amp_level='01',
        strategy=DDPStrategy(find_unused_parameters=True),
        )
    
    

    trainer.fit(
        model=model,
        train_dataloaders=trainloader,
        val_dataloaders=validloader,
        )


