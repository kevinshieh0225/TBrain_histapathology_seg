from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

def modeltrain(
            model,
            trainloader,
            validloader,
            epochs,
            expname,
            save_path,
            ):
    checkpoint_callback = ModelCheckpoint(
        dirpath=save_path,
        save_top_k=2,
        monitor="vloss"
        )
    wandb_logger = WandbLogger(
        project=expname,
        log_model="all"
        )
    trainer = Trainer(
        logger=wandb_logger,
        max_epochs=epochs,
        callbacks=[checkpoint_callback],
        accelerator='gpu', devices=1,
        amp_backend="native",
        )
    trainer.fit(
        model=model,
        train_dataloaders=trainloader,
        val_dataloaders=validloader,
        )


