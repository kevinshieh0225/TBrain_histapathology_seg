import torch
import pytorch_lightning as pl
import segmentation_models_pytorch as smp

class Litsmp(pl.LightningModule):
    def __init__(self, opts_dict):
        super().__init__()
        self.opts_dict = opts_dict.copy()
        model_type = self.opts_dict['model'].pop('type')
        if model_type == 'DeepLabV3Plus':
            self.model = smp.DeepLabV3Plus(
                    **self.opts_dict['model']
                )
        elif model_type == 'Unet':
            self.model = smp.Unet(
                    **self.opts_dict['model']
                )
        
        self.loss = smp.utils.losses.DiceLoss()
        self.metrics = [
            smp.utils.metrics.Fscore(),
            smp.utils.metrics.IoU(),
        ]

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        embedding = self.model(x)
        return embedding

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        record = {}
        x, y = batch
        y_pred = self.model(x)
        loss = self.loss(y_pred, y)
        self.log("train loss", loss, on_epoch=True)
        record.update({'loss': loss})
        # update metrics logs
        for metric_fn in self.metrics:
            metric_value = metric_fn(y_pred, y)
            self.log(f'train {metric_fn.__name__}', metric_value, on_epoch=True)
            record.update({f't{metric_fn.__name__}': metric_value})
        return record

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.model(x)
        loss = self.loss(y_pred, y)
        self.log("valid loss", loss, on_epoch=True, prog_bar=True)
        for metric_fn in self.metrics:
            metric_value = metric_fn(y_pred, y)
            self.log(f'valid {metric_fn.__name__}', metric_value, on_epoch=True)

    def configure_optimizers(self):
        optim_type = self.opts_dict['optim'].pop('type')
        if optim_type == 'Adam':
            optimizer = torch.optim.Adam(self.parameters(), **self.opts_dict['optim'])

        sched_type = self.opts_dict['sched'].pop('type')
        if sched_type == 'CosineAnnealingWR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                            optimizer,
                            **self.opts_dict['sched']
                        )
        return [optimizer], [scheduler]