import torch
import pytorch_lightning as pl
import segmentation_models_pytorch as smp

class Litsmp(pl.LightningModule):
    def __init__(self, smpmodel):
        super().__init__()
        self.model = smpmodel
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
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=4, T_mult=2)
        return [optimizer], [scheduler]