import torch
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from loss import TverskyLoss, FocalTverskyLoss
import ttach as tta
import copy

class Litsmp(pl.LightningModule):
    def __init__(self, opts_dict):
        super().__init__()
        self.opts_dict = copy.deepcopy(opts_dict)
        self.save_hyperparameters(opts_dict)
        # loss initial
        
        loss_type = self.opts_dict['loss'].pop('type')
        if loss_type == 'DiceLoss':
            self.loss = smp.utils.losses.DiceLoss()
            self.opts_dict['model']['activation'] = 'sigmoid'
        elif loss_type == 'CrossEntropyLoss':
            self.loss = smp.utils.losses.CrossEntropyLoss()
        elif loss_type == 'BCEWithLogitsLoss':
            self.loss = smp.utils.losses.BCEWithLogitsLoss()
        elif loss_type == 'TverskyLoss':
            self.loss = TverskyLoss()
        elif loss_type == 'FocalTverskyLoss':
            self.loss = FocalTverskyLoss()

        # model initial
        model_type = self.opts_dict['model'].pop('type')
        if model_type == 'DeepLabV3Plus':
            self.model = smp.DeepLabV3Plus(
                    **self.opts_dict['model']
                )
        elif model_type == 'Unet':
            self.model = smp.Unet(
                    **self.opts_dict['model']
                )
        elif model_type == 'UnetPlusPlus':
            self.model = smp.UnetPlusPlus(
                    **self.opts_dict['model']
                )

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
        self.log(f"train loss", loss, on_epoch=True)
        record.update({'loss': loss})
        # update metrics logs
        for metric_fn in self.metrics:
            metric_value = metric_fn(y_pred, y)
            self.log(f'train {metric_fn.__name__}', metric_value, on_epoch=True)
            record.update({f't{metric_fn.__name__}': metric_value})
        return record

    def validation_step(self, batch, batch_idx):
        x, y = batch
        tta_model = tta.SegmentationTTAWrapper(self.model, tta.aliases.d4_transform(), merge_mode='mean')
        y_pred = tta_model(x)
        loss = self.loss(y_pred, y)
        self.log(f"valid loss", loss, on_epoch=True, prog_bar=True)
        for metric_fn in self.metrics:
            metric_value = metric_fn(y_pred, y)
            self.log(f'valid {metric_fn.__name__}', metric_value, on_epoch=True)

    def configure_optimizers(self):
        # otimizer initial
        optim_type = self.opts_dict['optim'].pop('type')
        if optim_type == 'Adam':
            optimizer = torch.optim.Adam(self.parameters(), **self.opts_dict['optim'])
        elif optim_type == 'AdamW':
            optimizer = torch.optim.AdamW(self.parameters(), **self.opts_dict['optim'])
        elif optim_type == 'SGD':
            optimizer = torch.optim.SGD(self.parameters(), **self.opts_dict['optim'])

        # scheduler initial
        sched_type = self.opts_dict['sched'].pop('type')
        if sched_type == 'CosineAnnealingWR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                            optimizer,
                            **self.opts_dict['sched']
                        )
        return [optimizer], [scheduler]
