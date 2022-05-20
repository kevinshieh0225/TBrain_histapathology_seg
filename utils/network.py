import torch
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from utils.loss import TverskyLoss, FocalTverskyLoss, BCEFocalTverskyLoss, IoULoss
import ttach as tta
from  utils.scheduler import TimmCosineLRScheduler
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
        elif loss_type == 'IoULoss':
            self.loss = IoULoss(**self.opts_dict['loss'])
        elif loss_type == 'TverskyLoss':
            self.loss = TverskyLoss(**self.opts_dict['loss'])
        elif loss_type == 'FocalTverskyLoss':
            self.loss = FocalTverskyLoss(**self.opts_dict['loss'])
        elif loss_type == 'BCEFocalTverskyLoss':
            self.loss = BCEFocalTverskyLoss(**self.opts_dict['loss'])


        # model initial
        model_type = self.opts_dict['model'].pop('type')
        if model_type == 'DeepLabV3':
            self.model = smp.DeepLabV3(
                    **self.opts_dict['model']
                )
        elif model_type == 'DeepLabV3Plus':
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

        # learning rate scheduler update
        # schedulers = self.lr_schedulers()
        # for sch in schedulers:
        #     sch.step()
        # sch.get_lr()
        # update metrics logs
        for metric_fn in self.metrics:
            metric_value = metric_fn(y_pred, y)
            self.log(f'train {metric_fn.__name__}', metric_value, on_epoch=True)
            record.update({f't{metric_fn.__name__}': metric_value})
        return record

    def validation_step(self, batch, batch_idx):
        x, y = batch
        ttaD4Scale = tta.base.Compose(
        [
            tta.HorizontalFlip(),
            tta.Rotate90(angles=[0, 90, 180, 270]),
            tta.Scale([1, 0.84, 0.68], interpolation="nearest")
        ])
        tta_model = tta.SegmentationTTAWrapper(self.model, ttaD4Scale, merge_mode='mean')
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
        elif sched_type == 'ReduceLROnPlateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                            optimizer,
                            **self.opts_dict['sched']
                        )
        elif sched_type == 'MultiplicativeLR':
            def lambda_rule(epoch):
                multiplicative = 1
                if epoch == self.opts_dict['lrboost']: # set lrboost 100 to increase LR by 10 on epoch 100
                    multiplicative = 10
                return multiplicative
            scheduler = torch.optim.lr_scheduler.MultiplicativeLR(
                            optimizer,
                            **self.opts_dict['sched'],
                            lr_lambda = lambda_rule
                        )
        elif sched_type == 'timm_CosineLRScheduler':
            scheduler = TimmCosineLRScheduler(optimizer,
                            **self.opts_dict['sched'],
                        )
        return [optimizer], [scheduler]
