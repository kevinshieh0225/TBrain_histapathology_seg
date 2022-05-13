import torch
from  timm.scheduler.cosine_lr import CosineLRScheduler

class TimmCosineLRScheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optim, **kwargs):
        self.init_lr = optim.param_groups[0]["lr"]
        self.timmsteplr = CosineLRScheduler(optim, **kwargs)
        super().__init__(optim, self)

    def __call__(self, epoch):
        desired_lr = self.timmsteplr.get_epoch_values(epoch)[0]
        mult = desired_lr / self.init_lr
        return mult