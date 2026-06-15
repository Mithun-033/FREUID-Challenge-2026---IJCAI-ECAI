import torch
import torch.optim as optim
import torch.nn as nn

class MuonAdamW(nn.Module):
    def __init__(self,model,config,total_steps):
        super().__init__()

        self.config=config

        adam_params=[]
        muon_params=[]

        for param in model.parameters():
            if param.ndim>=2:
                muon_params.append(param)

            else:
                adam_params.apend(param)

        self.adam=optim.AdamW(
            adam_params,
            lr=config.lr,
            betas=(config.betas[0],config.betas[1]),
            fused=True,
            weight_decay=config.weight_decay
        )

        self.muon=optim.Muon(
            muon_params,
            lr=config.lr,
            betas=(config.betas[0],config.betas[1]),
            adjust_lr_fn="match_rms_adamw",
            weight_decay=config.weight_decay
        )

        self.adam_decay=optim.lr_scheduler.CosineAnnealingLR(
            self.adam,
            T_max=int(total_steps*0.9),
            eta_min=config.final_lr
        )

        self.muon_decay=optim.lr_scheduler.CosineAnnealingLR(
            self.muon,
            T_max=int(total_steps*0.9),
            eta_min=config.final_lr
        )

        self.adam_warmup=optim.lr_scheduler.LinearLR(
            self.adam,
            total_iters=int(total_steps*0.1),
            start_factor=config.lr*0.2,
            end_factor=config.lr
        )

        self.muon_warmup=optim.lr_scheduler.LinearLR(
            self.muon,
            total_iters=int(total_steps*0.1),
            start_factor=config.lr*0.2,
            end_factor=config.lr
        )

        self.Adam_scheduler=optim.lr_scheduler.SequentialLR(
            self.adam,
            [self.adam_warmup,self.adam_decay],
            milestones=[int(total_steps*0.1)]
        )
        

        self.Muon_scheduler=optim.lr_scheduler.SequentialLR(
            self.muon,
            [self.muon_warmup,self.muon_decay],
            milestones=[int(total_steps*0.1)]
        )

    def step(self):
        self.adam.step()
        self.muon.step()
        self.Adam_scheduler.step()
        self.Muon_scheduler.step()

    def zero_grad(self):
        self.adam.zero_grad()
        self.muon.zero_grad()

