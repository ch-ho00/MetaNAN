import torch
import torch.nn as nn 

from degae.uformer.model import Uformer
from degae.srgan.degrade_extractor import DegFeatureExtractor
from degae.decoder import DegAE_decoder


def de_parallel(model):
    return model.module if hasattr(model, 'module') else model


class DegAE(nn.Module):

    def __init__(self, args, train_scratch=False):    
        super().__init__()

        self.args = args 
        self.device = torch.device(f"cuda:{args.local_rank}")
        
        depths=[2, 2, 2, 2, 2, 2, 2, 2, 2]
        img_wh = [self.args.img_size, self.args.img_size] #[1024, 768]
        self.encoder = Uformer(img_wh=img_wh, embed_dim=16, depths=depths,
                    win_size=8, mlp_ratio=4., token_projection='linear', token_mlp='leff', modulator=True, shift_flag=False).to(self.device)

        self.degrep_extractor = DegFeatureExtractor(ckpt_path=args.degrep_ckpt, train_scratch=train_scratch).to(self.device)
        self.decoder = DegAE_decoder(rand_noise=args.rand_noise, skip_condition=args.skip_condition).to(self.device)
        self.optimizer, self.scheduler = self.create_optimizer()


    def create_optimizer(self):
        params_list = [{'params': self.encoder.parameters(), 'lr': self.args.lrate_feature},
                       {'params': self.degrep_extractor.degrep_conv.parameters(),  'lr': self.args.lrate_feature},
                       {'params': self.degrep_extractor.degrep_fc.parameters(),  'lr': self.args.lrate_feature},
                       {'params': self.decoder.parameters(),  'lr': self.args.lrate_feature},                       
                       ]

        optimizer = torch.optim.Adam(params_list)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=self.args.lrate_decay_steps,
                                                    gamma=self.args.lrate_decay_factor)

        return optimizer, scheduler

    def save_model(self, filename):
        to_save = {'optimizer'  : self.optimizer.state_dict(),
                   'scheduler'  : self.scheduler.state_dict(),
                   'model' : de_parallel(self).state_dict()}
        torch.save(to_save, filename)
        
    
    def forward(self, batch_data):

        img_embed = self.encoder(batch_data['noisy_rgb'])                                           # (B, 64, H, W)
        noise_vec_ref = None
        if not self.args.skip_condition:        
            noise_vec_ref = self.degrep_extractor(batch_data['ref_rgb'], batch_data['white_level']) # (B, 512)

        reconst_signal = self.decoder(img_embed, noise_vec_ref)                                     # (B, 3, H, W)

        return reconst_signal