import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.modules.distance import PairwiseDistance


class SiamusicLoss(nn.Module):
    def __init__(self,dim=1):
        super().__init__()
        self.dim = dim

    def neg_cos_sim(self,p,z):
        z = z.detach()
        p = F.normalize(p,dim=self.dim)
        z = F.normalize(z,dim=self.dim)
        return -torch.mean(torch.sum(p*z,dim=self.dim))
    
    def forward(self,p1,z2,p2,z1):
        L = self.neg_cos_sim(p1,z2)/2 + self.neg_cos_sim(p2,z1)/2
        return L