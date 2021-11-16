import torch
import torch.nn.functional as F
import torch.nn as nn


class SiamusicLoss(nn.Module):
    def __init__(self,dim=1):
        super().__init__()
        self.dim = dim

    def neg_cos_sim(self,p,z):
        z = z.detach()
        p = F.normalize(p,dim=self.dim) # default : L2 norm
        z = F.normalize(z,dim=self.dim)
        return -torch.mean(torch.sum(p*z,dim=self.dim))
    
    def forward(self,p1,z2,p2,z1):
        L = self.neg_cos_sim(p1,z2)/2 + self.neg_cos_sim(p2,z1)/2
        return L


if __name__ == '__main__':
    p1 = torch.randn((16,2048))
    p2 = torch.randn((16,2048))
    z1 = torch.randn((16,2048))
    z2 = torch.randn((16,2048))
    criterion = SiamusicLoss()
    loss = criterion(p1,z2,p2,z1)
    print(loss.item())