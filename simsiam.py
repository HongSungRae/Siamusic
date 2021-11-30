import torch.nn as nn
import torchaudio
import torchvision.models as models
import torch


class Siamusic(nn.Module):
    def __init__(self, backbone, dim=2048, pred_dim=512, sample_rate=16000, n_fft=512, f_min=0.0, f_max=8000.0, n_mels=96):                 
        # backbone에 원하는 backbone 입력 ex) 'resnet' 
        # dim: projection의 hidden fc dimension
        # pred_dim: predictor의 hidden dimension
        
        super(Siamusic, self).__init__()
    
        self.spec = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=n_fft, 
                                                        f_min=f_min, f_max=f_max, n_mels=n_mels)
        self.to_db = torchaudio.transforms.AmplitudeToDB()
        self.spec_bn = nn.BatchNorm2d(1)
        
        if backbone in ['resnet50','resnet101','resnet152']:
            self.encoder = models.__dict__[backbone](zero_init_residual=True,pretrained=False) # encoder: backbone + projector
            self.encoder.conv1 = nn.Conv2d(1, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)
        elif backbone == 'Transformer':
            pass
        #     self.encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        #     self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)
        # transformer 구성이랑... transformer와 projector 연결을 어떻게 해야할지 고민 중이었다....

        # encoder의 projector를 3layer MLP로 구성
        prev_dim = self.encoder.fc.weight.shape[-1]
        self.encoder.fc.bias.requires_grad = False
        self.encoder.fc = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                               nn.BatchNorm1d(prev_dim),
                                               nn.ReLU(inplace=True), #First Layer
                                               
                                               nn.Linear(prev_dim, prev_dim, bias=False),
                                               nn.BatchNorm1d(prev_dim),
                                               nn.ReLU(inplace=True), #Second Layer
                                               
                                               nn.Linear(prev_dim, prev_dim, bias=False),
                                               nn.BatchNorm1d(prev_dim),
                                               nn.ReLU(inplace=True), #Third Layer
                                               
                                               nn.BatchNorm1d(dim, affine=False) #Output Layer
                                               )
        
        self.predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                       nn.BatchNorm1d(pred_dim),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(pred_dim, dim)
                                       )
        
    def forward(self, x1, x2):
        # x1, x2 shape: [B, 1, 48000]
        x1, x2 = self.spec(x1), self.spec(x2)       #[B, 1, 96, 188]
        x1, x2 = self.to_db(x1), self.to_db(x2)     #[B, 1, 96, 188]
        x1, x2 = self.spec_bn(x1), self.spec_bn(x2) #[B, 1, 96, 188]
        
        z1 = self.encoder(x1)   # x1이 stop_gradient 쪽에 투입되는 경우 #[B,8,1000]
        z2 = self.encoder(x2)   # x2가 stop_gradient 쪽에 투입되는 경우 #[B,8,1000]
        
        p1 = self.predictor(z1) # x1이 predictor 쪽에 투입되는 경우
        p2 = self.predictor(z2) # x2가 predictor 쪽에 투입되는 경우
        
        # x1, x2 위치 변경 두 가지 경우에 대한 output 모두 반환
        # case1 : p1, z2.deatach() 이용
        # case2 : z1.detach(), p2 이용
        return p1, z2.detach(), p2, z1.detach() # z1, z2는 stop_gradient


if __name__ == '__main__':
    x1 = torch.randn((8,1,48000))
    x2 = torch.randn((8,1,48000))
    model = Siamusic('resnet152')
    p1,p2,z1,z2 = model(x1,x2)
    print(f'p : {p1.shape} | z : {z1.shape}')