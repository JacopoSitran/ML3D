import torch
import torch.nn as nn
from torch.nn.modules.activation import LeakyReLU
from torch.nn.modules.batchnorm import BatchNorm3d


class ThreeDEPN(nn.Module):
    def __init__(self):
        super(ThreeDEPN, self).__init__()

        self.num_features = 80

        # TODO: 4 Encoder layers
        self.encoder1 = nn.Sequential(
            nn.Conv3d(2,self.num_features,4,2,1),
            LeakyReLU(0.2)
        )
        self.encoder2 = nn.Sequential(
            nn.Conv3d(self.num_features,self.num_features*2,4,2,1),
            BatchNorm3d(self.num_features*2),
            LeakyReLU(0.2)
        )
        self.encoder3 = nn.Sequential(
            nn.Conv3d(self.num_features*2,self.num_features*4,4,2,1),
            BatchNorm3d(self.num_features*4),
            LeakyReLU(0.2)
        )
        self.encoder4 = nn.Sequential(
            nn.Conv3d(self.num_features*4,self.num_features*8,4,1),
            BatchNorm3d(self.num_features*8),
            LeakyReLU(0.2)
        )

        # TODO: 2 Bottleneck layers
        self.bottleneck = nn.Sequential(
            nn.Linear(self.num_features * 8,self.num_features * 8),
            nn.ReLU(),
            nn.Linear(self.num_features * 8,self.num_features * 8),
            nn.ReLU()
        )

        # TODO: 4 Decoder layers
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose3d(self.num_features*8*2,self.num_features*4,4,1),
            BatchNorm3d(self.num_features*4),
            nn.ReLU()
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose3d(self.num_features*4*2,self.num_features*2,4,2,1),
            BatchNorm3d(self.num_features*2),
            nn.ReLU()
        )
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose3d(self.num_features*2*2,self.num_features,4,2,1),
            BatchNorm3d(self.num_features),
            nn.ReLU()
        )
        self.decoder4 = nn.ConvTranspose3d(self.num_features*2,1,4,2,1)
        

    def forward(self, x):

        b = x.shape[0]
        # Encode
        # TODO: Pass x though encoder while keeping the intermediate outputs for the skip connections
        x_e1 = self.encoder1(x)
        x_e2 = self.encoder2(x_e1)
        x_e3 = self.encoder3(x_e2)
        x_e4 = self.encoder4(x_e3)

        # Reshape and apply bottleneck layers
        x = x_e4.view(b, -1)
        x = self.bottleneck(x)

        x = x.view(x.shape[0], x.shape[1], 1, 1, 1)
        # Decode

        # TODO: Pass x through the decoder, applying the skip connections in the process
        x = self.decoder1(torch.cat((x,x_e4),dim=1))
        x = self.decoder2(torch.cat((x,x_e3),dim=1))
        x = self.decoder3(torch.cat((x,x_e2),dim=1))
        x = self.decoder4(torch.cat((x,x_e1),dim=1))
        x = torch.squeeze(x, dim=1)
        # TODO: Log scaling
        x = torch.log1p(x.abs())
        

        return x
