import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules import batchnorm, linear


class TNet(nn.Module):
    def __init__(self, k):
        super(TNet, self).__init__()
        # TODO Add layers: Convolutional k->64, 64->128, 128->1024 with corresponding batch norms and ReLU
        upscale_mod = []
        upscale_mod.append(nn.Conv1d(k,64,1))
        upscale_mod.append(nn.BatchNorm1d(64))
        upscale_mod.append(nn.ReLU())
        
        upscale_mod.append(nn.Conv1d(64,128,1))
        upscale_mod.append(nn.BatchNorm1d(128))
        upscale_mod.append(nn.ReLU())
        
        upscale_mod.append(nn.Conv1d(128,1024,1))
        upscale_mod.append(nn.BatchNorm1d(1024))
        upscale_mod.append(nn.ReLU())
        
        self.upscale_1 = nn.Sequential(*upscale_mod)
        # TODO Add layers: Linear 1024->512, 512->256, 256->k^2 with corresponding batch norms and ReLU
        
        downscale = []
        self.downscale=nn.Sequential(
            nn.Linear(1024,512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256,(k*k))
        )
        
        
        self.register_buffer('identity', torch.from_numpy(np.eye(k).flatten().astype(np.float32)).view(1, k ** 2))
        self.k = k

    def forward(self, x):
        b = x.shape[0]
        
        x = self.upscale_1(x) 
        x = torch.max(x,2,keepdim=True)[0]
        x = x.view(-1,1024)
        x= self.downscale(x)
        # TODO Pass input through layers, applying the same max operation as in PointNetEncoder
        # TODO No batch norm and relu after the last Linear layer

        # Adding the identity to constrain the feature transformation matrix to be close to orthogonal matrix
        identity = self.identity.repeat(b,1)
        x = x + identity
        x = x.view(-1, self.k, self.k)
        return x

class PointNetEncoder(nn.Module):
    def __init__(self, return_point_features=False):
        super(PointNetEncoder, self).__init__()

        # TODO Define convolution layers, batch norm layers, and ReLU

        self.input_transform_net = TNet(k=3)
        self.feature_transform_net = TNet(k=64)
        
        # first set of conv layers
        first_1 = []
        first_1.append(nn.Conv1d(3,64,1))
        first_1.append(nn.BatchNorm1d(64))
        first_1.append(nn.ReLU())
        self.first_mod = nn.Sequential(*first_1)        
        #sec set of conv layers

        sec_1 = []
        sec_1.append(nn.Conv1d(64,128,1))
        sec_1.append(nn.BatchNorm1d(128))
        sec_1.append(nn.ReLU())
        sec_1.append(nn.Conv1d(128,1024,1))
        sec_1.append(nn.BatchNorm1d(1024))
        sec_1.append(nn.ReLU())
        self.sec_mod = nn.Sequential(*sec_1)
        self.inter = 0
        self.inter2 = 0
        self.return_point_features = return_point_features

    def forward(self, x):
        num_points = x.shape[2]

        input_transform = self.input_transform_net(x)
        x = torch.bmm(x.transpose(2, 1), input_transform.transpose(2,1)).transpose(2, 1)

        # TODO: First layer: 3->64
        
        x = self.first_mod(x)
        
        feature_transform = self.feature_transform_net(x)
        #print(x.shape,feature_transform.shape)
        x = torch.bmm(x.transpose(2, 1), feature_transform.transpose(2,1)).transpose(2, 1)
        point_features = x
        self.inter = x
        # TODO: Layers 2 and 3: 64->128, 128->1024
        
        x = self.sec_mod(x)

        
        # This is the symmetric max operation
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        self.inter2 = x
        if self.return_point_features:
            x = x.view(-1, 1024, 1).repeat(1, 1, num_points)
            return torch.cat([x, point_features], dim=1)
        else:
            return x


class PointNetClassification(nn.Module):
    def __init__(self, num_classes):
        super(PointNetClassification, self).__init__()
        self.encoder = PointNetEncoder(retu