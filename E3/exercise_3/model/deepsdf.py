import torch.nn as nn
import torch


class DeepSDFDecoder(nn.Module):

    def __init__(self, latent_size):
        """
        :param latent_size: latent code vector length
        """
        super().__init__()
        dropout_prob = 0.2

        # TODO: Define model
        self.model1 = nn.Sequential(
            
            nn.utils.weight_norm(nn.Linear(latent_size+3,512), name='weight'),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.utils.weight_norm(nn.Linear(512,512), name='weight'),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
           
            nn.utils.weight_norm( nn.Linear(512,512), name='weight'),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            
            nn.utils.weight_norm(nn.Linear(512,latent_size-3), name='weight'),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
        )
        self.model2 = nn.Sequential(
           
            nn.utils.weight_norm( nn.Linear(512,512), name='weight'),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            
            nn.utils.weight_norm(nn.Linear(512,512), name='weight'),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
           
            nn.utils.weight_norm( nn.Linear(512,512), name='weight'),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
           
            nn.utils.weight_norm( nn.Linear(512,512), name='weight'),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            
            nn.Linear(512,1)
        )
    def forward(self, x_in):
        """
        :param x_in: B x (latent_size + 3) tensor
        :return: B x 1 tensor
        """
        # TODO: implement forward pass
        x = self.model1(x_in)
        x = self.model2(torch.cat((x,x_in),dim = 1))
        return x
