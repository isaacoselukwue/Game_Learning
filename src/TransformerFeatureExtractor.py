import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# Transformer feature extractor for processing visual inputs
class TransformerFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256, n_heads=4, n_layers=2):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        
        patch_size = 8
        h, w = observation_space.shape[1:]
        n_patches = (h // patch_size) * (w // patch_size)
        self.patch_embed = nn.Conv2d(
            n_input_channels, features_dim, kernel_size=patch_size, stride=patch_size
        )
        
        self.pos_embed = nn.Parameter(th.zeros(1, n_patches, features_dim))
        nn.init.normal_(self.pos_embed, std=0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=features_dim, 
            nhead=n_heads,
            dim_feedforward=features_dim * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.fc = nn.Linear(features_dim, features_dim)

    def forward(self, observations):
        x = self.patch_embed(observations)
        b, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        
        x = x + self.pos_embed        
        x = self.transformer(x)        
        x = x.mean(dim=1)        
        x = self.fc(x)
        return x
