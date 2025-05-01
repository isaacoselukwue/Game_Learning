import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# Hybrid CNN-Transformer feature extractor
class HybridCNN_TransformerExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256, cnn_features=128, n_heads=4, n_layers=2):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, cnn_features, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        with th.no_grad():
            dummy_input = th.zeros(1, n_input_channels, observation_space.shape[1], observation_space.shape[2])
            cnn_output = self.cnn(dummy_input)
            self.n_patches = cnn_output.shape[2] * cnn_output.shape[3]
            self.cnn_output_dim = cnn_output.shape[1]
        
        self.projection = nn.Linear(self.cnn_output_dim, features_dim)
        
        self.pos_embed = nn.Parameter(th.zeros(1, self.n_patches, features_dim))
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
        x = self.cnn(observations)
        b, c, h, w = x.shape
        
        x = x.permute(0, 2, 3, 1).reshape(b, h * w, c)        
        x = self.projection(x)        
        x = x + self.pos_embed        
        x = self.transformer(x)        
        x = x.mean(dim=1)        
        x = self.fc(x)
        return x
