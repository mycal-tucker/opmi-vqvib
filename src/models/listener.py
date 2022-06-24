import torch
import torch.nn as nn
import torch.nn.functional as F


class Listener(nn.Module):
    def __init__(self, feature_dim, num_imgs, num_layers=2):
        super(Listener, self).__init__()
        self.feature_embed_dim = 16
        self.num_imgs = num_imgs
        self.feature_embedder = nn.Linear(feature_dim, self.feature_embed_dim)
        self.fc1 = nn.Linear(self.feature_embed_dim * (num_imgs + 1), 64)
        self.fc2 = nn.Linear(64, num_imgs)

    def forward(self, features):
        embedded_features = self.feature_embedder(features)
        features_reshaped = torch.reshape(embedded_features, (-1, self.feature_embed_dim * (self.num_imgs + 1)))
        hidden1 = F.relu(self.fc1(features_reshaped))
        logits = self.fc2(hidden1)
        return logits
