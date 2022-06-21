import torch.nn as nn
import torch.nn.functional as F
from src.models.network_utils import reparameterize, gumbel_softmax, onehot_from_logits
import torch
import src.settings as settings
import numpy as np


class Listener(nn.Module):
    def __init__(self, comm_dim, feature_dim, num_imgs, num_layers=2):
        super(Listener, self).__init__()
        self.feature_embed_dim = 16
        self.num_imgs = num_imgs
        self.feature_embedder = nn.Linear(feature_dim, self.feature_embed_dim)
        self.fc1 = nn.Linear(self.feature_embed_dim * num_imgs + comm_dim, 64)
        self.fc2 = nn.Linear(64, num_imgs)

    def forward(self, comms, features):
        embedded_features = self.feature_embedder(features)
        features_reshaped = torch.reshape(embedded_features, (-1, self.feature_embed_dim * self.num_imgs))
        catted = torch.hstack([comms, features_reshaped])
        hidden1 = F.relu(self.fc1(catted))
        logits = self.fc2(hidden1)
        return logits
