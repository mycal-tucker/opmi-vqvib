import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import multivariate_normal

import src.settings as settings
from src.models.network_utils import reparameterize, gumbel_softmax


class VQLayer(nn.Module):
    def __init__(self, num_protos, latent_dim, beta=0.25):
        super(VQLayer, self).__init__()
        self.num_protos = num_protos
        self.latent_dim = latent_dim
        self.beta = beta
        self.prototypes = nn.Parameter(data=torch.Tensor(num_protos, latent_dim))
        self.prototypes.data.uniform_(-1 / self.num_protos, 1 / self.num_protos)

    def forward(self, latents, sample=False):
        dists_to_protos = torch.sum(latents ** 2, dim=1, keepdim=True) + \
                          torch.sum(self.prototypes ** 2, dim=1) - 2 * \
                          torch.matmul(latents, self.prototypes.t())
        if sample:
            closest_encodings = gumbel_softmax(-dists_to_protos, hard=True)
            # get quantized latent vectors
            quantized_latents = torch.matmul(closest_encodings, self.prototypes).view(latents.shape)
        else:
            closest_protos = torch.argmin(dists_to_protos, dim=1).unsqueeze(1)
            encoding_one_hot = torch.zeros(closest_protos.size(0), self.num_protos).to(settings.device)
            encoding_one_hot.scatter_(1, closest_protos, 1)
            quantized_latents = torch.matmul(encoding_one_hot, self.prototypes)

        # Compute the VQ Losses
        commitment_loss = F.mse_loss(quantized_latents.detach(), latents)
        embedding_loss = F.mse_loss(quantized_latents, latents.detach())
        vq_loss = commitment_loss * self.beta + embedding_loss

        # Compute the entropy of the distribution for which prototypes are used.
        ent = self.get_categorical_ent(dists_to_protos)

        # Add the residue back to the latents
        quantized_latents = latents + (quantized_latents - latents).detach()
        return quantized_latents, vq_loss, ent

    def get_categorical_ent(self, distances):
        # Approximate the onehot of which prototype via a softmax of the negative distances
        logdist = torch.log_softmax(-distances, dim=1)
        soft_dist = torch.mean(logdist.exp(), dim=0)
        epsilon = 0.00000001  # Need a fuzz factor for numerical stability.
        soft_dist += epsilon
        soft_dist = soft_dist / torch.sum(soft_dist)
        logdist = soft_dist.log()
        entropy = torch.sum(-1 * soft_dist * logdist)
        return entropy


class VQ(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers, num_protos, variational=False):
        super(VQ, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = 64
        self.num_tokens = num_protos  # Need this general variable for num tokens
        self.variational = variational
        in_dim = input_dim
        out_dim = self.hidden_dim if num_layers > 1 else output_dim
        self.layers = nn.ModuleList()
        while len(self.layers) < num_layers:
            self.layers.append(nn.Linear(in_dim, out_dim))
            in_dim = out_dim
            out_dim = self.hidden_dim if len(self.layers) < num_layers - 1 else output_dim
        self.vq_layer = VQLayer(num_protos, output_dim)
        self.fc_mu = nn.Linear(output_dim, output_dim)
        if variational:
            # Learnable prior is initialized to match a unit Gaussian.
            if settings.learned_marginal:
                self.prior_mu = nn.Parameter(data=torch.Tensor(out_dim))
                self.prior_logvar = nn.Parameter(data=torch.Tensor(out_dim))
                torch.nn.init.constant_(self.prior_mu, 0)
                torch.nn.init.constant_(self.prior_logvar, 0)
        self.eval_mode = False

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            x = F.relu(x)
        if self.variational:
            mu = self.fc_mu(x)
            # Quantize the vectors
            output, proto_loss, kld_loss = self.vq_layer(mu, True)
            total_loss = settings.kl_weight * kld_loss + proto_loss
            capacity = kld_loss
        else:
            x = self.fc_mu(x)
            output, total_loss, _ = self.vq_layer(x)
            capacity = torch.tensor(0)
        return output, total_loss, capacity

    # Helper method calculates the distribution over prototypes given an input
    def get_token_dist(self, x):
        with torch.no_grad():
            for i, layer in enumerate(self.layers):
                x = layer(x)
                x = F.relu(x)
            samples = self.fc_mu(x)
            # Now discretize
            dists_to_protos = torch.sum(samples ** 2, dim=1, keepdim=True) + \
                              torch.sum(self.vq_layer.prototypes ** 2, dim=1) - 2 * \
                              torch.matmul(samples, self.vq_layer.prototypes.t())
            _, distribution = gumbel_softmax(-dists_to_protos, return_dist=True)
            likelihoods = np.mean(distribution.detach().cpu().numpy(), axis=0)
        return likelihoods
