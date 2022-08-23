import torch.nn as nn
import torch.nn.functional as F
from src.models.network_utils import reparameterize, gumbel_softmax, onehot_from_logits
import torch
import src.settings as settings
import numpy as np


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers, onehot, num_simultaneous_tokens=1, variational=True,
                 num_imgs=1):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = 512
        self.onehot = onehot
        self.onehot_output_dim = int(self.output_dim / num_simultaneous_tokens)
        self.num_simultaneous_tokens = num_simultaneous_tokens
        self.num_tokens = output_dim
        self.variational = variational
        self.num_imgs = num_imgs
        self.feature_embed_dim = 64
        self.feature_embedder = nn.Linear(input_dim, self.feature_embed_dim)
        in_dim = self.feature_embed_dim * num_imgs
        out_dim = self.hidden_dim if num_layers > 1 else output_dim
        self.layers = nn.ModuleList()
        while len(self.layers) < num_layers:
            self.layers.append(nn.Linear(in_dim, out_dim))
            in_dim = out_dim
            out_dim = self.hidden_dim if len(self.layers) < num_layers - 1 else output_dim
        self.fc_mu = nn.Linear(output_dim, output_dim)
        self.fc_var = nn.Linear(output_dim, output_dim)
        # Learnable prior is initialized to match a unit Gaussian.
        self.prior_mu = nn.Parameter(data=torch.Tensor(out_dim))
        self.prior_logvar = nn.Parameter(data=torch.Tensor(out_dim))
        torch.nn.init.constant_(self.prior_mu, 0)
        torch.nn.init.constant_(self.prior_logvar, 0)
        self.disc_loss_fn = nn.KLDivLoss(reduction='batchmean')
        self.eval_mode = False

    def forward(self, x):
        embedded_features = self.feature_embedder(x)
        x = torch.reshape(embedded_features, (-1, self.feature_embed_dim * self.num_imgs))
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i != len(self.layers) - 1:
                x = F.relu(x)
        if self.variational:
            if not self.onehot:
                mu = self.fc_mu(x)
                logvar = self.fc_var(x)
                output = reparameterize(mu, logvar) if not self.eval_mode else mu
                # For a fixed, unit gaussian
                if not settings.learned_marginal:
                    capacity = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)
                else:  # For a learned marginal
                    capacity = torch.mean(0.5 * torch.sum(
                        self.prior_logvar - logvar - 1 + logvar.exp() / self.prior_logvar.exp() + (
                                (self.prior_mu - mu) ** 2) / self.prior_logvar.exp(), dim=1), dim=0)
            else:
                # Create a onehot for each of the simultaneous tokens
                # Reshape the latent into the right number
                reshaped = torch.reshape(x, (-1, self.onehot_output_dim))
                output, logprobs = gumbel_softmax(reshaped, hard=True, return_dist=True)
                prior = torch.ones_like(logprobs) / logprobs.shape[1]  # Assume uniform prior for now.
                # The order of the arguments matters a lot.
                capacity = self.disc_loss_fn(logprobs, prior)
                # Now regroup into a single message
                output = torch.reshape(output, (-1, self.output_dim))

                # Also encourage unconditional entropy to boost number of vocab words.
                # unconditional = torch.mean(output, 0)
                # unconditional += 0.0001
                # unconditional = unconditional / torch.sum(unconditional)
                # uncond_ent = self.disc_loss_fn(unconditional.log(), torch.ones_like(unconditional) / len(unconditional))
                # if np.random.random() < 0.01:
                #     print("uncond_ent", uncond_ent)
            network_loss = settings.kl_weight * capacity  # - 1.0 * uncond_ent
        else:
            if not self.onehot:
                output = x
            else:
                output = gumbel_softmax(x, hard=True)
            capacity = torch.tensor(0)
            network_loss = torch.tensor(0)
        return output, network_loss, capacity

    def get_token_dist(self, x):
        assert self.onehot, "No categorical distribution if not onehot"
        with torch.no_grad():
            for i, layer in enumerate(self.layers):
                x = layer(x)
                if i != len(self.layers) - 1:
                    x = F.relu(x)
            _, logprobs = gumbel_softmax(x, hard=True, return_dist=True)
            dist = torch.exp(logprobs)
            likelihoods = np.mean(dist.detach().cpu().numpy(), axis=0)
        return likelihoods

    def snap_comms(self, x):
        if not self.onehot:
            return x
        # Reshape to the right number of tokens per message, then snap to onehot.
        reshaped = torch.reshape(x, (-1, self.onehot_output_dim))
        indices = torch.argmax(reshaped, dim=1)
        onehot = torch.zeros_like(reshaped).scatter(1, indices.unsqueeze(1), 1.)
        onehot = torch.reshape(onehot, (-1, self.output_dim))
        return onehot
