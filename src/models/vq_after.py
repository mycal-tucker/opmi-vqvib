import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import src.settings as settings
from src.models.network_utils import reparameterize, gumbel_softmax


class VQLayer(nn.Module):
    def __init__(self, num_protos, latent_dim, init_vectors=None, beta=0.25):
        super(VQLayer, self).__init__()
        self.num_protos = num_protos
        self.latent_dim = latent_dim
        self.trainable_quantizers = init_vectors is None
        self.beta = beta
        self.prototypes = nn.Parameter(data=torch.Tensor(num_protos, latent_dim))
        if init_vectors is not None:
            print("Manually specifying vector quantization for", len(init_vectors), "vectors.")
            self.prototypes.data = torch.from_numpy(init_vectors).type(torch.FloatTensor)
            self.prototypes.requires_grad = not settings.hardcoded_vq
        else:
            self.prototypes.data.uniform_(-1/ self.num_protos, 1/self.num_protos)

    def forward(self, latents):
        dists_to_protos = torch.sum(latents ** 2, dim=1, keepdim=True) + \
                          torch.sum(self.prototypes ** 2, dim=1) - 2 * \
                          torch.matmul(latents, self.prototypes.t())
        closest_protos = torch.argmin(dists_to_protos, dim=1).unsqueeze(1)
        encoding_one_hot = torch.zeros(closest_protos.size(0), self.num_protos).to(settings.device)
        encoding_one_hot.scatter_(1, closest_protos, 1)
        # encoding_one_hot = gumbel_softmax(-dists_to_protos, hard=True, temperature=0.1)
        epsilon = 0.00000001
        prior = torch.mean(encoding_one_hot + epsilon, dim=0)
        normalizer = torch.sum(prior)
        prior = prior / normalizer
        true_ent = torch.sum(-1 * prior * prior.log())

        do_print = np.random.random() < 0.01

        quantized_latents = torch.matmul(encoding_one_hot, self.prototypes)

        # Compute the VQ Losses
        commitment_loss = F.mse_loss(quantized_latents.detach(), latents)
        embedding_loss = F.mse_loss(quantized_latents, latents.detach())

        vq_loss = 1.0 * (commitment_loss * self.beta + 1.0 * embedding_loss)

        # Add the residue back to the latents
        quantized_latents = latents + (quantized_latents - latents).detach()
        return quantized_latents, vq_loss


class VQAfter(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers, num_protos, specified_tok=None, num_simultaneous_tokens=1, variational=True, num_imgs=1):
        super(VQAfter, self).__init__()
        self.input_dim = input_dim
        self.comm_dim = output_dim
        self.proto_latent_dim = int(self.comm_dim / num_simultaneous_tokens)
        self.hidden_dim = 64
        if specified_tok is not None:
            num_protos = specified_tok.shape[0]
        self.num_tokens = num_protos  # Need this general variable for num tokens
        self.num_simultaneous_tokens = num_simultaneous_tokens
        self.variational = variational
        self.num_imgs = num_imgs
        self.feature_embedder = nn.Linear(input_dim, self.hidden_dim)
        in_dim = self.hidden_dim * num_imgs
        out_dim = self.hidden_dim if num_layers > 1 else self.comm_dim
        self.layers = nn.ModuleList()
        while len(self.layers) < num_layers:
            self.layers.append(nn.Linear(in_dim, out_dim))
            in_dim = out_dim
            out_dim = self.hidden_dim if len(self.layers) < num_layers - 1 else self.comm_dim
        self.vq_layer = VQLayer(num_protos, self.proto_latent_dim, init_vectors=specified_tok)
        self.fc_mu = nn.Linear(self.proto_latent_dim, self.proto_latent_dim)
        if variational:
            self.fc_var = nn.Linear(self.proto_latent_dim, self.proto_latent_dim)
            # Learnable prior is initialized to match a unit Gaussian.
            if settings.learned_marginal:
                self.prior_mu = nn.Parameter(data=torch.Tensor(out_dim))
                self.prior_logvar = nn.Parameter(data=torch.Tensor(out_dim))
                torch.nn.init.constant_(self.prior_mu, 0)
                torch.nn.init.constant_(self.prior_logvar, 0)
        self.eval_mode = False

    def forward(self, x):
        embedded_features = self.feature_embedder(x)
        x = torch.reshape(embedded_features, (-1, self.hidden_dim * self.num_imgs))
        for i, layer in enumerate(self.layers):
            x = layer(x)
            x = F.relu(x)
        if self.variational:
            # Quantize and then sample
            # This reshaping handles the desire to have multiple tokens in a single message.
            reshaped = torch.reshape(x, (-1, self.proto_latent_dim))
            latent = self.fc_mu(reshaped)
            mu, proto_loss = self.vq_layer(latent)
            if self.eval_mode:
                return torch.reshape(mu, (-1, self.comm_dim)), None, None
            logvar = self.fc_var(reshaped)
            output = reparameterize(mu, logvar)
            # Regroup the tokens into messages, now with possibly multiple tokens.
            output = torch.reshape(output, (-1, self.comm_dim))
            relevant_mu = mu
            # Compute the KL divergence
            if not settings.learned_marginal:
                kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - relevant_mu ** 2 - logvar.exp(), dim=1), dim=0)
                regularization_loss = 0
            else:
                kld_loss = torch.mean(0.5 * torch.sum(
                    self.prior_logvar - logvar - 1 + logvar.exp() / self.prior_logvar.exp() + (
                                (self.prior_mu - relevant_mu) ** 2) / self.prior_logvar.exp(), dim=1), dim=0)
                regularization_loss = torch.mean(self.prior_logvar ** 2) + torch.mean(self.prior_mu ** 2)
            total_loss = settings.kl_weight * kld_loss + proto_loss + 0.01 * regularization_loss
            capacity = kld_loss
            # if np.random.random() < 0.01:
            #     print("KLD loss", kld_loss.item())
        else:
            x = self.fc_mu(x)
            output, total_loss = self.vq_layer(x)
            capacity = torch.tensor(0)
        return output, total_loss, capacity

    # Helper method calculates the distribution over prototypes given an input
    def get_token_dist(self, x):
        assert settings.sample_first, "There's no distribution over protos if you don't sample first."
        with torch.no_grad():
            for i, layer in enumerate(self.layers):
                x = layer(x)
                x = F.relu(x)
            if self.variational:
                logvar = self.fc_var(x)
                mu = self.fc_mu(x)
                samples = reparameterize(mu, logvar)
            else:
                samples = self.fc_mu(x)
            # Now discretize
            dists_to_protos = torch.sum(samples ** 2, dim=1, keepdim=True) + \
                              torch.sum(self.vq_layer.prototypes ** 2, dim=1) - 2 * \
                              torch.matmul(samples, self.vq_layer.prototypes.t())
            closest_protos = torch.argmin(dists_to_protos, dim=1).unsqueeze(1)
            encoding_one_hot = torch.zeros(closest_protos.size(0), self.vq_layer.num_protos).to(settings.device)
            encoding_one_hot.scatter_(1, closest_protos, 1)
            likelihoods = np.mean(encoding_one_hot.detach().cpu().numpy(), axis=0)
        return likelihoods

    def snap_comms(self, x):
        reshaped = torch.reshape(x, (-1, self.proto_latent_dim))
        quantized, _ = self.vq_layer(reshaped)
        return torch.reshape(quantized, (-1, self.comm_dim))

    def get_comm_id(self, comm):
        # Return the index of the closest prototype.
        # Or, if there are multiple tokens, do the clever multiplication
        vocab_size = self.num_tokens ** self.num_simultaneous_tokens
        reshaped = torch.reshape(comm, (-1, self.proto_latent_dim))
        dists_to_protos = torch.sum(reshaped ** 2, dim=1, keepdim=True) + \
                          torch.sum(self.vq_layer.prototypes ** 2, dim=1) - 2 * \
                          torch.matmul(reshaped, self.vq_layer.prototypes.t())

        closest_protos = torch.argmin(dists_to_protos, dim=1).unsqueeze(1)
        # idx = 0
        # for i, closest_proto in enumerate(closest_protos):
        #     idx += (self.num_tokens ** i) * closest_proto
        idx = closest_protos  # FIXME: do the math for multiple tokens
        onehot = torch.zeros((1, vocab_size))
        onehot[0, idx] = 1
        return onehot
