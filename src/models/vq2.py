import torch
import torch.nn as nn
import torch.nn.functional as F

import src.settings as settings
import numpy as np
from src.models.network_utils import gumbel_softmax, reparameterize


class VQLayer(nn.Module):
    def __init__(self, num_protos, latent_dim, init_vectors=None, beta=0.25):
        super(VQLayer, self).__init__()
        self.num_protos = num_protos
        self.latent_dim = latent_dim
        self.trainable_quantizers = init_vectors is None
        self.beta = beta
        self.prototypes = nn.Parameter(data=torch.Tensor(num_protos, latent_dim))

        self.ent_loss_fn = nn.KLDivLoss(reduction='batchmean')
        if init_vectors is not None:
            print("Manually specifying vector quantization for", len(init_vectors), "vectors.")
            self.prototypes.data = torch.from_numpy(init_vectors).type(torch.FloatTensor)
            self.prototypes.requires_grad = not settings.hardcoded_vq
        else:
            self.prototypes.data.uniform_(-1 / self.num_protos, 1 / self.num_protos)

    def forward(self, latents):
        vector_diffs = latents.unsqueeze(1) - self.prototypes
        normalized_dists = torch.sum(vector_diffs ** 2, dim=2)
        neg_dists = -1 * normalized_dists
        onehot, logprobs = gumbel_softmax(neg_dists, hard=True, return_dist=True)
        quantized_latents = torch.matmul(onehot, self.prototypes)

        # Compute the VQ Losses
        # commitment_loss = F.mse_loss(quantized_latents.detach(), latents)
        # embedding_loss = F.mse_loss(quantized_latents, latents.detach())
        # vq_loss = commitment_loss * self.beta + embedding_loss
        vq_loss = 0

        # Compute capacity as the KL divergence from the prior.
        epsilon = 0.000001  # Need a fuzz factor for numerical stability.
        prior = torch.mean(logprobs.exp() + epsilon, dim=0, keepdim=True)
        prior = prior.expand(logprobs.shape[0], -1)
        capacity = self.ent_loss_fn(logprobs, prior)
        vq_loss += settings.kl_weight * capacity

        # Penalize the entropy of the prior, just to reduce codebook size
        ent = torch.sum(-1 * prior[0] * prior[0].log())  # Just the first row, because it's repeated for batch size.
        if np.random.random() < 0.01:
            # print("Commitment loss", commitment_loss.item())
            # print("Embedding loss", embedding_loss.item())
            print("Capacity", capacity.item())
            print("Entropy", ent.item())
        if settings.epoch > 5000:
            vq_loss += 0.001 * ent
        else:
            vq_loss -= 0.001 * ent

        # Get the categorical entropy of all messages used.
        # We want each token to be individually peaky, but overall to use lots of tokens.
        # Discourage entropy of individual messages (make them very peaky)
        # indiv_dist = logprobs.exp() + epsilon
        # indiv_dist = indiv_dist / torch.sum(indiv_dist)
        # logdist = indiv_dist.log()
        # indiv_ent = torch.sum(-1 * indiv_dist * logdist) / indiv_dist.shape[0]
        # vq_loss += 0.01 * indiv_ent

        # Encourage use of many messages over the whole batch.
        # batch_dist = torch.mean(logprobs.exp(), dim=0)
        # batch_dist += epsilon
        # batch_dist = batch_dist / torch.sum(batch_dist)
        # logdist = batch_dist.log()
        # batch_ent = torch.sum(-1 * batch_dist * logdist)
        # # print("Batch ent", batch_ent)
        # vq_loss -= 0.01 * batch_ent  # Encourage entropy over the whole batch
        return quantized_latents, vq_loss


class VQ2(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers, num_protos, specified_tok=None, num_simultaneous_tokens=1, variational=True, num_imgs=1):
        super(VQ2, self).__init__()
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
        self.fc_var = nn.Linear(self.proto_latent_dim, self.proto_latent_dim)

    def forward(self, x):
        embedded_features = self.feature_embedder(x)
        x = torch.reshape(embedded_features, (-1, self.hidden_dim * self.num_imgs))
        for i, layer in enumerate(self.layers):
            x = layer(x)
            x = F.relu(x)

        # This reshaping handles the desire to have multiple tokens in a single message.
        reshaped = torch.reshape(x, (-1, self.proto_latent_dim))
        mu = self.fc_mu(reshaped)
        logvar = self.fc_var(reshaped)
        sample = reparameterize(mu, logvar)
        output, proto_loss = self.vq_layer(mu)
        # Regroup the tokens into messages, now with possibly multiple tokens.
        output = torch.reshape(output, (-1, self.comm_dim))

        # Compute the KL divergence
        # kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)
        # total_loss = settings.kl_weight * kld_loss + proto_loss
        kld_loss = 0
        total_loss = proto_loss
        capacity = kld_loss

        return output, total_loss, capacity

    def snap_comms(self, x):
        reshaped = torch.reshape(x, (-1, self.proto_latent_dim))
        dists_to_protos = torch.sum(reshaped ** 2, dim=1, keepdim=True) + \
                          torch.sum(self.vq_layer.prototypes ** 2, dim=1) - 2 * \
                          torch.matmul(reshaped, self.vq_layer.prototypes.t())
        closest_protos = torch.argmin(dists_to_protos, dim=1).unsqueeze(1)
        encoding_one_hot = torch.zeros(closest_protos.size(0), self.vq_layer.num_protos).to(settings.device)
        encoding_one_hot.scatter_(1, closest_protos, 1)
        quantized = torch.matmul(encoding_one_hot, self.vq_layer.prototypes)
        return torch.reshape(quantized, (-1, self.comm_dim))

    def get_comm_id(self, comm):
        # Return the index of the closest prototype. During training, we sample, but here we assume it's just
        # the closest one.
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
