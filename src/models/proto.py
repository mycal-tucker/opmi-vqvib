import torch.nn as nn
import torch
import torch.nn.functional as F
from src.models.network_utils import gumbel_softmax, reparameterize
import src.settings as settings


class ProtoNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers, num_protos, variational=False):
        super(ProtoNetwork, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = 64
        self.num_tokens = num_protos
        self.variational = variational
        in_dim = input_dim
        out_dim = self.hidden_dim if num_layers > 1 else output_dim
        self.layers = nn.ModuleList()
        while len(self.layers) < num_layers:
            self.layers.append(nn.Linear(in_dim, out_dim))
            in_dim = out_dim
            out_dim = self.hidden_dim if len(self.layers) < num_layers - 1 else num_protos
        self.prototypes = nn.Parameter(data=torch.Tensor(num_protos, output_dim))
        self.prototypes.data.uniform_(-1 / num_protos, 1 / num_protos)
        self.fc_var = nn.Linear(num_protos, output_dim)
        self.eval_mode = False
        # Learnable marginal is initialized to match a unit Gaussian.
        if settings.learned_marginal:
            self.prior_mu = nn.Parameter(data=torch.Tensor(out_dim))
            self.prior_logvar = nn.Parameter(data=torch.Tensor(out_dim))
            torch.nn.init.constant_(self.prior_mu, 0)
            torch.nn.init.constant_(self.prior_logvar, 0)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i != len(self.layers) - 1:
                x = F.relu(x)
        # Turn into onehot to multiply by protos.
        onehot_pred = gumbel_softmax(x, hard=True)
        proto = torch.matmul(onehot_pred, self.prototypes)
        if self.variational:
            logvar = self.fc_var(x)
            output = reparameterize(proto, logvar) if not self.eval_mode else proto
            # Compute the KL divergence
            if not settings.learned_marginal:
                kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - proto ** 2 - logvar.exp(), dim=1), dim=0)
                regularization_loss = 0
            else:
                kld_loss = torch.mean(0.5 * torch.sum(
                    self.prior_logvar - logvar - 1 + logvar.exp() / self.prior_logvar.exp() + (
                                (self.prior_mu - proto) ** 2) / self.prior_logvar.exp(), dim=1), dim=0)
                regularization_loss = torch.mean(self.prior_logvar ** 2) + torch.mean(self.prior_mu ** 2)
            total_loss = settings.kl_weight * kld_loss + 0.01 * regularization_loss
            capacity = kld_loss
        else:
            output = proto
            total_loss = torch.tensor(0)
            capacity = torch.tensor(0)
        return output, total_loss, capacity
