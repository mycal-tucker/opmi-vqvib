import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import src.settings as settings
from src.data_utils.helper_fns import gen_batch


class Net(nn.Module):
    def __init__(self, input1_size, input2_size):
        super(Net, self).__init__()
        self.fc_1 = nn.Linear(input1_size + input2_size, 128)
        self.fc_2 = nn.Linear(128, 128)
        self.fc_3 = nn.Linear(128, 128)
        self.fc_4 = nn.Linear(128, 1)

    def forward(self, x, y):
        catted = torch.hstack((x, y))
        h1 = F.relu(self.fc_1(catted))
        h2 = F.relu(self.fc_2(h1))
        h3 = F.relu(self.fc_3(h2))
        output = self.fc_4(h3)
        return output


def get_info(model, dataset, targ_dim, comm_targ=False, batch_size=1024):
    # Define a network that takes in the two variables to calculate the MI of.
    mine_net = Net(512, targ_dim)
    mine_net.to(settings.device)
    optimizer = optim.Adam(mine_net.parameters())
    num_epochs = 200
    for epoch in range(num_epochs):
        speaker_obs, listener_obs, labels, _ = gen_batch(dataset, batch_size)
        with torch.no_grad():
            if comm_targ:
                targ_var, _, _ = model.speaker(speaker_obs)  # Communication
            else:
                _, _, _, targ_var = model(speaker_obs, listener_obs)  # Reconstruction
        # Shuffle the target variable so we can get a marginal of sorts.
        targ_shuffle = torch.Tensor(np.random.permutation(targ_var.cpu().numpy())).to(settings.device)
        optimizer.zero_grad()

        pred_xy = mine_net(speaker_obs, targ_var)
        pred_x_y = mine_net(speaker_obs, targ_shuffle)
        ret = torch.mean(pred_xy) - torch.log(torch.mean(torch.exp(pred_x_y)))
        loss = -ret  # maximize
        loss.backward()
        optimizer.step()
    # Now evaluate the mutual information instead of actively training.
    summed_loss = 0
    num_eval_epochs = 20
    for epoch in range(num_eval_epochs):
        speaker_obs, listener_obs, labels, _ = gen_batch(dataset, 1024)
        with torch.no_grad():
            if comm_targ:
                targ_var, _, _ = model.speaker(speaker_obs)  # Communication
            else:
                _, _, _, targ_var = model(speaker_obs, listener_obs)  # Reconstruction
        targ_shuffle = torch.Tensor(np.random.permutation(targ_var.cpu().numpy())).to(settings.device)
        optimizer.zero_grad()

        pred_xy = mine_net(speaker_obs, targ_var)
        pred_x_y = mine_net(speaker_obs, targ_shuffle)
        ret = torch.mean(pred_xy) - torch.log(torch.mean(torch.exp(pred_x_y)))
        summed_loss += ret.item()
    mutual_info = summed_loss / num_eval_epochs
    return mutual_info


