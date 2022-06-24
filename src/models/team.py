import torch
import torch.nn as nn


class Team(nn.Module):
    def __init__(self, speaker, listener, decoder):
        super(Team, self).__init__()
        self.speaker = speaker
        self.listener = listener
        self.decoder = decoder

    def forward(self, speaker_x, listener_x):
        comm, speaker_loss, info = self.speaker(speaker_x)
        recons = self.decoder(comm)
        unsqueezed = torch.unsqueeze(recons, 1)
        listener_obs = torch.hstack([unsqueezed, listener_x])
        prediction = self.listener(listener_obs)
        return prediction, speaker_loss, info, recons
