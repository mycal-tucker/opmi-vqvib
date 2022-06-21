import torch.nn as nn


class Team(nn.Module):
    def __init__(self, speaker, listener, decoder):
        super(Team, self).__init__()
        self.speaker = speaker
        self.listener = listener
        self.decoder = decoder

    def forward(self, speaker_x, listener_x):
        comm, speaker_loss, info = self.speaker(speaker_x)
        prediction = self.listener(comm, listener_x)
        recons = self.decoder(comm)
        return prediction, speaker_loss, info, recons
