from src.models.vq import VQ
from src.models.team import Team
from src.models.mlp import MLP
from src.models.listener import Listener
from src.models.decoder import Decoder
import src.settings as settings
from src.data_utils.read_data import get_feature_data
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


def gen_batch(all_features):
    # Given the dataset of all features, creates a batch of inputs.
    # That's:
    # 1) The speaker's observation
    # 2) The listener's observation
    # 3) The label (which is the index of the speaker's observation).
    speaker_obs = []
    listener_obs = []
    labels = []
    for _ in range(batch_size):
        targ_idx = int(np.random.random() * len(all_features))
        targ_features = all_features[targ_idx]
        distractor_features = [all_features[int(np.random.random() * len(all_features))] for _ in range(num_distractors)]
        obs_targ_idx = int(np.random.random() * (num_distractors + 1))  # Pick where to slide the target observation into.

        speaker_obs.append(targ_features)
        l_obs = np.expand_dims(np.vstack(distractor_features[:obs_targ_idx] + [targ_features] + distractor_features[obs_targ_idx:]), axis=0)
        listener_obs.append(l_obs)
        labels.append(obs_targ_idx)
    speaker_tensor = torch.Tensor(np.vstack(speaker_obs)).to(settings.device)
    listener_tensor = torch.Tensor(np.vstack(listener_obs)).to(settings.device)
    label_tensor = torch.Tensor(labels).long().to(settings.device)
    return speaker_tensor, listener_tensor, label_tensor


def train(model, data):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    running_acc = 0
    for epoch in range(num_epochs):
        settings.kl_weight += kl_incr
        print('epoch', epoch, 'of', num_epochs)
        print("Kl weight", settings.kl_weight)
        speaker_obs, listener_obs, labels = gen_batch(data)
        optimizer.zero_grad()
        outputs, speaker_loss, info, recons = model(speaker_obs, listener_obs)
        loss = criterion(outputs, labels)
        print("Classification loss", loss)
        recons_loss = torch.mean(((speaker_obs - recons) ** 2))
        print("Recons loss", recons_loss)
        loss += settings.alpha * recons_loss
        loss += speaker_loss
        loss.backward()
        optimizer.step()
        print("Loss", loss)

        # Metrics
        pred_labels = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        num_correct = np.sum(pred_labels == labels.cpu().numpy())
        num_total = pred_labels.size
        running_acc = running_acc * 0.95 + 0.05 * num_correct / num_total
        print("Running acc", running_acc)


def run():
    speaker_inp_dim = feature_len if not see_distractor else (num_distractors + 1) * feature_len
    # speaker = MLP(speaker_inp_dim, comm_dim, num_layers=3, onehot=False, variational=variational)
    speaker = VQ(speaker_inp_dim, comm_dim, num_layers=3, num_protos=330, variational=variational)
    listener = Listener(comm_dim, feature_len, num_distractors + 1, num_layers=2)
    decoder = Decoder(comm_dim, speaker_inp_dim, num_layers=3)
    model = Team(speaker, listener, decoder)
    model.to(settings.device)

    all_features = get_feature_data(features_filename)
    train(model, all_features)


if __name__ == '__main__':
    feature_len = 512
    see_distractor = False
    num_distractors = 1
    num_epochs = 5000
    batch_size = 1024
    comm_dim = 32
    kl_incr = 0.00001
    variational = True
    settings.alpha = 1
    settings.sample_first = True
    settings.kl_weight = 0.0
    settings.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    settings.learned_marginal = False
    with_bbox = False
    features_filename = 'data/features.csv' if with_bbox else 'data/features_nobox.csv'
    run()
