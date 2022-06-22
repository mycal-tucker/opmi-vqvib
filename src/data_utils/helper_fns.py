import numpy as np
import src.settings as settings
import torch


def gen_batch(all_features, batch_size, num_distractors):
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
