import src.settings as settings
from src.data_utils.helper_fns import get_unique_labels
from src.data_utils.read_data import get_feature_data
from src.models.decoder import Decoder
from src.models.listener import Listener
from src.models.mlp import MLP
from src.models.team import Team
from src.models.vq import VQ
import numpy as np
from src.models.vae import VAE
import torch
import random
from src.scripts.main import train


def run_trial():
    num_imgs = 1 if not settings.see_distractor else (num_distractors + 1)
    if speaker_type == 'cont':
        speaker = MLP(feature_len, comm_dim, num_layers=3, onehot=False, variational=variational, num_imgs=num_imgs)
    elif speaker_type == 'onehot':
        speaker = MLP(feature_len, comm_dim, num_layers=3, onehot=True, variational=variational, num_imgs=num_imgs)
    elif speaker_type == 'vq':
        speaker = VQ(feature_len, comm_dim, num_layers=3, num_protos=num_prototypes, num_simultaneous_tokens=num_tokens, variational=variational, num_imgs=num_imgs)
    listener = Listener(feature_len,  num_imgs + num_distractors + 1, num_distractors + 1, num_layers=2)
    decoder = Decoder(comm_dim, feature_len, num_layers=3, num_imgs=num_imgs)
    model = Team(speaker, listener, decoder)
    model.to(settings.device)

    train_data = get_feature_data(features_filename, selected_fraction=train_fraction)
    train_topnames, train_responses = get_unique_labels(train_data)
    val_data = get_feature_data(features_filename, excluded_names=train_responses)
    # viz_data = get_feature_data(features_filename, desired_names=viz_names, max_per_class=40)
    viz_data = val_data  # Turn off viz data because we don't use it during trials.
    train(model, train_data, val_data, viz_data, vae=vae, savepath=savepath, comm_dim=comm_dim, num_epochs=num_epochs,
          batch_size=batch_size, burnin_epochs=num_burnin, val_period=val_period,
          plot_comms_flag=False, calculate_complexity=False)


if __name__ == '__main__':
    feature_len = 512
    settings.see_distractor = False
    num_distractors = 1
    num_epochs = 5000
    num_burnin = 500
    val_period = 500  # How often to test on the validation set and calculate various info metrics.
    batch_size = 1024
    comm_dim = 128
    features_filename = 'data/features_nobox.csv'

    viz_names = ['airplane', 'plane',
                 'animal', 'cow', 'dog', 'cat']
    train_fraction = 0.5
    settings.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    settings.kl_weight = 0.00001  # For cont
    settings.kl_incr = 0.0
    settings.num_distractors = num_distractors
    settings.learned_marginal = False
    settings.embedding_cache = {}
    settings.sample_first = True
    settings.alpha = 10
    variational = True

    vae = VAE(512, 32)
    vae.load_state_dict(torch.load('saved_models/vae0.001.pt'))
    vae.to(settings.device)

    num_tokens = 1
    num_unique_messages = 3 ** 8
    num_prototypes = int(num_unique_messages ** (1 / num_tokens))

    seeds = [i for i in range(3)]
    # comm_types = ['vq', 'cont']
    comm_types = ['vq']
    for seed in seeds:
        for speaker_type in comm_types:
            print("Training comm type", speaker_type, "seed", seed)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            savepath = 'saved_models/alpha' + str(settings.alpha) + '_noent_' + str(num_tokens) + 'tok_fixed/' + speaker_type + '/seed' + str(seed) + '/'
            run_trial()
