from src.models.decoder import Decoder
from src.models.mlp import MLP
from src.models.vae import VAE
from src.models.vq import VQ
from src.models.listener import Listener
from src.models.team import Team
import torch
import torch.nn as nn
import torch.optim as optim
import os
from src.data_utils.read_data import get_feature_data, get_glove_vectors
import src.settings as settings
from src.data_utils.helper_fns import gen_batch
from src.utils.mine import get_info


def run_seed(speaker, decoder, team):
    optimizer = optim.Adam(decoder.parameters())  # Only update the decoder

    running_mse = 0
    for epoch in range(300):
        # if epoch % 50 == 0:
        #     print("Epoch", epoch)
        #     print("Running mse", running_mse)
        optimizer.zero_grad()
        speaker_obs, _, _, _ = gen_batch(train_data, batch_size, fieldname='vg_domain', vae=vae, glove_data=glove_data)
        with torch.no_grad():
            comm, _, _ = speaker(speaker_obs)
        recons = decoder(comm)
        loss = torch.mean(((speaker_obs - torch.squeeze(recons, 1)) ** 2))
        loss.backward()
        optimizer.step()
        running_mse = running_mse * 0.95 + 0.05 * loss.item()
    print("Distortion", running_mse)
    complexity = get_info(team, train_data, targ_dim=comm_dim, glove_data=glove_data, num_epochs=200)
    print("Complexity", complexity)
    return running_mse, complexity


def run():
    infos = []
    comps = []
    for s in seeds:
        # Load the last checkpoint
        seedpath = base + '/seed' + str(s)
        list_of_files = os.listdir(seedpath)
        checkpoints = sorted([int(elt) for elt in list_of_files])
        last_checkpoint = checkpoints[-1]
        print("Loading checkpoint", last_checkpoint)

        checkpoint_dir = seedpath + '/' + str(last_checkpoint)

        decoder = Decoder(comm_dim, feature_len, num_layers=3, num_imgs=1)
        if speaker_type == 'vq':            speaker = VQ(feature_len, comm_dim, num_layers=3, num_protos=1024, specified_tok=None,
                         num_simultaneous_tokens=num_tok,
                         variational=True, num_imgs=1)
        elif speaker_type == 'onehot':
            speaker = MLP(feature_len, comm_dim, num_layers=3, onehot=True, num_simultaneous_tokens=num_tok,
                          variational=True, num_imgs=1)
        listener = Listener(feature_len)
        team = Team(speaker, listener, decoder)
        team.load_state_dict(torch.load(checkpoint_dir + '/model.pt'))
        speaker = team.speaker
        # decoder = team.decoder  # If you want, can initialize to the previous decoder
        team.decoder = decoder
        speaker.trainable = False
        distortion, comp = run_seed(speaker, decoder, team)
        infos.append(-1 * distortion)
        comps.append(comp)
    print(infos)
    print(comps)
    print()


if __name__ == '__main__':
    settings.device = 'cpu'
    features_filename = 'data/features_nobox.csv'
    train_data = get_feature_data(features_filename)
    settings.embedding_cache = {}
    vae = VAE(512, 32)
    vae.load_state_dict(torch.load('saved_models/vae0.001.pt'))
    vae.to(settings.device)
    batch_size = 256


    settings.distinct_word = False
    settings.num_distractors = 0
    settings.entropy_weight = 0
    settings.kl_weight = 0
    settings.epoch = 0
    settings.alpha = 0

    feature_len = 512
    fieldname = 'vg_domain'
    speaker_type = 'vq'
    comm_dim = 1024 if speaker_type == 'onehot' else 64
    entropyweight = '0.0'
    alpha = 10
    num_tok = 8
    seeds = [0, 1, 2, 3, 4]

    glove_data = get_glove_vectors(comm_dim)

    base = '/'.join(['saved_models', fieldname, 'trainfrac1.0', speaker_type, 'alpha' + str(alpha),
                     str(num_tok) + 'tok', 'entropyweight0.0'])
    run()
