import os
import random

import numpy as np
import torch

import src.settings as settings
from src.data_utils.helper_fns import get_unique_labels, get_entry_for_labels
from src.data_utils.read_data import get_feature_data, get_glove_vectors
from src.models.decoder import Decoder
from src.models.listener import Listener
from src.models.mlp import MLP
from src.models.team import Team
from src.models.vae import VAE
from src.models.vq import VQ
from src.models.vq2 import VQ2
from src.scripts.main import eval_model, get_embedding_alignment, evaluate_with_english
from src.utils.performance_metrics import PerformanceMetrics
from src.utils.plotting import plot_scatter


# For a given particular setup, load and evaluate all the checkpoints
def eval_run(basepath, num_tok, speaker_type):
    list_of_files = os.listdir(basepath)
    checkpoints = sorted([int(elt) for elt in list_of_files])

    mean_top_snapped = []
    mean_top_nosnap = []
    mean_syn_snapped = []
    mean_syn_nosnap = []
    mean_top_eng = []
    mean_syn_eng = []
    num_cand_to_metrics = {2: [], 8: [], 16: []}
    for empty_list in num_cand_to_metrics.values():
        empty_list.extend([PerformanceMetrics(), PerformanceMetrics()])  # Train metrics, validation metrics
    complexities = []
    mses = []
    eng_decoder = torch.load('english64.pt').to(settings.device)
    eng_listener = torch.load('english_list64.pt').to(settings.device)
    for idx, checkpoint in enumerate(checkpoints):
        print("\n\n\nCheckpoint", checkpoint)
        # Load the model
        try:
            team = torch.load(basepath + str(checkpoint) + '/model_obj.pt')
        except FileNotFoundError:
            print("Failed to load the preferred full model; falling back to statedict.")
            feature_len = 512
            if speaker_type == 'vq':
                speaker = VQ(feature_len, comm_dim, num_layers=3, num_protos=1024, specified_tok=None,
                             num_simultaneous_tokens=num_tok,
                             variational=True, num_imgs=1)
            if speaker_type == 'vq2':
                speaker = VQ2(feature_len, comm_dim, num_layers=3, num_protos=1024, specified_tok=None,
                             num_simultaneous_tokens=num_tok,
                             variational=True, num_imgs=1)
            elif speaker_type == 'onehot':
                speaker = MLP(feature_len, comm_dim, num_layers=3, onehot=True, num_simultaneous_tokens=num_tok,
                              variational=True, num_imgs=1)
            listener = Listener(feature_len)
            decoder = Decoder(comm_dim, feature_len, num_layers=3, num_imgs=1)
            team = Team(speaker, listener, decoder)
            team.load_state_dict(torch.load(basepath + str(checkpoint) + '/model.pt'))
            team.to(settings.device)
        # And evaluate it
        metric = PerformanceMetrics.from_file(basepath + str(checkpoint) + '/train_2_metrics')
        comps = metric.complexities
        if comps[-1] is not None:
            complexities.append(comps[-1])
            mses.append(-1 * metric.recons[-1])
        else:  # If we didn't calculate complexity during the training run
            print("Running full eval to get complexity")
            mses.append(None)
            eval_model(team, vae, comm_dim, train_data, train_data, None, glove_data,
                       num_cand_to_metrics=num_cand_to_metrics, savepath=basepath, epoch=checkpoint,
                       calculate_complexity=True, alignment_dataset=alignment_dataset, save_model=False)
            metric = PerformanceMetrics.from_file(basepath + str(checkpoint) + '/train_2_metrics')
            complexities.append(metric.complexities[-1])
        top_snap_accs = []
        top_nosnap_accs = []
        syn_snap_accs = []
        syn_nosnap_accs = []
        top_eng_accs = []
        syn_eng_accs = []
        for align_data in alignment_datasets:
            tok_to_embed, embed_to_tok, _, _, comm_map = get_embedding_alignment(team, align_data, glove_data, use_comm_idx=True)
            nosnap, snap, eng_stuff = evaluate_with_english(team, train_data, vae, embed_to_tok, glove_data,
                                                            use_top=True,
                                                            num_dist=1,
                                                            eng_dec=eng_decoder,
                                                            eng_list=eng_listener,
                                                            tok_to_embed=tok_to_embed, use_comm_idx=True, comm_map=comm_map)
            top_snap_accs.append(snap)
            top_nosnap_accs.append(nosnap)
            top_eng_accs.append(eng_stuff)

            tok_to_embed, embed_to_tok, _, _, _ = get_embedding_alignment(team, align_data, glove_data, use_comm_idx=False)
            nosnap, snap, eng_stuff = evaluate_with_english(team, train_data, vae, embed_to_tok, glove_data,
                                                            use_top=True,
                                                            num_dist=1,
                                                            eng_dec=eng_decoder,
                                                            eng_list=eng_listener,
                                                            tok_to_embed=tok_to_embed, use_comm_idx=False)
            syn_snap_accs.append(snap)
            syn_nosnap_accs.append(nosnap)
            syn_eng_accs.append(eng_stuff)
        mean_top_snapped.append(np.mean(top_snap_accs))
        mean_top_nosnap.append(np.mean(top_nosnap_accs))
        mean_syn_snapped.append(np.mean(syn_snap_accs))
        mean_syn_nosnap.append(np.mean(syn_nosnap_accs))
        mean_top_eng.append(np.mean(top_eng_accs))
        mean_syn_eng.append(np.mean(syn_eng_accs))
        print("Mean top snap", np.mean(top_snap_accs), np.std(top_snap_accs))
        print("Mean top no snap", np.mean(top_nosnap_accs), np.std(top_snap_accs))
        print("Mean syn snap", np.mean(syn_snap_accs), np.std(syn_snap_accs))
        print("Mean syn no snap", np.mean(syn_nosnap_accs), np.std(syn_nosnap_accs))
        print("Mean top eng", np.mean(top_eng_accs), np.std(top_eng_accs))
        print("Mean syn eng", np.mean(syn_eng_accs), np.std(syn_eng_accs))
        print("Complexities", complexities)
        plot_scatter([complexities, mean_top_snapped], ['Complexity', 'English to tok snap top'],
                     savepath=basepath + '../top_snap.png')
        plot_scatter([complexities, mean_top_nosnap], ['Complexity', 'English to tok nosnap top'],
                     savepath=basepath + '../top_nosnap.png')
        plot_scatter([complexities, mean_syn_snapped], ['Complexity', 'English to tok snap syn'],
                     savepath=basepath + '../syn_snap.png')
        plot_scatter([complexities, mean_syn_nosnap], ['Complexity', 'English to tok nosnap syn'],
                     savepath=basepath + '../syn_nosnap.png')
        plot_scatter([complexities, mean_top_eng], ['Complexity', 'Tok to Eng top'],
                     savepath=basepath + '../top_eng.png')
        plot_scatter([mses, mean_top_eng], ['MSE', 'Tok to Eng top'],
                     savepath=basepath + '../top_eng_mse.png')
        plot_scatter([complexities, mean_syn_eng], ['Complexity', 'Tok to Eng syn'],
                     savepath=basepath + '../syn_eng.png')
        plot_scatter([mses, mean_syn_eng], ['MSE', 'Tok to Eng syn'],
                     savepath=basepath + '../syn_eng_mse.png')


# Iterate over combinations of hyperparameters and seeds.
def run():
    base = 'saved_models/beta0.001'
    for speaker_type in model_types:
        for s in seeds:
            for alpha in alphas:
                for num_tok in num_tokens:
                    setup = 'alpha' + str(alpha) + '_' + str(num_tok) + 'tok'
                    basepath = '/'.join([base, setup, speaker_type, 'seed' + str(s)]) + '/'
                    eval_run(basepath, num_tok, speaker_type)


if __name__ == '__main__':
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    settings.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    settings.num_distractors = 1
    settings.see_distractor = False
    settings.learned_marginal = False
    settings.embedding_cache = {}
    settings.sample_first = True
    settings.hardcoded_vq = False
    settings.kl_weight = 0.0
    settings.epoch = 0

    # comm_dim = 64
    comm_dim = 64
    features_filename = 'data/features_nobox.csv'

    # Load the dataset
    embed_dim = 64
    glove_data = get_glove_vectors(embed_dim)
    train_data = get_feature_data(features_filename, selected_fraction=1.0)
    # Use hardcoded subsets based on index
    # num_align_data = 32
    # alignment_dataset = train_data[:num_align_data]
    # alignment_datasets = [train_data[i * num_align_data: (i + 1) * num_align_data] for i in range(3)]
    # Use a dataset generated as one label for each English topname
    unique_topnames, _ = get_unique_labels(train_data)
    alignment_datasets = [get_entry_for_labels(train_data, unique_topnames) for i in range(1)]
    subdata = get_entry_for_labels(train_data, unique_topnames)
    alignment_dataset = subdata

    vae = VAE(512, 32)
    vae_beta = 0.001
    vae.load_state_dict(torch.load('saved_models/vae' + str(vae_beta) + '.pt'))
    vae.to(settings.device)

    candidates = [2, 8, 16]
    model_types = ['vq2']
    seeds = [1]
    alphas = [10]
    num_tokens = [1]
    run()
