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
    num_cand_to_metrics = {2: [], 8: [], 16: []}
    for empty_list in num_cand_to_metrics.values():
        empty_list.extend([PerformanceMetrics(), PerformanceMetrics()])  # Train metrics, validation metrics
    complexities = []
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
        else:  # If we didn't calculate complexity during the training run
            print("Running full eval to get complexity")
            eval_model(team, vae, comm_dim, train_data, train_data, None, glove_data,
                       num_cand_to_metrics=num_cand_to_metrics, savepath=basepath, epoch=checkpoint,
                       calculate_complexity=True, alignment_dataset=alignment_dataset, save_model=False)
            metric = PerformanceMetrics.from_file(basepath + str(checkpoint) + '/train_2_metrics')
            complexities.append(metric.complexities[-1])
        top_snap_accs = []
        top_nosnap_accs = []
        syn_snap_accs = []
        syn_nosnap_accs = []
        for align_data in alignment_datasets:
            _, embed_to_tok, _, _ = get_embedding_alignment(team, align_data, glove_data)
            nosnap, snap = evaluate_with_english(team, train_data, vae, embed_to_tok, glove_data,
                                                        use_top=True,
                                                        num_dist=1)
            top_snap_accs.append(snap)
            top_nosnap_accs.append(nosnap)
            nosnap, snap = evaluate_with_english(team, train_data, vae, embed_to_tok, glove_data,
                                                        use_top=False,
                                                        num_dist=1)
            syn_snap_accs.append(snap)
            syn_nosnap_accs.append(nosnap)
        mean_top_snapped.append(np.mean(top_snap_accs))
        mean_top_nosnap.append(np.mean(top_nosnap_accs))
        mean_syn_snapped.append(np.mean(syn_snap_accs))
        mean_syn_nosnap.append(np.mean(syn_nosnap_accs))
        print("Mean top snap", np.mean(top_snap_accs))
        print("Mean top no snap", np.mean(top_nosnap_accs))
        print("Mean syn snap", np.mean(syn_snap_accs))
        print("Mean syn no snap", np.mean(syn_nosnap_accs))
        plot_scatter([complexities, mean_top_snapped], ['Complexity', 'English snap top'], savepath=basepath + '../top_snap.png')
        plot_scatter([complexities, mean_top_nosnap], ['Complexity', 'English nosnap top'], savepath=basepath + '../top_nosnap.png')
        plot_scatter([complexities, mean_syn_snapped], ['Complexity', 'English snap top'], savepath=basepath + '../syn_snap.png')
        plot_scatter([complexities, mean_syn_nosnap], ['Complexity', 'English nosnap top'], savepath=basepath + '../syn_nosnap.png')


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

    comm_dim = 64
    # comm_dim = 512
    features_filename = 'data/features_nobox.csv'

    # Load the dataset
    glove_data = get_glove_vectors(comm_dim)
    train_data = get_feature_data(features_filename, selected_fraction=1.0)
    # Use hardcoded subsets based on index
    # num_align_data = 32
    # alignment_dataset = train_data[:num_align_data]
    # alignment_datasets = [train_data[i * num_align_data: (i + 1) * num_align_data] for i in range(3)]
    # Use a dataset generated as one label for each English topname
    unique_topnames, _ = get_unique_labels(train_data)
    alignment_datasets = [get_entry_for_labels(train_data, unique_topnames) for i in range(3)]
    subdata = get_entry_for_labels(train_data, unique_topnames)
    alignment_dataset = subdata

    vae = VAE(512, 32)
    vae_beta = 0.001
    vae.load_state_dict(torch.load('saved_models/vae' + str(vae_beta) +'.pt'))
    vae.to(settings.device)

    candidates = [2, 8, 16]
    model_types = ['onehot']
    seeds = [1]
    alphas = [10]
    num_tokens = [16]
    run()
