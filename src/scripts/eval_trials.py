import os
import torch
from src.data_utils.read_data import get_feature_data, get_glove_vectors
import src.settings as settings
from src.models.vae import VAE
from src.scripts.main import eval_model, get_embedding_alignment, evaluate_with_english
from src.utils.performance_metrics import PerformanceMetrics
import numpy as np
from src.data_utils.helper_fns import get_unique_labels, get_entry_for_labels
import random
from src.utils.plotting import plot_scatter
from src.models.decoder import Decoder
from src.models.listener import Listener
from src.models.team import Team
from src.models.vq import VQ

# For a given particular setup, load and evaluate all the checkpoints
def eval_run(basepath, num_tok):
    list_of_files = os.listdir(basepath)
    checkpoints = sorted([int(elt) for elt in list_of_files])

    mean_snapped = []
    mean_nosnap = []
    comps = None
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
            speaker = VQ(feature_len, comm_dim, num_layers=3, num_protos=1024, specified_tok=None,
                         num_simultaneous_tokens=num_tok,
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
            eval_model(team, vae, comm_dim, train_data, train_data, None, glove_data, num_cand_to_metrics=num_cand_to_metrics, savepath=basepath, epoch=checkpoint, calculate_complexity=True, alignment_dataset=alignment_dataset, save_model=False)
            metric = PerformanceMetrics.from_file(basepath + str(checkpoint) + '/train_2_metrics')
            complexities.append(metric.complexities[-1])
        snap_accs = []
        nosnap_accs = []
        for align_data in alignment_datasets:
            _, embed_to_tok, _, _ = get_embedding_alignment(team, align_data, glove_data)
            nosnap, snap = evaluate_with_english(team, train_data, vae, embed_to_tok, glove_data,
                                                        use_top=True,
                                                        num_dist=1)
            snap_accs.append(snap)
            nosnap_accs.append(nosnap)
        mean_snapped.append(np.mean(snap_accs))
        mean_nosnap.append(np.mean(nosnap_accs))
        print("Mean snap", np.mean(snap_accs))
        print("Mean no snap", np.mean(nosnap_accs))
        plot_scatter([complexities, mean_snapped], ['Complexity', 'English snap top'], savepath=basepath + '../snap.png')
        plot_scatter([complexities, mean_nosnap], ['Complexity', 'English nosnap top'], savepath=basepath + '../nosnap.png')


# Iterate over combinations of hyperparameters and seeds.
def run():
    base = 'saved_models/beta0.001'
    for speaker_type in model_types:
        for s in seeds:
            for alpha in alphas:
                for num_tok in num_tokens:
                    setup = 'alpha' + str(alpha) + '_' + str(num_tok) + 'tok'
                    basepath = '/'.join([base, setup, speaker_type, 'seed' + str(s)]) + '/'
                    eval_run(basepath, num_tok)


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
    features_filename = 'data/features_nobox.csv'

    # Load the dataset
    glove_data = get_glove_vectors(comm_dim)
    train_data = get_feature_data(features_filename, selected_fraction=1.0)
    num_align_data = 32
    unique_topnames, _ = get_unique_labels(train_data)
    # Get a dataset of
    subdata = get_entry_for_labels(train_data, unique_topnames)
    # alignment_dataset = train_data[:num_align_data]
    alignment_dataset = subdata
    alignment_datasets = [get_entry_for_labels(train_data, unique_topnames) for i in range(3)]

    vae = VAE(512, 32)
    vae_beta = 0.001
    vae.load_state_dict(torch.load('saved_models/vae' + str(vae_beta) +'.pt'))
    vae.to(settings.device)

    candidates = [2, 8, 16]
    model_types = ['vq']
    seeds = [3]
    alphas = [10]
    num_tokens = [8]
    run()
