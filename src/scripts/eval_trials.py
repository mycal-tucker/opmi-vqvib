import os
import random

import numpy as np
import torch

import src.settings as settings
from src.data_utils.helper_fns import get_unique_labels, get_entry_for_labels, get_unique_by_field
from src.data_utils.read_data import get_feature_data, get_glove_vectors
from src.models.decoder import Decoder
from src.models.listener import Listener
from src.models.mlp import MLP
from src.models.team import Team
from src.models.vae import VAE
from src.models.vq import VQ
from src.models.vq2 import VQ2
from src.models.vq_after import VQAfter
from src.scripts.main import eval_model, get_embedding_alignment, evaluate_with_english
from src.utils.performance_metrics import PerformanceMetrics
from src.utils.plotting import plot_scatter, plot_multi_trials


# For a given particular setup, load and evaluate all the checkpoints
def eval_run(basepath, num_tok, speaker_type):
    list_of_files = os.listdir(basepath)
    checkpoints = sorted([int(elt) for elt in list_of_files])

    mean_top_snapped = []
    mean_top_nosnap = []
    mean_syn_snapped = []
    mean_syn_nosnap = []
    mean_top_eng_comm_id = []
    mean_syn_eng_comm_id = []
    mean_top_eng = []
    mean_syn_eng = []
    complexities = []
    mses = []
    eng_decoder = torch.load('english_vg_dec64.pt').to(settings.device)
    eng_listener = torch.load('english_vg_list64.pt').to(settings.device)
    for idx, checkpoint in enumerate(checkpoints):  # TODO: just use the last checkpoint instead of iterating through?
        if checkpoint != 9999:
            print("Skipping checkpoint", checkpoint)
            continue
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
            if speaker_type == 'vq_after':
                speaker = VQAfter(feature_len, comm_dim, num_layers=3, num_protos=1024, specified_tok=None,
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
        metric = PerformanceMetrics.from_file(basepath + str(checkpoint) + '/train_True_2_metrics')
        comps = metric.complexities
        if comps[-1] is not None:
            complexities.append(comps[-1])
            mses.append(-1 * metric.recons[-1])
        else:  # If we didn't calculate complexity during the training run
            print("Running full eval to get complexity")
            mses.append(None)
            complexities.append(None)
            # eval_model(team, vae, comm_dim, train_data, train_data, None, glove_data,
            #            num_cand_to_metrics=num_cand_to_metrics, savepath=basepath, epoch=checkpoint,
            #            calculate_complexity=True, alignment_dataset=alignment_dataset, save_model=False)
            # metric = PerformanceMetrics.from_file(basepath + str(checkpoint) + '/train_2_metrics')
            # complexities.append(metric.complexities[-1])
        top_snap_accs = []
        top_nosnap_accs = []
        syn_snap_accs = []
        syn_nosnap_accs = []
        top_eng_comm_id_accs = []
        syn_eng_comm_id_accs = []
        top_eng_accs = []
        syn_eng_accs = []
        for align_data in alignment_datasets:
            for use_comm_idx in [True, False]:
                tok_to_embed, embed_to_tok, _, _, comm_map = get_embedding_alignment(team, align_data, glove_data,
                                                                                     fieldname=experiment_fieldname)
                for use_top in [True, False]:
                    nosnap, snap, eng_stuff = evaluate_with_english(team, train_data, vae, embed_to_tok, glove_data,
                                                                    fieldname=experiment_fieldname,
                                                                    use_top=True,
                                                                    num_dist=1,
                                                                    eng_dec=eng_decoder,
                                                                    eng_list=eng_listener,
                                                                    tok_to_embed=tok_to_embed,
                                                                    use_comm_idx=use_comm_idx, comm_map=comm_map)
                    if use_comm_idx:
                        if use_top:
                            top_snap_accs.append(snap)
                            top_nosnap_accs.append(nosnap)
                            top_eng_comm_id_accs.append(eng_stuff)
                        else:
                            syn_snap_accs.append(snap)
                            syn_nosnap_accs.append(nosnap)
                            syn_eng_comm_id_accs.append(eng_stuff)
                    else:
                        if use_top:
                            top_eng_accs.append(eng_stuff)
                        else:
                            syn_eng_accs.append(eng_stuff)
        mean_top_snapped.append(np.mean(top_snap_accs))
        mean_top_nosnap.append(np.mean(top_nosnap_accs))
        mean_syn_snapped.append(np.mean(syn_snap_accs))
        mean_syn_nosnap.append(np.mean(syn_nosnap_accs))
        mean_top_eng_comm_id.append(np.mean(top_eng_comm_id_accs))
        mean_syn_eng_comm_id.append(np.mean(syn_eng_comm_id_accs))
        mean_top_eng.append(np.mean(top_eng_accs))
        mean_syn_eng.append(np.mean(syn_eng_accs))
        print("Mean top snap", np.mean(top_snap_accs), np.std(top_snap_accs))
        print("Mean top no snap", np.mean(top_nosnap_accs), np.std(top_snap_accs))
        print("Mean syn snap", np.mean(syn_snap_accs), np.std(syn_snap_accs))
        print("Mean syn no snap", np.mean(syn_nosnap_accs), np.std(syn_nosnap_accs))
        print("Mean top eng", np.mean(top_eng_accs), np.std(top_eng_accs))
        print("Mean syn eng", np.mean(syn_eng_accs), np.std(syn_eng_accs))
        print("Complexities", complexities)
        # plot_scatter([complexities, mean_top_snapped], ['Complexity', 'English to tok snap top'],
        #              savepath=basepath + '../eng_to_ec_top_snap.png')
        # plot_scatter([complexities, mean_top_nosnap], ['Complexity', 'English to tok nosnap top'],
        #              savepath=basepath + '../eng_to_ec_top_nosnap.png')
        # plot_scatter([complexities, mean_syn_snapped], ['Complexity', 'English to tok snap syn'],
        #              savepath=basepath + '../eng_to_ec_syn_snap.png')
        # plot_scatter([complexities, mean_syn_nosnap], ['Complexity', 'English to tok nosnap syn'],
        #              savepath=basepath + '../eng_to_ec_syn_nosnap.png')
        # plot_scatter([complexities, mean_top_eng], ['Complexity', 'Tok to Eng top'],
        #              savepath=basepath + '../ec_to_eng_top.png')
        # plot_scatter([complexities, mean_syn_eng], ['Complexity', 'Tok to Eng syn'],
        #              savepath=basepath + '../ec_to_eng_syn.png')
        # plot_scatter([complexities, mean_top_eng_comm_id], ['Complexity', 'Tok to Eng top'],
        #              savepath=basepath + '../ec_to_eng_top_comm_id.png')
        # plot_scatter([complexities, mean_syn_eng_comm_id], ['Complexity', 'Tok to Eng syn'],
        #              savepath=basepath + '../ec_to_eng_syn_comm_id.png')
    num_runs = len(top_eng_accs)
    return complexities[-1], (np.mean(top_eng_comm_id_accs), np.std(top_eng_comm_id_accs) / np.sqrt(num_runs)),\
           (np.mean(top_eng_accs), np.std(top_eng_accs) / np.sqrt(num_runs)), (np.mean(top_nosnap_accs), np.std(top_nosnap_accs) / np.sqrt(num_runs))


# Iterate over combinations of hyperparameters and seeds.
def run():
    # base = 'saved_models/topname/trainfrac1.0/'
    base = 'saved_models/all/trainfrac1.0/'
    for speaker_type in model_types:
        for alpha in alphas:
            for num_tok in num_tokens:
                all_comps = []
                all_top_ec_to_eng_comm_ids = [[], []]  # Mean, std.
                all_top_ec_to_eng = [[], []]
                all_top_eng_to_ec = [[], []]
                labels = []
                for entropy_weight in entropy_weights:
                    comps = []
                    top_ec_to_eng_comm_ids = []
                    top_ec_to_engs = []
                    top_eng_to_ecs = []
                    basepath = '/'.join([base, speaker_type, 'alpha' + str(alpha), str(num_tok) + 'tok', 'entropyweight' + str(entropy_weight)]) + '/'
                    for s in seeds:
                        comp, top_ec_to_eng_comm_id, top_ec_to_eng, top_eng_to_ec = eval_run(basepath + '/seed' + str(s) + '/', num_tok, speaker_type)
                        comps.append(comp)
                        top_ec_to_eng_comm_ids.append(top_ec_to_eng_comm_id)
                        top_ec_to_engs.append(top_ec_to_eng)
                        top_eng_to_ecs.append(top_eng_to_ec)
                    all_comps.append(comps)
                    for i in range(2):
                        all_top_ec_to_eng_comm_ids[i].append([elt[i] for elt in top_ec_to_eng_comm_ids])
                        all_top_ec_to_eng[i].append([elt[i] for elt in top_ec_to_engs])
                        all_top_eng_to_ec[i].append([elt[i] for elt in top_eng_to_ecs])
                    labels.append('H Weight ' + str(entropy_weight))
                    plot_multi_trials([all_comps, all_top_ec_to_eng_comm_ids[0], all_top_ec_to_eng_comm_ids[1]],
                                      labels,
                                      [20 for _ in labels],
                                      ylabel='Utility EC to Eng Top Comm ID',
                                      filename='/'.join([basepath, 'EC_to_Eng_Top_Comm_ID']))
                    plot_multi_trials([all_comps, all_top_ec_to_eng[0], all_top_ec_to_eng[1]],
                                      labels,
                                      [20 for _ in labels],
                                      ylabel='Utility EC to Eng Top',
                                      filename='/'.join([basepath, 'EC_to_Eng_Top']))
                    plot_multi_trials([all_comps, all_top_eng_to_ec[0], all_top_eng_to_ec[1]],
                                      labels,
                                      [20 for _ in labels],
                                      ylabel='Utility Eng to EC Top',
                                      filename='/'.join([basepath, 'Eng_to_EC_Top']))


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
    english_fieldname = 'vg_domain'
    experiment_fieldname = 'topname'  # FIXME
    settings.distinct_words = False  # FIXME
    settings.entropy_weight = 0.0

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
    # Use a dataset generated as one label for each English topname or vg_domain label
    unique_topnames = get_unique_by_field(train_data, english_fieldname)  # FIXME: can generate alignment data using one dataset and English words from another.
    print(len(unique_topnames), "unique classes for fieldname", english_fieldname)
    alignment_datasets = [get_entry_for_labels(train_data, unique_topnames, fieldname=english_fieldname, num_repeats=1) for i in range(3)]

    vae = VAE(512, 32)
    vae_beta = 0.001
    vae.load_state_dict(torch.load('saved_models/vae' + str(vae_beta) + '.pt'))
    vae.to(settings.device)

    candidates = [2]
    model_types = ['vq']
    seeds = [0, 1, 2, 3, 4]
    entropy_weights = [0.01, 0.03, 0.05, 0.06]
    # entropy_weights = [0.0, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
    alphas = [10]
    num_tokens = [1]
    run()
