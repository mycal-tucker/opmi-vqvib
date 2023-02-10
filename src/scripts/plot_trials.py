import os
from src.utils.performance_metrics import PerformanceMetrics
from src.utils.plotting import plot_scatter, plot_multi_trials, plot_multi_metrics
import numpy as np


def gen_plots(basepath):
    all_complexities = []
    all_informativeness = []
    all_eng = []
    all_accs = {}  # model type to two lists: train and val accs, as a list for each epoch
    snap_eng_accs = {}
    nosnap_eng_accs = {}
    full_paths = []
    # for base in ['train', 'val']:
    for num_candidates in candidates:
        full_paths.append('_'.join([datasplit, distinct, str(num_candidates), 'metrics']))
    epochs = None
    speaker_metrics = [[] for _ in range(len(full_paths))]  # Eval type to seed to metrics
    for seed in seeds:
        seed_dir = basepath + '/seed' + str(seed) + '/'
        if not os.path.exists(seed_dir):
            print("Path doesn't exist", seed_dir)
            continue
        list_of_files = os.listdir(seed_dir)
        # last_seed = 99999 if 'topname' in fieldname else 19999
        last_seed = 99999 if 'topname' in fieldname else 19999
        # print("Using checkpoint", last_seed)
        try:
            for i, metric_path in enumerate(full_paths):
                metric = PerformanceMetrics.from_file(seed_dir + str(last_seed) + '/' + metric_path)
                speaker_metrics[i].append(
                    metric)
                # if metric.complexities[0] is not None:
                #     plot_scatter([metric.complexities, metric.recons], ['Complexity', 'Informativeness'])
        except FileNotFoundError:
            print("Problem loading for", seed_dir)
            continue
        if epochs is None:
            epochs = speaker_metrics[0][-1].epoch_idxs[-burnin:]
    train_complexities = [metric.complexities for metric in speaker_metrics[0]]
    train_info = [metric.recons for metric in speaker_metrics[0]]
    all_complexities.append([item for sublist in train_complexities for item in sublist])
    all_informativeness.append([item for sublist in train_info for item in sublist])
    speaker_accs = [[metric.comm_accs for metric in eval_type] for eval_type in speaker_metrics]
    speaker_accs.append(epochs)  # Need to track this too
    all_accs[speaker_type] = speaker_accs
    # labels = ['snap', 'no_snap']
    # for eng_cand_idx in [2, 1, 0]:
    #     for idx, data_dict in enumerate([snap_eng_accs, nosnap_eng_accs]):
    #         top_eng_accs = [[elt[idx] for elt in metric.top_eng_acc] for metric in speaker_metrics[eng_cand_idx]]
    #         syn_eng_accs = [[elt[idx] for elt in metric.syn_eng_acc] for metric in speaker_metrics[eng_cand_idx]]
    #         top_val_eng_accs = [[elt[idx] for elt in metric.top_val_eng_acc] for metric in speaker_metrics[eng_cand_idx]]
    #         syn_val_eng_accs = [[elt[idx] for elt in metric.syn_val_eng_acc] for metric in speaker_metrics[eng_cand_idx]]
    #         data_dict['vq'] = [top_eng_accs, syn_eng_accs, top_val_eng_accs, syn_val_eng_accs, epochs]
    #         for i in range(len(train_complexities)):
    #             plot_scatter([train_complexities[i], top_eng_accs[i]], ['Complexity', 'English Top'],
    #                          savepath=basepath + labels[idx] + str(eng_cand_idx) + '_eng_comp.png')
    #             plot_scatter([train_complexities[i], syn_eng_accs[i]], ['Complexity', 'English Synonyms'],
    #                          savepath=basepath + labels[idx] + str(eng_cand_idx) + '_syn_eng_comp.png')
    # Add English data, gathered from english_analysis.py
    good_comps = []
    good_infos = []
    for i, c in enumerate(all_complexities[0]):
        if c is not None:
            good_comps.append(c)
            good_infos.append(all_informativeness[0][i])
    # print("Complexities:\n", ', '.join([str(np.round(c, 3)) for c in good_comps]))
    # print("Complexities:\n", '\n'.join([str(np.round(c, 3)) for c in good_comps]))
    # print('\n'.join([str(np.round(c, 3)) for c in good_comps]))
    # print("Infos:\n", ', '.join([str(np.round(elt, 3)) for elt in good_infos]))

    # Get the accuracies at the last measurement.
    good_accs = []
    for a in all_accs[speaker_type][0]:
        final_epoch = a[-1]
        good_accs.append(final_epoch)

    print("Accuracies:\n", ', '.join([str(np.round(a, 3)) for a in good_accs]))
    # print("Accuracies:\n", '\n'.join([str(np.round(a, 3)) for a in good_accs]))
    # print('\n'.join([str(np.round(a, 3)) for a in good_accs]))

    all_complexities.append([1.9])
    all_informativeness.append([-0.20])
    sizes = [20, 20, 60]
    # plot_multi_trials([all_complexities, all_informativeness],
    #                   ['VQ-VIB', 'English'],
    #                   sizes, filename=basepath + 'info_plane.png')
    # plot_multi_trials([all_complexities, all_eng],
    #                   ['VQ-VIB'],
    #                   sizes, filename=basepath + 'eng_')
    plot_multi_metrics(all_accs, labels=['$C=2$', '$C=16$', '$C=32$'], file_root=basepath + distinct + datasplit)


def run():
    base = 'saved_models/' + fieldname + '/trainfrac0.2/' + speaker_type + '/alpha' + str(alpha) + '/' + str(num_tok) + 'tok/klweight' + klweight + '/'
    gen_plots(base)


if __name__ == '__main__':
    # distinct = 'True'
    # candidates = [2]
    distinct = 'False'
    candidates = [32]
    # datasplit = 'train'
    datasplit = 'val'

    fieldname = 'topname'
    # fieldname = 'vg_domain'

    speaker_type = 'vq'
    klweight = '0.01'  # For vq
    # speaker_type = 'onehot'
    # klweight = '0.0'  # For onehot
    # speaker_type = 'proto'
    # klweight = '0.0'  # For proto
    num_tok = 8
    # seeds = [0]
    seeds = [0, 1, 2, 3, 4]
    burnin = 0
    for alpha in [0, 0.1, 0.5, 1, 1.5, 2, 3, 10, 100]:
        print("Alpha", alpha)
        run()
