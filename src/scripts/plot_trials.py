import os
from src.utils.performance_metrics import PerformanceMetrics
from src.utils.plotting import plot_scatter, plot_multi_trials, plot_multi_metrics


def gen_plots(basepath):
    all_complexities = []
    all_informativeness = []
    all_eng = []
    all_accs = {}  # model type to two lists: train and val accs, as a list for each epoch
    snap_eng_accs = {}
    nosnap_eng_accs = {}
    full_paths = []
    # for base in ['train', 'val']:
    for base in ['train']:
        for num_candidates in candidates:
            full_paths.append('_'.join([base, distinct, str(num_candidates), 'metrics']))
    epochs = None
    speaker_metrics = [[] for _ in range(len(full_paths))]  # Eval type to seed to metrics
    for seed in seeds:
        seed_dir = basepath + '/seed' + str(seed) + '/'
        if not os.path.exists(seed_dir):
            print("Path doesn't exist", seed_dir)
            continue
        list_of_files = os.listdir(seed_dir)
        last_seed = max([int(f) for f in list_of_files])
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
    all_complexities.append([1.9])
    all_informativeness.append([-0.20])
    sizes = [20, 20, 60]
    plot_multi_trials([all_complexities, all_informativeness],
                      ['VQ-VIB', 'English'],
                      sizes, filename=basepath + 'info_plane.png')
    plot_multi_trials([all_complexities, all_eng],
                      ['VQ-VIB'],
                      sizes, filename=basepath + 'eng_')
    plot_multi_metrics(all_accs, labels=['2', '16', '32', 'OOD 16', 'OOD 32'], file_root=basepath + distinct)
    plot_multi_metrics(snap_eng_accs, labels=['Train Top', 'Train Syn', 'Val Top', 'Val Syn'], file_root=basepath + 'snap_eng_')
    plot_multi_metrics(nosnap_eng_accs, labels=['Train Top', 'Train Syn', 'Val Top', 'Val Syn'], file_root=basepath + 'nosnap_eng_')


def run():
    base = 'saved_models/' + fieldname + '/trainfrac1.0/' + speaker_type + '/alpha' + str(alpha) + '/' + str(num_tok) + 'tok/entropyweight' + entropyweight + '/'
    gen_plots(base)


if __name__ == '__main__':
    # distinct = 'True'
    # candidates = [2]
    distinct = 'False'
    candidates = [2, 16, 32]

    fieldname = 'all'

    speaker_type = 'vq'
    entropyweight = '0.0'
    alpha = 10
    num_tok = 1
    # seeds = [1, 2]
    seeds = [0, 1, 2, 3, 4]
    burnin = 0
    run()
