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
    for base in ['train', 'val']:
        for num_candidates in candidates:
            full_paths.append('_'.join([base, str(num_candidates), 'metrics']))
    for speaker_type, burnin in zip(model_types, burnins):
        epochs = None
        speaker_metrics = [[] for _ in range(len(full_paths))]  # Eval type to seed to metrics
        for seed in seeds:
            seed_dir = basepath + speaker_type + '/seed' + str(seed) + '/'
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
        eng_acc = [metric.top_eng_acc for metric in speaker_metrics[0]]  # Index into having [2, 4, 16] candidates
        all_eng.append([item[0] for sublist in eng_acc for item in sublist])  # Snap to (index 0) or no snap (idx 1)
        all_complexities.append([item for sublist in train_complexities for item in sublist])
        all_informativeness.append([item for sublist in train_info for item in sublist])
        speaker_accs = [[metric.comm_accs for metric in eval_type] for eval_type in speaker_metrics]
        speaker_accs.append(epochs)  # Need to track this too
        all_accs[speaker_type] = speaker_accs
        labels = ['snap', 'no_snap']
        for eng_cand_idx in [2, 1, 0]:
            for idx, data_dict in enumerate([snap_eng_accs, nosnap_eng_accs]):
                top_eng_accs = [[elt[idx] for elt in metric.top_eng_acc] for metric in speaker_metrics[eng_cand_idx]]
                syn_eng_accs = [[elt[idx] for elt in metric.syn_eng_acc] for metric in speaker_metrics[eng_cand_idx]]
                top_val_eng_accs = [[elt[idx] for elt in metric.top_val_eng_acc] for metric in speaker_metrics[eng_cand_idx]]
                syn_val_eng_accs = [[elt[idx] for elt in metric.syn_val_eng_acc] for metric in speaker_metrics[eng_cand_idx]]
                data_dict['vq'] = [top_eng_accs, syn_eng_accs, top_val_eng_accs, syn_val_eng_accs, epochs]
                for i in range(len(train_complexities)):
                    plot_scatter([train_complexities[i], top_eng_accs[i]], ['Complexity', 'English Top'],
                                 savepath=basepath + labels[idx] + str(eng_cand_idx) + '_eng_comp.png')
                    plot_scatter([train_complexities[i], syn_eng_accs[i]], ['Complexity', 'English Synonyms'],
                                 savepath=basepath + labels[idx] + str(eng_cand_idx) + '_syn_eng_comp.png')
    # Add English data, gathered from english_analysis.py
    all_complexities.append([1.9])
    all_informativeness.append([-0.20])
    sizes = [20, 20, 60]
    plot_multi_trials([all_complexities, all_informativeness],
                      ['VQ-VIB', 'English'],
                      sizes, file_root=basepath)
    plot_multi_trials([all_complexities, all_eng],
                      ['VQ-VIB'],
                      sizes, file_root=basepath + 'eng_')
    plot_multi_metrics(all_accs, labels=['Train 2', 'Train 8', 'Train 16', 'Val 2', 'Val 8', 'Val 16'], file_root=basepath)
    plot_multi_metrics(snap_eng_accs, labels=['Train Top', 'Train Syn', 'Val Top', 'Val Syn'], file_root=basepath + 'snap_eng_')
    plot_multi_metrics(nosnap_eng_accs, labels=['Train Top', 'Train Syn', 'Val Top', 'Val Syn'], file_root=basepath + 'nosnap_eng_')


def run():
    base = 'saved_models/beta0.001'
    for alpha in [10]:
        # for num_tok in [1, 2, 4, 8]:
        for num_tok in [8]:
            setup = 'alpha' + str(alpha) + '_' + str(num_tok) + 'tok'
            basepath = base + '/' + setup + '/'
            gen_plots(basepath)


if __name__ == '__main__':
    # candidates = [2, 4, 8]
    candidates = [2, 8, 16]
    # model_types = ['cont', 'vq']
    model_types = ['vq']
    # seeds = [1, 2]
    seeds = [0, 1, 3]
    burnins = [0, 0, 0, 0, 0]
    run()
