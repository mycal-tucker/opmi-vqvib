import os
from src.utils.performance_metrics import PerformanceMetrics
from src.utils.plotting import plot_scatter, plot_multi_trials, plot_multi_metrics


def gen_plots(basepath):
    all_complexities = []
    all_informativeness = []
    all_accs = {}  # model type to two lists: train and val accs, as a list for each epoch
    eng_accs = {}
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
                    if metric.complexities[0] is not None:
                        plot_scatter([metric.complexities, metric.recons], ['Complexity', 'Informativeness'])
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
        top_eng_accs = [metric.top_eng_acc for metric in speaker_metrics[0]]
        syn_eng_accs = [metric.syn_eng_acc for metric in speaker_metrics[0]]
        top_val_eng_accs = [metric.top_val_eng_acc for metric in speaker_metrics[0]]
        syn_val_eng_accs = [metric.syn_val_eng_acc for metric in speaker_metrics[0]]
        eng_accs['vq'] = [top_eng_accs, syn_eng_accs, top_val_eng_accs, syn_val_eng_accs, epochs]
    # Add English data, gathered from english_analysis.py
    all_complexities.append([1.54])
    all_informativeness.append([-0.24])
    sizes = [20, 20, 60]
    plot_multi_trials([all_complexities, all_informativeness],
                      ['VQ-VIB', 'English'],
                      sizes, file_root=basepath)
    plot_multi_metrics(all_accs, labels=['Train 2', 'Train 8', 'Train 16', 'Val 2', 'Val 8', 'Val 16'], file_root=basepath)
    plot_multi_metrics(eng_accs, labels=['Train Top', 'Train Syn', 'Val Top', 'Val Syn'], file_root=basepath + 'eng_')


def run():
    base = 'saved_models/beta0.001'
    for alpha in [10]:
        # for num_tok in [1, 2, 4, 8]:
        for num_tok in [1]:
            setup = 'alpha' + str(alpha) + '_' + str(num_tok) + 'tok'
            basepath = base + '/' + setup + '/'
            gen_plots(basepath)


if __name__ == '__main__':
    # candidates = [2, 4, 8]
    candidates = [2, 8, 16]
    # model_types = ['cont', 'vq']
    model_types = ['vq']
    # seeds = [1, 2]
    seeds = [0, 1]
    burnins = [0, 0, 0, 0, 0]
    run()
