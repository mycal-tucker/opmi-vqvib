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
                    speaker_metrics[i].append(
                        PerformanceMetrics.from_file(seed_dir + str(last_seed) + '/' + metric_path))
            except FileNotFoundError:
                print("Problem loading for", seed_dir)
                continue
            if epochs is None:
                epochs = speaker_metrics[0][-1].epoch_idxs[-burnin:]
        # plot_scatter([train_complexities, train_informativeness], ['Complexity', 'Informativeness'])
        # all_complexities.append(train_complexities)
        # all_informativeness.append(train_informativeness)
        speaker_accs = [[metric.comm_accs for metric in eval_type] for eval_type in speaker_metrics]
        speaker_accs.append(epochs)  # Need to track this too
        all_accs[speaker_type] = speaker_accs
        top_eng_accs = [[metric.top_eng_acc for metric in eval_type] for eval_type in speaker_metrics]
        syn_eng_accs = [[metric.syn_eng_acc for metric in eval_type] for eval_type in speaker_metrics]
        top_eng_accs.append(epochs)
        syn_eng_accs.append(epochs)
        eng_accs['top'] = top_eng_accs
        eng_accs['syn'] = syn_eng_accs
    # Add English data, gathered from english_analysis.py
    all_complexities.append([1.54])
    all_informativeness.append([-0.00011])
    sizes = [20, 20, 60]
    plot_multi_trials([all_complexities, all_informativeness],
                      ['Cont.', 'VQ-VIB', 'English'],
                      sizes)
    plot_multi_metrics(all_accs, file_root=basepath)
    plot_multi_metrics(eng_accs, file_root=basepath + 'eng_')


def run():
    base = 'saved_models/beta0.01'
    for alpha in [10]:
        # for num_tok in [1, 2, 4, 8]:
        for num_tok in [1, 2, 8]:
            setup = 'alpha' + str(alpha) + '_' + str(num_tok) + 'tok'
            basepath = base + '/' + setup + '/'
            gen_plots(basepath)


if __name__ == '__main__':
    # candidates = [2, 4, 8]
    candidates = [2, 8, 16]
    # model_types = ['cont', 'vq']
    model_types = ['vq']
    seeds = [0, 1, 2, 3, 4]
    # seeds = [0]
    burnins = [0, 0, 0, 0, 0]
    run()
