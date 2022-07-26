import os
from src.utils.performance_metrics import PerformanceMetrics
from src.utils.plotting import plot_scatter, plot_multi_trials, plot_multi_metrics


def gen_plots(basepath):
    all_complexities = []
    all_informativeness = []
    all_accs = {}  # model type to two lists: train and val accs, as a list for each epoch
    for speaker_type, burnin in zip(model_types, burnins):
        epochs = None
        speaker_metrics = [[] for _ in range(6)]  # Eval type to seed to metrics
        for seed in seeds:
            seed_dir = basepath + speaker_type + '/seed' + str(seed) + '/'
            if not os.path.exists(seed_dir):
                print("Path doesn't exist", seed_dir)
                continue
            list_of_files = os.listdir(seed_dir)
            last_seed = max([int(f) for f in list_of_files])
            try:
                for i, metric_path in enumerate(
                        ['train_2_metrics', 'train_4_metrics', 'train_8_metrics', 'val_2_metrics', 'val_4_metrics',
                         'val_8_metrics']):
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
    # Add English data, gathered from english_analysis.py
    all_complexities.append([1.54])
    all_informativeness.append([-0.00011])
    sizes = [20, 20, 60]
    plot_multi_trials([all_complexities, all_informativeness],
                      ['Cont.', 'VQ-VIB', 'English'],
                      sizes)
    plot_multi_metrics(all_accs, file_root=basepath)


def run():
    base = 'saved_models'
    for alpha in [0, 10]:
        for num_tok in [1, 2, 4, 8]:
            setup = 'alpha' + str(alpha) + '_' + str(num_tok) + 'tok'
            basepath = base + '/' + setup + '/'
            gen_plots(basepath)


if __name__ == '__main__':
    # model_types = ['cont', 'vq']
    model_types = ['vq']
    # seeds = [0, 1, 2, 3, 4]
    seeds = [0]
    burnins = [0, 0, 0, 0, 0]
    run()
