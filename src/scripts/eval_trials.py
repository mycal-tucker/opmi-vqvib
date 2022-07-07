import os
from src.utils.performance_metrics import PerformanceMetrics
from src.utils.plotting import plot_scatter, plot_multi_trials, plot_multi_metrics


def run():
    all_complexities = []
    all_informativeness = []
    all_accs = {}  # model type to two lists: train and val accs, as a list for each epoch
    for speaker_type, burnin in zip(model_types, burnins):
        train_complexities = []
        train_informativeness = []
        train_accs = []
        val_complexities = []
        val_informativeness = []
        val_accs = []
        epochs = []
        all_accs[speaker_type] = [train_accs, val_accs, epochs]
        for seed in seeds:
            # basepath = 'saved_models/alpha0_noent/' + speaker_type + '/seed' + str(seed) + '/'
            basepath = 'saved_models/alpha10_noent_' + str(speaker_type) + 'tok_fixed/vq/seed' + str(seed) + '/'
            if not os.path.exists(basepath):
                print("Path doesn't exist", basepath)
                continue
            list_of_files = os.listdir(basepath)
            last_seed = max([int(f) for f in list_of_files])
            try:
                train_metrics = PerformanceMetrics.from_file(basepath + str(last_seed) + '/train_metrics')
                val_metrics = PerformanceMetrics.from_file(basepath + str(last_seed) + '/val_metrics')
            except FileNotFoundError:
                print("Problem loading for", basepath)
                continue
            train_complexities.extend(train_metrics.complexities[-burnin:])
            train_informativeness.extend(train_metrics.recons[-burnin:])
            train_accs.append(train_metrics.comm_accs[-burnin:])
            val_complexities.extend(val_metrics.complexities[-burnin:])
            val_informativeness.extend(val_metrics.recons[-burnin:])
            val_accs.append(val_metrics.comm_accs[-burnin:])
            epochs.append(val_metrics.epoch_idxs[-burnin:])
        plot_scatter([train_complexities, train_informativeness], ['Complexity', 'Informativeness'])
        all_complexities.append(train_complexities)
        all_informativeness.append(train_informativeness)
    # Add English data, gathered from english_analysis.py
    all_complexities.append([1.54])
    all_informativeness.append([-0.00011])
    sizes = [20, 20, 60]
    plot_multi_trials([all_complexities, all_informativeness],
                      ['Cont.', 'VQ-VIB', 'English'],
                      sizes)
    plot_multi_metrics(all_accs)


if __name__ == '__main__':
    # model_types = ['cont', 'vq']
    # model_types = ['vq']
    model_types = ['1', '2', '4', '8']
    seeds = [0, 1, 2]
    burnins = [0, 0, 0, 0]
    run()
