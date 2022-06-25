import os
from src.utils.performance_metrics import PerformanceMetrics
from src.utils.plotting import plot_scatter, plot_multi_trials


def run():
    all_complexities = []
    all_informativeness = []
    for speaker_type in model_types:
        train_complexities = []
        train_informativeness = []
        val_complexities = []
        val_informativeness = []
        for seed in seeds:
            basepath = 'saved_models/' + speaker_type + '/seed' + str(seed) + '/'
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
            train_complexities.extend(train_metrics.complexities)
            train_informativeness.extend(train_metrics.recons)
            val_complexities.extend(val_metrics.complexities)
            val_informativeness.extend(val_metrics.recons)
        plot_scatter([train_complexities, train_informativeness], ['Complexity', 'Informativeness'])
        all_complexities.append(train_complexities)
        all_informativeness.append(train_informativeness)
    plot_multi_trials([all_complexities, all_informativeness],
                      ['Cont.', 'VQ-VIB'])


if __name__ == '__main__':
    model_types = ['cont', 'vq']
    seeds = [0, 1]
    run()
