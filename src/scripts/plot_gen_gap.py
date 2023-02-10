from src.scripts.plot_ood_comp import run as run_ood
from src.scripts.plot_nondist_comp import run as run_train
from src.utils.plotting import plot_multi_trials


def plot_diff():
    small_size = 50
    comps, ood2, ood16, ood32 = run_ood()
    _, train2, train16, train32 = run_train()
    diffs = []
    sizes = []
    setup_len = None
    idx = 1
    for ood_data, train_data in zip([ood2, ood16, ood32], [train2, train16, train32]):
        diff = [[t - o for o, t in zip(o_data, t_data)] for o_data, t_data in zip(ood_data, train_data)]
        diffs.extend(diff)
        setup_len = len(diff)
        sizes.extend([small_size * 2 for _ in diff])
        idx += 1
    labels = ['$C=2$'] + ['' for _ in range(setup_len - 1)] + ['$C=16$'] + ['' for _ in range(setup_len - 1)] + [
        '$C=32$'] + ['' for _ in range(setup_len - 1)]
    # sizes = [4 * small_size for _ in diffs[0]] + [2 * small_size for _ in diffs[1]] + [small_size for _ in
    #                                                                                                 diffs[2]]
    plot_multi_trials([comps + comps + comps, diffs], labels, sizes,
                      ylabel='Generalization Gap', colors=None, filename='ood_train_gap.png')


if __name__ == '__main__':
    plot_diff()
