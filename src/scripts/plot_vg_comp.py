from src.utils.plotting import plot_multi_trials


def run():
    labels = []
    utilities2 = []
    utilities16 = []
    utilities32 = []
    comps = []
    sizes = []
    colors = []
    small_size = 50

    ############################################################################
    #####           VG
    ############################################################################
    labels.append('$\lambda_I = 0.1 n=1')
    utilities2.append([0.856, 0.868, 0.861, 0.851, 0.868])
    utilities16.append([0.273, 0.286, 0.259, 0.282, 0.262])
    utilities32.append([0.151, 0.161, 0.14, 0.155, 0.15])
    comps.append([1.154, 1.097, 1.166, 1.064, 1.046])
    sizes.append(small_size)
    labels.append('$\lambda_I = 0.1$ n=4')
    utilities2.append([ 0.859, 0.887, 0.857, 0.878, 0.881])
    utilities16.append([0.274, 0.337, 0.268, 0.318, 0.306])
    utilities32.append([0.152, 0.191, 0.143, 0.169, 0.172])
    comps.append([1.389, 1.407, 1.332, 1.314, 1.322])
    sizes.append(small_size)

    labels.append('$\lambda_I = 1.0 n=1')
    utilities2.append([0.923, 0.911, 0.924, 0.923, 0.932])
    utilities16.append([ 0.464, 0.461, 0.477, 0.494, 0.486])
    utilities32.append([0.302, 0.303, 0.32, 0.349, 0.306])
    comps.append([1.872, 1.853, 1.883, 1.957, 1.882])
    sizes.append(small_size)

    labels.append('$\lambda_I = 1.0$ n=4')
    utilities2.append([0.955, 0.966, 0.951, 0.942, 0.959])
    utilities16.append([0.588, 0.604, 0.609, 0.588, 0.595])
    utilities32.append([0.444, 0.434, 0.453, 0.436, 0.427])
    comps.append([2.642, 2.551, 2.651, 2.694, 2.458])
    sizes.append(small_size)

    for data, suffix in zip([utilities2, utilities16, utilities32], ['2', '16', '32']):
        plot_multi_trials([comps, data], labels, sizes, ylabel='Utility', colors=None, filename='vg_comp' + suffix + '.png')
    labels = ['$C=2$'] + ['' for _ in utilities2[:-1]] + ['$C=16$'] + ['' for _ in utilities16[:-1]] + ['$C=32$'] + ['' for _ in utilities32[:-1]]
    sizes = [4 * small_size for _ in utilities2] + [2 * small_size for _ in utilities16] + [small_size for _ in utilities32]
    plot_multi_trials([comps + comps + comps, utilities2 + utilities16 + utilities32], labels, sizes, ylabel='VG Utility', colors=None, filename='vg_comp_all.png')
    return comps, utilities2, utilities16, utilities32


if __name__ == '__main__':
    run()
