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

    # labels.append('$\lambda_I = 0.1; n = 1$')
    # utilities2.append([0.879, 0.85, 0.878, 0.812, 0.865])
    # utilities16.append([0.308, 0.252, 0.351, 0.211, 0.312])
    # utilities32.append([0.173, 0.134, 0.195, 0.106, 0.177])
    # comps.append([1.284, 1.141, 1.347, 0.995, 1.142])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 1; n = 1$')
    # utilities2.append([0.947, 0.941, 0.936, 0.923, 0.933])
    # utilities16.append([0.593, 0.55, 0.559, 0.541, 0.539])
    # utilities32.append([0.432, 0.388, 0.418, 0.384, 0.377])
    # comps.append([2.13, 1.899, 2.113, 1.94, 2.031])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')

    labels.append('$\lambda_I = 0; n = 4$')
    utilities2.append([0.909, 0.504, 0.912, 0.494, 0.498])
    utilities16.append([0.361, 0.07, 0.39, 0.066, 0.059])
    utilities32.append([0.2, 0.028, 0.248, 0.031, 0.031])
    comps.append([1.507, -0.0, 1.517, -0.0, 0.0])
    sizes.append(small_size)
    colors.append('xkcd:blue')

    labels.append('$\lambda_I = 0.1; n = 4$')
    utilities2.append([0.94, 0.899, 0.923, 0.865, 0.907])
    utilities16.append([0.459, 0.364, 0.452, 0.29, 0.415])
    utilities32.append([0.288, 0.218, 0.289, 0.166, 0.261])
    comps.append([1.748, 1.476, 1.751, 1.149, 1.65])
    sizes.append(small_size)
    colors.append('xkcd:blue')

    labels.append('$\lambda_I = 1; n = 4$')
    utilities2.append([0.98, 0.961, 0.972, 0.977, 0.974])
    utilities16.append([ 0.763, 0.675, 0.724, 0.779, 0.722])
    utilities32.append([0.63, 0.535, 0.592, 0.634, 0.598])
    comps.append([2.734, 2.652, 2.683, 2.763, 2.809])
    sizes.append(small_size)
    colors.append('xkcd:blue')

    # Just for 100k
    labels.append('$\lambda_I = 10; n = 4$')
    utilities2.append([ 0.996, 0.99, 0.996, 0.998, 0.994])
    utilities16.append([0.944, 0.929, 0.958, 0.975, 0.935])
    utilities32.append([0.915, 0.865, 0.932, 0.954, 0.895])
    comps.append([4.634, 4.924, 4.915, 4.245, 3.565])
    sizes.append(small_size)
    colors.append('xkcd:blue')

    # labels.append('$\lambda_I = 100; n = 4$')
    # utilities2.append([0.996, 0.995, 0.996, 0.999, 0.995])
    # utilities16.append([ 0.938, 0.927, 0.963, 0.979, 0.95])
    # utilities32.append([0.9, 0.877, 0.941, 0.958, 0.916])
    # comps.append([ 4.812, 4.593, 3.557, 4.942, 4.373])
    # sizes.append(small_size)
    # colors.append('xkcd:orange')


    # labels.append('$\lambda_I = 1; n = 8$')
    # utilities2.append([0.971, 0.985, 0.984, 0.976])
    # utilities16.append([0.727, 0.838, 0.84, 0.811])
    # utilities32.append([0.611, 0.738, 0.721, 0.704])
    # comps.append([3.092, 3.625, 3.259, 3.166])
    # sizes.append(small_size)
    # colors.append('xkcd:green')
    #
    # labels.append('$\lambda_I = 10; n = 8$')
    # utilities2.append([ 0.997, 0.996, 0.999, 0.998, 0.996])
    # utilities16.append([0.976, 0.967, 0.985, 0.988, 0.973])
    # utilities32.append([0.958, 0.947, 0.977, 0.978, 0.955])
    # comps.append([4.681, 4.35, 4.025, 4.363, 4.747])
    # sizes.append(small_size)
    # colors.append('xkcd:green')
    #
    #
    # labels.append('$\lambda_I = 100; n = 8$')
    # utilities2.append([0.998, 0.998, 0.991, 0.999, 0.997])
    # utilities16.append([0.985, 0.985, 0.891, 0.993, 0.979])
    # utilities32.append([ 0.973, 0.97, 0.83, 0.984, 0.962])
    # comps.append([4.038, 5.248, 3.155, 5.119, 3.319])
    # sizes.append(small_size)
    # colors.append('xkcd:black')

    for data, suffix in zip([utilities2, utilities16, utilities32], ['2', '16', '32']):
        plot_multi_trials([comps, data], labels, sizes, ylabel='Utility', colors=None, filename='nondist_comp' + suffix + '.png')

    labels = ['$C=2$'] + ['' for _ in utilities2[:-1]] + ['$C=16$'] + ['' for _ in utilities16[:-1]] + ['$C=32$'] + ['' for _ in utilities32[:-1]]
    sizes = [4 * small_size for _ in utilities2] + [2 * small_size for _ in utilities16] + [small_size for _ in utilities32]
    plot_multi_trials([comps + comps + comps, utilities2 + utilities16 + utilities32], labels, sizes, ylabel='In-Distribution Utility', colors=None, filename='nondist_comp_all.png')
    return comps, utilities2, utilities16, utilities32

if __name__ == '__main__':
    run()
