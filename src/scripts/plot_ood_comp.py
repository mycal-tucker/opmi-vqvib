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
    #####           N = 1
    ############################################################################
    # labels.append('$\lambda_I = 0; n = 1$')
    # utilities2.append([0.817, 0.498, 0.504, 0.509, 0.492])
    # utilities16.append([0.207, 0.066, 0.068, 0.07, 0.063])
    # utilities32.append([0.106, 0.034, 0.029, 0.028, 0.03])
    # comps.append([1.084, 0.0, -0.0, -0.0, 0.0])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 0.1; n = 1$')
    # utilities2.append([0.833, 0.757, 0.71, 0.58, 0.695])
    # utilities16.append([0.265, 0.22, 0.202, 0.112, 0.197])
    # utilities32.append([0.146, 0.123, 0.131, 0.056, 0.12])
    # comps.append([1.284, 1.141, 1.347, 0.995, 1.142])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 0.5; n = 1$')
    # utilities2.append([0.873, 0.756, 0.775, 0.806, 0.853])
    # utilities16.append([0.329, 0.266, 0.267, 0.278, 0.339])
    # utilities32.append([0.207, 0.143, 0.176, 0.182, 0.225])
    # comps.append([1.823, 1.586, 1.659, 1.662, 1.816])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 1; n = 1$')
    # utilities2.append([ 0.892, 0.856, 0.822, 0.846, 0.872])
    # utilities16.append([0.431, 0.343, 0.323, 0.338, 0.386])
    # utilities32.append([0.244, 0.218, 0.208, 0.186, 0.262])
    # comps.append([2.13, 1.899, 2.113, 1.94, 2.031])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 1.5; n = 1$')
    # utilities2.append([0.892, 0.87, 0.821, 0.84, 0.871])
    # utilities16.append([0.437, 0.389, 0.32, 0.377, 0.411])
    # utilities32.append([0.288, 0.252, 0.206, 0.232, 0.266])
    # comps.append([2.113, 2.218, 2.153, 2.234, 2.136])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 2.0; n = 1$')
    # utilities2.append([0.888, 0.876, 0.8, 0.868, 0.89])
    # utilities16.append([0.429, 0.383, 0.317, 0.373, 0.432])
    # utilities32.append([0.281, 0.261, 0.218, 0.241, 0.305])
    # comps.append([2.196, 2.206, 2.232, 2.203, 2.277])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 3.0; n = 1$')
    # utilities2.append([0.896, 0.881, 0.784, 0.86, 0.896])
    # utilities16.append([0.44, 0.407, 0.321, 0.376, 0.478])
    # utilities32.append([ 0.289, 0.256, 0.207, 0.244, 0.332])
    # comps.append([2.307, 2.257, 2.198, 2.247, 2.312])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 10; n = 1$')
    # utilities2.append([])
    # utilities16.append([])
    # utilities32.append([])
    # comps.append([])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')

    ############################################################################
    #####           N = 4
    ############################################################################
    labels.append('$\lambda_I = 0; n = 4$')
    utilities2.append([0.867, 0.498, 0.823, 0.51, 0.491])
    utilities16.append([0.306, 0.067, 0.304, 0.071, 0.062])
    utilities32.append([0.178, 0.034, 0.167, 0.027, 0.031])
    comps.append([1.507, -0.0, 1.517, -0.0, 0.0])
    sizes.append(small_size)
    colors.append('xkcd:blue')

    labels.append('$\lambda_I = 0.1; n = 4$')
    utilities2.append([0.884, 0.782, 0.82, 0.766, 0.83])
    utilities16.append([0.346, 0.283, 0.274, 0.234, 0.293])
    utilities32.append([0.188, 0.156, 0.171, 0.124, 0.184])
    comps.append([1.748, 1.476, 1.751, 1.149, 1.65])
    sizes.append(small_size)
    colors.append('xkcd:blue')

    labels.append('$\lambda_I = 0.5; n = 4$')
    utilities2.append([0.918, 0.898, 0.845, 0.877, 0.902])
    utilities16.append([0.505, 0.435, 0.372, 0.398, 0.441])
    utilities32.append([0.325, 0.293, 0.257, 0.263, 0.318])
    comps.append([2.382, 2.199, 2.216, 2.225, 2.125])
    sizes.append(small_size)
    colors.append('xkcd:blue')

    labels.append('$\lambda_I = 1; n = 4$')
    utilities2.append([ 0.933, 0.918, 0.859, 0.902, 0.941])
    utilities16.append([0.547, 0.509, 0.404, 0.5, 0.588])
    utilities32.append([0.393, 0.357, 0.288, 0.346, 0.458])
    comps.append([2.734, 2.652, 2.683, 2.763, 2.809])
    sizes.append(small_size)
    colors.append('xkcd:blue')

    labels.append('$\lambda_I = 1.5; n = 4$')
    utilities2.append([ 0.946, 0.92, 0.909, 0.927, 0.959])
    utilities16.append([0.604, 0.525, 0.508, 0.565, 0.66])
    utilities32.append([0.427, 0.361, 0.379, 0.432, 0.525])
    comps.append([2.962, 2.748, 3.395, 2.98, 2.942])
    sizes.append(small_size)
    colors.append('xkcd:blue')

    labels.append('$\lambda_I = 2.0; n = 4$')
    utilities2.append([ 0.958, 0.921, 0.916, 0.938, 0.97])
    utilities16.append([0.652, 0.524, 0.559, 0.628, 0.724])
    utilities32.append([0.481, 0.381, 0.437, 0.475, 0.605])
    comps.append([3.596, 3.426, 3.096, 3.679, 2.883])
    sizes.append(small_size)
    colors.append('xkcd:blue')

    labels.append('$\lambda_I = 3.0; n = 4$')
    utilities2.append([0.969, 0.948, 0.923, 0.964, 0.981])
    utilities16.append([0.733, 0.619, 0.582, 0.728, 0.805])
    utilities32.append([0.582, 0.495, 0.456, 0.592, 0.689])
    comps.append([3.123, 3.455, 3.419, 3.86, 3.31])
    sizes.append(small_size)
    colors.append('xkcd:blue')

    labels.append('$\lambda_I = 10; n = 4$')
    utilities2.append([0.982, 0.965, 0.95, 0.985, 0.99])
    utilities16.append([0.833, 0.731, 0.698, 0.863, 0.89])
    utilities32.append([0.704, 0.628, 0.605, 0.78, 0.818])
    comps.append([ 4.634, 4.924, 4.915, 4.245, 3.565])
    sizes.append(small_size)
    colors.append('xkcd:blue')

    # labels.append('$\lambda_I = 1; n = 8$')
    # utilities2.append([0.924, 0.882, 0.933, 0.962])
    # utilities16.append([0.524, 0.503, 0.591, 0.679])
    # utilities32.append([0.389, 0.379, 0.424, 0.574])
    # comps.append([3.092, 3.625, 3.259, 3.166])
    # sizes.append(small_size)
    # colors.append('xkcd:green')
    #
    # labels.append('$\lambda_I = 10; n = 8$')
    # utilities2.append([0.992, 0.981, 0.984, 0.986, 0.996])
    # utilities16.append([ 0.924, 0.863, 0.868, 0.881, 0.957])
    # utilities32.append([0.869, 0.796, 0.797, 0.827, 0.93])
    # comps.append([4.681, 4.35, 4.025, 4.363, 4.747])
    # sizes.append(small_size)
    # colors.append('xkcd:green')
    #
    # labels.append('$\lambda_I = 100; n = 8$')
    # utilities2.append([0.997, 0.991, 0.895, 0.994, 0.998])
    # utilities16.append([0.948, 0.905, 0.507, 0.954, 0.969])
    # utilities32.append([0.911, 0.858, 0.436, 0.916, 0.939])
    # comps.append([4.038, 5.248, 3.155, 5.119, 3.319])
    # sizes.append(small_size)
    # colors.append('xkcd:black')

    for data, suffix in zip([utilities2, utilities16, utilities32], ['2', '16', '32']):
        plot_multi_trials([comps, data], labels, sizes, ylabel='Utility', colors=None, filename='ood_comp' + suffix + '.png')
    labels = ['$C=2$'] + ['' for _ in utilities2[:-1]] + ['$C=16$'] + ['' for _ in utilities16[:-1]] + ['$C=32$'] + ['' for _ in utilities32[:-1]]
    sizes = [4 * small_size for _ in utilities2] + [2 * small_size for _ in utilities16] + [small_size for _ in utilities32]
    plot_multi_trials([comps + comps + comps, utilities2 + utilities16 + utilities32], labels, sizes, ylabel='OOD Utility', colors=None, filename='ood_comp_all.png')
    return comps, utilities2, utilities16, utilities32


if __name__ == '__main__':
    run()
