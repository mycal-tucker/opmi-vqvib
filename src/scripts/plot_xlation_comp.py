from src.utils.plotting import plot_multi_trials

# No snap, with 100


def run():
    labels = []
    utilities2_100 = []
    utilities16_100 = []
    utilities32_100 = []
    utilities32_1000 = []
    comps = []
    sizes = []
    colors = []
    small_size = 50

    # labels.append('$\lambda_I = 0.1; n = 1$')
    # utilities2_100.append([0.86, 0.83, 0.80, 0.83, 0.86])
    # utilities16_100.append([0.26, 0.23, 0.27, 0.16, 0.28])
    # utilities32_100.append([0.13, 0.15, 0.16, 0.11, 0.17])
    # utilities32_1000.append([])
    # comps.append([1.284, 1.141, 1.347, 0.995, 1.142])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')

    ############################################################################
    #####           N = 4
    ############################################################################
    labels.append('$\lambda_I = 0.0; n = 4$')
    utilities2_100.append([0.825, 0.525, 0.81, 0.485, 0.435])
    utilities16_100.append([0.225, 0.06, 0.22, 0.065, 0.025])
    utilities32_100.append([0.135, 0.035, 0.145, 0.04,  0.01])
    utilities32_1000.append([])
    comps.append([1.507, -0.0, 1.517, -0.0, 0.0])
    sizes.append(small_size)
    colors.append(0)

    labels.append('$\lambda_I = 0.1; n = 4$')
    utilities2_100.append([0.92, 0.83, 0.84, 0.82, 0.85])
    utilities16_100.append([0.41, 0.27, 0.28, 0.19, 0.36])
    utilities32_100.append([0.25, 0.11, 0.2,  0.12, 0.25])
    utilities32_1000.append([0.32, 0.13, 0.2, 0.12, 0.26])
    comps.append([1.748, 1.476, 1.751, 1.149, 1.65])
    sizes.append(small_size)
    colors.append(0.1)

    labels.append('$\lambda_I = 0.5; n = 4$')
    utilities2_100.append([0.92, 0.865, 0.87, 0.84, 0.895])
    utilities16_100.append([0.525, 0.295, 0.375, 0.275, 0.43])
    utilities32_100.append([0.285, 0.15, 0.225, 0.165, 0.265])
    comps.append([2.382, 2.199, 2.216, 2.225, 2.125])
    sizes.append(small_size)
    colors.append('xkcd:blue')

    labels.append('$\lambda_I = 1.0; n = 4$')
    utilities2_100.append([0.93, 0.9,  0.87, 0.84, 0.89])
    utilities16_100.append([0.51, 0.37, 0.41, 0.31, 0.44])
    utilities32_100.append([0.27, 0.18, 0.24, 0.16, 0.29])
    utilities32_1000.append([0.18, 0.29, 0.26, 0.33])
    comps.append([2.734, 2.652, 2.683, 2.763, 2.809])
    sizes.append(small_size)
    colors.append(1)

    labels.append('$\lambda_I = 1.5; n = 4$')
    utilities2_100.append([0.925, 0.885, 0.88, 0.87, 0.905])
    utilities16_100.append([0.545, 0.345, 0.4, 0.315, 0.455])
    utilities32_100.append([0.33, 0.2, 0.26, 0.21, 0.29])
    comps.append([2.962, 2.748, 3.395, 2.98, 2.942])
    sizes.append(small_size)
    colors.append('xkcd:blue')

    labels.append('$\lambda_I = 2.0; n = 4$')
    utilities2_100.append([0.93, 0.88, 0.89, 0.855, 0.9])
    utilities16_100.append([0.545, 0.325, 0.395, 0.3, 0.455])
    utilities32_100.append([0.33, 0.205, 0.265, 0.16, 0.28])
    comps.append([3.596, 3.426, 3.096, 3.679, 2.883])
    sizes.append(small_size)
    colors.append('xkcd:blue')

    labels.append('$\lambda_I = 3.0; n = 4$')
    utilities2_100.append([0.94, 0.885, 0.89, 0.88, 0.915])
    utilities16_100.append([0.575, 0.325, 0.415, 0.335, 0.475])
    utilities32_100.append([0.36, 0.185, 0.27, 0.22, 0.295])
    comps.append([3.123, 3.455, 3.419, 3.86, 3.31])
    sizes.append(small_size)
    colors.append('xkcd:blue')

    labels.append('$\lambda_I = 10; n = 4$')
    utilities2_100.append([0.95, 0.88, 0.89, 0.87, 0.91])
    utilities16_100.append([0.59, 0.37, 0.4,  0.36, 0.5])
    utilities32_100.append([0.32, 0.22, 0.25, 0.24, 0.34])
    utilities32_1000.append([0.39,  0.265, 0.275, 0.28,  0.36])
    comps.append([4.634, 4.924, 4.915, 4.245, 3.565])
    sizes.append(small_size)
    colors.append(10)

    # labels.append('$\lambda_I = 100; n = 4$')
    # utilities2_100.append([0.95, 0.86, 0.9,  0.87, 0.92])
    # utilities16_100.append([0.58, 0.38, 0.4,  0.36, 0.47])
    # utilities32_100.append([0.34, 0.22, 0.24, 0.25, 0.31])
    # utilities32_1000.append([0.395, 0.265, 0.295, 0.27,  0.355])
    # comps.append([4.812, 4.593, 3.557, 4.942, 4.373])
    # sizes.append(small_size)
    # colors.append('xkcd:orange')

    # labels.append('$\lambda_I = 1; n = 8$')
    # utilities2_100.append([0.87, 0.89, 0.86, 0.92])
    # comps.append([3.092, 3.625, 3.259, 3.166])
    # sizes.append(small_size)
    # colors.append('xkcd:green')
    #
    # labels.append('$\lambda_I = 10; n = 8$')
    # utilities2_100.append([0.95, 0.91, 0.88, 0.85, 0.91])
    # comps.append([4.681, 4.35, 4.025, 4.363, 4.747])
    # sizes.append(small_size)
    # colors.append('xkcd:green')
    #
    # labels.append('$\lambda_I = 100; n = 8$')
    # utilities2_100.append([0.95, 0.89, 0.89, 0.88, 0.92])
    # comps.append([4.038, 5.248, 3.155, 5.119, 3.319])
    # sizes.append(small_size)
    # colors.append('xkcd:black')

    for data, suffix in zip([utilities2_100, utilities16_100, utilities32_100], ['2', '16', '32']):
        plot_multi_trials([comps, data], labels, sizes, ylabel='Utility', colors=colors, filename='xlation_train_comp' + suffix + '.png')
    labels = ['$C=2$'] + ['' for _ in utilities2_100[:-1]] + ['$C=16$'] + ['' for _ in utilities16_100[:-1]] + ['$C=32$'] + ['' for _ in utilities32_100[:-1]]
    sizes = [4 * small_size for _ in utilities2_100] + [2 * small_size for _ in utilities16_100] + [small_size for _ in utilities32_100]
    plot_multi_trials([comps + comps + comps, utilities2_100 + utilities16_100 + utilities32_100], labels, sizes, ylabel='Translation Utility', colors=None, filename='xlation_comp_all.png')


if __name__ == '__main__':
    run()
