from src.utils.plotting import plot_multi_trials

# No snap, with 100


def run():
    labels = []
    utilities2_10 = []
    utilities2_50 = []
    utilities2_100 = []
    utilities2_1000 = []
    utilities16_10 = []
    utilities16_50 = []
    utilities16_100 = []
    utilities16_1000 = []
    utilities32_10 = []
    utilities32_50 = []
    utilities32_100 = []
    utilities32_1000 = []
    comps = []
    sizes = []
    colors = []
    small_size = 50
    ############################################################################
    #####           Onehot
    ############################################################################
    # n = 'onehot'
    # labels.append('$\lambda_I = 0.1$')
    # utilities2_100.append([0.53, 0.715, 0.47, 0.455, 0.435])
    # utilities16_100.append([0.03, 0.185, 0.04, 0.045, 0.07])
    # utilities32_100.append([0.01, 0.065, 0.03, 0.02, 0.04])
    # comps.append([0.001, 0.637, 0.298, 0.003, 0.0])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 1.0$')
    # utilities2_100.append([0.755, 0.545, 0.585, 0.65, 0.66])
    # utilities16_100.append([0.19, 0.06, 0.1, 0.115, 0.085])
    # utilities32_100.append([0.11, 0.035, 0.045, 0.045, 0.025])
    # comps.append([1.393, 0.489, 0.782, 1.268, 0.651])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 10.0$')
    # utilities2_100.append([0.76, 0.805, 0.705, 0.745, 0.655])
    # utilities16_100.append([0.195, 0.225, 0.165, 0.17, 0.14])
    # utilities32_100.append([0.115, 0.12, 0.105, 0.085, 0.075])
    # comps.append([1.22, 1.432, 1.346, 1.38, 1.105])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 100$')
    # utilities2_100.append([0.725, 0.815, 0.665, 0.76, 0.715])
    # utilities16_100.append([0.16, 0.255, 0.165, 0.215, 0.16])
    # utilities32_100.append([0.085, 0.12, 0.09, 0.11, 0.09])
    # comps.append([1.201, 1.492, 1.216, 1.353, 1.047])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')

    ############################################################################
    #####           N = 1
    ############################################################################
    n = 1
    labels.append('$\lambda_I = 0.0; n = 1$')
    utilities2_10.append([])
    utilities16_10.append([])
    utilities32_10.append([])
    utilities2_50.append([])
    utilities16_50.append([])
    utilities32_50.append([])
    utilities2_100.append([0.655, 0.485, 0.44, 0.475, 0.44])
    utilities16_100.append([0.105, 0.07, 0.05, 0.07, 0.025])
    utilities32_100.append([0.04, 0.045, 0.03, 0.03, 0.01])
    utilities2_1000.append([])
    utilities16_1000.append([])
    utilities32_1000.append([])
    comps.append([1.084, 0.0, -0.0, -0.0, 0.0])
    sizes.append(small_size)
    colors.append('xkcd:blue')

    labels.append('$\lambda_I = 0.1; n = 1$')
    utilities2_10.append([])
    utilities16_10.append([])
    utilities32_10.append([])
    utilities2_50.append([])
    utilities16_50.append([])
    utilities32_50.append([])
    utilities2_100.append([0.8, 0.84, 0.87, 0.825, 0.855])
    utilities16_100.append([0.175, 0.24, 0.24, 0.205, 0.225])
    utilities32_100.append([0.085, 0.115, 0.14, 0.11, 0.1])
    utilities2_1000.append([])
    utilities16_1000.append([])
    utilities32_1000.append([])
    comps.append([1.284, 1.141, 1.347, 0.995, 1.142])
    sizes.append(small_size)
    colors.append('xkcd:blue')

    labels.append('$\lambda_I = 0.5; n = 1$')
    utilities2_10.append([])
    utilities16_10.append([])
    utilities32_10.append([])
    utilities2_50.append([])
    utilities16_50.append([])
    utilities32_50.append([])
    utilities2_100.append([0.82, 0.855, 0.86, 0.865, 0.855])
    utilities16_100.append([0.235, 0.24, 0.305, 0.325, 0.24])
    utilities32_100.append([0.15, 0.115, 0.17, 0.19, 0.12])
    utilities2_1000.append([])
    utilities16_1000.append([])
    utilities32_1000.append([])
    comps.append([1.823, 1.586, 1.659, 1.662, 1.816])
    sizes.append(small_size)
    colors.append('xkcd:blue')

    labels.append('$\lambda_I = 1.0; n = 1$')
    utilities2_10.append([])
    utilities16_10.append([])
    utilities32_10.append([])
    utilities2_50.append([])
    utilities16_50.append([])
    utilities32_50.append([])
    utilities2_100.append([0.815, 0.865, 0.87, 0.855, 0.875])
    utilities16_100.append([0.265, 0.245, 0.28, 0.31, 0.28])
    utilities32_100.append([0.145, 0.135, 0.155, 0.175, 0.15])
    utilities2_1000.append([])
    utilities16_1000.append([])
    utilities32_1000.append([])
    comps.append([2.13, 1.899, 2.113, 1.94, 2.031])
    sizes.append(small_size)
    colors.append('xkcd:blue')

    labels.append('$\lambda_I = 1.5; n = 1$')
    utilities2_10.append([])
    utilities16_10.append([])
    utilities32_10.append([])
    utilities2_50.append([])
    utilities16_50.append([])
    utilities32_50.append([])
    utilities2_100.append([0.82, 0.875, 0.865, 0.845, 0.87])
    utilities16_100.append([0.26, 0.27, 0.325, 0.31, 0.29])
    utilities32_100.append([0.15, 0.145, 0.19, 0.19, 0.145])
    utilities2_1000.append([])
    utilities16_1000.append([])
    utilities32_1000.append([])
    comps.append([2.113, 2.218, 2.153, 2.234, 2.136])
    sizes.append(small_size)
    colors.append('xkcd:blue')

    labels.append('$\lambda_I = 2.0; n = 1$')
    utilities2_10.append([])
    utilities16_10.append([])
    utilities32_10.append([])
    utilities2_50.append([])
    utilities16_50.append([])
    utilities32_50.append([])
    utilities2_100.append([0.79, 0.845, 0.865, 0.855, 0.845])
    utilities16_100.append([0.18, 0.27, 0.345, 0.275, 0.3])
    utilities32_100.append([0.095, 0.165, 0.21, 0.16, 0.185])
    utilities2_1000.append([])
    utilities16_1000.append([])
    utilities32_1000.append([])
    comps.append([2.196, 2.206, 2.232, 2.203, 2.277])
    sizes.append(small_size)
    colors.append('xkcd:blue')

    labels.append('$\lambda_I = 3.0; n = 1$')
    utilities2_10.append([])
    utilities16_10.append([])
    utilities32_10.append([])
    utilities2_50.append([])
    utilities16_50.append([])
    utilities32_50.append([])
    utilities2_100.append([0.78, 0.87, 0.86, 0.855, 0.83])
    utilities16_100.append([0.205, 0.27, 0.3, 0.3, 0.3])
    utilities32_100.append([0.1, 0.16, 0.165, 0.15, 0.18])
    utilities2_1000.append([])
    utilities16_1000.append([])
    utilities32_1000.append([])
    comps.append([2.307, 2.257, 2.198, 2.247, 2.312])
    sizes.append(small_size)
    colors.append('xkcd:blue')

    labels.append('$\lambda_I = 10.0; n = 1$')
    utilities2_10.append([])
    utilities16_10.append([])
    utilities32_10.append([])
    utilities2_50.append([])
    utilities16_50.append([])
    utilities32_50.append([])
    utilities2_100.append([0.79, 0.865, 0.86, 0.845, 0.83])
    utilities16_100.append([0.215, 0.28, 0.315, 0.32, 0.265])
    utilities32_100.append([0.105, 0.15, 0.16, 0.16, 0.155])
    utilities2_1000.append([])
    utilities16_1000.append([])
    utilities32_1000.append([])
    comps.append([2.598, 2.487, 2.324, 2.482, 2.665])
    sizes.append(small_size)
    colors.append('xkcd:blue')
    ############################################################################
    #####           N = 4
    ############################################################################
    # n = 4
    # labels.append('$\lambda_I = 0.0; n = 4$')
    # utilities2_10.append([])
    # utilities16_10.append([])
    # utilities32_10.append([])
    # utilities2_50.append([])
    # utilities16_50.append([])
    # utilities32_50.append([])
    # utilities2_100.append([0.61, 0.48, 0.785, 0.49, 0.44])
    # utilities16_100.append([0.12, 0.07, 0.165, 0.07, 0.025])
    # utilities32_100.append([0.07, 0.05, 0.105, 0.03, 0.01])
    # utilities2_1000.append([])
    # utilities16_1000.append([])
    # utilities32_1000.append([])
    # comps.append([1.507, -0.0, 1.517, -0.0, 0.0])
    # sizes.append(small_size)
    # colors.append(0)
    #
    # labels.append('$\lambda_I = 0.1; n = 4$')
    # utilities2_10.append([])
    # utilities16_10.append([])
    # utilities32_10.append([])
    # utilities2_50.append([])
    # utilities16_50.append([])
    # utilities32_50.append([])
    # utilities2_100.append([0.845, 0.85, 0.86, 0.815, 0.875])
    # utilities16_100.append([0.26, 0.23, 0.25, 0.2, 0.265])
    # utilities32_100.append([0.13, 0.09, 0.105, 0.12, 0.13])
    # utilities2_1000.append([])
    # utilities16_1000.append([])
    # utilities32_1000.append([])
    # comps.append([1.748, 1.476, 1.751, 1.149, 1.65])
    # sizes.append(small_size)
    # colors.append(0)
    #
    # labels.append('$\lambda_I = 0.5; n = 4$')
    # utilities2_10.append([])
    # utilities16_10.append([])
    # utilities32_10.append([])
    # utilities2_50.append([])
    # utilities16_50.append([])
    # utilities32_50.append([])
    # utilities2_100.append([0.825, 0.87, 0.865, 0.865, 0.79])
    # utilities16_100.append([0.215, 0.27, 0.285, 0.315, 0.27])
    # utilities32_100.append([0.13, 0.135, 0.185, 0.19, 0.155])
    # utilities2_1000.append([])
    # utilities16_1000.append([])
    # utilities32_1000.append([])
    # comps.append([2.382, 2.199, 2.216, 2.225, 2.125])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 1.0; n = 4$')
    # utilities2_10.append([])
    # utilities16_10.append([])
    # utilities32_10.append([])
    # utilities2_50.append([])
    # utilities16_50.append([])
    # utilities32_50.append([])
    # utilities2_100.append([0.755, 0.88, 0.875, 0.86, 0.815])
    # utilities16_100.append([0.2, 0.3, 0.33, 0.315, 0.29])
    # utilities32_100.append([0.135, 0.175, 0.195, 0.205, 0.165])
    # utilities2_1000.append([])
    # utilities16_1000.append([])
    # utilities32_1000.append([])
    # comps.append([2.734, 2.652, 2.683, 2.763, 2.809])
    # sizes.append(small_size)
    # colors.append(1)
    #
    # labels.append('$\lambda_I = 1.5; n = 4$')
    # utilities2_10.append([])
    # utilities16_10.append([])
    # utilities32_10.append([])
    # utilities2_50.append([])
    # utilities16_50.append([])
    # utilities32_50.append([])
    # utilities2_100.append([0.77, 0.885, 0.875, 0.865, 0.81])
    # utilities16_100.append([0.215, 0.32, 0.365, 0.335, 0.275])
    # utilities32_100.append([0.125, 0.195, 0.195, 0.205, 0.165])
    # utilities2_1000.append([])
    # utilities16_1000.append([])
    # utilities32_1000.append([])
    # comps.append([2.962, 2.748, 3.395, 2.98, 2.942])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 2.0; n = 4$')
    # utilities2_10.append([])
    # utilities16_10.append([])
    # utilities32_10.append([])
    # utilities2_50.append([])
    # utilities16_50.append([])
    # utilities32_50.append([])
    # utilities2_100.append([0.78, 0.885, 0.88, 0.87, 0.855])
    # utilities16_100.append([0.21, 0.28, 0.33, 0.325, 0.3])
    # utilities32_100.append([0.11, 0.17, 0.18, 0.185, 0.16])
    # utilities2_1000.append([])
    # utilities16_1000.append([])
    # utilities32_1000.append([])
    # comps.append([3.596, 3.426, 3.096, 3.679, 2.883])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 3.0; n = 4$')
    # utilities2_10.append([])
    # utilities16_10.append([])
    # utilities32_10.append([])
    # utilities2_50.append([])
    # utilities16_50.append([])
    # utilities32_50.append([])
    # utilities2_100.append([0.795, 0.87, 0.88, 0.865, 0.87])
    # utilities16_100.append([0.22, 0.275, 0.305, 0.325, 0.295])
    # utilities32_100.append([0.135, 0.19, 0.16, 0.17, 0.19])
    # utilities2_1000.append([])
    # utilities16_1000.append([])
    # utilities32_1000.append([])
    # comps.append([3.123, 3.455, 3.419, 3.86, 3.31])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 10; n = 4$')
    # utilities2_10.append([])
    # utilities16_10.append([])
    # utilities32_10.append([])
    # utilities2_50.append([])
    # utilities16_50.append([])
    # utilities32_50.append([])
    # utilities2_100.append([0.82, 0.87, 0.895, 0.865, 0.875])
    # utilities16_100.append([0.26, 0.315, 0.325, 0.355, 0.3])
    # utilities32_100.append([0.145, 0.195, 0.17, 0.22, 0.165])
    # utilities2_1000.append([])
    # utilities16_1000.append([])
    # utilities32_1000.append([])
    # comps.append([4.634, 4.924, 4.915, 4.245, 3.565])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')

    num_align = 100
    if num_align == 10:
        utilities2 = utilities2_10
        utilities16 = utilities16_10
        utilities32 = utilities32_10
    if num_align == 50:
        utilities2 = utilities2_50
        utilities16 = utilities16_50
        utilities32 = utilities32_50
    if num_align == 100:
        utilities2 = utilities2_100
        utilities16 = utilities16_100
        utilities32 = utilities32_100
    if num_align == 1000:
        utilities2 = utilities2_1000
        utilities16 = utilities16_1000
        utilities32 = utilities32_1000
    for data, suffix in zip([utilities2, utilities16, utilities32], ['2', '16', '32']):
        plot_multi_trials([comps, data], labels, sizes, ylabel='Utility', colors=None, filename='xlation_train_comp' + suffix + '.png')
    labels = ['$C=2$'] + ['' for _ in utilities2[:-1]] + ['$C=16$'] + ['' for _ in utilities16[:-1]] + ['$C=32$'] + ['' for _ in utilities32[:-1]]
    sizes = [2 * small_size for _ in utilities2] + [2 * small_size for _ in utilities16] + [2 * small_size for _ in utilities32]
    plot_multi_trials([comps + comps + comps, utilities2 + utilities16 + utilities32], labels, sizes, ylabel='Translation Utility', colors=None, filename='vg_xlation_comp_all_n' + str(n) + '_numalign' + str(num_align) + '.png')


if __name__ == '__main__':
    run()
