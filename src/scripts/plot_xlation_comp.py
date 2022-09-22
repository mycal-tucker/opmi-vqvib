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

    labels.append('$\lambda_I = 1.0; n = 4$')
    utilities2_100.append([0.93, 0.9,  0.87, 0.84, 0.89])
    utilities16_100.append([0.51, 0.37, 0.41, 0.31, 0.44])
    utilities32_100.append([0.27, 0.18, 0.24, 0.16, 0.29])
    utilities32_1000.append([0.18, 0.29, 0.26, 0.33])
    comps.append([2.734, 2.652, 2.683, 2.763, 2.809])
    sizes.append(small_size)
    colors.append(1)

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


if __name__ == '__main__':
    run()
