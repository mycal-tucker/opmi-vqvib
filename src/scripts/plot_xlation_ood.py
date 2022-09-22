from src.utils.plotting import plot_multi_trials


def run():
    labels = []
    xlation_utilities2_100 = []
    xlation_utilities16_100 = []
    xlation_utilities32_100 = []
    ood_utilities32 = []
    sizes = []
    colors = []
    small_size = 50

    labels.append('$\lambda_I = 0; n = 4$')
    ood_utilities32.append([0.146, 0.123, 0.131, 0.056, 0.12])
    xlation_utilities2_100.append([0.825, 0.525, 0.81, 0.485, 0.435])
    xlation_utilities16_100.append([0.225, 0.06, 0.22, 0.065, 0.025])
    xlation_utilities32_100.append([0.135, 0.035, 0.145, 0.04,  0.01])
    sizes.append(small_size)
    colors.append('xkcd:blue')

    labels.append('$\lambda_I = 0.1; n = 4$')
    ood_utilities32.append([0.188, 0.156, 0.171, 0.124, 0.184])
    xlation_utilities2_100.append([0.92, 0.83, 0.84, 0.82, 0.85])
    xlation_utilities16_100.append([0.41, 0.27, 0.28, 0.19, 0.36])
    xlation_utilities32_100.append([0.25, 0.11, 0.2,  0.12, 0.25])
    sizes.append(small_size)
    colors.append('xkcd:blue')

    labels.append('$\lambda_I = 1; n = 4$')
    ood_utilities32.append([0.393, 0.357, 0.288, 0.346, 0.458])
    xlation_utilities2_100.append([0.93, 0.9,  0.87, 0.84, 0.89])
    xlation_utilities16_100.append([0.51, 0.37, 0.41, 0.31, 0.44])
    xlation_utilities32_100.append([0.27, 0.18, 0.24, 0.16, 0.29])
    sizes.append(small_size)
    colors.append('xkcd:blue')

    labels.append('$\lambda_I = 10; n = 4$')
    ood_utilities32.append([0.746, 0.691, 0.611, 0.832, 0.838])
    xlation_utilities2_100.append([0.94, 0.86, 0.86, 0.84, 0.92])
    xlation_utilities16_100.append([0.58, 0.39, 0.41, 0.35, 0.5])
    xlation_utilities32_100.append([0.36, 0.22, 0.23, 0.24, 0.3])
    sizes.append(small_size)
    colors.append('xkcd:blue')

    # labels.append('$\lambda_I = 100; n = 4$')
    # ood_utilities32.append([0.772, 0.714, 0.611, 0.807, 0.875])
    # xlation_utilities2_100.append([0.95, 0.87, 0.89, 0.84, 0.92])
    # xlation_utilities16_100.append([0.58, 0.36, 0.42, 0.36, 0.48])
    # xlation_utilities32_100.append([0.33, 0.2, 0.25, 0.21, 0.32])
    # sizes.append(small_size)
    # colors.append('xkcd:orange')

    # labels.append('$\lambda_I = 1; n = 8$')
    # ood_utilities32.append([0.389, 0.379, 0.424, 0.574])
    # xlation_utilities2_100.append([0.87, 0.89, 0.86, 0.92])
    # sizes.append(small_size)
    # colors.append('xkcd:green')
    #
    # labels.append('$\lambda_I = 10; n = 8$')
    # ood_utilities32.append([0.869, 0.796, 0.797, 0.827, 0.93])
    # xlation_utilities2_100.append([0.95, 0.91, 0.88, 0.85, 0.91])
    # sizes.append(small_size)
    # colors.append('xkcd:green')
    #
    # labels.append('$\lambda_I = 100; n = 8$')
    # ood_utilities32.append([0.911, 0.858, 0.436, 0.916, 0.939])
    # xlation_utilities2_100.append([0.95, 0.89, 0.89, 0.88, 0.92])
    # sizes.append(small_size)
    # colors.append('xkcd:black')

    for data, suffix in zip([xlation_utilities2_100, xlation_utilities16_100, xlation_utilities32_100], ['2', '16', '32']):
        plot_multi_trials([ood_utilities32, data], labels, sizes, ylabel='Translation Utility', xlabel='OOD Utility', colors=None, filename='xlation' + suffix + '_ood.png')

if __name__ == '__main__':
    run()
