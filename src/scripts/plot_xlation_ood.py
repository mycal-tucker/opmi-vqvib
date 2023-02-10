from src.utils.plotting import plot_multi_trials


def run():
    labels = []
    xlation_utilities2_100 = []
    xlation_utilities16_100 = []
    xlation_utilities32_100 = []
    ood_utilities2 = []
    ood_utilities16 = []
    ood_utilities32 = []
    sizes = []
    colors = []
    small_size = 50

    ############################################################################
    #####           N = 1
    ############################################################################
    n = 1
    labels.append('$\lambda_I = 0; n = 1$')
    ood_utilities2.append([0.817, 0.498, 0.504, 0.509, 0.492])
    ood_utilities16.append([0.207, 0.066, 0.068, 0.07, 0.063])
    ood_utilities32.append([0.106, 0.034, 0.029, 0.028, 0.03])
    xlation_utilities2_100.append([0.8, 0.485, 0.44, 0.49, 0.435])
    xlation_utilities16_100.append([0.195, 0.07, 0.05, 0.07, 0.025])
    xlation_utilities32_100.append([0.105, 0.045, 0.03, 0.03, 0.01])
    sizes.append(small_size)
    colors.append('xkcd:blue')

    labels.append('$\lambda_I = 0.1; n = 1$')
    ood_utilities2.append([0.833, 0.757, 0.71, 0.58, 0.695])
    ood_utilities16.append([0.265, 0.22, 0.202, 0.112, 0.197])
    ood_utilities32.append([0.146, 0.123, 0.131, 0.056, 0.12])
    xlation_utilities2_100.append([0.85, 0.84, 0.825, 0.825, 0.875])
    xlation_utilities16_100.append([0.25, 0.23, 0.285, 0.195, 0.34])
    xlation_utilities32_100.append([0.14, 0.12, 0.17, 0.09, 0.18])
    sizes.append(small_size)
    colors.append('xkcd:blue')

    labels.append('$\lambda_I = 0.5; n = 1$')
    ood_utilities2.append([0.873, 0.756, 0.775, 0.806, 0.853])
    ood_utilities16.append([0.329, 0.266, 0.267, 0.278, 0.339])
    ood_utilities32.append([0.207, 0.143, 0.176, 0.182, 0.225])
    xlation_utilities2_100.append([0.91, 0.855, 0.865, 0.85, 0.915])
    xlation_utilities16_100.append([0.465, 0.245, 0.355, 0.285, 0.44])
    xlation_utilities32_100.append([0.32, 0.13, 0.225, 0.16, 0.265])
    sizes.append(small_size)
    colors.append('xkcd:blue')

    labels.append('$\lambda_I = 1; n = 1$')
    ood_utilities2.append([ 0.892, 0.856, 0.822, 0.846, 0.872])
    ood_utilities16.append([0.431, 0.343, 0.323, 0.338, 0.386])
    ood_utilities32.append([0.244, 0.218, 0.208, 0.186, 0.262])
    xlation_utilities2_100.append([0.92, 0.85, 0.89, 0.835, 0.895])
    xlation_utilities16_100.append([0.5, 0.265, 0.32, 0.255, 0.455])
    xlation_utilities32_100.append([0.295, 0.155, 0.195, 0.145, 0.26])
    sizes.append(small_size)
    colors.append('xkcd:blue')

    labels.append('$\lambda_I = 1.5; n = 1$')
    ood_utilities2.append([0.892, 0.87, 0.821, 0.84, 0.871])
    ood_utilities16.append([0.437, 0.389, 0.32, 0.377, 0.411])
    ood_utilities32.append([0.288, 0.252, 0.206, 0.232, 0.266])
    xlation_utilities2_100.append([0.91, 0.87, 0.84, 0.84, 0.895])
    xlation_utilities16_100.append([0.485, 0.27, 0.375, 0.265, 0.44])
    xlation_utilities32_100.append([0.345, 0.165, 0.25, 0.16, 0.28])
    sizes.append(small_size)
    colors.append('xkcd:blue')

    labels.append('$\lambda_I = 2.0; n = 1$')
    ood_utilities2.append([0.888, 0.876, 0.8, 0.868, 0.89])
    ood_utilities16.append([0.429, 0.383, 0.317, 0.373, 0.432])
    ood_utilities32.append([0.281, 0.261, 0.218, 0.241, 0.305])
    xlation_utilities2_100.append([0.895, 0.85, 0.85, 0.835, 0.905])
    xlation_utilities16_100.append([0.49, 0.29, 0.395, 0.245, 0.47])
    xlation_utilities32_100.append([0.3, 0.195, 0.225, 0.14, 0.295])
    sizes.append(small_size)
    colors.append('xkcd:blue')

    labels.append('$\lambda_I = 3.0; n = 1$')
    ood_utilities2.append([0.896, 0.881, 0.784, 0.86, 0.896])
    ood_utilities16.append([0.44, 0.407, 0.321, 0.376, 0.478])
    ood_utilities32.append([ 0.289, 0.256, 0.207, 0.244, 0.332])
    xlation_utilities2_100.append([0.885, 0.865, 0.86, 0.865, 0.895])
    xlation_utilities16_100.append([0.475, 0.28, 0.34, 0.285, 0.46])
    xlation_utilities32_100.append([0.295, 0.18, 0.22, 0.165, 0.275])
    sizes.append(small_size)
    colors.append('xkcd:blue')

    labels.append('$\lambda_I = 10; n = 1$')
    ood_utilities2.append([0.897, 0.848, 0.814, 0.882, 0.912])
    ood_utilities16.append([0.473, 0.354, 0.344, 0.417, 0.5])
    ood_utilities32.append([0.328, 0.246, 0.232, 0.292, 0.365])
    xlation_utilities2_100.append([0.895, 0.855, 0.84, 0.85, 0.9])
    xlation_utilities16_100.append([0.49, 0.31, 0.355, 0.32, 0.5])
    xlation_utilities32_100.append([0.315, 0.19, 0.25, 0.185, 0.305])
    sizes.append(small_size)
    colors.append('xkcd:blue')

    ############################################################################
    #####           N = 2
    ############################################################################
    # n = 2
    # labels.append('$\lambda_I = 0; n = 2$')
    # ood_utilities2.append([0.51, 0.493, 0.506, 0.509, 0.493])
    # ood_utilities16.append([0.067, 0.066, 0.071, 0.07, 0.063])
    # ood_utilities32.append([0.038, 0.034, 0.03, 0.027, 0.031])
    # xlation_utilities2_100.append([0.455, 0.485, 0.45, 0.485, 0.44])
    # xlation_utilities16_100.append([0.07, 0.07, 0.055, 0.07, 0.025])
    # xlation_utilities32_100.append([0.04, 0.045, 0.02, 0.03, 0.01])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 0.1; n = 2$')
    # ood_utilities2.append([ 0.866, 0.727, 0.8, 0.705, 0.722])
    # ood_utilities16.append([ 0.296, 0.214, 0.253, 0.185, 0.194])
    # ood_utilities32.append([ 0.157, 0.11, 0.15, 0.091, 0.127])
    # xlation_utilities2_100.append([0.87, 0.825, 0.85, 0.82, 0.885])
    # xlation_utilities16_100.append([0.355, 0.195, 0.27, 0.24, 0.295])
    # xlation_utilities32_100.append([0.2, 0.125, 0.155, 0.145, 0.175])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 0.5; n = 2$')
    # ood_utilities2.append([ 0.907, 0.873, 0.824, 0.823, 0.879])
    # ood_utilities16.append([0.441, 0.375, 0.342, 0.318, 0.382])
    # ood_utilities32.append([0.28, 0.229, 0.219, 0.189, 0.25])
    # xlation_utilities2_100.append([0.9, 0.84, 0.86, 0.87, 0.905])
    # xlation_utilities16_100.append([0.5, 0.28, 0.335, 0.32, 0.44])
    # xlation_utilities32_100.append([0.335, 0.125, 0.225, 0.205, 0.26])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 1; n = 2$')
    # ood_utilities2.append([ 0.922, 0.888, 0.838, 0.862, 0.892])
    # ood_utilities16.append([0.497, 0.422, 0.353, 0.392, 0.443])
    # ood_utilities32.append([0.321, 0.293, 0.248, 0.264, 0.314])
    # xlation_utilities2_100.append([0.92, 0.855, 0.86, 0.84, 0.9])
    # xlation_utilities16_100.append([0.525, 0.325, 0.345, 0.27, 0.465])
    # xlation_utilities32_100.append([0.345, 0.19, 0.22, 0.185, 0.28])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 1.5; n = 2$')
    # ood_utilities2.append([ 0.931, 0.902, 0.851, 0.884, 0.92])
    # ood_utilities16.append([0.499, 0.452, 0.402, 0.426, 0.522])
    # ood_utilities32.append([ 0.345, 0.282, 0.284, 0.296, 0.405])
    # xlation_utilities2_100.append([0.895, 0.87, 0.855, 0.86, 0.9])
    # xlation_utilities16_100.append([0.495, 0.305, 0.37, 0.295, 0.515])
    # xlation_utilities32_100.append([0.33, 0.18, 0.235, 0.205, 0.305])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 2.0; n = 2$')
    # ood_utilities2.append([ 0.932, 0.909, 0.852, 0.891, 0.932])
    # ood_utilities16.append([ 0.53, 0.477, 0.417, 0.49, 0.534])
    # ood_utilities32.append([0.367, 0.334, 0.274, 0.311, 0.404])
    # xlation_utilities2_100.append([0.91, 0.865, 0.865, 0.855, 0.9])
    # xlation_utilities16_100.append([0.5, 0.315, 0.355, 0.28, 0.49])
    # xlation_utilities32_100.append([0.325, 0.185, 0.23, 0.175, 0.285])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 3.0; n = 2$')
    # ood_utilities2.append([ 0.947, 0.902, 0.869, 0.904, 0.945])
    # ood_utilities16.append([0.581, 0.475, 0.404, 0.529, 0.589])
    # ood_utilities32.append([0.405, 0.343, 0.302, 0.369, 0.437])
    # xlation_utilities2_100.append([0.915, 0.86, 0.865, 0.86, 0.915])
    # xlation_utilities16_100.append([0.5, 0.32, 0.365, 0.33, 0.5])
    # xlation_utilities32_100.append([0.31, 0.2, 0.255, 0.18, 0.3])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 10; n = 2$')
    # ood_utilities2.append([ 0.951, 0.918, 0.9, 0.917, 0.959])
    # ood_utilities16.append([ 0.622, 0.481, 0.485, 0.543, 0.642])
    # ood_utilities32.append([ 0.46, 0.368, 0.375, 0.4, 0.516])
    # xlation_utilities2_100.append([0.905, 0.855, 0.875, 0.85, 0.91])
    # xlation_utilities16_100.append([0.525, 0.315, 0.385, 0.32, 0.49])
    # xlation_utilities32_100.append([0.33, 0.2, 0.245, 0.21, 0.315])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    ############################################################################
    #####           N = 4
    ############################################################################
    # n = 4
    # labels.append('$\lambda_I = 0; n = 4$')
    # ood_utilities2.append([0.867, 0.498, 0.823, 0.51, 0.491])
    # ood_utilities16.append([0.306, 0.067, 0.304, 0.071, 0.062])
    # ood_utilities32.append([0.178, 0.034, 0.167, 0.027, 0.031])
    # xlation_utilities2_100.append([0.825, 0.525, 0.81, 0.485, 0.435])
    # xlation_utilities16_100.append([0.225, 0.06, 0.22, 0.065, 0.025])
    # xlation_utilities32_100.append([0.135, 0.035, 0.145, 0.04,  0.01])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 0.1; n = 4$')
    # ood_utilities2.append([0.884, 0.782, 0.82, 0.766, 0.83])
    # ood_utilities16.append([0.346, 0.283, 0.274, 0.234, 0.293])
    # ood_utilities32.append([0.188, 0.156, 0.171, 0.124, 0.184])
    # xlation_utilities2_100.append([0.92, 0.83, 0.84, 0.82, 0.85])
    # xlation_utilities16_100.append([0.41, 0.27, 0.28, 0.19, 0.36])
    # xlation_utilities32_100.append([0.25, 0.11, 0.2,  0.12, 0.25])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 0.5; n = 4$')
    # ood_utilities2.append([0.918, 0.898, 0.845, 0.877, 0.902])
    # ood_utilities16.append([0.505, 0.435, 0.372, 0.398, 0.441])
    # ood_utilities32.append([0.325, 0.293, 0.257, 0.263, 0.318])
    # xlation_utilities2_100.append([0.92, 0.865, 0.87, 0.84, 0.895])
    # xlation_utilities16_100.append([0.525, 0.295, 0.375, 0.275, 0.43])
    # xlation_utilities32_100.append([0.285, 0.15, 0.225, 0.165, 0.265])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 1; n = 4$')
    # ood_utilities2.append([ 0.933, 0.918, 0.859, 0.902, 0.941])
    # ood_utilities16.append([0.547, 0.509, 0.404, 0.5, 0.588])
    # ood_utilities32.append([0.393, 0.357, 0.288, 0.346, 0.458])
    # xlation_utilities2_100.append([0.93, 0.9,  0.87, 0.84, 0.89])
    # xlation_utilities16_100.append([0.51, 0.37, 0.41, 0.31, 0.44])
    # xlation_utilities32_100.append([0.27, 0.18, 0.24, 0.16, 0.29])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 1.5; n = 4$')
    # ood_utilities2.append([ 0.946, 0.92, 0.909, 0.927, 0.959])
    # ood_utilities16.append([0.604, 0.525, 0.508, 0.565, 0.66])
    # ood_utilities32.append([0.427, 0.361, 0.379, 0.432, 0.525])
    # xlation_utilities2_100.append([0.925, 0.885, 0.88, 0.87, 0.905])
    # xlation_utilities16_100.append([0.545, 0.345, 0.4, 0.315, 0.455])
    # xlation_utilities32_100.append([0.33, 0.2, 0.26, 0.21, 0.29])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 2.0; n = 4$')
    # ood_utilities2.append([ 0.958, 0.921, 0.916, 0.938, 0.97])
    # ood_utilities16.append([0.652, 0.524, 0.559, 0.628, 0.724])
    # ood_utilities32.append([0.481, 0.381, 0.437, 0.475, 0.605])
    # xlation_utilities2_100.append([0.93, 0.88, 0.89, 0.855, 0.9])
    # xlation_utilities16_100.append([0.545, 0.325, 0.395, 0.3, 0.455])
    # xlation_utilities32_100.append([0.33, 0.205, 0.265, 0.16, 0.28])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 3.0; n = 4$')
    # ood_utilities2.append([0.969, 0.948, 0.923, 0.964, 0.981])
    # ood_utilities16.append([0.733, 0.619, 0.582, 0.728, 0.805])
    # ood_utilities32.append([0.582, 0.495, 0.456, 0.592, 0.689])
    # xlation_utilities2_100.append([0.94, 0.885, 0.89, 0.88, 0.915])
    # xlation_utilities16_100.append([0.575, 0.325, 0.415, 0.335, 0.475])
    # xlation_utilities32_100.append([0.36, 0.185, 0.27, 0.22, 0.295])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 10; n = 4$')
    # ood_utilities2.append([0.982, 0.965, 0.95, 0.985, 0.99])
    # ood_utilities16.append([0.833, 0.731, 0.698, 0.863, 0.89])
    # ood_utilities32.append([0.704, 0.628, 0.605, 0.78, 0.818])
    # xlation_utilities2_100.append([0.94, 0.86, 0.86, 0.84, 0.92])
    # xlation_utilities16_100.append([0.58, 0.39, 0.41, 0.35, 0.5])
    # xlation_utilities32_100.append([0.36, 0.22, 0.23, 0.24, 0.3])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')

    # labels.append('$\lambda_I = 100; n = 4$')
    # ood_utilities2.append([0.982, 0.966, 0.952, 0.984, 0.993])
    # ood_utilities16.append([ 0.805, 0.73, 0.733, 0.851, 0.903])
    # ood_utilities32.append([0.716, 0.623, 0.644, 0.763, 0.851])
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
    labels = ['$C=2$'] + ['' for _ in xlation_utilities2_100[:-1]] + ['$C=16$'] + ['' for _ in xlation_utilities16_100[:-1]] + ['$C=32$'] + ['' for _ in xlation_utilities32_100[:-1]]
    sizes = [4 * small_size for _ in xlation_utilities2_100] + [2 * small_size for _ in xlation_utilities16_100] + [small_size for _ in xlation_utilities32_100]
    ood_c = 16
    if ood_c == 2:
        plot_ood_data = ood_utilities2
    elif ood_c == 16:
        plot_ood_data = ood_utilities16
    elif ood_c == 32:
        plot_ood_data = ood_utilities32
    # plot_multi_trials([plot_ood_data + plot_ood_data + plot_ood_data, xlation_utilities2_100 + xlation_utilities16_100 + xlation_utilities32_100], labels, sizes, ylabel='Translation Utility', xlabel='OOD Utility; $C=' + str(ood_c) + '$', colors=None, filename='xlation_ood_all' + str(ood_c) + 'n' + str(n) + '.png')
    plot_multi_trials([ood_utilities2 + ood_utilities16 + ood_utilities32, xlation_utilities2_100 + xlation_utilities16_100 + xlation_utilities32_100], labels, sizes, ylabel='Translation Utility', xlabel='OOD Utility', colors=None, filename='xlation_ood_all_n' + str(n) + '.png')

if __name__ == '__main__':
    run()
