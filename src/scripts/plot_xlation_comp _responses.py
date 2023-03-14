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
    # utilities2_100.append([])
    # utilities16_100.append([])
    # utilities32_100.append([])
    # comps.append([0.001, 0.637, 0.298, 0.003, 0.0])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 1.0$')
    # utilities2_100.append([])
    # utilities16_100.append([])
    # utilities32_100.append([])
    # comps.append([1.393, 0.489, 0.782, 1.268, 0.651])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 10.0$')
    # utilities2_100.append([])
    # utilities16_100.append([])
    # utilities32_100.append([])
    # comps.append([1.22, 1.432, 1.346, 1.38, 1.105])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 100$')
    # utilities2_100.append([])
    # utilities16_100.append([])
    # utilities32_100.append([])
    # comps.append([1.201, 1.492, 1.216, 1.353, 1.047])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')

    ############################################################################
    #####           Proto
    ############################################################################
    # n = 'proto'
    # labels.append('$\lambda_I = 0.1$')
    # utilities2_100.append([])
    # utilities16_100.append([])
    # utilities32_100.append([])
    # comps.append([1.076, 1.311, 1.254, 1.203, 1.128])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 1.0$')
    # utilities2_100.append([])
    # utilities16_100.append([])
    # utilities32_100.append([])
    # comps.append([1.3, 1.684, 1.455, 1.599, 1.635])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 10.0$')
    # utilities2_100.append([])
    # utilities16_100.append([])
    # utilities32_100.append([])
    # comps.append([1.86, 1.962, 1.89, 1.983, 1.874])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 100$')
    # utilities2_100.append([])
    # utilities16_100.append([])
    # utilities32_100.append([])
    # comps.append([2.448, 2.474, 2.426, 2.525, 2.605])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')

    ############################################################################
    #####           N = 1
    ############################################################################
    # n = 1
    # labels.append('$\lambda_I = 0.0; n = 1$')
    # utilities2_100.append([0.77, 0.49, 0.45, 0.51, 0.39])
    # utilities16_100.append([0.18, 0.08, 0.05, 0.07, 0.02])
    # utilities32_100.append([0.11, 0.03, 0.03, 0.03, 0.02])
    # comps.append([1.084, 0.0, -0.0, -0.0, 0.0])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 0.1; n = 1$')
    # utilities2_100.append([0.83, 0.83, 0.75, 0.75, 0.8])
    # utilities16_100.append([0.26, 0.22, 0.22, 0.18, 0.3])
    # utilities32_100.append([0.14, 0.12, 0.11, 0.07, 0.16])
    # comps.append([1.284, 1.141, 1.347, 0.995, 1.142])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 0.5; n = 1$')
    # utilities2_100.append([0.87, 0.81, 0.79, 0.8, 0.77])
    # utilities16_100.append([0.38, 0.25, 0.28, 0.27, 0.39])
    # utilities32_100.append([0.29, 0.16, 0.17, 0.12, 0.24])
    # comps.append([1.823, 1.586, 1.659, 1.662, 1.816])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 1.0; n = 1$')
    # utilities2_100.append([0.88, 0.82, 0.78, 0.82, 0.83])
    # utilities16_100.append([0.44, 0.24, 0.27, 0.29, 0.36])
    # utilities32_100.append([0.32, 0.14, 0.17, 0.14, 0.26])
    # comps.append([2.13, 1.899, 2.113, 1.94, 2.031])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 1.5; n = 1$')
    # utilities2_100.append([0.88, 0.81, 0.79, 0.79, 0.83])
    # utilities16_100.append([0.44, 0.29, 0.34, 0.3, 0.36])
    # utilities32_100.append([0.28, 0.13, 0.19, 0.16, 0.26])
    # comps.append([2.113, 2.218, 2.153, 2.234, 2.136])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 2.0; n = 1$')
    # utilities2_100.append([0.91, 0.8, 0.79, 0.78, 0.82])
    # utilities16_100.append([0.46, 0.3, 0.32, 0.27, 0.36])
    # utilities32_100.append([0.25, 0.15, 0.19, 0.12, 0.25])
    # comps.append([2.196, 2.206, 2.232, 2.203, 2.277])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 3.0; n = 1$')
    # utilities2_100.append([0.88, 0.81, 0.76, 0.8, 0.83])
    # utilities16_100.append([0.41, 0.28, 0.31, 0.31, 0.39])
    # utilities32_100.append([0.29, 0.15, 0.15, 0.14, 0.26])
    # comps.append([2.307, 2.257, 2.198, 2.247, 2.312])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 10.0; n = 1$')
    # utilities2_100.append([0.91, 0.81, 0.72, 0.8, 0.87])
    # utilities16_100.append([0.41, 0.3, 0.35, 0.31, 0.41])
    # utilities32_100.append([0.29, 0.19, 0.17, 0.15, 0.3])
    # comps.append([2.598, 2.487, 2.324, 2.482, 2.665])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')

    ############################################################################
    #####           N = 2
    ############################################################################
    # n = 2
    # labels.append('$\lambda_I = 0.0; n = 1$')
    # utilities2_100.append([0.45, 0.5, 0.45, 0.5, 0.4])
    # utilities16_100.append([0.09, 0.08, 0.06, 0.07, 0.02])
    # utilities32_100.append([0.04, 0.04, 0.03, 0.03, 0.02])
    # comps.append([ -0.0, -0.0, 0.0, 0.0, -0.0])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 0.1; n = 1$')
    # utilities2_100.append([0.85, 0.78, 0.77, 0.79, 0.79])
    # utilities16_100.append([0.31, 0.2, 0.23, 0.21, 0.3])
    # utilities32_100.append([0.16, 0.09, 0.12, 0.1, 0.2])
    # comps.append([1.681, 1.2, 1.549, 1.316, 1.493])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 0.5; n = 1$')
    # utilities2_100.append([0.88, 0.83, 0.78, 0.8, 0.81])
    # utilities16_100.append([0.43, 0.25, 0.28, 0.34, 0.39])
    # utilities32_100.append([0.35, 0.12, 0.17, 0.2, 0.21])
    # comps.append([2.043, 1.908, 2.073, 1.881, 1.993])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 1.0; n = 1$')
    # utilities2_100.append([0.89, 0.81, 0.78, 0.81, 0.78])
    # utilities16_100.append([0.46, 0.27, 0.28, 0.33, 0.4])
    # utilities32_100.append([0.33, 0.15, 0.2, 0.14, 0.28])
    # comps.append([2.403, 2.3, 2.391, 2.254, 2.298])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 1.5; n = 1$')
    # utilities2_100.append([0.87, 0.81, 0.75, 0.83, 0.79])
    # utilities16_100.append([0.49, 0.34, 0.32, 0.3, 0.42])
    # utilities32_100.append([0.27, 0.17, 0.19, 0.18, 0.28])
    # comps.append([2.425, 2.458, 2.722, 2.454, 2.48])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 2.0; n = 1$')
    # utilities2_100.append([0.87, 0.82, 0.73, 0.82, 0.79])
    # utilities16_100.append([0.45, 0.29, 0.28, 0.32, 0.4])
    # utilities32_100.append([0.31, 0.19, 0.18, 0.18, 0.26])
    # comps.append([2.587, 2.758, 2.467, 2.566, 2.66])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 3.0; n = 1$')
    # utilities2_100.append([0.89, 0.84, 0.76, 0.81, 0.8])
    # utilities16_100.append([0.46, 0.35, 0.3, 0.3, 0.37])
    # utilities32_100.append([0.35, 0.17, 0.18, 0.18, 0.27])
    # comps.append([2.867, 2.847, 2.612, 3.257, 3.258])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 10.0; n = 1$')
    # utilities2_100.append([0.91, 0.84, 0.75, 0.83, 0.82])
    # utilities16_100.append([0.47, 0.34, 0.31, 0.3, 0.38])
    # utilities32_100.append([0.35, 0.19, 0.17, 0.18, 0.26])
    # comps.append([3.491, 3.085, 3.435, 3.279, 3.665])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    ############################################################################
    #####           N = 4
    ############################################################################
    n = 4
    labels.append('$\lambda_I = 0.0; n = 4$')
    utilities2_10.append([0.63, 0.46, 0.67, 0.51, 0.39])
    utilities16_10.append([0.16, 0.06, 0.13, 0.08, 0.03])
    utilities32_10.append([0.06, 0.04, 0.05, 0.02, 0.01])
    utilities2_50.append([0.82, 0.46, 0.71, 0.52, 0.4])
    utilities16_50.append([0.24, 0.08, 0.13, 0.07, 0.03])
    utilities32_50.append([0.12, 0.04, 0.07, 0.03, 0.01])
    utilities2_100.append([0.78, 0.5, 0.76, 0.51, 0.4])
    utilities16_100.append([0.23, 0.08, 0.17, 0.07, 0.02])
    utilities32_100.append([0.12, 0.04, 0.11, 0.03, 0.02])
    utilities2_1000.append([0.82, 0.47, 0.74, 0.5, 0.39])
    utilities16_1000.append([0.27, 0.06, 0.19, 0.07, 0.02])
    utilities32_1000.append([0.13, 0.04, 0.12, 0.02, 0.02])
    comps.append([1.507, -0.0, 1.517, -0.0, 0.0])
    sizes.append(small_size)
    colors.append(0)

    labels.append('$\lambda_I = 0.1; n = 4$')
    utilities2_10.append([0.74, 0.77, 0.58, 0.54, 0.66])
    utilities16_10.append([0.2, 0.18, 0.12, 0.11, 0.24])
    utilities32_10.append([0.11, 0.06, 0.05, 0.04, 0.14])
    utilities2_50.append([0.83, 0.8, 0.77, 0.75, 0.8])
    utilities16_50.append([0.36, 0.21, 0.24, 0.18, 0.31])
    utilities32_50.append([0.24, 0.11, 0.1, 0.12, 0.18])
    utilities2_100.append([0.87, 0.81, 0.76, 0.77, 0.8])
    utilities16_100.append([0.42, 0.22, 0.23, 0.16, 0.34])
    utilities32_100.append([0.24, 0.1, 0.12, 0.09, 0.2])
    utilities2_1000.append([0.88, 0.8, 0.74, 0.78, 0.85])
    utilities16_1000.append([0.4, 0.19, 0.25, 0.22, 0.31])
    utilities32_1000.append([0.3, 0.12, 0.17, 0.1, 0.21])
    comps.append([1.748, 1.476, 1.751, 1.149, 1.65])
    sizes.append(small_size)
    colors.append(0.1)

    labels.append('$\lambda_I = 0.5; n = 4$')
    utilities2_10.append([0.75, 0.71, 0.78, 0.57, 0.72])
    utilities16_10.append([0.25, 0.21, 0.15, 0.12, 0.26])
    utilities32_10.append([0.16, 0.05, 0.08, 0.08, 0.09])
    utilities2_50.append([0.88, 0.75, 0.77, 0.79, 0.79])
    utilities16_50.append([0.41, 0.29, 0.26, 0.28, 0.3])
    utilities32_50.append([0.27, 0.15, 0.19, 0.15, 0.2])
    utilities2_100.append([0.9, 0.85, 0.81, 0.82, 0.79])
    utilities16_100.append([0.45, 0.27, 0.36, 0.3, 0.36])
    utilities32_100.append([0.32, 0.15, 0.21, 0.17, 0.2])
    utilities2_1000.append([0.9, 0.78, 0.81, 0.82, 0.84])
    utilities16_1000.append([0.47, 0.23, 0.3, 0.32, 0.41])
    utilities32_1000.append([0.35, 0.14, 0.2, 0.15, 0.28])
    comps.append([2.382, 2.199, 2.216, 2.225, 2.125])
    sizes.append(small_size)
    colors.append('xkcd:blue')

    labels.append('$\lambda_I = 1.0; n = 4$')
    utilities2_10.append([0.71, 0.72, 0.69, 0.62, 0.66])
    utilities16_10.append([0.25, 0.22, 0.18, 0.13, 0.28])
    utilities32_10.append([0.15, 0.11, 0.09, 0.06, 0.11])
    utilities2_50.append([0.87, 0.78, 0.77, 0.83, 0.83])
    utilities16_50.append([0.46, 0.3, 0.29, 0.27, 0.37])
    utilities32_50.append([0.31, 0.15, 0.18, 0.13, 0.22])
    utilities2_100.append([0.89, 0.81, 0.74, 0.83, 0.78])
    utilities16_100.append([0.45, 0.29, 0.29, 0.29, 0.38])
    utilities32_100.append([0.3, 0.17, 0.16, 0.18, 0.31])
    utilities2_1000.append([0.88, 0.83, 0.82, 0.81, 0.85])
    utilities16_1000.append([0.5, 0.32, 0.35, 0.33, 0.36])
    utilities32_1000.append([0.33, 0.16, 0.23, 0.19, 0.24])
    comps.append([2.734, 2.652, 2.683, 2.763, 2.809])
    sizes.append(small_size)
    colors.append(1)

    labels.append('$\lambda_I = 1.5; n = 4$')
    utilities2_10.append([0.69, 0.65, 0.63, 0.74, 0.7])
    utilities16_10.append([0.24, 0.17, 0.17, 0.14, 0.29])
    utilities32_10.append([0.17, 0.11, 0.07, 0.09, 0.1])
    utilities2_50.append([0.85, 0.76, 0.73, 0.8, 0.8])
    utilities16_50.append([0.47, 0.32, 0.28, 0.27, 0.33])
    utilities32_50.append([0.29, 0.15, 0.16, 0.16, 0.26])
    utilities2_100.append([0.91, 0.81, 0.77, 0.81, 0.83])
    utilities16_100.append([0.45, 0.33, 0.29, 0.3, 0.34])
    utilities32_100.append([0.33, 0.19, 0.18, 0.22, 0.26])
    utilities2_1000.append([0.91, 0.83, 0.79, 0.79, 0.86])
    utilities16_1000.append([0.49, 0.37, 0.36, 0.38, 0.39])
    utilities32_1000.append([0.38, 0.16, 0.23, 0.21, 0.23])
    comps.append([2.962, 2.748, 3.395, 2.98, 2.942])
    sizes.append(small_size)
    colors.append('xkcd:blue')

    labels.append('$\lambda_I = 2.0; n = 4$')
    utilities2_10.append([0.77, 0.71, 0.68, 0.58, 0.7])
    utilities16_10.append([0.32, 0.23, 0.14, 0.12, 0.29])
    utilities32_10.append([0.17, 0.12, 0.07, 0.06, 0.15])
    utilities2_50.append([0.85, 0.8, 0.76, 0.79, 0.79])
    utilities16_50.append([0.48, 0.29, 0.26, 0.27, 0.36])
    utilities32_50.append([0.33, 0.16, 0.14, 0.11, 0.22])
    utilities2_100.append([0.91, 0.8, 0.76, 0.78, 0.8])
    utilities16_100.append([0.47, 0.31, 0.3, 0.3, 0.38])
    utilities32_100.append([0.32, 0.18, 0.18, 0.16, 0.27])
    utilities2_1000.append([0.9, 0.84, 0.75, 0.82, 0.84])
    utilities16_1000.append([0.48, 0.34, 0.36, 0.33, 0.43])
    utilities32_1000.append([0.39, 0.16, 0.25, 0.21, 0.25])
    comps.append([3.596, 3.426, 3.096, 3.679, 2.883])
    sizes.append(small_size)
    colors.append('xkcd:blue')

    labels.append('$\lambda_I = 3.0; n = 4$')
    utilities2_10.append([0.77, 0.68, 0.7, 0.61, 0.71])
    utilities16_10.append([0.27, 0.26, 0.14, 0.16, 0.26])
    utilities32_10.append([0.14, 0.1, 0.06, 0.1, 0.17])
    utilities2_50.append([0.86, 0.76, 0.75, 0.82, 0.79])
    utilities16_50.append([0.48, 0.31, 0.29, 0.31, 0.36])
    utilities32_50.append([0.29, 0.15, 0.18, 0.15, 0.23])
    utilities2_100.append([0.91, 0.8, 0.79, 0.82, 0.79])
    utilities16_100.append([0.46, 0.32, 0.33, 0.32, 0.39])
    utilities32_100.append([0.35, 0.19, 0.19, 0.19, 0.27])
    utilities2_1000.append([0.92, 0.78, 0.83, 0.84, 0.87])
    utilities16_1000.append([0.54, 0.3, 0.38, 0.36, 0.42])
    utilities32_1000.append([0.38, 0.15, 0.23, 0.23, 0.29])
    comps.append([3.123, 3.455, 3.419, 3.86, 3.31])
    sizes.append(small_size)
    colors.append('xkcd:blue')

    labels.append('$\lambda_I = 10; n = 4$')
    utilities2_10.append([0.75, 0.73, 0.66, 0.55, 0.65])
    utilities16_10.append([0.27, 0.25, 0.12, 0.12, 0.22])
    utilities32_10.append([0.17, 0.13, 0.06, 0.05, 0.14])
    utilities2_50.append([0.84, 0.76, 0.77, 0.81, 0.81])
    utilities16_50.append([0.5, 0.34, 0.26, 0.32, 0.37])
    utilities32_50.append([0.27, 0.19, 0.14, 0.17, 0.27])
    utilities2_100.append([0.89, 0.82, 0.77, 0.84, 0.83])
    utilities16_100.append([0.46, 0.37, 0.34, 0.33, 0.4])
    utilities32_100.append([0.36, 0.19, 0.19, 0.2, 0.3])
    utilities2_1000.append([0.92, 0.8, 0.81, 0.8, 0.86])
    utilities16_1000.append([0.51, 0.37, 0.37, 0.41, 0.43])
    utilities32_1000.append([0.39, 0.2, 0.26, 0.25, 0.29])
    comps.append([4.634, 4.924, 4.915, 4.245, 3.565])
    sizes.append(small_size)
    colors.append('xkcd:blue')

    ############################################################################
    #####           N = 8
    ############################################################################
    # n = 8
    # labels.append('$\lambda_I = 0.0; n = 8$')
    # utilities2_100.append([])
    # utilities16_100.append([])
    # utilities32_100.append([])
    # comps.append([1.553, -0.0, -0.0, -0.0, 1.622])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 0.1; n = 8$')
    # utilities2_100.append([])
    # utilities16_100.append([])
    # utilities32_100.append([])
    # comps.append([1.813, 1.53, 1.728, 1.776, 1.589])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 0.5; n = 8$')
    # utilities2_100.append([])
    # utilities16_100.append([])
    # utilities32_100.append([])
    # comps.append([ 2.126, 2.28, 2.201, 2.502, 2.827])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 1.0; n = 8$')
    # utilities2_100.append([])
    # utilities16_100.append([])
    # utilities32_100.append([])
    # comps.append([ 2.884, 3.16, 3.221, 3.308, 3.062])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 1.5; n = 8$')
    # utilities2_100.append([])
    # utilities16_100.append([])
    # utilities32_100.append([])
    # comps.append([3.124, 3.448, 4.164, 3.883, 3.793])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 2.0; n = 8$')
    # utilities2_100.append([])
    # utilities16_100.append([])
    # utilities32_100.append([])
    # comps.append([ 3.612, 3.618, 3.957, 4.084, 3.863])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 3.0; n = 8$')
    # utilities2_100.append([])
    # utilities16_100.append([])
    # utilities32_100.append([])
    # comps.append([ 4.456, 3.311, 4.16, 4.543, 4.493])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 10.0; n = 8$')
    # utilities2_100.append([])
    # utilities16_100.append([])
    # utilities32_100.append([])
    # comps.append([4.703, 4.502, 5.082, 5.502, 4.655])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')

    ############################################################################
    #####           N = 16
    ############################################################################
    # n = 16
    # labels.append('$\lambda_I = 0.0; n = 16$')
    # utilities2_100.append([])
    # utilities16_100.append([])
    # utilities32_100.append([])
    # comps.append([1.928, 1.227, 0.0, -0.0, 1.932])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 0.1; n = 16$')
    # utilities2_100.append([])
    # utilities16_100.append([])
    # utilities32_100.append([])
    # comps.append([1.862, 1.579, 1.955, -0.0, 1.679])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 0.5; n = 16$')
    # utilities2_100.append([])
    # utilities16_100.append([])
    # utilities32_100.append([])
    # comps.append([2.394, 2.143, 2.707, 2.454, 2.685])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 1.0; n = 16$')
    # utilities2_100.append([])
    # utilities16_100.append([])
    # utilities32_100.append([])
    # comps.append([2.915, 2.485, 3.422, 2.821, 3.037])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 1.5; n = 16$')
    # utilities2_100.append([])
    # utilities16_100.append([])
    # utilities32_100.append([])
    # comps.append([3.31, 3.021, 4.041, 3.5, 3.319])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 2.0; n = 8$')
    # utilities2_100.append([])
    # utilities16_100.append([])
    # utilities32_100.append([])
    # comps.append([ 3.847, 3.184, 4.175, 3.823, 3.878])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 3.0; n = 16$')
    # utilities2_100.append([])
    # utilities16_100.append([])
    # utilities32_100.append([])
    # comps.append([4.35, 4.017, 4.986, 3.709, 2.619])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 10.0; n = 16$')
    # utilities2_100.append([])
    # utilities16_100.append([])
    # utilities32_100.append([])
    # comps.append([4.31, 4.316, 4.773, 3.781, 4.276])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')

    num_align = 50
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
        plot_multi_trials([comps, data], labels, sizes, ylabel='Utility', colors=None, filename='xlation_train_responses_comp' + suffix + '.png')
    labels = ['$C=2$'] + ['' for _ in utilities2[:-1]] + ['$C=16$'] + ['' for _ in utilities16[:-1]] + ['$C=32$'] + ['' for _ in utilities32[:-1]]
    sizes = [2 * small_size for _ in utilities2] + [2 * small_size for _ in utilities16] + [2 * small_size for _ in utilities32]
    plot_multi_trials([comps + comps + comps, utilities2 + utilities16 + utilities32], labels, sizes, ylabel='Translation Utility', colors=None, filename='xlation_comp_responses_all_n' + str(n) + '_numalign' + str(num_align) + '.png')


if __name__ == '__main__':
    run()
