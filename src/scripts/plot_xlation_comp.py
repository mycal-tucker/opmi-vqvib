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
    #####           Proto
    ############################################################################
    # n = 'proto'
    # labels.append('$\lambda_I = 0.1$')
    # utilities2_100.append([0.545, 0.715, 0.705, 0.705, 0.565])
    # utilities16_100.append([0.065, 0.135, 0.135, 0.12, 0.115])
    # utilities32_100.append([0.035, 0.07, 0.095, 0.055, 0.0])
    # comps.append([1.076, 1.311, 1.254, 1.203, 1.128])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 1.0$')
    # utilities2_100.append([0.745, 0.825, 0.68, 0.61, 0.57])
    # utilities16_100.append([0.205, 0.225, 0.155, 0.125, 0.12])
    # utilities32_100.append([0.115, 0.145, 0.095, 0.055, 0.08])
    # comps.append([1.3, 1.684, 1.455, 1.599, 1.635])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 10.0$')
    # utilities2_100.append([0.785, 0.83, 0.65, 0.725, 0.69])
    # utilities16_100.append([0.235, 0.29, 0.215, 0.22, 0.19])
    # utilities32_100.append([0.145, 0.165, 0.115, 0.11, 0.11])
    # comps.append([1.86, 1.962, 1.89, 1.983, 1.874])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 100$')
    # utilities2_100.append([0.69, 0.75, 0.72, 0.74, 0.71])
    # utilities16_100.append([0.135, 0.22, 0.16, 0.17, 0.23])
    # utilities32_100.append([0.085, 0.095, 0.085, 0.075, 0.13])
    # comps.append([2.448, 2.474, 2.426, 2.525, 2.605])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')

    ############################################################################
    #####           N = 1
    ############################################################################
    # n = 1
    # labels.append('$\lambda_I = 0.0; n = 1$')
    # utilities2_100.append([0.8, 0.485, 0.44, 0.49, 0.435])
    # utilities16_100.append([0.195, 0.07, 0.05, 0.07, 0.025])
    # utilities32_100.append([0.105, 0.045, 0.03, 0.03, 0.01])
    # comps.append([1.084, 0.0, -0.0, -0.0, 0.0])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 0.1; n = 1$')
    # utilities2_100.append([0.85, 0.84, 0.825, 0.825, 0.875])
    # utilities16_100.append([0.25, 0.23, 0.285, 0.195, 0.34])
    # utilities32_100.append([0.14, 0.12, 0.17, 0.09, 0.18])
    # comps.append([1.284, 1.141, 1.347, 0.995, 1.142])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 0.5; n = 1$')
    # utilities2_100.append([0.91, 0.855, 0.865, 0.85, 0.915])
    # utilities16_100.append([0.465, 0.245, 0.355, 0.285, 0.44])
    # utilities32_100.append([0.32, 0.13, 0.225, 0.16, 0.265])
    # comps.append([1.823, 1.586, 1.659, 1.662, 1.816])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 1.0; n = 1$')
    # utilities2_100.append([0.92, 0.85, 0.89, 0.835, 0.895])
    # utilities16_100.append([0.5, 0.265, 0.32, 0.255, 0.455])
    # utilities32_100.append([0.295, 0.155, 0.195, 0.145, 0.26])
    # comps.append([2.13, 1.899, 2.113, 1.94, 2.031])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 1.5; n = 1$')
    # utilities2_100.append([0.91, 0.87, 0.84, 0.84, 0.895])
    # utilities16_100.append([0.485, 0.27, 0.375, 0.265, 0.44])
    # utilities32_100.append([0.345, 0.165, 0.25, 0.16, 0.28])
    # comps.append([2.113, 2.218, 2.153, 2.234, 2.136])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 2.0; n = 1$')
    # utilities2_100.append([0.895, 0.85, 0.85, 0.835, 0.905])
    # utilities16_100.append([0.49, 0.29, 0.395, 0.245, 0.47])
    # utilities32_100.append([0.3, 0.195, 0.225, 0.14, 0.295])
    # comps.append([2.196, 2.206, 2.232, 2.203, 2.277])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 3.0; n = 1$')
    # utilities2_100.append([0.885, 0.865, 0.86, 0.865, 0.895])
    # utilities16_100.append([0.475, 0.28, 0.34, 0.285, 0.46])
    # utilities32_100.append([0.295, 0.18, 0.22, 0.165, 0.275])
    # comps.append([2.307, 2.257, 2.198, 2.247, 2.312])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 10.0; n = 1$')
    # utilities2_100.append([0.895, 0.855, 0.84, 0.85, 0.9])
    # utilities16_100.append([0.49, 0.31, 0.355, 0.32, 0.5])
    # utilities32_100.append([0.315, 0.19, 0.25, 0.185, 0.305])
    # comps.append([2.598, 2.487, 2.324, 2.482, 2.665])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')

    ############################################################################
    #####           N = 2
    ############################################################################
    # n = 2
    # labels.append('$\lambda_I = 0.0; n = 1$')
    # utilities2_100.append([0.455, 0.485, 0.45, 0.485, 0.44])
    # utilities16_100.append([0.07, 0.07, 0.055, 0.07, 0.025])
    # utilities32_100.append([0.04, 0.045, 0.02, 0.03, 0.01])
    # comps.append([ -0.0, -0.0, 0.0, 0.0, -0.0])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 0.1; n = 1$')
    # utilities2_100.append([0.87, 0.825, 0.85, 0.82, 0.885])
    # utilities16_100.append([0.355, 0.195, 0.27, 0.24, 0.295])
    # utilities32_100.append([0.2, 0.125, 0.155, 0.145, 0.175])
    # comps.append([1.681, 1.2, 1.549, 1.316, 1.493])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 0.5; n = 1$')
    # utilities2_100.append([0.9, 0.84, 0.86, 0.87, 0.905])
    # utilities16_100.append([0.5, 0.28, 0.335, 0.32, 0.44])
    # utilities32_100.append([0.335, 0.125, 0.225, 0.205, 0.26])
    # comps.append([2.043, 1.908, 2.073, 1.881, 1.993])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 1.0; n = 1$')
    # utilities2_100.append([0.92, 0.855, 0.86, 0.84, 0.9])
    # utilities16_100.append([0.525, 0.325, 0.345, 0.27, 0.465])
    # utilities32_100.append([0.345, 0.19, 0.22, 0.185, 0.28])
    # comps.append([2.403, 2.3, 2.391, 2.254, 2.298])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 1.5; n = 1$')
    # utilities2_100.append([0.895, 0.87, 0.855, 0.86, 0.9])
    # utilities16_100.append([0.495, 0.305, 0.37, 0.295, 0.515])
    # utilities32_100.append([0.33, 0.18, 0.235, 0.205, 0.305])
    # comps.append([2.425, 2.458, 2.722, 2.454, 2.48])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 2.0; n = 1$')
    # utilities2_100.append([0.91, 0.865, 0.865, 0.855, 0.9])
    # utilities16_100.append([0.5, 0.315, 0.355, 0.28, 0.49])
    # utilities32_100.append([0.325, 0.185, 0.23, 0.175, 0.285])
    # comps.append([2.587, 2.758, 2.467, 2.566, 2.66])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 3.0; n = 1$')
    # utilities2_100.append([0.915, 0.86, 0.865, 0.86, 0.915])
    # utilities16_100.append([0.5, 0.32, 0.365, 0.33, 0.5])
    # utilities32_100.append([0.31, 0.2, 0.255, 0.18, 0.3])
    # comps.append([2.867, 2.847, 2.612, 3.257, 3.258])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 10.0; n = 1$')
    # utilities2_100.append([0.905, 0.855, 0.875, 0.85, 0.91])
    # utilities16_100.append([0.525, 0.315, 0.385, 0.32, 0.49])
    # utilities32_100.append([0.33, 0.2, 0.245, 0.21, 0.315])
    # comps.append([3.491, 3.085, 3.435, 3.279, 3.665])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    ############################################################################
    #####           N = 4
    ############################################################################
    # n = 4
    # labels.append('$\lambda_I = 0.0; n = 4$')
    # utilities2_10.append([0.64, 0.52, 0.65, 0.475, 0.4])
    # utilities16_10.append([0.14, 0.055, 0.11, 0.08, 0.03])
    # utilities32_10.append([0.085, 0.03, 0.065, 0.04, 0.01])
    # utilities2_50.append([0.825, 0.475, 0.785, 0.49, 0.415])
    # utilities16_50.append([0.25, 0.065, 0.19, 0.07, 0.02])
    # utilities32_50.append([0.115, 0.04, 0.095, 0.04, 0.01])
    # utilities2_100.append([0.825, 0.525, 0.81, 0.485, 0.435])
    # utilities16_100.append([0.225, 0.06, 0.22, 0.065, 0.025])
    # utilities32_100.append([0.135, 0.035, 0.145, 0.04,  0.01])
    # utilities2_1000.append([0.835, 0.5, 0.835, 0.46, 0.395])
    # utilities16_1000.append([0.255, 0.08, 0.235, 0.065, 0.03])
    # utilities32_1000.append([0.16, 0.04, 0.15, 0.03, 0.01])
    # comps.append([1.507, -0.0, 1.517, -0.0, 0.0])
    # sizes.append(small_size)
    # colors.append(0)
    #
    # labels.append('$\lambda_I = 0.1; n = 4$')
    # utilities2_10.append([0.77, 0.81, 0.69, 0.655, 0.695])
    # utilities16_10.append([0.185, 0.195, 0.165, 0.145, 0.225])
    # utilities32_10.append([0.145, 0.105, 0.1, 0.065, 0.145])
    # utilities2_50.append([0.845, 0.81, 0.795, 0.815, 0.87])
    # utilities16_50.append([0.37, 0.225, 0.265, 0.18, 0.345])
    # utilities32_50.append([0.22, 0.125, 0.135, 0.085, 0.235])
    # utilities2_100.append([0.92, 0.83, 0.84, 0.82, 0.85])
    # utilities16_100.append([0.41, 0.27, 0.28, 0.19, 0.36])
    # utilities32_100.append([0.25, 0.11, 0.2,  0.12, 0.25])
    # utilities2_1000.append([0.905, 0.855, 0.895, 0.835, 0.885])
    # utilities16_1000.append([0.475, 0.24, 0.315, 0.205, 0.365])
    # utilities32_1000.append([0.305, 0.115, 0.17, 0.095, 0.24])
    # comps.append([1.748, 1.476, 1.751, 1.149, 1.65])
    # sizes.append(small_size)
    # colors.append(0.1)
    #
    # labels.append('$\lambda_I = 0.5; n = 4$')
    # utilities2_10.append([0.765, 0.8, 0.725, 0.665, 0.745])
    # utilities16_10.append([0.21, 0.235, 0.245, 0.165, 0.265])
    # utilities32_10.append([0.185, 0.11, 0.105, 0.1, 0.155])
    # utilities2_50.append([0.885, 0.835, 0.83, 0.835, 0.865])
    # utilities16_50.append([0.465, 0.255, 0.32, 0.275, 0.38])
    # utilities32_50.append([0.29, 0.155, 0.215, 0.16, 0.255])
    # utilities2_100.append([0.92, 0.865, 0.87, 0.84, 0.895])
    # utilities16_100.append([0.525, 0.295, 0.375, 0.275, 0.43])
    # utilities32_100.append([0.285, 0.15, 0.225, 0.165, 0.265])
    # utilities2_1000.append([0.94, 0.89, 0.905, 0.875, 0.915])
    # utilities16_1000.append([0.535, 0.28, 0.39, 0.355, 0.47])
    # utilities32_1000.append([0.335, 0.17, 0.295, 0.2, 0.285])
    # comps.append([2.382, 2.199, 2.216, 2.225, 2.125])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 1.0; n = 4$')
    # utilities2_10.append([0.725, 0.81, 0.715, 0.68, 0.75])
    # utilities16_10.append([0.24, 0.215, 0.16, 0.15, 0.305])
    # utilities32_10.append([0.2, 0.15, 0.085, 0.1, 0.15])
    # utilities2_50.append([0.9, 0.85, 0.845, 0.835, 0.87])
    # utilities16_50.append([0.5, 0.305, 0.355, 0.29, 0.42])
    # utilities32_50.append([0.3, 0.17, 0.225, 0.175, 0.285])
    # utilities2_100.append([0.93, 0.9,  0.87, 0.84, 0.89])
    # utilities16_100.append([0.51, 0.37, 0.41, 0.31, 0.44])
    # utilities32_100.append([0.27, 0.18, 0.24, 0.16, 0.29])
    # utilities2_1000.append([0.94, 0.89, 0.905, 0.885, 0.915])
    # utilities16_1000.append([0.55, 0.37, 0.4, 0.36, 0.48])
    # utilities32_1000.append([0.345, 0.22, 0.295, 0.21, 0.33])
    # comps.append([2.734, 2.652, 2.683, 2.763, 2.809])
    # sizes.append(small_size)
    # colors.append(1)
    #
    # labels.append('$\lambda_I = 1.5; n = 4$')
    # utilities2_10.append([0.715, 0.83, 0.68, 0.685, 0.77])
    # utilities16_10.append([0.255, 0.23, 0.16, 0.175, 0.29])
    # utilities32_10.append([0.19, 0.145, 0.095, 0.115, 0.15])
    # utilities2_50.append([0.89, 0.84, 0.84, 0.835, 0.87])
    # utilities16_50.append([0.485, 0.34, 0.365, 0.3, 0.425])
    # utilities32_50.append([0.33, 0.19, 0.26, 0.175, 0.255])
    # utilities2_100.append([0.925, 0.885, 0.88, 0.87, 0.905])
    # utilities16_100.append([0.545, 0.345, 0.4, 0.315, 0.455])
    # utilities32_100.append([0.33, 0.2, 0.26, 0.21, 0.29])
    # utilities2_1000.append([0.94, 0.895, 0.91, 0.885, 0.915])
    # utilities16_1000.append([0.56, 0.37, 0.455, 0.385, 0.465])
    # utilities32_1000.append([0.385, 0.235, 0.325, 0.23, 0.325])
    # comps.append([2.962, 2.748, 3.395, 2.98, 2.942])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 2.0; n = 4$')
    # utilities2_10.append([0.765, 0.825, 0.67, 0.68, 0.79])
    # utilities16_10.append([0.28, 0.25, 0.18, 0.195, 0.34])
    # utilities32_10.append([0.165, 0.16, 0.09, 0.105, 0.17])
    # utilities2_50.append([0.875, 0.86, 0.865, 0.82, 0.88])
    # utilities16_50.append([0.495, 0.31, 0.385, 0.27, 0.41])
    # utilities32_50.append([0.355, 0.175, 0.23, 0.175, 0.245])
    # utilities2_100.append([0.93, 0.88, 0.89, 0.855, 0.9])
    # utilities16_100.append([0.545, 0.325, 0.395, 0.3, 0.455])
    # utilities32_100.append([0.33, 0.205, 0.265, 0.16, 0.28])
    # utilities2_1000.append([0.95, 0.895, 0.915, 0.895, 0.915])
    # utilities16_1000.append([0.545, 0.36, 0.45, 0.385, 0.49])
    # utilities32_1000.append([0.385, 0.225, 0.305, 0.225, 0.34])
    # comps.append([3.596, 3.426, 3.096, 3.679, 2.883])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 3.0; n = 4$')
    # utilities2_10.append([0.785, 0.805, 0.76, 0.705, 0.755])
    # utilities16_10.append([0.27, 0.23, 0.16, 0.19, 0.29])
    # utilities32_10.append([0.17, 0.12, 0.095, 0.115, 0.155])
    # utilities2_50.append([0.905, 0.86, 0.84, 0.825, 0.845])
    # utilities16_50.append([0.515, 0.3, 0.37, 0.3, 0.46])
    # utilities32_50.append([0.35, 0.175, 0.24, 0.17, 0.27])
    # utilities2_100.append([0.94, 0.885, 0.89, 0.88, 0.915])
    # utilities16_100.append([0.575, 0.325, 0.415, 0.335, 0.475])
    # utilities32_100.append([0.36, 0.185, 0.27, 0.22, 0.295])
    # utilities2_1000.append([0.945, 0.895, 0.9, 0.895, 0.925])
    # utilities16_1000.append([0.565, 0.35, 0.455, 0.37, 0.48])
    # utilities32_1000.append([0.39, 0.225, 0.32, 0.245, 0.35])
    # comps.append([3.123, 3.455, 3.419, 3.86, 3.31])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 10; n = 4$')
    # utilities2_10.append([0.755, 0.815, 0.72, 0.66, 0.765])
    # utilities16_10.append([0.26, 0.255, 0.17, 0.17, 0.275])
    # utilities32_10.append([0.195, 0.17, 0.095, 0.1, 0.15])
    # utilities2_50.append([0.885, 0.855, 0.855, 0.84, 0.875])
    # utilities16_50.append([0.505, 0.325, 0.365, 0.33, 0.45])
    # utilities32_50.append([0.33, 0.2, 0.24, 0.2, 0.245])
    # utilities2_100.append([0.95, 0.88, 0.89, 0.87, 0.91])
    # utilities16_100.append([0.59, 0.37, 0.4,  0.36, 0.5])
    # utilities32_100.append([0.32, 0.22, 0.25, 0.24, 0.34])
    # utilities2_1000.append([0.95, 0.9, 0.92, 0.895, 0.93])
    # utilities16_1000.append([0.58, 0.41, 0.465, 0.415, 0.53])
    # utilities32_1000.append([0.41, 0.255, 0.325, 0.275, 0.37])
    # comps.append([4.634, 4.924, 4.915, 4.245, 3.565])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')

    ############################################################################
    #####           N = 8
    ############################################################################
    # n = 8
    # labels.append('$\lambda_I = 0.0; n = 8$')
    # utilities2_100.append([0.855, 0.44, 0.53, 0.495, 0.86])
    # utilities16_100.append([0.31, 0.055, 0.06, 0.055, 0.285])
    # utilities32_100.append([0.17, 0.02, 0.03, 0.02, 0.135])
    # comps.append([1.553, -0.0, -0.0, -0.0, 1.622])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 0.1; n = 8$')
    # utilities2_100.append([0.905, 0.825, 0.845, 0.855, 0.895])
    # utilities16_100.append([0.445, 0.24, 0.335, 0.26, 0.43])
    # utilities32_100.append([0.285, 0.135, 0.185, 0.155, 0.21])
    # comps.append([1.813, 1.53, 1.728, 1.776, 1.589])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 0.5; n = 8$')
    # utilities2_100.append([0.9, 0.855, 0.85, 0.825, 0.92])
    # utilities16_100.append([0.49, 0.31, 0.35, 0.26, 0.51])
    # utilities32_100.append([0.32, 0.19, 0.22, 0.175, 0.31])
    # comps.append([ 2.126, 2.28, 2.201, 2.502, 2.827])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 1.0; n = 8$')
    # utilities2_100.append([0.915, 0.865, 0.865, 0.845, 0.91])
    # utilities16_100.append([0.515, 0.33, 0.385, 0.315, 0.51])
    # utilities32_100.append([0.315, 0.2, 0.235, 0.155, 0.31])
    # comps.append([ 2.884, 3.16, 3.221, 3.308, 3.062])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 1.5; n = 8$')
    # utilities2_100.append([0.92, 0.865, 0.875, 0.87, 0.92])
    # utilities16_100.append([0.535, 0.325, 0.38, 0.31, 0.49])
    # utilities32_100.append([0.385, 0.205, 0.22, 0.195, 0.305])
    # comps.append([3.124, 3.448, 4.164, 3.883, 3.793])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 2.0; n = 8$')
    # utilities2_100.append([0.905, 0.87, 0.88, 0.86, 0.92])
    # utilities16_100.append([0.515, 0.32, 0.355, 0.355, 0.525])
    # utilities32_100.append([0.335, 0.215, 0.22, 0.21, 0.325])
    # comps.append([ 3.612, 3.618, 3.957, 4.084, 3.863])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 3.0; n = 8$')
    # utilities2_100.append([0.925, 0.87, 0.87, 0.855, 0.925])
    # utilities16_100.append([0.515, 0.32, 0.355, 0.355, 0.525])
    # utilities32_100.append([0.35, 0.235, 0.245, 0.2, 0.325])
    # comps.append([ 4.456, 3.311, 4.16, 4.543, 4.493])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 10.0; n = 8$')
    # utilities2_100.append([0.915, 0.87, 0.885, 0.85, 0.92])
    # utilities16_100.append([0.54, 0.335, 0.37, 0.35, 0.52])
    # utilities32_100.append([0.37, 0.21, 0.235, 0.22, 0.31])
    # comps.append([4.703, 4.502, 5.082, 5.502, 4.655])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')

    ############################################################################
    #####           N = 16
    ############################################################################
    n = 16
    labels.append('$\lambda_I = 0.0; n = 16$')
    utilities2_100.append([0.85, 0.83, 0.53, 0.5, 0.905])
    utilities16_100.append([0.33, 0.245, 0.06, 0.05, 0.4])
    utilities32_100.append([0.19, 0.145, 0.03, 0.02, 0.22])
    comps.append([1.928, 1.227, 0.0, -0.0, 1.932])
    sizes.append(small_size)
    colors.append('xkcd:blue')

    labels.append('$\lambda_I = 0.1; n = 16$')
    utilities2_100.append([0.91, 0.84, 0.87, 0.5, 0.9])
    utilities16_100.append([0.43, 0.27, 0.365, 0.05, 0.44])
    utilities32_100.append([0.275, 0.15, 0.22, 0.02, 0.235])
    comps.append([1.862, 1.579, 1.955, -0.0, 1.679])
    sizes.append(small_size)
    colors.append('xkcd:blue')

    labels.append('$\lambda_I = 0.5; n = 16$')
    utilities2_100.append([0.91, 0.87, 0.87, 0.85, 0.91])
    utilities16_100.append([0.495, 0.305, 0.365, 0.295, 0.48])
    utilities32_100.append([0.285, 0.21, 0.23, 0.18, 0.3])
    comps.append([2.394, 2.143, 2.707, 2.454, 2.685])
    sizes.append(small_size)
    colors.append('xkcd:blue')

    labels.append('$\lambda_I = 1.0; n = 16$')
    utilities2_100.append([0.895, 0.855, 0.87, 0.855, 0.915])
    utilities16_100.append([0.51, 0.305, 0.335, 0.325, 0.5])
    utilities32_100.append([0.355, 0.17, 0.22, 0.18, 0.3])
    comps.append([2.915, 2.485, 3.422, 2.821, 3.037])
    sizes.append(small_size)
    colors.append('xkcd:blue')

    labels.append('$\lambda_I = 1.5; n = 16$')
    utilities2_100.append([0.91, 0.86, 0.88, 0.845, 0.91])
    utilities16_100.append([0.525, 0.325, 0.38, 0.34, 0.51])
    utilities32_100.append([0.355, 0.19, 0.245, 0.215, 0.305])
    comps.append([3.31, 3.021, 4.041, 3.5, 3.319])
    sizes.append(small_size)
    colors.append('xkcd:blue')

    labels.append('$\lambda_I = 2.0; n = 8$')
    utilities2_100.append([0.91, 0.86, 0.87, 0.84, 0.91])
    utilities16_100.append([0.545, 0.31, 0.4, 0.29, 0.51])
    utilities32_100.append([0.355, 0.195, 0.26, 0.185, 0.33])
    comps.append([ 3.847, 3.184, 4.175, 3.823, 3.878])
    sizes.append(small_size)
    colors.append('xkcd:blue')

    labels.append('$\lambda_I = 3.0; n = 16$')
    utilities2_100.append([0.925, 0.87, 0.87, 0.85, 0.915])
    utilities16_100.append([0.535, 0.34, 0.365, 0.32, 0.51])
    utilities32_100.append([0.365, 0.22, 0.24, 0.215, 0.325])
    comps.append([4.35, 4.017, 4.986, 3.709, 2.619])
    sizes.append(small_size)
    colors.append('xkcd:blue')

    labels.append('$\lambda_I = 10.0; n = 16$')
    utilities2_100.append([0.915, 0.88, 0.875, 0.85, 0.915])
    utilities16_100.append([0.54, 0.31, 0.41, 0.35, 0.51])
    utilities32_100.append([0.365, 0.22, 0.255, 0.24, 0.33])
    comps.append([4.31, 4.316, 4.773, 3.781, 4.276])
    sizes.append(small_size)
    colors.append('xkcd:blue')

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
    plot_multi_trials([comps + comps + comps, utilities2 + utilities16 + utilities32], labels, sizes, ylabel='Translation Utility', colors=None, filename='xlation_comp_all_n' + str(n) + '_numalign' + str(num_align) + '.png')


if __name__ == '__main__':
    run()
