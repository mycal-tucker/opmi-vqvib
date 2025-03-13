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
    n = 'onehot'
    labels.append('$\lambda_I = 0.1$')
    utilities2_10.append([0.5, 0.5, 0.41, 0.47, 0.43])
    utilities16_10.append([0.04, 0.06, 0.05, 0.05, 0.04])
    utilities32_10.append([0.01, 0.02, 0.02, 0.03, 0.02])
    utilities2_50.append([0.5, 0.63, 0.46, 0.46, 0.43])
    utilities16_50.append([0.04, 0.16, 0.04, 0.04, 0.04])
    utilities32_50.append([0.01, 0.06, 0.03, 0.03, 0.02])
    utilities2_1000.append([0.5, 0.68, 0.42, 0.47, 0.4])
    utilities16_1000.append([0.04, 0.15, 0.03, 0.04, 0.04])
    utilities32_1000.append([0.01, 0.05, 0.04, 0.03, 0.02])
    utilities2_100.append([0.5, 0.65, 0.42, 0.46, 0.43])
    utilities16_100.append([0.04, 0.11, 0.04, 0.04, 0.04])
    utilities32_100.append([0.01, 0.08, 0.03, 0.03, 0.02])
    comps.append([-0.317, -0.286, -0.266, -0.347, -0.311])
    sizes.append(small_size)
    colors.append('xkcd:blue')

    labels.append('$\lambda_I = 0.5$')
    utilities2_10.append([0.62, 0.67, 0.55, 0.49, 0.54])
    utilities16_10.append([0.13, 0.15, 0.11, 0.07, 0.1])
    utilities32_10.append([0.06, 0.09, 0.04, 0.03, 0.06])
    utilities2_50.append([0.71, 0.69, 0.65, 0.65, 0.62])
    utilities16_50.append([0.15, 0.17, 0.14, 0.12, 0.15])
    utilities32_50.append([0.11, 0.09, 0.08, 0.06, 0.07])
    utilities2_1000.append([0.78, 0.68, 0.71, 0.68, 0.64])
    utilities16_1000.append([0.22, 0.2, 0.13, 0.12, 0.15])
    utilities32_1000.append([0.14, 0.11, 0.09, 0.08, 0.08])
    utilities2_100.append([0.78, 0.7, 0.66, 0.7, 0.6])
    utilities16_100.append([0.2, 0.18, 0.15, 0.12, 0.13])
    utilities32_100.append([0.14, 0.1, 0.08, 0.06, 0.05])
    comps.append([-0.214, -0.245, -0.204, -0.266, -0.222])
    sizes.append(small_size)
    colors.append('xkcd:blue')

    labels.append('$\lambda_I = 1.0$')
    utilities2_10.append([0.64, 0.59, 0.45, 0.51, 0.54])
    utilities16_10.append([0.12, 0.07, 0.09, 0.05, 0.04])
    utilities32_10.append([0.06, 0.05, 0.01, 0.03, 0.03])
    utilities2_50.append([0.71, 0.56, 0.54, 0.58, 0.64])
    utilities16_50.append([0.16, 0.08, 0.1, 0.1, 0.05])
    utilities32_50.append([0.09, 0.04, 0.06, 0.08, 0.03])
    utilities2_1000.append([0.72, 0.56, 0.54, 0.6, 0.63])
    utilities16_1000.append([0.21, 0.09, 0.09, 0.12, 0.08])
    utilities32_1000.append([0.1, 0.06, 0.06, 0.08, 0.05])
    utilities2_100.append([0.72, 0.54, 0.5, 0.63, 0.64])
    utilities16_100.append([0.2, 0.09, 0.09, 0.11, 0.07])
    utilities32_100.append([0.12, 0.04, 0.04, 0.07, 0.03])
    comps.append([-0.194, -0.274, -0.242, -0.229, -0.254])
    sizes.append(small_size)
    colors.append('xkcd:blue')

    labels.append('$\lambda_I = 1.5$')
    utilities2_10.append([0.65, 0.62, 0.5, 0.47, 0.6])
    utilities16_10.append([0.11, 0.07, 0.12, 0.05, 0.09])
    utilities32_10.append([0.07, 0.04, 0.06, 0.03, 0.04])
    utilities2_50.append([0.64, 0.72, 0.62, 0.67, 0.68])
    utilities16_50.append([0.14, 0.14, 0.13, 0.16, 0.13])
    utilities32_50.append([0.07, 0.05, 0.08, 0.07, 0.07])
    utilities2_1000.append([0.72, 0.76, 0.71, 0.73, 0.67])
    utilities16_1000.append([0.18, 0.15, 0.19, 0.23, 0.13])
    utilities32_1000.append([0.1, 0.07, 0.11, 0.14, 0.05])
    utilities2_100.append([0.7, 0.75, 0.67, 0.73, 0.69])
    utilities16_100.append([0.14, 0.13, 0.14, 0.16, 0.12])
    utilities32_100.append([0.1, 0.07, 0.06, 0.08, 0.06])
    comps.append([-0.203, -0.248, -0.212, -0.221, -0.262])
    sizes.append(small_size)
    colors.append('xkcd:blue')

    labels.append('$\lambda_I = 2.0$')
    utilities2_10.append([0.59, 0.68, 0.53, 0.51, 0.51])
    utilities16_10.append([0.09, 0.11, 0.05, 0.04, 0.07])
    utilities32_10.append([0.07, 0.06, 0.03, 0.02, 0.03])
    utilities2_50.append([0.7, 0.73, 0.55, 0.71, 0.52])
    utilities16_50.append([0.15, 0.18, 0.07, 0.15, 0.1])
    utilities32_50.append([0.11, 0.08, 0.03, 0.07, 0.07])
    utilities2_1000.append([0.77, 0.75, 0.57, 0.68, 0.64])
    utilities16_1000.append([0.26, 0.16, 0.07, 0.14, 0.12])
    utilities32_1000.append([0.15, 0.05, 0.04, 0.1, 0.08])
    utilities2_100.append([0.66, 0.75, 0.52, 0.73, 0.53])
    utilities16_100.append([0.18, 0.15, 0.06, 0.15, 0.11])
    utilities32_100.append([0.09, 0.06, 0.02, 0.07, 0.04])
    comps.append([-0.193, -0.237, -0.219, -0.234, -0.21])
    sizes.append(small_size)
    colors.append('xkcd:blue')

    labels.append('$\lambda_I = 3.0$')
    utilities2_10.append([0.69, 0.52, 0.49, 0.54, 0.58])
    utilities16_10.append([0.16, 0.08, 0.12, 0.07, 0.07])
    utilities32_10.append([0.06, 0.01, 0.04, 0.04, 0.03])
    utilities2_50.append([0.74, 0.65, 0.7, 0.71, 0.6])
    utilities16_50.append([0.16, 0.16, 0.18, 0.17, 0.11])
    utilities32_50.append([0.12, 0.09, 0.11, 0.08, 0.07])
    utilities2_1000.append([0.73, 0.68, 0.74, 0.73, 0.66])
    utilities16_1000.append([0.19, 0.19, 0.2, 0.15, 0.11])
    utilities32_1000.append([0.1, 0.11, 0.13, 0.07, 0.05])
    utilities2_100.append([0.7, 0.65, 0.69, 0.7, 0.65])
    utilities16_100.append([0.17, 0.18, 0.17, 0.16, 0.1])
    utilities32_100.append([0.1, 0.08, 0.11, 0.08, 0.05])
    comps.append([-0.236, -0.243, -0.191, -0.2, -0.197])
    sizes.append(small_size)
    colors.append('xkcd:blue')

    labels.append('$\lambda_I = 10.0$')
    utilities2_10.append([0.62, 0.69, 0.47, 0.53, 0.61])
    utilities16_10.append([0.13, 0.14, 0.07, 0.06, 0.1])
    utilities32_10.append([0.06, 0.08, 0.03, 0.04, 0.08])
    utilities2_50.append([0.68, 0.71, 0.57, 0.69, 0.59])
    utilities16_50.append([0.17, 0.16, 0.1, 0.14, 0.14])
    utilities32_50.append([0.16, 0.08, 0.08, 0.1, 0.07])
    utilities2_1000.append([0.75, 0.73, 0.7, 0.7, 0.71])
    utilities16_1000.append([0.23, 0.19, 0.14, 0.18, 0.2])
    utilities32_1000.append([0.12, 0.12, 0.06, 0.11, 0.1])
    utilities2_100.append([0.71, 0.74, 0.57, 0.73, 0.66])
    utilities16_100.append([0.21, 0.17, 0.12, 0.18, 0.14])
    utilities32_100.append([0.12, 0.09, 0.06, 0.08, 0.07])
    comps.append([-0.177, -0.215, -0.191, -0.199, -0.216])
    sizes.append(small_size)
    colors.append('xkcd:blue')

    labels.append('$\lambda_I = 100$')
    utilities2_10.append([0.7, 0.67, 0.55, 0.48, 0.57])
    utilities16_10.append([0.13, 0.09, 0.07, 0.07, 0.07])
    utilities32_10.append([0.06, 0.06, 0.05, 0.05, 0.03])
    utilities2_50.append([0.69, 0.72, 0.58, 0.74, 0.6])
    utilities16_50.append([0.13, 0.23, 0.1, 0.15, 0.1])
    utilities32_50.append([0.06, 0.13, 0.07, 0.08, 0.04])
    utilities2_1000.append([0.7, 0.77, 0.65, 0.76, 0.67])
    utilities16_1000.append([0.15, 0.23, 0.17, 0.17, 0.17])
    utilities32_1000.append([0.07, 0.09, 0.1, 0.11, 0.1])
    utilities2_100.append([0.7, 0.68, 0.58, 0.75, 0.66])
    utilities16_100.append([0.13, 0.27, 0.11, 0.19, 0.1])
    utilities32_100.append([0.06, 0.14, 0.08, 0.1, 0.09])
    comps.append([-0.208, -0.197, -0.205, -0.21, -0.211])
    sizes.append(small_size)
    colors.append('xkcd:blue')
    onehot_util2 = utilities2_100
    onehot_util16 = utilities16_100
    onehot_util32 = utilities32_100
    onehot_comps = comps
    utilities2_100 = []
    utilities16_100 = []
    utilities32_100 = []
    comps = []
    ############################################################################
    #####           Proto
    ############################################################################
    n = 'proto'
    labels.append('$\lambda_I = 0.1$')
    utilities2_1000.append([0.455, 0.41, 0.408, 0.49, 0.54])
    utilities16_10.append([])
    utilities32_10.append([])
    utilities2_50.append([0.455, 0.41, 0.408, 0.49, 0.54])
    utilities16_50.append([])
    utilities32_50.append([])
    utilities2_10.append([0.455, 0.41, 0.408, 0.49, 0.54])
    utilities16_1000.append([])
    utilities32_1000.append([])
    utilities2_100.append([0.455, 0.41, 0.408, 0.49, 0.54])
    utilities16_100.append([])
    utilities32_100.append([])
    comps.append([-0.322, -0.373, -0.347, -0.32, -0.323])
    sizes.append(small_size)
    colors.append('xkcd:blue')

    labels.append('$\lambda_I = 0.5$')
    utilities2_1000.append([0.444, 0.41, 0.418, 0.49, 0.55])
    utilities16_10.append([])
    utilities32_10.append([])
    utilities2_50.append([0.444, 0.41, 0.418, 0.49, 0.55])
    utilities16_50.append([])
    utilities32_50.append([])
    utilities2_10.append([0.444, 0.41, 0.418, 0.49, 0.55])
    utilities16_1000.append([])
    utilities32_1000.append([])
    utilities2_100.append([0.444, 0.41, 0.418, 0.49, 0.55])
    utilities16_100.append([])
    utilities32_100.append([])
    comps.append([-0.322, -0.373, -0.347, -0.32, -0.323])
    sizes.append(small_size)
    colors.append('xkcd:blue')

    labels.append('$\lambda_I = 1.0$')
    utilities2_1000.append([0.556, 0.45, 0.571, 0.49, 0.66])
    utilities16_10.append([])
    utilities32_10.append([])
    utilities2_50.append([0.485, 0.58, 0.52, 0.49, 0.57])
    utilities16_50.append([])
    utilities32_50.append([])
    utilities2_10.append([0.495, 0.6, 0.561, 0.49, 0.6])
    utilities16_1000.append([])
    utilities32_1000.append([])
    utilities2_100.append([0.495, 0.6, 0.541, 0.49, 0.56])
    utilities16_100.append([])
    utilities32_100.append([])
    comps.append([-0.294, -0.333, -0.294, -0.317, -0.271])
    sizes.append(small_size)
    colors.append('xkcd:blue')

    labels.append('$\lambda_I = 1.5$')
    utilities2_1000.append([0.586, 0.47, 0.51, 0.59, 0.63])
    utilities16_10.append([])
    utilities32_10.append([])
    utilities2_50.append([0.495, 0.56, 0.439, 0.55, 0.61])
    utilities16_50.append([])
    utilities32_50.append([])
    utilities2_10.append([0.495, 0.59, 0.429, 0.54, 0.58])
    utilities16_1000.append([])
    utilities32_1000.append([])
    utilities2_100.append([0.505, 0.57, 0.418, 0.56, 0.54])
    utilities16_100.append([])
    utilities32_100.append([])
    comps.append([ -0.281, -0.331, -0.256, -0.253, -0.265])
    sizes.append(small_size)
    colors.append('xkcd:blue')

    labels.append('$\lambda_I = 2.0$')
    utilities2_1000.append([0.636, 0.49, 0.52, 0.58, 0.63])
    utilities16_10.append([])
    utilities32_10.append([])
    utilities2_50.append([0.525, 0.59, 0.459, 0.58, 0.56])
    utilities16_50.append([])
    utilities32_50.append([])
    utilities2_10.append([0.525, 0.58, 0.459, 0.51, 0.56])
    utilities16_1000.append([])
    utilities32_1000.append([])
    utilities2_100.append([0.515, 0.58, 0.449, 0.54, 0.55])
    utilities16_100.append([])
    utilities32_100.append([])
    comps.append([-0.253, -0.326, -0.249, -0.248, -0.246])
    sizes.append(small_size)
    colors.append('xkcd:blue')

    labels.append('$\lambda_I = 3.0$')
    utilities2_1000.append([0.616, 0.58, 0.49, 0.56, 0.63])
    utilities16_10.append([])
    utilities32_10.append([])
    utilities2_50.append([0.556, 0.52, 0.459, 0.5, 0.55])
    utilities16_50.append([])
    utilities32_50.append([])
    utilities2_10.append([0.545, 0.56, 0.459, 0.45, 0.53])
    utilities16_1000.append([])
    utilities32_1000.append([])
    utilities2_100.append([0.545, 0.57, 0.569, 0.46, 0.53])
    utilities16_100.append([])
    utilities32_100.append([])
    comps.append([-0.252, -0.296, -0.244, -0.233, -0.24])
    sizes.append(small_size)
    colors.append('xkcd:blue')

    labels.append('$\lambda_I = 10.0$')
    utilities2_1000.append([0.657, 0.62, 0.551, 0.6, 0.67])
    utilities16_10.append([])
    utilities32_10.append([])
    utilities2_50.append([0.465, 0.55, 0.49, 0.47, 0.55])
    utilities16_50.append([])
    utilities32_50.append([])
    utilities2_10.append([0.495, 0.49, 0.418, 0.42, 0.48])
    utilities16_1000.append([])
    utilities32_1000.append([])
    utilities2_100.append([0.495, 0.5, 0.469, 0.66, 0.61])
    utilities16_100.append([])
    utilities32_100.append([])
    comps.append([-0.225, -0.253, -0.226, -0.208, -0.218])
    sizes.append(small_size)
    colors.append('xkcd:blue')

    labels.append('$\lambda_I = 100$')
    utilities2_1000.append([0.626, 0.55, 0.591, 0.56, 0.69])
    utilities16_10.append([])
    utilities32_10.append([])
    utilities2_50.append([0.596, 0.62, 0.52, 0.51, 0.59])
    utilities16_50.append([])
    utilities32_50.append([])
    utilities2_10.append([0.495, 0.52, 0.48, 0.46, 0.44])
    utilities16_1000.append([])
    utilities32_1000.append([])
    utilities2_100.append([0.525, 0.54, 0.51, 0.65, 0.68])
    utilities16_100.append([])
    utilities32_100.append([])
    comps.append([-0.212, -0.215, -0.197, -0.188, -0.2])
    sizes.append(small_size)
    colors.append('xkcd:blue')

    proto_util2 = utilities2_100
    proto_util16 = utilities16_100
    proto_util32 = utilities32_100
    proto_comps = comps
    utilities2_100 = []
    utilities16_100 = []
    utilities32_100 = []
    comps = []
    ############################################################################
    #####           VQ-VIB N = 1
    ############################################################################
    n = 1
    labels.append('$\lambda_I = 0.0; n = 1$')
    utilities2_10.append([0.67, 0.54, 0.44, 0.56, 0.42])
    utilities16_10.append([0.09, 0.09, 0.06, 0.08, 0.04])
    utilities32_10.append([])
    utilities2_50.append([0.75, 0.51, 0.44, 0.56, 0.46])
    utilities16_50.append([0.18, 0.08, 0.06, 0.07, 0.04])
    utilities32_50.append([])
    utilities2_1000.append([0.81, 0.5, 0.49, 0.49, 0.41])
    utilities16_1000.append([])
    utilities32_1000.append([])
    utilities2_100.append([0.77, 0.49, 0.45, 0.51, 0.39])
    utilities16_100.append([0.18, 0.08, 0.05, 0.07, 0.02])
    utilities32_100.append([0.11, 0.03, 0.03, 0.03, 0.02])
    comps.append([-0.397, -0.354, -0.358, -0.361, -0.358])
    sizes.append(small_size)
    colors.append('xkcd:blue')

    labels.append('$\lambda_I = 0.1; n = 1$')
    utilities2_10.append([0.74, 0.71, 0.7, 0.62, 0.7])
    utilities16_10.append([0.22, 0.17, 0.13, 0.07, 0.21])
    utilities32_10.append([])
    utilities2_50.append([0.78, 0.82, 0.78, 0.73, 0.79])
    utilities16_50.append([0.22, 0.27, 0.2, 0.18, 0.28])
    utilities32_50.append([])
    utilities2_1000.append([0.84, 0.83, 0.78, 0.78, 0.79])
    utilities16_1000.append([])
    utilities32_1000.append([])
    utilities2_100.append([0.83, 0.83, 0.75, 0.75, 0.8])
    utilities16_100.append([0.26, 0.22, 0.22, 0.18, 0.3])
    utilities32_100.append([0.14, 0.12, 0.11, 0.07, 0.16])
    comps.append([ -0.251, -0.247, -0.234, -0.274, -0.239])
    sizes.append(small_size)
    colors.append('xkcd:blue')

    labels.append('$\lambda_I = 0.5; n = 1$')
    utilities2_10.append([0.77, 0.75, 0.7, 0.56, 0.62])
    utilities16_10.append([0.25, 0.19, 0.11, 0.08, 0.2])
    utilities32_10.append([])
    utilities2_50.append([0.77, 0.84, 0.75, 0.77, 0.79])
    utilities16_50.append([0.39, 0.27, 0.24, 0.26, 0.32])
    utilities32_50.append([])
    utilities2_1000.append([0.89, 0.81, 0.78, 0.83, 0.82])
    utilities16_1000.append([])
    utilities32_1000.append([])
    utilities2_100.append([0.87, 0.81, 0.79, 0.8, 0.77])
    utilities16_100.append([0.38, 0.25, 0.28, 0.27, 0.39])
    utilities32_100.append([0.29, 0.16, 0.17, 0.12, 0.24])
    comps.append([-0.189, -0.202, -0.197, -0.199, -0.182])
    sizes.append(small_size)
    colors.append('xkcd:blue')

    labels.append('$\lambda_I = 1.0; n = 1$')
    utilities2_10.append([0.7, 0.66, 0.6, 0.62, 0.62])
    utilities16_10.append([0.2, 0.14, 0.12, 0.08, 0.18])
    utilities32_10.append([])
    utilities2_50.append([0.8, 0.77, 0.77, 0.8, 0.82])
    utilities16_50.append([0.43, 0.23, 0.22, 0.29, 0.33])
    utilities32_50.append([])
    utilities2_1000.append([0.89, 0.8, 0.78, 0.84, 0.84])
    utilities16_1000.append([])
    utilities32_1000.append([])
    utilities2_100.append([0.88, 0.82, 0.78, 0.82, 0.83])
    utilities16_100.append([0.44, 0.24, 0.27, 0.29, 0.36])
    utilities32_100.append([0.32, 0.14, 0.17, 0.14, 0.26])
    comps.append([ -0.168, -0.164, -0.167, -0.174, -0.167])
    sizes.append(small_size)
    colors.append('xkcd:blue')

    labels.append('$\lambda_I = 1.5; n = 1$')
    utilities2_10.append([0.76, 0.7, 0.62, 0.6, 0.64])
    utilities16_10.append([0.25, 0.21, 0.13, 0.12, 0.22])
    utilities32_10.append([])
    utilities2_50.append([0.86, 0.78, 0.75, 0.83, 0.8])
    utilities16_50.append([0.4, 0.23, 0.25, 0.31, 0.3])
    utilities32_50.append([])
    utilities2_1000.append([0.91, 0.79, 0.74, 0.84, 0.86])
    utilities16_1000.append([])
    utilities32_1000.append([])
    utilities2_100.append([0.88, 0.81, 0.79, 0.79, 0.83])
    utilities16_100.append([0.44, 0.29, 0.34, 0.3, 0.36])
    utilities32_100.append([0.28, 0.13, 0.19, 0.16, 0.26])
    comps.append([-0.158, -0.149, -0.155, -0.161, -0.156 ])
    sizes.append(small_size)
    colors.append('xkcd:blue')

    labels.append('$\lambda_I = 2.0; n = 1$')
    utilities2_10.append([0.76, 0.75, 0.63, 0.59, 0.61])
    utilities16_10.append([0.26, 0.18, 0.14, 0.16, 0.25])
    utilities32_10.append([])
    utilities2_50.append([0.85, 0.77, 0.78, 0.78, 0.78])
    utilities16_50.append([0.42, 0.26, 0.26, 0.25, 0.34])
    utilities32_50.append([])
    utilities2_1000.append([0.88, 0.79, 0.77, 0.83, 0.82])
    utilities16_1000.append([])
    utilities32_1000.append([])
    utilities2_100.append([0.91, 0.8, 0.79, 0.78, 0.82])
    utilities16_100.append([0.46, 0.3, 0.32, 0.27, 0.36])
    utilities32_100.append([0.25, 0.15, 0.19, 0.12, 0.25])
    comps.append([-0.158, -0.15, -0.149, -0.149, -0.151])
    sizes.append(small_size)
    colors.append('xkcd:blue')

    labels.append('$\lambda_I = 3.0; n = 1$')
    utilities2_10.append([0.74, 0.74, 0.68, 0.61, 0.67])
    utilities16_10.append([0.25, 0.21, 0.13, 0.09, 0.24])
    utilities32_10.append([])
    utilities2_50.append([0.81, 0.76, 0.71, 0.82, 0.81])
    utilities16_50.append([0.37, 0.23, 0.25, 0.27, 0.34])
    utilities32_50.append([])
    utilities2_1000.append([0.86, 0.83, 0.81, 0.84, 0.88])
    utilities16_1000.append([])
    utilities32_1000.append([])
    utilities2_100.append([0.88, 0.81, 0.76, 0.8, 0.83])
    utilities16_100.append([0.41, 0.28, 0.31, 0.31, 0.39])
    utilities32_100.append([0.29, 0.15, 0.15, 0.14, 0.26])
    comps.append([ -0.149, -0.141, -0.147, -0.142, -0.141])
    sizes.append(small_size)
    colors.append('xkcd:blue')

    labels.append('$\lambda_I = 10.0; n = 1$')
    utilities2_10.append([0.76, 0.68, 0.65, 0.53, 0.65])
    utilities16_10.append([0.29, 0.2, 0.13, 0.09, 0.26])
    utilities32_10.append([])
    utilities2_50.append([0.85, 0.79, 0.72, 0.76, 0.81])
    utilities16_50.append([0.43, 0.24, 0.26, 0.28, 0.36])
    utilities32_50.append([])
    utilities2_1000.append([0.83, 0.83, 0.74, 0.83, 0.84])
    utilities16_1000.append([])
    utilities32_1000.append([])
    utilities2_100.append([0.91, 0.81, 0.72, 0.8, 0.87])
    utilities16_100.append([0.41, 0.3, 0.35, 0.31, 0.41])
    utilities32_100.append([0.29, 0.19, 0.17, 0.15, 0.3])
    comps.append([-0.128, -0.125, -0.127, -0.127, -0.124])
    sizes.append(small_size)
    colors.append('xkcd:blue')
    vqvib_util2 = utilities2_100
    vqvib_util16 = utilities16_100
    vqvib_util32 = utilities32_100
    vqvib_comps = comps
    utilities2_100 = []
    utilities16_100 = []
    utilities32_100 = []
    comps = []
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
    # n = 4
    # labels.append('$\lambda_I = 0.0; n = 4$')
    # utilities2_10.append([0.63, 0.46, 0.67, 0.51, 0.39])
    # utilities16_10.append([0.16, 0.06, 0.13, 0.08, 0.03])
    # utilities32_10.append([0.06, 0.04, 0.05, 0.02, 0.01])
    # utilities2_50.append([0.82, 0.46, 0.71, 0.52, 0.4])
    # utilities16_50.append([0.24, 0.08, 0.13, 0.07, 0.03])
    # utilities32_50.append([0.12, 0.04, 0.07, 0.03, 0.01])
    # utilities2_100.append([0.78, 0.5, 0.76, 0.51, 0.4])
    # utilities16_100.append([0.23, 0.08, 0.17, 0.07, 0.02])
    # utilities32_100.append([0.12, 0.04, 0.11, 0.03, 0.02])
    # utilities2_1000.append([0.82, 0.47, 0.74, 0.5, 0.39])
    # utilities16_1000.append([0.27, 0.06, 0.19, 0.07, 0.02])
    # utilities32_1000.append([0.13, 0.04, 0.12, 0.02, 0.02])
    # comps.append([-0.333, -0.356, -0.317, -0.358, -0.349])
    # sizes.append(small_size)
    # colors.append(0)
    #
    # labels.append('$\lambda_I = 0.1; n = 4$')
    # utilities2_10.append([0.74, 0.77, 0.58, 0.54, 0.66])
    # utilities16_10.append([0.2, 0.18, 0.12, 0.11, 0.24])
    # utilities32_10.append([0.11, 0.06, 0.05, 0.04, 0.14])
    # utilities2_50.append([0.83, 0.8, 0.77, 0.75, 0.8])
    # utilities16_50.append([0.36, 0.21, 0.24, 0.18, 0.31])
    # utilities32_50.append([0.24, 0.11, 0.1, 0.12, 0.18])
    # utilities2_100.append([0.87, 0.81, 0.76, 0.77, 0.8])
    # utilities16_100.append([0.42, 0.22, 0.23, 0.16, 0.34])
    # utilities32_100.append([0.24, 0.1, 0.12, 0.09, 0.2])
    # utilities2_1000.append([0.88, 0.8, 0.74, 0.78, 0.85])
    # utilities16_1000.append([0.4, 0.19, 0.25, 0.22, 0.31])
    # utilities32_1000.append([0.3, 0.12, 0.17, 0.1, 0.21])
    # comps.append([ -0.196, -0.212, -0.202, -0.248, -0.199])
    # sizes.append(small_size)
    # colors.append(0.1)
    #
    # labels.append('$\lambda_I = 0.5; n = 4$')
    # utilities2_10.append([0.75, 0.71, 0.78, 0.57, 0.72])
    # utilities16_10.append([0.25, 0.21, 0.15, 0.12, 0.26])
    # utilities32_10.append([0.16, 0.05, 0.08, 0.08, 0.09])
    # utilities2_50.append([0.88, 0.75, 0.77, 0.79, 0.79])
    # utilities16_50.append([0.41, 0.29, 0.26, 0.28, 0.3])
    # utilities32_50.append([0.27, 0.15, 0.19, 0.15, 0.2])
    # utilities2_100.append([0.9, 0.85, 0.81, 0.82, 0.79])
    # utilities16_100.append([0.45, 0.27, 0.36, 0.3, 0.36])
    # utilities32_100.append([0.32, 0.15, 0.21, 0.17, 0.2])
    # utilities2_1000.append([0.9, 0.78, 0.81, 0.82, 0.84])
    # utilities16_1000.append([0.47, 0.23, 0.3, 0.32, 0.41])
    # utilities32_1000.append([0.35, 0.14, 0.2, 0.15, 0.28])
    # comps.append([ -0.137, -0.144, -0.138, -0.143, -0.141])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 1.0; n = 4$')
    # utilities2_10.append([0.71, 0.72, 0.69, 0.62, 0.66])
    # utilities16_10.append([0.25, 0.22, 0.18, 0.13, 0.28])
    # utilities32_10.append([0.15, 0.11, 0.09, 0.06, 0.11])
    # utilities2_50.append([0.87, 0.78, 0.77, 0.83, 0.83])
    # utilities16_50.append([0.46, 0.3, 0.29, 0.27, 0.37])
    # utilities32_50.append([0.31, 0.15, 0.18, 0.13, 0.22])
    # utilities2_100.append([0.89, 0.81, 0.74, 0.83, 0.78])
    # utilities16_100.append([0.45, 0.29, 0.29, 0.29, 0.38])
    # utilities32_100.append([0.3, 0.17, 0.16, 0.18, 0.31])
    # utilities2_1000.append([0.88, 0.83, 0.82, 0.81, 0.85])
    # utilities16_1000.append([0.5, 0.32, 0.35, 0.33, 0.36])
    # utilities32_1000.append([0.33, 0.16, 0.23, 0.19, 0.24])
    # comps.append([-0.107, -0.1, -0.113, -0.109, -0.107])
    # sizes.append(small_size)
    # colors.append(1)
    #
    # labels.append('$\lambda_I = 1.5; n = 4$')
    # utilities2_10.append([0.69, 0.65, 0.63, 0.74, 0.7])
    # utilities16_10.append([0.24, 0.17, 0.17, 0.14, 0.29])
    # utilities32_10.append([0.17, 0.11, 0.07, 0.09, 0.1])
    # utilities2_50.append([0.85, 0.76, 0.73, 0.8, 0.8])
    # utilities16_50.append([0.47, 0.32, 0.28, 0.27, 0.33])
    # utilities32_50.append([0.29, 0.15, 0.16, 0.16, 0.26])
    # utilities2_100.append([0.91, 0.81, 0.77, 0.81, 0.83])
    # utilities16_100.append([0.45, 0.33, 0.29, 0.3, 0.34])
    # utilities32_100.append([0.33, 0.19, 0.18, 0.22, 0.26])
    # utilities2_1000.append([0.91, 0.83, 0.79, 0.79, 0.86])
    # utilities16_1000.append([0.49, 0.37, 0.36, 0.38, 0.39])
    # utilities32_1000.append([0.38, 0.16, 0.23, 0.21, 0.23])
    # comps.append([-0.092, -0.091, -0.083, -0.083, -0.088])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 2.0; n = 4$')
    # utilities2_10.append([0.77, 0.71, 0.68, 0.58, 0.7])
    # utilities16_10.append([0.32, 0.23, 0.14, 0.12, 0.29])
    # utilities32_10.append([0.17, 0.12, 0.07, 0.06, 0.15])
    # utilities2_50.append([0.85, 0.8, 0.76, 0.79, 0.79])
    # utilities16_50.append([0.48, 0.29, 0.26, 0.27, 0.36])
    # utilities32_50.append([0.33, 0.16, 0.14, 0.11, 0.22])
    # utilities2_100.append([0.91, 0.8, 0.76, 0.78, 0.8])
    # utilities16_100.append([0.47, 0.31, 0.3, 0.3, 0.38])
    # utilities32_100.append([0.32, 0.18, 0.18, 0.16, 0.27])
    # utilities2_1000.append([0.9, 0.84, 0.75, 0.82, 0.84])
    # utilities16_1000.append([0.48, 0.34, 0.36, 0.33, 0.43])
    # utilities32_1000.append([0.39, 0.16, 0.25, 0.21, 0.25])
    # comps.append([-0.083, -0.076, -0.07, -0.082, -0.075])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 3.0; n = 4$')
    # utilities2_10.append([0.77, 0.68, 0.7, 0.61, 0.71])
    # utilities16_10.append([0.27, 0.26, 0.14, 0.16, 0.26])
    # utilities32_10.append([0.14, 0.1, 0.06, 0.1, 0.17])
    # utilities2_50.append([0.86, 0.76, 0.75, 0.82, 0.79])
    # utilities16_50.append([0.48, 0.31, 0.29, 0.31, 0.36])
    # utilities32_50.append([0.29, 0.15, 0.18, 0.15, 0.23])
    # utilities2_100.append([0.91, 0.8, 0.79, 0.82, 0.79])
    # utilities16_100.append([0.46, 0.32, 0.33, 0.32, 0.39])
    # utilities32_100.append([0.35, 0.19, 0.19, 0.19, 0.27])
    # utilities2_1000.append([0.92, 0.78, 0.83, 0.84, 0.87])
    # utilities16_1000.append([0.54, 0.3, 0.38, 0.36, 0.42])
    # utilities32_1000.append([0.38, 0.15, 0.23, 0.23, 0.29])
    # comps.append([-0.062, -0.061, -0.063, -0.061, -0.063])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 10; n = 4$')
    # utilities2_10.append([0.75, 0.73, 0.66, 0.55, 0.65])
    # utilities16_10.append([0.27, 0.25, 0.12, 0.12, 0.22])
    # utilities32_10.append([0.17, 0.13, 0.06, 0.05, 0.14])
    # utilities2_50.append([0.84, 0.76, 0.77, 0.81, 0.81])
    # utilities16_50.append([0.5, 0.34, 0.26, 0.32, 0.37])
    # utilities32_50.append([0.27, 0.19, 0.14, 0.17, 0.27])
    # utilities2_100.append([0.89, 0.82, 0.77, 0.84, 0.83])
    # utilities16_100.append([0.46, 0.37, 0.34, 0.33, 0.4])
    # utilities32_100.append([0.36, 0.19, 0.19, 0.2, 0.3])
    # utilities2_1000.append([0.92, 0.8, 0.81, 0.8, 0.86])
    # utilities16_1000.append([0.51, 0.37, 0.37, 0.41, 0.43])
    # utilities32_1000.append([0.39, 0.2, 0.26, 0.25, 0.29])
    # comps.append([-0.042, -0.041, -0.044, -0.041, -0.045])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')


    ############################################################################
    #####         VQ-VIB2  N = 1
    ############################################################################
    n = 1
    # labels.append('$\lambda_I = 0; n = 1$')
    # utilities2_10.append([0.68, 0.48, 0.69, 0.61, 0.62])
    # utilities16_10.append([0.1, 0.06, 0.13, 0.07, 0.11])
    # utilities32_10.append([0.06, 0.02, 0.04, 0.03, 0.05])
    # utilities2_50.append([0.69, 0.46, 0.66, 0.71, 0.68])
    # utilities16_50.append([0.12, 0.04, 0.09, 0.15, 0.11])
    # utilities32_50.append([0.05, 0.02, 0.03, 0.11, 0.05])
    # utilities2_100.append([0.71, 0.51, 0.64, 0.73, 0.75])
    # utilities16_100.append([0.15, 0.07, 0.08, 0.19, 0.15])
    # utilities32_100.append([0.09, 0.02, 0.03, 0.08, 0.07])
    # utilities2_1000.append([0.75, 0.47, 0.66, 0.79, 0.69])
    # utilities16_1000.append([0.16, 0.05, 0.11, 0.19, 0.13])
    # utilities32_1000.append([0.08, 0.02, 0.04, 0.11, 0.05])
    # comps.append([ ])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')

    labels.append('$\lambda_I = 0.1; n = 1$')
    utilities2_10.append([0.49, 0.48, 0.5, 0.54, 0.44])
    utilities16_10.append([0.06, 0.07, 0.05, 0.06, 0.04])
    utilities32_10.append([0.04, 0.03, 0.02, 0.04, 0.03])
    utilities2_50.append([0.48, 0.48, 0.47, 0.51, 0.47])
    utilities16_50.append([0.05, 0.07, 0.07, 0.07, 0.04])
    utilities32_50.append([0.04, 0.04, 0.02, 0.03, 0.02])
    utilities2_100.append([0.45, 0.54, 0.49, 0.52, 0.43])
    utilities16_100.append([0.07, 0.06, 0.06, 0.06, 0.03])
    utilities32_100.append([0.04, 0.04, 0.02, 0.02, 0.02])
    utilities2_1000.append([0.49, 0.52, 0.55, 0.48, 0.43])
    utilities16_1000.append([0.06, 0.07, 0.07, 0.07, 0.03])
    utilities32_1000.append([0.04, 0.06, 0.04, 0.04, 0.03])
    comps.append([-0.375, -0.364, -0.355, -0.307, -0.318])
    sizes.append(small_size)
    colors.append('xkcd:blue')

    labels.append('$\lambda_I = 0.5; n = 1$')
    utilities2_10.append([0.51, 0.47, 0.5, 0.55, 0.45])
    utilities16_10.append([0.06, 0.06, 0.05, 0.06, 0.03])
    utilities32_10.append([0.03, 0.03, 0.02, 0.05, 0.03])
    utilities2_50.append([0.49, 0.47, 0.48, 0.5, 0.44])
    utilities16_50.append([0.06, 0.06, 0.07, 0.07, 0.03])
    utilities32_50.append([0.03, 0.02, 0.03, 0.02, 0.03])
    utilities2_100.append([0.46, 0.53, 0.5, 0.52, 0.43])
    utilities16_100.append([0.07, 0.06, 0.06, 0.06, 0.03])
    utilities32_100.append([0.02, 0.03, 0.03, 0.03, 0.03])
    utilities2_1000.append([0.5, 0.51, 0.56, 0.46, 0.41])
    utilities16_1000.append([0.07, 0.06, 0.07, 0.07, 0.02])
    utilities32_1000.append([0.04, 0.04, 0.03, 0.04, 0.03])
    comps.append([-0.375, -0.364, -0.355, -0.307, -0.318])
    sizes.append(small_size)
    colors.append('xkcd:blue')

    labels.append('$\lambda_I = 1; n = 1$')
    utilities2_10.append([0.62, 0.63, 0.61, 0.51, 0.65])
    utilities16_10.append([0.18, 0.08, 0.09, 0.07, 0.1])
    utilities32_10.append([0.07, 0.03, 0.05, 0.03, 0.05])
    utilities2_50.append([0.7, 0.66, 0.68, 0.74, 0.71])
    utilities16_50.append([0.15, 0.1, 0.09, 0.12, 0.16])
    utilities32_50.append([0.06, 0.04, 0.04, 0.05, 0.06])
    utilities2_100.append([0.77, 0.73, 0.67, 0.71, 0.71])
    utilities16_100.append([0.18, 0.13, 0.11, 0.14, 0.14])
    utilities32_100.append([0.1, 0.05, 0.05, 0.09, 0.07])
    utilities2_1000.append([0.78, 0.71, 0.64, 0.74, 0.71])
    utilities16_1000.append([0.18, 0.11, 0.1, 0.18, 0.15])
    utilities32_1000.append([0.12, 0.04, 0.05, 0.06, 0.1])
    comps.append([ -0.317, -0.292, -0.293, -0.266, -0.266])
    sizes.append(small_size)
    colors.append('xkcd:blue')

    labels.append('$\lambda_I = 1.5; n = 1$')
    utilities2_10.append([0.77, 0.61, 0.7, 0.56, 0.55])
    utilities16_10.append([0.22, 0.08, 0.17, 0.17, 0.1])
    utilities32_10.append([0.12, 0.04, 0.12, 0.11, 0.05])
    utilities2_50.append([0.75, 0.59, 0.74, 0.79, 0.7])
    utilities16_50.append([0.28, 0.08, 0.24, 0.31, 0.14])
    utilities32_50.append([0.12, 0.04, 0.14, 0.21, 0.09])
    utilities2_100.append([0.8, 0.68, 0.73, 0.79, 0.65])
    utilities16_100.append([0.26, 0.09, 0.27, 0.32, 0.17])
    utilities32_100.append([0.16, 0.04, 0.16, 0.16, 0.09])
    utilities2_1000.append([0.82, 0.66, 0.82, 0.82, 0.72])
    utilities16_1000.append([0.29, 0.1, 0.32, 0.36, 0.18])
    utilities32_1000.append([0.19, 0.06, 0.22, 0.22, 0.09])
    comps.append([ -0.214, -0.276, -0.226, -0.203, -0.25])
    sizes.append(small_size)
    colors.append('xkcd:blue')

    labels.append('$\lambda_I = 2.0; n = 1$')
    utilities2_10.append([0.7, 0.67, 0.68, 0.54, 0.52])
    utilities16_10.append([0.12, 0.17, 0.13, 0.09, 0.07])
    utilities32_10.append([0.06, 0.08, 0.09, 0.05, 0.03])
    utilities2_50.append([0.77, 0.68, 0.72, 0.78, 0.68])
    utilities16_50.append([0.22, 0.18, 0.21, 0.24, 0.14])
    utilities32_50.append([0.1, 0.14, 0.13, 0.11, 0.05])
    utilities2_100.append([0.77, 0.72, 0.71, 0.77, 0.69])
    utilities16_100.append([0.2, 0.19, 0.18, 0.25, 0.14])
    utilities32_100.append([0.11, 0.11, 0.12, 0.12, 0.04])
    utilities2_1000.append([0.78, 0.69, 0.74, 0.81, 0.7])
    utilities16_1000.append([0.24, 0.19, 0.23, 0.21, 0.12])
    utilities32_1000.append([0.12, 0.1, 0.15, 0.11, 0.05])
    comps.append([  -0.243, -0.28, -0.225, -0.228, -0.255])
    sizes.append(small_size)
    colors.append('xkcd:blue')


    labels.append('$\lambda_I = 3.0; n = 1$')
    utilities2_10.append([0.72, 0.58, 0.74, 0.47, 0.64])
    utilities16_10.append([0.19, 0.06, 0.12, 0.07, 0.17])
    utilities32_10.append([0.11, 0.03, 0.04, 0.03, 0.08])
    utilities2_50.append([0.79, 0.63, 0.69, 0.7, 0.75])
    utilities16_50.append([0.26, 0.08, 0.18, 0.12, 0.23])
    utilities32_50.append([0.16, 0.06, 0.1, 0.08, 0.12])
    utilities2_100.append([0.79, 0.65, 0.74, 0.65, 0.76])
    utilities16_100.append([0.27, 0.08, 0.2, 0.15, 0.28])
    utilities32_100.append([0.18, 0.05, 0.09, 0.06, 0.17])
    utilities2_1000.append([0.81, 0.63, 0.77, 0.69, 0.79])
    utilities16_1000.append([0.31, 0.06, 0.15, 0.13, 0.25])
    utilities32_1000.append([0.2, 0.05, 0.12, 0.05, 0.15])
    comps.append([-0.196, -0.281, -0.218, -0.268, -0.22])
    sizes.append(small_size)
    colors.append('xkcd:blue')

    labels.append('$\lambda_I = 10; n = 1$')
    utilities2_10.append([0.77, 0.65, 0.74, 0.51, 0.67])
    utilities16_10.append([0.29, 0.1, 0.12, 0.03, 0.18])
    utilities32_10.append([0.14, 0.03, 0.09, 0.04, 0.1])
    utilities2_50.append([0.83, 0.72, 0.67, 0.62, 0.79])
    utilities16_50.append([0.38, 0.17, 0.14, 0.09, 0.36])
    utilities32_50.append([0.21, 0.07, 0.07, 0.04, 0.18])
    utilities2_100.append([0.85, 0.79, 0.69, 0.68, 0.79])
    utilities16_100.append([0.41, 0.16, 0.15, 0.1, 0.34])
    utilities32_100.append([0.28, 0.06, 0.07, 0.06, 0.21])
    utilities2_1000.append([0.88, 0.69, 0.71, 0.7, 0.85])
    utilities16_1000.append([0.47, 0.15, 0.16, 0.11, 0.4])
    utilities32_1000.append([0.3, 0.1, 0.11, 0.05, 0.25])
    comps.append([-0.181, -0.22, -0.215, -0.275, -0.183])
    sizes.append(small_size)
    colors.append('xkcd:blue')

    labels.append('$\lambda_I = 100; n = 1$')
    utilities2_10.append([0.78, 0.69, 0.66, 0.59, 0.67])
    utilities16_10.append([0.22, 0.16, 0.16, 0.14, 0.2])
    utilities32_10.append([0.15, 0.08, 0.1, 0.07, 0.09])
    utilities2_50.append([0.78, 0.78, 0.75, 0.77, 0.73])
    utilities16_50.append([0.4, 0.25, 0.25, 0.28, 0.22])
    utilities32_50.append([0.25, 0.15, 0.14, 0.15, 0.16])
    utilities2_100.append([0.89, 0.84, 0.73, 0.8, 0.76])
    utilities16_100.append([0.39, 0.28, 0.25, 0.34, 0.23])
    utilities32_100.append([0.31, 0.13, 0.17, 0.19, 0.16])
    utilities2_1000.append([0.88, 0.8, 0.76, 0.83, 0.8])
    utilities16_1000.append([0.48, 0.28, 0.34, 0.38, 0.31])
    utilities32_1000.append([0.31, 0.18, 0.22, 0.2, 0.17])
    comps.append([  -0.132, -0.136, -0.137, -0.134, -0.156])
    sizes.append(small_size)
    colors.append('xkcd:blue')

    vqvib2_util2 = utilities2_100
    vqvib2_util16 = utilities16_100
    vqvib2_util32 = utilities32_100
    vqvib2_comps = comps
    utilities2_100 = []
    utilities16_100 = []
    utilities32_100 = []
    comps = []

    ############################################################################
    #####           VQ-VIB2 n = 2
    ############################################################################
    # n = 2
    # labels.append('$\lambda_I = 0.0; n = 2$')
    # utilities2_100.append([])
    # utilities16_100.append([])
    # utilities32_100.append([])
    # comps.append([])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 0.1; n = 2$')
    # utilities2_100.append([0.48, 0.5, 0.51, 0.5, 0.43])
    # utilities16_100.append([0.06, 0.07, 0.06, 0.07, 0.02])
    # utilities32_100.append([0.03, 0.02, 0.03, 0.02, 0.02])
    # comps.append([-0.375, -0.364, -0.355, -0.307, -0.318])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 0.5; n = 2$')
    # utilities2_100.append([0.79, 0.82, 0.51, 0.77, 0.82])
    # utilities16_100.append([0.19, 0.25, 0.07, 0.23, 0.31])
    # utilities32_100.append([0.12, 0.1, 0.03, 0.12, 0.23])
    # comps.append([-0.225, -0.19, -0.355, -0.22, -0.201])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 1.0; n = 2$')
    # utilities2_100.append([0.86, 0.76, 0.73, 0.79, 0.74])
    # utilities16_100.append([0.35, 0.24, 0.23, 0.3, 0.31])
    # utilities32_100.append([0.27, 0.12, 0.1, 0.16, 0.19])
    # comps.append([-0.18, -0.221, -0.224, -0.206, -0.188])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 1.5; n = 2$')
    # utilities2_100.append([0.88, 0.69, 0.71, 0.8, 0.81])
    # utilities16_100.append([0.36, 0.07, 0.23, 0.23, 0.32])
    # utilities32_100.append([0.24, 0.05, 0.14, 0.14, 0.22])
    # comps.append([-0.174, -0.221, -0.173, -0.208, -0.191])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 2.0; n = 2$')
    # utilities2_100.append([0.87, 0.83, 0.75, 0.77, 0.81])
    # utilities16_100.append([0.43, 0.29, 0.3, 0.26, 0.32])
    # utilities32_100.append([0.28, 0.15, 0.16, 0.15, 0.23])
    # comps.append([-0.163, -0.164, -0.191, -0.189, -0.176])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 3.0; n = 2$')
    # utilities2_100.append([0.86, 0.84, 0.76, 0.8, 0.81])
    # utilities16_100.append([0.4, 0.31, 0.33, 0.33, 0.39])
    # utilities32_100.append([0.25, 0.15, 0.19, 0.19, 0.28])
    # comps.append([-0.156, -0.139, -0.173, -0.146, -0.151])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 10.0; n = 2$')
    # utilities2_100.append([0.9, 0.84, 0.76, 0.81, 0.8])
    # utilities16_100.append([0.46, 0.32, 0.25, 0.3, 0.39])
    # utilities32_100.append([0.31, 0.15, 0.19, 0.17, 0.27])
    # comps.append([-0.125, -0.116, -0.132, -0.117, -0.126])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 100.0; n = 2$')
    # utilities2_100.append([0.9, 0.86, 0.77, 0.84, 0.79])
    # utilities16_100.append([0.46, 0.31, 0.32, 0.35, 0.41])
    # utilities32_100.append([0.35, 0.18, 0.17, 0.18, 0.33])
    # comps.append([-0.104, -0.099, -0.105, -0.1, -0.1])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    # vqvib2_util2 = utilities2_100
    # vqvib2_util16 = utilities16_100
    # vqvib2_util32 = utilities32_100
    # vqvib2_comps = comps
    # utilities2_100 = []
    # utilities16_100 = []
    # utilities32_100 = []
    # comps = []

    ############################################################################
    #####           VQ-VIB2 n = 4
    ############################################################################
    # n = 4
    # labels.append('$\lambda_I = 0.0; n = 4$')
    # utilities2_100.append([])
    # utilities16_100.append([])
    # utilities32_100.append([])
    # comps.append([])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 0.1; n = 4$')
    # utilities2_100.append([0.49, 0.52, 0.51, 0.51, 0.43])
    # utilities16_100.append([0.08, 0.06, 0.06, 0.07, 0.03])
    # utilities32_100.append([0.03, 0.03, 0.02, 0.01, 0.02])
    # comps.append([ -0.375, -0.364, -0.355, -0.308, -0.318])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 0.5; n = 4$')
    # utilities2_100.append([0.87, 0.8, 0.74, 0.69, 0.82])
    # utilities16_100.append([0.4, 0.29, 0.27, 0.15, 0.36])
    # utilities32_100.append([0.27, 0.13, 0.15, 0.1, 0.18])
    # comps.append([-0.191, -0.176, -0.192, -0.24, -0.197])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 1.0; n = 4$')
    # utilities2_100.append([0.87, 0.85, 0.74, 0.83, 0.82])
    # utilities16_100.append([0.42, 0.26, 0.33, 0.31, 0.37])
    # utilities32_100.append([0.3, 0.13, 0.19, 0.15, 0.28])
    # comps.append([-0.164, -0.153, -0.164, -0.151, -0.171])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 1.5; n = 4$')
    # utilities2_100.append([0.9, 0.81, 0.76, 0.83, 0.8])
    # utilities16_100.append([0.43, 0.27, 0.25, 0.32, 0.39])
    # utilities32_100.append([0.3, 0.13, 0.16, 0.19, 0.31])
    # comps.append([ -0.143, -0.136, -0.142, -0.142, -0.141])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 2.0; n = 4$')
    # utilities2_100.append([0.89, 0.83, 0.74, 0.83, 0.79])
    # utilities16_100.append([0.48, 0.28, 0.27, 0.32, 0.39])
    # utilities32_100.append([0.31, 0.17, 0.13, 0.2, 0.29])
    # comps.append([ -0.133, -0.14, -0.136, -0.132, -0.131])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 3.0; n = 4$')
    # utilities2_100.append([0.89, 0.82, 0.76, 0.84, 0.83])
    # utilities16_100.append([0.44, 0.3, 0.29, 0.34, 0.4])
    # utilities32_100.append([0.34, 0.17, 0.17, 0.2, 0.33])
    # comps.append([-0.121, -0.113, -0.124, -0.117, -0.124])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 10.0; n = 4$')
    # utilities2_100.append([0.88, 0.85, 0.73, 0.82, 0.81])
    # utilities16_100.append([0.45, 0.3, 0.28, 0.35, 0.41])
    # utilities32_100.append([0.36, 0.17, 0.14, 0.2, 0.31])
    # comps.append([-0.099, -0.092, -0.099, -0.098, -0.096])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 100.0; n = 4$')
    # utilities2_100.append([0.91, 0.83, 0.76, 0.83, 0.8])
    # utilities16_100.append([0.44, 0.31, 0.37, 0.33, 0.4])
    # utilities32_100.append([0.36, 0.17, 0.17, 0.22, 0.32])
    # comps.append([ -0.082, -0.075, -0.083, -0.08, -0.079])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    # vqvib2_util2 = utilities2_100
    # vqvib2_util16 = utilities16_100
    # vqvib2_util32 = utilities32_100
    # vqvib2_comps = comps
    # utilities2_100 = []
    # utilities16_100 = []
    # utilities32_100 = []
    # comps = []

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
    # for data, suffix in zip([utilities2, utilities16, utilities32], ['2', '16', '32']):
    #     plot_multi_trials([comps, data], labels, sizes, ylabel='Utility', colors=None, filename='xlation_train_responses_info' + suffix + '.png')
    # Plotting different $C$ for the same model type
    # labels = ['$C=2$'] + ['' for _ in utilities2[:-1]] + ['$C=16$'] + ['' for _ in utilities16[:-1]] + ['$C=32$'] + ['' for _ in utilities32[:-1]]
    # sizes = [2 * small_size for _ in utilities2] + [2 * small_size for _ in utilities16] + [2 * small_size for _ in utilities32]
    # plot_multi_trials([comps + comps + comps, utilities2 + utilities16 + utilities32], labels, sizes, ylabel='Translation Utility', xlabel='Distortion (MSE)', colors=None, filename='xlation_info_responses_vqvib2_n' + str(n) + '_numalign' + str(num_align) + '.png')

    # Plotting for different $N$ for the same model type
    # labels = ['$N=10$'] + ['' for _ in utilities2_100[:-1]] + ['$N=50$'] + ['' for _ in utilities2_100[:-1]] + ['$N=100$'] + ['' for _ in utilities2_100[:-1]] + ['$N=1000$'] + ['' for _ in utilities2_100[:-1]]
    # sizes = [2 * small_size for _ in range(4 * len(utilities2))]
    # plot_multi_trials([comps + comps + comps + comps, utilities2_10 + utilities2_50 + utilities2_100 + utilities2_1000], labels, sizes, ylabel='Functional Alignment', xlabel='Distortion (MSE)', colors=None, filename='xlation_info_responses_proto_n' + str(n) + '_varyingalign.png')

    labels = ['Onehot'] + ['' for _ in onehot_comps] + ['Proto.'] + ['' for _ in proto_comps] + ['VQ-VIB$_{\mathcal{N}}$'] + ['' for _ in vqvib_comps[:-1]] +  ['VQ-VIB$_{\mathcal{C}}$'] + ['' for _ in vqvib2_comps[:-1]]
    sizes = [2 * small_size for _ in labels]
    num_candidates = 2
    filename = 'xlation_info_multiimodel_' + str(num_candidates) + 'full'
    if num_candidates == 2:
        onehot_util = onehot_util2
        proto_util = proto_util2
        vqvib_util = vqvib_util2
        vqvib2_util = vqvib2_util2
    elif num_candidates == 16:
        onehot_util = onehot_util16
        proto_util = proto_util16
        vqvib_util = vqvib_util16
        vqvib2_util = vqvib2_util16
    elif num_candidates == 32:
        onehot_util = onehot_util32
        proto_util = proto_util32
        vqvib_util = vqvib_util32
        vqvib2_util = vqvib2_util32
    plot_multi_trials([onehot_comps + proto_comps + vqvib_comps + vqvib2_comps, onehot_util + proto_util + vqvib_util + vqvib2_util], labels, sizes,
                      ylabel='Functional Alignment', xlabel='Distortion (MSE)', colors=None, filename=('%s.png' % filename))

if __name__ == '__main__':
    run()
