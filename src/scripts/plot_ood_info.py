from src.utils.plotting import plot_multi_trials


def run():
    labels = []
    utilities2 = []
    utilities16 = []
    utilities32 = []
    infos = []
    sizes = []
    small_size = 100

    use_lambdas = False
    ############################################################################
    #####           Proto Redone March 2023
    ############################################################################
    # n = 'proto'
    # labels.append('$\lambda_I = 0.1 Proto')
    # utilities2.append([ 0.49, 0.497, 0.491, 0.506, 0.483])
    # utilities16.append([ 0.07, 0.068, 0.056, 0.071, 0.064])
    # utilities32.append([ 0.03, 0.03, 0.03, 0.034, 0.04])
    # if not use_lambdas:
    #     infos.append([ -0.322, -0.373, -0.347, -0.32, -0.323])
    # else:
    #     infos.append([0.1 for _ in range(5)])  # Hacking in lambdas
    # sizes.append(small_size)
    #
    # labels.append('$\lambda_I = 0.5$ proto')
    # utilities2.append([ 0.491, 0.5, 0.492, 0.505, 0.482])
    # utilities16.append([ 0.069, 0.069, 0.057, 0.072, 0.062])
    # utilities32.append([ 0.03, 0.03, 0.03, 0.033, 0.039])
    # if not use_lambdas:
    #     infos.append([ -0.322, -0.373, -0.347, -0.32, -0.323])
    # else:
    #     infos.append([0.5 for _ in range(5)])  # Hacking in lambdas
    # sizes.append(small_size)
    #
    # labels.append('$\lambda_I = 1.0$ proto')
    # utilities2.append([ 0.655, 0.684, 0.662, 0.504, 0.573])
    # utilities16.append([  0.137, 0.133, 0.126, 0.072, 0.101])
    # utilities32.append([ 0.063, 0.066, 0.074, 0.034, 0.059])
    # if not use_lambdas:
    #     infos.append([-0.294, -0.333, -0.294, -0.317, -0.271])
    # else:
    #     infos.append([1.0 for _ in range(5)])  # Hacking in lambdas
    # sizes.append(small_size)
    #
    # labels.append('$\lambda_I = 1.5$ proto')
    # utilities2.append([ 0.721, 0.745, 0.689, 0.604, 0.541])
    # utilities16.append([ 0.149, 0.138, 0.153, 0.112, 0.097])
    # utilities32.append([ 0.085, 0.07, 0.099, 0.056, 0.05])
    # if not use_lambdas:
    #     infos.append([ -0.281, -0.331, -0.256, -0.253, -0.265])
    # else:
    #     infos.append([1.5 for _ in range(5)])  # Hacking in lambdas
    # sizes.append(small_size)
    #
    # labels.append('$\lambda_I = 2$ proto')
    # utilities2.append([  0.782, 0.745, 0.716, 0.664, 0.559])
    # utilities16.append([ 0.188, 0.143, 0.175, 0.128, 0.106])
    # utilities32.append([ 0.112, 0.076, 0.102, 0.072, 0.074])
    # if not use_lambdas:
    #     infos.append([ -0.253, -0.326, -0.249, -0.248, -0.246])
    # else:
    #     infos.append([2 for _ in range(5)])  # Hacking in lambdas
    # sizes.append(small_size)
    #
    # labels.append('$\lambda_I = 3$ proto')
    # utilities2.append([ 0.78, 0.786, 0.73, 0.714, 0.566])
    # utilities16.append([ 0.187, 0.176, 0.189, 0.153, 0.099])
    # utilities32.append([ 0.102, 0.114, 0.109, 0.096, 0.07])
    # if not use_lambdas:
    #     infos.append([ -0.252, -0.296, -0.244, -0.233, -0.24])
    # else:
    #     infos.append([3 for _ in range(5)])  # Hacking in lambdas
    # sizes.append(small_size)
    #
    # labels.append('$\lambda_I = 10$ proto')
    # utilities2.append([0.811, 0.783, 0.765, 0.74, 0.604])
    # utilities16.append([ 0.236, 0.185, 0.203, 0.165, 0.12])
    # utilities32.append([  0.148, 0.109, 0.125, 0.106, 0.082])
    # if not use_lambdas:
    #     infos.append([ -0.225, -0.253, -0.226, -0.208, -0.218])
    # else:
    #     infos.append([10 for _ in range(5)])  # Hacking in lambdas
    # sizes.append(small_size)
    #
    # labels.append('$\lambda_I = 100$ proto')
    # utilities2.append([ 0.768, 0.759, 0.738, 0.767, 0.705])
    # utilities16.append([  0.205, 0.219, 0.209, 0.217, 0.182])
    # utilities32.append([  0.14, 0.103, 0.126, 0.108, 0.08])
    # if not use_lambdas:
    #     infos.append([ -0.212, -0.215, -0.197, -0.188, -0.2])
    # else:
    #     infos.append([100 for _ in range(5)])  # Hacking in lambdas
    # sizes.append(small_size)
    #
    # proto_util2 = utilities2
    # proto_util16 = utilities16
    # proto_util32 = utilities32
    # proto_comps = infos
    # utilities2 = []
    # utilities16 = []
    # utilities32 = []
    # infos = []
    ############################################################################
    #####           Onehot
    ############################################################################
    # n = 'onehot'
    # labels.append('$\lambda_I = 0.1 Onehot')
    # utilities2.append([0.499, 0.661, 0.52, 0.502, 0.486])
    # utilities16.append([0.058, 0.116, 0.066, 0.057, 0.061])
    # utilities32.append([0.026, 0.054, 0.03, 0.033, 0.029])
    # if not use_lambdas:
    #     infos.append([-0.317, -0.286, -0.266, -0.347, -0.311])
    # else:
    #     infos.append([0.1 for _ in range(5)])  # Hacking in lambdas
    # sizes.append(small_size)
    #
    # labels.append('$\lambda_I = 0.5 Onehot')
    # utilities2.append([0.734, 0.743, 0.705, 0.77, 0.653])
    # utilities16.append([0.249, 0.234, 0.217, 0.213, 0.185])
    # utilities32.append([0.129, 0.136, 0.13, 0.114, 0.112])
    # if not use_lambdas:
    #     infos.append([-0.214, -0.245, -0.204, -0.266, -0.222])
    # else:
    #     infos.append([0.5 for _ in range(5)])  # Hacking in lambdas
    # sizes.append(small_size)
    #
    # labels.append('$\lambda_I = 1.0$ Onehot')
    # utilities2.append([0.797, 0.617, 0.583, 0.822, 0.716])
    # utilities16.append([0.253, 0.109, 0.095, 0.274, 0.134])
    # utilities32.append([0.142, 0.056, 0.054, 0.151, 0.08])
    # if not use_lambdas:
    #     infos.append([-0.194, -0.274, -0.242, -0.229, -0.254])
    # else:
    #     infos.append([1.0 for _ in range(5)])  # Hacking in lambdas
    # sizes.append(small_size)
    #
    # labels.append('$\lambda_I = 1.5$ Onehot')
    # utilities2.append([0.801, 0.601, 0.799, 0.844, 0.638])
    # utilities16.append([0.259, 0.14, 0.241, 0.289, 0.148])
    # utilities32.append([0.142, 0.09, 0.154, 0.164, 0.097])
    # if not use_lambdas:
    #     infos.append([-0.203, -0.248, -0.212, -0.221, -0.262])
    # else:
    #     infos.append([1.5 for _ in range(5)])  # Hacking in lambdas
    # sizes.append(small_size)
    #
    # labels.append('$\lambda_I = 2.0 Onehot')
    # utilities2.append([0.777, 0.785, 0.763, 0.82, 0.784])
    # utilities16.append([0.252, 0.214, 0.219, 0.253, 0.243])
    # utilities32.append([0.162, 0.122, 0.12, 0.131, 0.137])
    # if not use_lambdas:
    #     infos.append([-0.193, -0.237, -0.219, -0.234, -0.21])
    # else:
    #     infos.append([2.0 for _ in range(5)])  # Hacking in lambdas
    # sizes.append(small_size)
    #
    # labels.append('$\lambda_I = 3.0 Onehot')
    # utilities2.append([0.74, 0.803, 0.725, 0.862, 0.777])
    # utilities16.append([0.213, 0.271, 0.237, 0.327, 0.276])
    # utilities32.append([0.121, 0.156, 0.157, 0.198, 0.174])
    # if not use_lambdas:
    #     infos.append([-0.236, -0.243, -0.191, -0.2, -0.197])
    # else:
    #     infos.append([3.0 for _ in range(5)])  # Hacking in lambdas
    # sizes.append(small_size)
    #
    # labels.append('$\lambda_I = 10$ Onehot')
    # utilities2.append([0.78, 0.818, 0.821, 0.77, 0.753])
    # utilities16.append([0.262, 0.246, 0.299, 0.266, 0.208])
    # utilities32.append([0.164, 0.152, 0.184, 0.163, 0.134])
    # if not use_lambdas:
    #     infos.append([-0.177, -0.215, -0.191, -0.199, -0.216])
    # else:
    #     infos.append([10 for _ in range(5)])  # Hacking in lambdas
    # sizes.append(small_size)
    #
    # labels.append('$\lambda_I = 100$ Onehot')
    # utilities2.append([0.706, 0.847, 0.693, 0.831, 0.752])
    # utilities16.append([0.186, 0.311, 0.217, 0.259, 0.223])
    # utilities32.append([0.13, 0.182, 0.141, 0.176, 0.141])
    # if not use_lambdas:
    #     infos.append([-0.208, -0.197, -0.205, -0.21, -0.211])
    # else:
    #     infos.append([100 for _ in range(5)])  # Hacking in lambdas
    # sizes.append(small_size)
    #
    # onehot_util2 = utilities2
    # onehot_util16 = utilities16
    # onehot_util32 = utilities32
    # onehot_comps = infos
    # utilities2 = []
    # utilities16 = []
    # utilities32 = []
    # infos = []
    ############################################################################
    #####           Onehot x2
    ############################################################################
    # n = 'onehot x2'
    # labels.append('$\lambda_I = 0.1 Onehot x2')
    # utilities2.append([ 0.731, 0.794, 0.846, 0.815, 0.844])
    # utilities16.append([ 0.245, 0.254, 0.316, 0.289, 0.279])
    # utilities32.append([ 0.124, 0.146, 0.182, 0.162, 0.156])
    # if not use_lambdas:
    #     infos.append([ -0.217, -0.226, -0.247, -0.231, -0.239])
    # else:
    #     infos.append([0.1 for _ in range(5)])  # Hacking in lambdas
    # sizes.append(small_size)
    #
    # labels.append('$\lambda_I = 0.5 Onehot x2')
    # utilities2.append([0.768, 0.806, 0.775, 0.836, 0.804])
    # utilities16.append([ 0.276, 0.269, 0.236, 0.273, 0.28])
    # utilities32.append([ 0.167, 0.161, 0.141, 0.172, 0.166])
    # if not use_lambdas:
    #     infos.append([ -0.213, -0.23, -0.214, -0.251, -0.219])
    # else:
    #     infos.append([0.5 for _ in range(5)])  # Hacking in lambdas
    # sizes.append(small_size)
    #
    # labels.append('$\lambda_I = 1.0$ Onehot x2')
    # utilities2.append([ 0.796, 0.81, 0.848, 0.876, 0.791])
    # utilities16.append([ 0.293, 0.233, 0.326, 0.301, 0.248])
    # utilities32.append([ 0.202, 0.162, 0.211, 0.189, 0.161])
    # if not use_lambdas:
    #     infos.append([ -0.196, -0.205, -0.215, -0.221, -0.213])
    # else:
    #     infos.append([1.0 for _ in range(5)])  # Hacking in lambdas
    # sizes.append(small_size)
    #
    # labels.append('$\lambda_I = 1.5$ Onehot x2')
    # utilities2.append([ 0.784, 0.834, 0.787, 0.881, 0.825])
    # utilities16.append([ 0.29, 0.264, 0.299, 0.382, 0.27])
    # utilities32.append([ 0.196, 0.166, 0.192, 0.227, 0.167])
    # if not use_lambdas:
    #     infos.append([ -0.19, -0.205, -0.204, -0.19, -0.213])
    # else:
    #     infos.append([1.5 for _ in range(5)])  # Hacking in lambdas
    # sizes.append(small_size)
    #
    # labels.append('$\lambda_I = 2.0 Onehot x2')
    # utilities2.append([ 0.786, 0.841, 0.804, 0.902, 0.777])
    # utilities16.append([ 0.293, 0.316, 0.313, 0.407, 0.256])
    # utilities32.append([ 0.2, 0.185, 0.186, 0.266, 0.141])
    # if not use_lambdas:
    #     infos.append([ -0.176, -0.194, -0.21, -0.192, -0.21])
    # else:
    #     infos.append([2.0 for _ in range(5)])  # Hacking in lambdas
    # sizes.append(small_size)
    #
    # labels.append('$\lambda_I = 3.0 Onehot x2')
    # utilities2.append([ 0.795, 0.848, 0.826, 0.904, 0.827])
    # utilities16.append([ 0.316, 0.338, 0.316, 0.405, 0.301])
    # utilities32.append([ 0.218, 0.198, 0.214, 0.266, 0.207])
    # if not use_lambdas:
    #     infos.append([ -0.176, -0.189, -0.184, -0.178, -0.186])
    # else:
    #     infos.append([3.0 for _ in range(5)])  # Hacking in lambdas
    # sizes.append(small_size)
    #
    # labels.append('$\lambda_I = 10$ Onehot x2')
    # utilities2.append([ 0.807, 0.87, 0.83, 0.9, 0.83])
    # utilities16.append([ 0.318, 0.356, 0.309, 0.41, 0.304])
    # utilities32.append([ 0.208, 0.227, 0.213, 0.273, 0.202])
    # if not use_lambdas:
    #     infos.append([ -0.164, -0.176, -0.17, -0.172, -0.174])
    # else:
    #     infos.append([10 for _ in range(5)])  # Hacking in lambdas
    # sizes.append(small_size)
    #
    # labels.append('$\lambda_I = 100$ Onehot x2')
    # utilities2.append([ 0.805, 0.857, 0.81, 0.891, 0.855])
    # utilities16.append([ 0.316, 0.33, 0.311, 0.405, 0.359])
    # utilities32.append([ 0.231, 0.209, 0.207, 0.253, 0.227])
    # if not use_lambdas:
    #     infos.append([ -0.163, -0.18, -0.177, -0.166, -0.171])
    # else:
    #     infos.append([100 for _ in range(5)])  # Hacking in lambdas
    # sizes.append(small_size)
    #
    # onehot2_util2 = utilities2
    # onehot2_util16 = utilities16
    # onehot2_util32 = utilities32
    # onehot2_comps = infos
    # utilities2 = []
    # utilities16 = []
    # utilities32 = []
    # infos = []
    ############################################################################
    #####           Onehot x4
    ############################################################################
    # n = 'onehot x4'
    # labels.append('$\lambda_I = 0.1 Onehot x4')
    # utilities2.append([ 0.814, 0.757, 0.833, 0.896, 0.886])
    # utilities16.append([ 0.32, 0.212, 0.317, 0.384, 0.363])
    # utilities32.append([ 0.198, 0.114, 0.186, 0.206, 0.219])
    # if not use_lambdas:
    #     infos.append([ -0.242, -0.258, -0.256, -0.268, -0.233])
    # else:
    #     infos.append([0.1 for _ in range(5)])  # Hacking in lambdas
    # sizes.append(small_size)
    #
    # labels.append('$\lambda_I = 0.5 Onehot x4')
    # utilities2.append([ 0.782, 0.862, 0.879, 0.898, 0.832])
    # utilities16.append([0.305, 0.346, 0.414, 0.414, 0.33])
    # utilities32.append([ 0.219, 0.217, 0.276, 0.269, 0.207])
    # if not use_lambdas:
    #     infos.append([ -0.2, -0.199, -0.199, -0.189, -0.2])
    # else:
    #     infos.append([0.5 for _ in range(5)])  # Hacking in lambdas
    # sizes.append(small_size)
    #
    # labels.append('$\lambda_I = 1.0$ Onehot x4')
    # utilities2.append([ 0.793, 0.867, 0.807, 0.934, 0.804])
    # utilities16.append([ 0.332, 0.357, 0.297, 0.486, 0.295])
    # utilities32.append([ 0.234, 0.228, 0.189, 0.342, 0.192])
    # if not use_lambdas:
    #     infos.append([ -0.168, -0.194, -0.189, -0.172, -0.187])
    # else:
    #     infos.append([1.0 for _ in range(5)])  # Hacking in lambdas
    # sizes.append(small_size)
    #
    # labels.append('$\lambda_I = 1.5$ Onehot x4')
    # utilities2.append([ 0.786, 0.87, 0.816, 0.928, 0.848])
    # utilities16.append([ 0.302, 0.378, 0.35, 0.478, 0.348])
    # utilities32.append([ 0.218, 0.253, 0.227, 0.344, 0.234])
    # if not use_lambdas:
    #     infos.append([ -0.169, -0.183, -0.169, -0.169, -0.179])
    # else:
    #     infos.append([1.5 for _ in range(5)])  # Hacking in lambdas
    # sizes.append(small_size)
    #
    # labels.append('$\lambda_I = 2.0 Onehot x4')
    # utilities2.append([ 0.816, 0.855, 0.834, 0.922, 0.845])
    # utilities16.append([ 0.359, 0.364, 0.341, 0.44, 0.312])
    # utilities32.append([ 0.25, 0.235, 0.203, 0.292, 0.203])
    # if not use_lambdas:
    #     infos.append([ -0.158, -0.175, -0.164, -0.171, -0.185])
    # else:
    #     infos.append([2.0 for _ in range(5)])  # Hacking in lambdas
    # sizes.append(small_size)
    #
    # labels.append('$\lambda_I = 3.0 Onehot x4')
    # utilities2.append([ 0.806, 0.861, 0.85, 0.923, 0.85])
    # utilities16.append([ 0.328, 0.397, 0.356, 0.476, 0.363])
    # utilities32.append([ 0.249, 0.281, 0.244, 0.324, 0.252])
    # if not use_lambdas:
    #     infos.append([ -0.151, -0.162, -0.161, -0.159, -0.17])
    # else:
    #     infos.append([3.0 for _ in range(5)])  # Hacking in lambdas
    # sizes.append(small_size)
    #
    # labels.append('$\lambda_I = 10$ Onehot x4')
    # utilities2.append([ 0.812, 0.907, 0.887, 0.928, 0.916])
    # utilities16.append([ 0.366, 0.458, 0.439, 0.488, 0.459])
    # utilities32.append([ 0.251, 0.319, 0.312, 0.338, 0.308])
    # if not use_lambdas:
    #     infos.append([ -0.145, -0.151, -0.145, -0.159, -0.153])
    # else:
    #     infos.append([10 for _ in range(5)])  # Hacking in lambdas
    # sizes.append(small_size)
    #
    # labels.append('$\lambda_I = 100$ Onehot x4')
    # utilities2.append([ 0.82, 0.871, 0.862, 0.906, 0.915])
    # utilities16.append([ 0.341, 0.414, 0.409, 0.44, 0.492])
    # utilities32.append([ 0.238, 0.294, 0.268, 0.292, 0.323])
    # if not use_lambdas:
    #     infos.append([ -0.147, -0.154, -0.144, -0.145, -0.15])
    # else:
    #     infos.append([100 for _ in range(5)])  # Hacking in lambdas
    # sizes.append(small_size)
    #
    # onehot4_util2 = utilities2
    # onehot4_util16 = utilities16
    # onehot4_util32 = utilities32
    # onehot4_comps = infos
    # utilities2 = []
    # utilities16 = []
    # utilities32 = []
    # infos = []
    ############################################################################
    #####           VQVIB N = 1
    ############################################################################
    # n = 1
    # labels.append('$\lambda_I = 0.1; n = 1$')
    # utilities2.append([0.833, 0.757, 0.71, 0.58, 0.695])
    # utilities16.append([0.265, 0.22, 0.202, 0.112, 0.197])
    # utilities32.append([0.146, 0.123, 0.131, 0.056, 0.12])
    # if not use_lambdas:
    #     infos.append([ -0.251, -0.247, -0.234, -0.274, -0.239])
    # else:
    #     infos.append([0.1 for _ in range(5)])  # Hacking in lambdas
    # sizes.append(small_size)
    #
    # labels.append('$\lambda_I = 0.5; n = 1$')
    # utilities2.append([0.873, 0.756, 0.775, 0.806, 0.853])
    # utilities16.append([0.329, 0.266, 0.267, 0.278, 0.339])
    # utilities32.append([0.207, 0.143, 0.176, 0.182, 0.225])
    # if not use_lambdas:
    #     infos.append([ -0.189, -0.202, -0.197, -0.199, -0.182])
    # else:
    #     infos.append([0.5 for _ in range(5)])  # Hacking in lambdas
    # sizes.append(small_size)
    #
    # labels.append('$\lambda_I = 1; n = 1$')
    # utilities2.append([ 0.892, 0.856, 0.822, 0.846, 0.872])
    # utilities16.append([0.431, 0.343, 0.323, 0.338, 0.386])
    # utilities32.append([0.244, 0.218, 0.208, 0.186, 0.262])
    # if not use_lambdas:
    #     infos.append([ -0.168, -0.164, -0.167, -0.174, -0.167])
    # else:
    #     infos.append([1 for _ in range(5)])  # Hacking in lambdas
    # sizes.append(small_size)
    #
    # labels.append('$\lambda_I = 1.5; n = 1$')
    # utilities2.append([0.892, 0.87, 0.821, 0.84, 0.871])
    # utilities16.append([0.437, 0.389, 0.32, 0.377, 0.411])
    # utilities32.append([0.288, 0.252, 0.206, 0.232, 0.266])
    # if not use_lambdas:
    #     infos.append([ -0.158, -0.149, -0.155, -0.161, -0.156 ])
    # else:
    #     infos.append([1.5 for _ in range(5)])  # Hacking in lambdas
    # sizes.append(small_size)
    #
    # labels.append('$\lambda_I = 2.0; n = 1$')
    # utilities2.append([0.888, 0.876, 0.8, 0.868, 0.89])
    # utilities16.append([0.429, 0.383, 0.317, 0.373, 0.432])
    # utilities32.append([0.281, 0.261, 0.218, 0.241, 0.305])
    # if not use_lambdas:
    #     infos.append([-0.158, -0.15, -0.149, -0.149, -0.151])
    # else:
    #     infos.append([2 for _ in range(5)])  # Hacking in lambdas
    # sizes.append(small_size)
    #
    # labels.append('$\lambda_I = 3.0; n = 1$')
    # utilities2.append([0.896, 0.881, 0.784, 0.86, 0.896])
    # utilities16.append([0.44, 0.407, 0.321, 0.376, 0.478])
    # utilities32.append([ 0.289, 0.256, 0.207, 0.244, 0.332])
    # if not use_lambdas:
    #     infos.append([ -0.149, -0.141, -0.147, -0.142, -0.141])
    # else:
    #     infos.append([3 for _ in range(5)])  # Hacking in lambdas
    # sizes.append(small_size)
    #
    # labels.append('$\lambda_I = 10; n = 1$')
    # utilities2.append([0.897, 0.848, 0.814, 0.882, 0.912])
    # utilities16.append([0.473, 0.354, 0.344, 0.417, 0.5])
    # utilities32.append([0.328, 0.246, 0.232, 0.292, 0.365])
    # if not use_lambdas:
    #     infos.append([ -0.128, -0.125, -0.127, -0.127, -0.124])
    # else:
    #     infos.append([10 for _ in range(5)])  # Hacking in lambdas
    # sizes.append(small_size)
    #
    # labels.append('$\lambda_I = 100; n = 1$')
    # utilities2.append([ 0.905, 0.871, 0.793, 0.863, 0.912])
    # utilities16.append([  0.417, 0.357, 0.279, 0.364, 0.453])
    # utilities32.append([ 0.276, 0.225, 0.19, 0.251, 0.316])
    # if not use_lambdas:
    #     infos.append([ -0.154, -0.168, -0.142, -0.132])
    # else:
    #     infos.append([100 for _ in range(5)])  # Hacking in lambdas
    # sizes.append(small_size)

    # vq_util2 = utilities2
    # vq_util16 = utilities16
    # vq_util32 = utilities32
    # vq_comps = infos
    # utilities2 = []
    # utilities16 = []
    # utilities32 = []
    # infos = []
    #

    ############################################################################
    #####           VQVIB N = 2
    ############################################################################
    # n = 2
    # labels.append('$\lambda_I = 0.1; n = 2$')
    # utilities2.append([ 0.807, 0.68, 0.791, 0.78, 0.757])
    # utilities16.append([ 0.201, 0.173, 0.221, 0.2, 0.249])
    # utilities32.append([ 0.11, 0.114, 0.124, 0.112, 0.142])
    # if not use_lambdas:
    #     infos.append([  -0.237, -0.237, -0.246, -0.26, -0.252])
    # else:
    #     infos.append([0.1 for _ in range(5)])  # Hacking in lambdas
    # sizes.append(small_size)
    #
    # labels.append('$\lambda_I = 0.5; n = 2$')
    # utilities2.append([ 0.894, 0.87, 0.775, 0.801, 0.803])
    # utilities16.append([ 0.375, 0.35, 0.257, 0.288, 0.251])
    # utilities32.append([ 0.21, 0.212, 0.167, 0.175, 0.167])
    # if not use_lambdas:
    #     infos.append([  -0.183, -0.177, -0.181, -0.188, -0.19])
    # else:
    #     infos.append([0.5 for _ in range(5)])  # Hacking in lambdas
    # sizes.append(small_size)
    #
    # labels.append('$\lambda_I = 1; n = 2$')
    # utilities2.append([ 0.896, 0.871, 0.823, 0.855, 0.896])
    # utilities16.append([ 0.363, 0.366, 0.298, 0.362, 0.393])
    # utilities32.append([ 0.225, 0.224, 0.211, 0.236, 0.257])
    # if not use_lambdas:
    #     infos.append([  -0.16, -0.154, -0.154, -0.156, -0.157])
    # else:
    #     infos.append([1 for _ in range(5)])  # Hacking in lambdas
    # sizes.append(small_size)
    #
    # labels.append('$\lambda_I = 1.5; n = 2$')
    # utilities2.append([ 0.92, 0.897, 0.851, 0.847, 0.912])
    # utilities16.append([ 0.449, 0.408, 0.355, 0.351, 0.462])
    # utilities32.append([ 0.296, 0.287, 0.254, 0.238, 0.319])
    # if not use_lambdas:
    #     infos.append([ -0.141, -0.135, -0.14, -0.144, -0.141])
    # else:
    #     infos.append([1.5 for _ in range(5)])  # Hacking in lambdas
    # sizes.append(small_size)
    #
    # labels.append('$\lambda_I = 2.0; n = 2$')
    # utilities2.append([ 0.928, 0.904, 0.84, 0.869, 0.916])
    # utilities16.append([ 0.461, 0.415, 0.359, 0.396, 0.499])
    # utilities32.append([ 0.285, 0.288, 0.247, 0.278, 0.361])
    # if not use_lambdas:
    #     infos.append([ -0.133, -0.125, -0.128, -0.128, -0.127])
    # else:
    #     infos.append([2 for _ in range(5)])  # Hacking in lambdas
    # sizes.append(small_size)
    #
    # labels.append('$\lambda_I = 3.0; n = 2$')
    # utilities2.append([ 0.935, 0.916, 0.858, 0.912, 0.927])
    # utilities16.append([ 0.511, 0.491, 0.357, 0.475, 0.543])
    # utilities32.append([ 0.341, 0.321, 0.265, 0.36, 0.391])
    # if not use_lambdas:
    #     infos.append([-0.118, -0.112, -0.119, -0.116, -0.119])
    # else:
    #     infos.append([3 for _ in range(5)])  # Hacking in lambdas
    # sizes.append(small_size)
    #
    # labels.append('$\lambda_I = 10; n = 2$')
    # utilities2.append([ 0.941, 0.932, 0.879, 0.929, 0.948])
    # utilities16.append([ 0.554, 0.519, 0.455, 0.555, 0.649])
    # utilities32.append([ 0.389, 0.376, 0.352, 0.427, 0.504])
    # if not use_lambdas:
    #     infos.append([ -0.099, -0.097, -0.1, -0.094, -0.098])
    # else:
    #     infos.append([10 for _ in range(5)])  # Hacking in lambdas
    # sizes.append(small_size)
    #
    #
    # labels.append('$\lambda_I = 100; n = 2$')
    # utilities2.append([ 0.911, 0.942, 0.899, 0.939, 0.968])
    # utilities16.append([ 0.574, 0.539, 0.465, 0.585, 0.659])
    # utilities32.append([ 0.399, 0.376, 0.392, 0.447, 0.514])
    # if not use_lambdas:
    #     infos.append([ -0.091, -0.092, -0.091, -0.091, -0.092])
    # else:
    #     infos.append([100 for _ in range(5)])  # Hacking in lambdas
    # sizes.append(small_size)
    #

    # vq_util2 = utilities2
    # vq_util16 = utilities16
    # vq_util32 = utilities32
    # vq_comps = infos
    # utilities2 = []
    # utilities16 = []
    # utilities32 = []
    # infos = []


    ############################################################################
    #####           VQVIB N = 4
    ############################################################################
    # n = 4
    # labels.append('$\lambda_I = 0.1; n = 4$')
    # utilities2.append([  0.839, 0.671, 0.74, 0.775, 0.714])
    # utilities16.append([ 0.265, 0.173, 0.23, 0.228, 0.201])
    # utilities32.append([  0.145, 0.105, 0.145, 0.144, 0.134])
    # if not use_lambdas:
    #     infos.append([   -0.217, -0.221, -0.225, -0.245, -0.248])
    # else:
    #     infos.append([0.1 for _ in range(5)])  # Hacking in lambdas
    # sizes.append(small_size)
    #
    # labels.append('$\lambda_I = 0.5; n = 4$')
    # utilities2.append([ 0.909, 0.825, 0.768, 0.886, 0.856])
    # utilities16.append([  0.393, 0.321, 0.275, 0.379, 0.352])
    # utilities32.append([  0.24, 0.198, 0.197, 0.268, 0.239])
    # if not use_lambdas:
    #     infos.append([  -0.156, -0.161, -0.162, -0.171, -0.168])
    # else:
    #     infos.append([0.5 for _ in range(5)])  # Hacking in lambdas
    # sizes.append(small_size)
    #
    # labels.append('$\lambda_I = 1; n = 4$')
    # utilities2.append([  0.929, 0.916, 0.825, 0.88, 0.921])
    # utilities16.append([  0.488, 0.479, 0.332, 0.417, 0.489])
    # utilities32.append([ 0.317, 0.329, 0.223, 0.302, 0.34])
    # if not use_lambdas:
    #     infos.append([  -0.134, -0.122, -0.14, -0.139, -0.134])
    # else:
    #     infos.append([1 for _ in range(5)])  # Hacking in lambdas
    # sizes.append(small_size)
    #
    # labels.append('$\lambda_I = 1.5; n = 4$')
    # utilities2.append([  0.939, 0.915, 0.873, 0.902, 0.929])
    # utilities16.append([  0.517, 0.457, 0.438, 0.468, 0.559])
    # utilities32.append([ 0.331, 0.346, 0.304, 0.34, 0.413])
    # if not use_lambdas:
    #     infos.append([  -0.114, -0.11, -0.112, -0.11, -0.114])
    # else:
    #     infos.append([1.5 for _ in range(5)])  # Hacking in lambdas
    # sizes.append(small_size)
    #
    # labels.append('$\lambda_I = 2.0; n = 4$')
    # utilities2.append([  0.955, 0.929, 0.891, 0.906, 0.941])
    # utilities16.append([  0.552, 0.536, 0.457, 0.496, 0.609])
    # utilities32.append([ 0.398, 0.4, 0.345, 0.367, 0.465])
    # if not use_lambdas:
    #     infos.append([ -0.102, -0.097, -0.099, -0.106, -0.101])
    # else:
    #     infos.append([2 for _ in range(5)])  # Hacking in lambdas
    # sizes.append(small_size)
    #
    # labels.append('$\lambda_I = 3.0; n = 4$')
    # utilities2.append([ 0.955, 0.938, 0.865, 0.948, 0.968])
    # utilities16.append([  0.639, 0.579, 0.407, 0.598, 0.716])
    # utilities32.append([  0.445, 0.423, 0.305, 0.48, 0.566])
    # if not use_lambdas:
    #     infos.append([ -0.093, -0.088, -0.091, -0.091, -0.093])
    # else:
    #     infos.append([3 for _ in range(5)])  # Hacking in lambdas
    # sizes.append(small_size)
    #
    # labels.append('$\lambda_I = 10; n = 4$')
    # utilities2.append([  0.977, 0.955, 0.938, 0.97, 0.986])
    # utilities16.append([  0.785, 0.632, 0.646, 0.726, 0.85])
    # utilities32.append([  0.65, 0.493, 0.556, 0.618, 0.773])
    # if not use_lambdas:
    #     infos.append([  -0.067, -0.066, -0.067, -0.067, -0.057])
    # else:
    #     infos.append([10 for _ in range(5)])  # Hacking in lambdas
    # sizes.append(small_size)
    #
    #
    # labels.append('$\lambda_I = 100; n = 4$')
    # utilities2.append([  0.976, 0.957, 0.927, 0.969, 0.984])
    # utilities16.append([  0.752, 0.67, 0.605, 0.727, 0.841])
    # utilities32.append([ 0.612, 0.529, 0.516, 0.629, 0.725])
    # if not use_lambdas:
    #     infos.append([  -0.077, -0.07, -0.072, -0.068, -0.07])
    # else:
    #     infos.append([100 for _ in range(5)])  # Hacking in lambdas
    # sizes.append(small_size)


    # vq_util2 = utilities2
    # vq_util16 = utilities16
    # vq_util32 = utilities32
    # vq_comps = infos
    # utilities2 = []
    # utilities16 = []
    # utilities32 = []
    # infos = []

    ############################################################################
    #####         VQ-VIB2  N = 1
    ############################################################################
    # n = 1
    # labels.append('$\lambda_I = 0.1; n = 1$')
    # utilities2.append([ 0.497, 0.495, 0.514, 0.512, 0.487])
    # utilities16.append([0.067, 0.071, 0.058, 0.064, 0.069])
    # utilities32.append([0.036, 0.033, 0.029, 0.04, 0.028])
    # if not use_lambdas:
    #     infos.append([ -0.375, -0.364, -0.355, -0.307, -0.318])
    # else:
    #     infos.append([0.1 for _ in range(5)])  # Hacking in lambdas
    # sizes.append(small_size)
    #
    # labels.append('$\lambda_I = 0.5; n = 1$')
    # utilities2.append([0.494, 0.495, 0.512, 0.509, 0.487])
    # utilities16.append([ 0.069, 0.07, 0.061, 0.065, 0.07])
    # utilities32.append([0.037, 0.03, 0.033, 0.041, 0.03])
    # if not use_lambdas:
    #     infos.append([ -0.375, -0.364, -0.355, -0.307, -0.318])
    # else:
    #     infos.append([0.5 for _ in range(5)])  # Hacking in lambdas
    # sizes.append(small_size)
    #
    # labels.append('$\lambda_I = 1; n = 1$')
    # utilities2.append([0.689, 0.571, 0.535, 0.693, 0.716])
    # utilities16.append([0.157, 0.106, 0.1, 0.162, 0.152])
    # utilities32.append([ 0.073, 0.054, 0.05, 0.081, 0.088])
    # if not use_lambdas:
    #     infos.append([-0.317, -0.292, -0.293, -0.266, -0.266])  # Done
    # else:
    #     infos.append([1.0 for _ in range(5)])  # Hacking in lambdas
    # sizes.append(small_size)
    # #
    # # labels.append('$\lambda_I = 1.5; n = 1$')
    # # utilities2.append([0.791, 0.647, 0.687, 0.746, 0.711])
    # # utilities16.append([0.195, 0.134, 0.17, 0.232, 0.15])
    # # utilities32.append([ 0.109, 0.071, 0.115, 0.154, 0.077])
    # # if not use_lambdas:
    # #     infos.append([  -0.214, -0.276, -0.226, -0.203, -0.25])
    # # else:
    # #     infos.append([1.5 for _ in range(5)])  # Hacking in lambdas
    # # sizes.append(small_size)
    # #
    # # labels.append('$\lambda_I = 2.0; n = 1$')
    # # utilities2.append([ 0.688, 0.655, 0.707, 0.695, 0.557])
    # # utilities16.append([0.153, 0.155, 0.182, 0.204, 0.097])
    # # utilities32.append([ 0.085, 0.084, 0.111, 0.135, 0.059])
    # # if not use_lambdas:
    # #     infos.append([  -0.243, -0.28, -0.225, -0.228, -0.255])
    # # else:
    # #     infos.append([2.0 for _ in range(5)])  # Hacking in lambdas
    # # sizes.append(small_size)
    # labels.append('$\lambda_I = 2.0; n = 1$')
    # utilities2.append([ 0.688, 0.655, 0.707, 0.695, 0.557])
    # utilities16.append([0.153, 0.155, 0.182, 0.204, 0.097])
    # utilities32.append([ 0.085, 0.084, 0.111, 0.135, 0.059])
    # if not use_lambdas:
    #     infos.append([  -0.243, -0.28, -0.225, -0.228, -0.255])
    # else:
    #     infos.append([1.5 for _ in range(5)])  # Hacking in lambdas
    # sizes.append(small_size)
    #
    # labels.append('$\lambda_I = 1.5; n = 1$')
    # utilities2.append([0.791, 0.647, 0.687, 0.746, 0.711])
    # utilities16.append([0.195, 0.134, 0.17, 0.232, 0.15])
    # utilities32.append([ 0.109, 0.071, 0.115, 0.154, 0.077])
    # if not use_lambdas:
    #     infos.append([  -0.214, -0.276, -0.226, -0.203, -0.25])
    # else:
    #     infos.append([2.0 for _ in range(5)])  # Hacking in lambdas
    # sizes.append(small_size)
    #
    #
    #
    # labels.append('$\lambda_I = 3.0; n = 1$')
    # utilities2.append([0.707, 0.67, 0.662, 0.744, 0.65])
    # utilities16.append([ 0.176, 0.144, 0.18, 0.145, 0.153])
    # utilities32.append([ 0.107, 0.077, 0.111, 0.071, 0.097])
    # if not use_lambdas:
    #     infos.append([   -0.196, -0.281, -0.218, -0.268, -0.22])
    # else:
    #     infos.append([3.0 for _ in range(5)])  # Hacking in lambdas
    # sizes.append(small_size)
    #
    # labels.append('$\lambda_I = 10; n = 1$')
    # utilities2.append([ 0.86, 0.576, 0.559, 0.741, 0.817])
    # utilities16.append([ 0.341, 0.12, 0.111, 0.146, 0.288])
    # utilities32.append([0.2, 0.091, 0.072, 0.075, 0.159])
    # if not use_lambdas:
    #     infos.append([ -0.181, -0.22, -0.215, -0.275, -0.183])
    # else:
    #     infos.append([10 for _ in range(5)])  # Hacking in lambdas
    # sizes.append(small_size)
    #
    # labels.append('$\lambda_I = 100; n = 1$')
    # utilities2.append([0.877, 0.865, 0.783, 0.868, 0.843])
    # utilities16.append([0.364, 0.35, 0.29, 0.396, 0.331])
    # utilities32.append([0.234, 0.23, 0.197, 0.284, 0.216])
    # if not use_lambdas:
    #     infos.append([-0.132, -0.136, -0.137, -0.134, -0.156])
    # else:
    #     infos.append([100 for _ in range(5)])  # Hacking in lambdas
    # sizes.append(small_size)

    # vqvib2_util2 = utilities2
    # vqvib2_util16 = utilities16
    # vqvib2_util32 = utilities32
    # vqvib2_comps = infos
    # utilities2 = []
    # utilities16 = []
    # utilities32 = []
    # infos = []

    ############################################################################
    #####         VQ-VIB2  N = 2
    ############################################################################
    # n = 2
    # labels.append('$\lambda_I = 0.1; n = 2$')
    # utilities2.append([ 0.495, 0.492, 0.48, 0.497, 0.486])
    # utilities16.append([0.064, 0.063, 0.049, 0.054, 0.069])
    # utilities32.append([0.036, 0.062, 0.101, 0.029, 0.025])
    # if not use_lambdas:
    #     infos.append([-0.375, -0.364, -0.355, -0.307, -0.318])
    # else:
    #     infos.append([0.1 for _ in range(5)])  # Hacking in lambdas
    # sizes.append(small_size)
    #
    # labels.append('$\lambda_I = 0.5; n = 2$')
    # utilities2.append([0.785, 0.72, 0.477, 0.716, 0.69])
    # utilities16.append([0.22, 0.223, 0.051, 0.184, 0.18])
    # utilities32.append([0.127, 0.122, 0.034, 0.122, 0.108])
    # if not use_lambdas:
    #     infos.append([-0.225, -0.19, -0.355, -0.22, -0.201])
    # else:
    #     infos.append([0.5 for _ in range(5)])  # Hacking in lambdas
    # sizes.append(small_size)
    #
    # labels.append('$\lambda_I = 1; n = 2$')
    # utilities2.append([0.864, 0.737, 0.696, 0.746, 0.794])
    # utilities16.append([ 0.317, 0.184, 0.161, 0.206, 0.242])
    # utilities32.append([ 0.186, 0.102, 0.087, 0.131, 0.145])
    # if not use_lambdas:
    #     infos.append([-0.18, -0.221, -0.224, -0.206, -0.188])
    # else:
    #     infos.append([1.0 for _ in range(5)])  # Hacking in lambdas
    # sizes.append(small_size)
    #
    # labels.append('$\lambda_I = 1.5; n = 2$')
    # utilities2.append([0.854, 0.712, 0.7, 0.754, 0.703])
    # utilities16.append([0.303, 0.152, 0.21, 0.215, 0.184])
    # utilities32.append([ 0.185, 0.08, 0.12, 0.146, 0.11])
    # if not use_lambdas:
    #     infos.append([-0.174, -0.221, -0.173, -0.208, -0.191])
    # else:
    #     infos.append([1.5 for _ in range(5)])  # Hacking in lambdas
    # sizes.append(small_size)
    #
    # labels.append('$\lambda_I = 2.0; n = 2$')
    # utilities2.append([0.883, 0.798, 0.654, 0.829, 0.813])
    # utilities16.append([0.356, 0.289, 0.19, 0.258, 0.282])
    # utilities32.append([ 0.218, 0.169, 0.126, 0.152, 0.185])
    # if not use_lambdas:
    #     infos.append([-0.163, -0.164, -0.191, -0.189, -0.176])
    # else:
    #     infos.append([2.0 for _ in range(5)])  # Hacking in lambdas
    # sizes.append(small_size)
    #
    # labels.append('$\lambda_I = 3.0; n = 2$')
    # utilities2.append([0.879, 0.843, 0.728, 0.85, 0.852])
    # utilities16.append([0.342, 0.359, 0.2, 0.33, 0.354])
    # utilities32.append([0.225, 0.218, 0.126, 0.224, 0.245])
    # if not use_lambdas:
    #     infos.append([ -0.156, -0.139, -0.173, -0.146, -0.151])
    # else:
    #     infos.append([3.0 for _ in range(5)])  # Hacking in lambdas
    # sizes.append(small_size)
    #
    # labels.append('$\lambda_I = 10; n = 2$')
    # utilities2.append([0.911, 0.883, 0.798, 0.905, 0.897])
    # utilities16.append([0.435, 0.396, 0.299, 0.434, 0.452])
    # utilities32.append([0.299, 0.255, 0.2, 0.297, 0.302])
    # if not use_lambdas:
    #     infos.append([-0.125, -0.116, -0.132, -0.117, -0.126])
    # else:
    #     infos.append([10 for _ in range(5)])  # Hacking in lambdas
    # sizes.append(small_size)
    #
    # labels.append('$\lambda_I = 100; n = 2$')
    # utilities2.append([0.915, 0.897, 0.812, 0.931, 0.944])
    # utilities16.append([ 0.469, 0.443, 0.311, 0.519, 0.602])
    # utilities32.append([ 0.347, 0.294, 0.22, 0.384, 0.47])
    # if not use_lambdas:
    #     infos.append([-0.104, -0.099, -0.105, -0.1, -0.1])
    # else:
    #     infos.append([100 for _ in range(5)])  # Hacking in lambdas
    # sizes.append(small_size)
    #
    # vqvib2_util2 = utilities2
    # vqvib2_util16 = utilities16
    # vqvib2_util32 = utilities32
    # vqvib2_comps = infos
    # utilities2 = []
    # utilities16 = []
    # utilities32 = []
    # infos = []

    ############################################################################
    #####         VQ-VIB2  N = 4
    ############################################################################
    n = 4
    labels.append('$\lambda_I = 0.1; n = 4$')
    utilities2.append([0.495, 0.49, 0.482, 0.495, 0.486])
    utilities16.append([0.062, 0.062, 0.052, 0.055, 0.064])
    utilities32.append([ 0.032, 0.029, 0.033, 0.029, 0.024])
    if not use_lambdas:
        infos.append([ -0.375, -0.364, -0.355, -0.308, -0.318])
    else:
        infos.append([0.1 for _ in range(5)])  # Hacking in lambdas
    sizes.append(small_size)

    labels.append('$\lambda_I = 0.5; n = 4$')
    utilities2.append([0.825, 0.808, 0.745, 0.777, 0.76])
    utilities16.append([0.245, 0.273, 0.231, 0.184, 0.219])
    utilities32.append([0.141, 0.148, 0.148, 0.108, 0.146])
    if not use_lambdas:
        infos.append([ -0.191, -0.176, -0.192, -0.24, -0.197])
    else:
        infos.append([0.5 for _ in range(5)])  # Hacking in lambdas
    sizes.append(small_size)

    labels.append('$\lambda_I = 1; n = 4$')
    utilities2.append([0.877, 0.83, 0.775, 0.845, 0.825])
    utilities16.append([0.326, 0.326, 0.27, 0.345, 0.311])
    utilities32.append([0.199, 0.198, 0.173, 0.222, 0.185])
    if not use_lambdas:
        infos.append([ -0.164, -0.153, -0.164, -0.151, -0.171])
    else:
        infos.append([1.0 for _ in range(5)])  # Hacking in lambdas
    sizes.append(small_size)

    labels.append('$\lambda_I = 1.5; n = 4$')
    utilities2.append([0.904, 0.875, 0.817, 0.859, 0.907])
    utilities16.append([0.408, 0.391, 0.302, 0.34, 0.454])
    utilities32.append([0.286, 0.24, 0.199, 0.229, 0.313])
    if not use_lambdas:
        infos.append([ -0.143, -0.136, -0.142, -0.142, -0.141])
    else:
        infos.append([1.5 for _ in range(5)])  # Hacking in lambdas
    sizes.append(small_size)

    labels.append('$\lambda_I = 2.0; n = 4$')
    utilities2.append([0.904, 0.841, 0.814, 0.879, 0.903])
    utilities16.append([0.401, 0.347, 0.308, 0.375, 0.462])
    utilities32.append([0.28, 0.2, 0.195, 0.255, 0.328])
    if not use_lambdas:
        infos.append([ -0.133, -0.14, -0.136, -0.132, -0.131])
    else:
        infos.append([2.0 for _ in range(5)])  # Hacking in lambdas
    sizes.append(small_size)

    labels.append('$\lambda_I = 3.0; n = 4$')
    utilities2.append([ 0.91, 0.889, 0.818, 0.89, 0.893])
    utilities16.append([0.465, 0.427, 0.32, 0.405, 0.457])
    utilities32.append([0.324, 0.292, 0.203, 0.286, 0.329])
    if not use_lambdas:
        infos.append([ -0.121, -0.113, -0.124, -0.117, -0.124])
    else:
        infos.append([3.0 for _ in range(5)])  # Hacking in lambdas
    sizes.append(small_size)

    labels.append('$\lambda_I = 10; n = 4$')
    utilities2.append([0.936, 0.909, 0.865, 0.916, 0.929])
    utilities16.append([0.518, 0.471, 0.404, 0.492, 0.562])
    utilities32.append([ 0.386, 0.348, 0.288, 0.366, 0.422])
    if not use_lambdas:
        infos.append([ -0.099, -0.092, -0.099, -0.098, -0.096])
    else:
        infos.append([10 for _ in range(5)])  # Hacking in lambdas
    sizes.append(small_size)

    labels.append('$\lambda_I = 100; n = 4$')
    utilities2.append([ 0.935, 0.941, 0.873, 0.946, 0.97])
    utilities16.append([ 0.561, 0.58, 0.454, 0.621, 0.736])
    utilities32.append([0.42, 0.43, 0.342, 0.497, 0.61])
    if not use_lambdas:
        infos.append([ -0.082, -0.075, -0.083, -0.08, -0.079])
    else:
        infos.append([100 for _ in range(5)])  # Hacking in lambdas
    sizes.append(small_size)

    # vqvib2_util2 = utilities2
    # vqvib2_util16 = utilities16
    # vqvib2_util32 = utilities32
    # vqvib2_comps = infos
    # utilities2 = []
    # utilities16 = []
    # utilities32 = []
    # infos = []

    xlabel = 'Distortion (MSE)' if not use_lambdas else '$\lambda_I$'

    # for data, suffix in zip([utilities2, utilities16, utilities32], ['2', '16', '32']):
    #     plot_multi_trials([infos, data], labels, sizes, ylabel='Utility', colors=None, filename='ood_comp' + suffix + '.png')

    # sizes = [2 * small_size for _ in utilities2] + [2 * small_size for _ in utilities16] + [2 * small_size for _ in utilities32]
    # For just a single architecture.
    labels = ['$C=2$'] + ['' for _ in utilities2[:-1]] + ['$C=16$'] + ['' for _ in utilities16[:-1]] + ['$C=32$'] + ['' for _ in utilities32[:-1]]
    plot_multi_trials([infos + infos + infos, utilities2 + utilities16 + utilities32], labels, [small_size for _ in range(len(sizes) * 3)], ylabel='OOD Utility', xlabel=xlabel, colors=None, filename='ood_info_vqvib_n' + str(n) + '.png')

    labels = ['Onehot'] + ['' for _ in onehot_comps[:-1]] + \
             ['Proto.'] + ['' for _ in proto_comps[:-1]] + \
             ['VQ-VIB$_{\mathcal{N}}$'] + ['' for _ in vq_comps[:-1]] +\
             ['VQ-VIB$_{\mathcal{C}}$'] + ['' for _ in vqvib2_comps[:-1]]
    sizes = [small_size for _ in labels]
    num_candidates = 2
    filename = 'ood_info_all_multimodel_' + str(num_candidates)
    filename = filename + '_lambda' if use_lambdas else filename
    if num_candidates == 2:
        proto_util = proto_util2
        onehot_util = onehot_util2
        # onehot2_util = onehot2_util2
        # onehot4_util = onehot4_util2
        vq_util = vq_util2
        vqvib2_util = vqvib2_util2
    elif num_candidates == 16:
        proto_util = proto_util16
        onehot_util = onehot_util16
        # onehot2_util = onehot2_util16
        # onehot4_util = onehot4_util16
        vq_util = vq_util16
        vqvib2_util = vqvib2_util16
    elif num_candidates == 32:
        proto_util = proto_util32
        onehot_util = onehot_util32
        # onehot2_util = onehot2_util32
        # onehot4_util = onehot4_util32
        vq_util = vq_util32
        vqvib2_util = vqvib2_util32

    plot_multi_trials([onehot_comps + proto_comps + vq_comps + vqvib2_comps, onehot_util + proto_util + vq_util + vqvib2_util], labels, sizes,
                      ylabel='OOD Utility', xlabel=xlabel, colors=None, filename=('%s.png' % filename))
    # plot_multi_trials([proto_comps, proto_util], labels, sizes,
    #                   ylabel='OOD Utility', xlabel=xlabel, colors=None, filename=('%s.png' % filename))

    # labels = ['Onehot'] + ['' for _ in proto_comps[:-1]] + \
    #          ['Proto.'] + ['' for _ in proto_comps[:-1]] + \
    #          ['VQ-VIB$_{\mathcal{N}}$'] + ['' for _ in vq_comps[:-1]]
    # sizes = [small_size for _ in range(3 * len(proto_comps))]
    # plot_multi_trials([onehot_comps + proto_comps + vq_comps, onehot_util + proto_util + vq_util], labels, sizes,
    #                   ylabel='OOD Utility', colors=None, filename='ood_comp_all_multimodel_' + str(num_candidates) + '.png')

    return infos, utilities2, utilities16, utilities32


if __name__ == '__main__':
    run()
