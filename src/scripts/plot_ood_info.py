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
    n = 'proto'
    labels.append('$\lambda_I = 0.1 Proto')
    utilities2.append([ 0.49, 0.497, 0.491, 0.506, 0.483])
    utilities16.append([ 0.07, 0.068, 0.056, 0.071, 0.064])
    utilities32.append([ 0.03, 0.03, 0.03, 0.034, 0.04])
    if not use_lambdas:
        infos.append([ -0.322, -0.373, -0.347, -0.32, -0.323])
    else:
        infos.append([0.1 for _ in range(5)])  # Hacking in lambdas
    sizes.append(small_size)

    labels.append('$\lambda_I = 0.5$ proto')
    utilities2.append([ 0.491, 0.5, 0.492, 0.505, 0.482])
    utilities16.append([ 0.069, 0.069, 0.057, 0.072, 0.062])
    utilities32.append([ 0.03, 0.03, 0.03, 0.033, 0.039])
    if not use_lambdas:
        infos.append([ -0.322, -0.373, -0.347, -0.32, -0.323])
    else:
        infos.append([0.5 for _ in range(5)])  # Hacking in lambdas
    sizes.append(small_size)

    labels.append('$\lambda_I = 1.0$ proto')
    utilities2.append([ 0.705, 0.724, 0.712, 0.504, 0.573])
    utilities16.append([  0.137, 0.133, 0.126, 0.072, 0.101])
    utilities32.append([ 0.063, 0.066, 0.074, 0.034, 0.059])
    if not use_lambdas:
        infos.append([-0.294, -0.333, -0.294, -0.317, -0.271])
    else:
        infos.append([1.0 for _ in range(5)])  # Hacking in lambdas
    sizes.append(small_size)

    labels.append('$\lambda_I = 1.5$ proto')
    utilities2.append([ 0.721, 0.745, 0.689, 0.604, 0.541])
    utilities16.append([ 0.149, 0.138, 0.153, 0.112, 0.097])
    utilities32.append([ 0.085, 0.07, 0.099, 0.056, 0.05])
    if not use_lambdas:
        infos.append([ -0.281, -0.331, -0.256, -0.253, -0.265])
    else:
        infos.append([1.5 for _ in range(5)])  # Hacking in lambdas
    sizes.append(small_size)

    labels.append('$\lambda_I = 2$ proto')
    utilities2.append([  0.782, 0.745, 0.716, 0.664, 0.559])
    utilities16.append([ 0.188, 0.143, 0.175, 0.128, 0.106])
    utilities32.append([ 0.112, 0.076, 0.102, 0.072, 0.074])
    if not use_lambdas:
        infos.append([ -0.253, -0.326, -0.249, -0.248, -0.246])
    else:
        infos.append([2 for _ in range(5)])  # Hacking in lambdas
    sizes.append(small_size)

    labels.append('$\lambda_I = 3$ proto')
    utilities2.append([ 0.78, 0.786, 0.73, 0.714, 0.566])
    utilities16.append([ 0.187, 0.176, 0.189, 0.153, 0.099])
    utilities32.append([ 0.102, 0.114, 0.109, 0.096, 0.07])
    if not use_lambdas:
        infos.append([ -0.252, -0.296, -0.244, -0.233, -0.24])
    else:
        infos.append([3 for _ in range(5)])  # Hacking in lambdas
    sizes.append(small_size)

    labels.append('$\lambda_I = 10$ proto')
    utilities2.append([0.811, 0.783, 0.765, 0.74, 0.604])
    utilities16.append([ 0.236, 0.185, 0.203, 0.165, 0.12])
    utilities32.append([  0.148, 0.109, 0.125, 0.106, 0.082])
    if not use_lambdas:
        infos.append([ -0.225, -0.253, -0.226, -0.208, -0.218])
    else:
        infos.append([10 for _ in range(5)])  # Hacking in lambdas
    sizes.append(small_size)

    labels.append('$\lambda_I = 100$ proto')
    utilities2.append([ 0.818, 0.809, 0.788, 0.817, 0.725])
    utilities16.append([  0.245, 0.219, 0.239, 0.257, 0.182])
    utilities32.append([  0.17, 0.143, 0.166, 0.158, 0.12])
    if not use_lambdas:
        infos.append([ -0.212, -0.215, -0.197, -0.188, -0.2])
    else:
        infos.append([100 for _ in range(5)])  # Hacking in lambdas
    sizes.append(small_size)

    proto_util2 = utilities2
    proto_util16 = utilities16
    proto_util32 = utilities32
    proto_comps = infos
    utilities2 = []
    utilities16 = []
    utilities32 = []
    infos = []
    ############################################################################
    #####           Onehot
    ############################################################################
    n = 'onehot'
    labels.append('$\lambda_I = 0.1 Onehot')
    utilities2.append([0.499, 0.661, 0.52, 0.502, 0.486])
    utilities16.append([0.058, 0.116, 0.066, 0.057, 0.061])
    utilities32.append([ 0.026, 0.054, 0.03, 0.033, 0.029])
    if not use_lambdas:
        infos.append([ -0.317, -0.286, -0.266, -0.347, -0.311])
    else:
        infos.append([0.1 for _ in range(5)])  # Hacking in lambdas
    sizes.append(small_size)

    labels.append('$\lambda_I = 0.5 Onehot')
    utilities2.append([ 0.784, 0.793, 0.755, 0.82, 0.723])
    utilities16.append([ 0.249, 0.234, 0.217, 0.213, 0.185])
    utilities32.append([ 0.129, 0.136, 0.13, 0.114, 0.112])
    if not use_lambdas:
        infos.append([ -0.214, -0.245, -0.204, -0.266, -0.222])
    else:
        infos.append([0.5 for _ in range(5)])  # Hacking in lambdas
    sizes.append(small_size)

    labels.append('$\lambda_I = 1.0$ Onehot')
    utilities2.append([0.797, 0.617, 0.583, 0.822, 0.716])
    utilities16.append([0.253, 0.109, 0.095, 0.274, 0.134])
    utilities32.append([0.142, 0.056, 0.054, 0.151, 0.08])
    if not use_lambdas:
        infos.append([-0.194, -0.274, -0.242, -0.229, -0.254])
    else:
        infos.append([1.0 for _ in range(5)])  # Hacking in lambdas
    sizes.append(small_size)

    labels.append('$\lambda_I = 1.5$ Onehot')
    utilities2.append([0.801, 0.601, 0.799, 0.844, 0.638])
    utilities16.append([0.259, 0.14, 0.241, 0.289, 0.148])
    utilities32.append([0.142, 0.09, 0.154, 0.164, 0.097])
    if not use_lambdas:
        infos.append([ -0.203, -0.248, -0.212, -0.221, -0.262])
    else:
        infos.append([1.5 for _ in range(5)])  # Hacking in lambdas
    sizes.append(small_size)

    labels.append('$\lambda_I = 2.0 Onehot')
    utilities2.append([ 0.777, 0.785, 0.763, 0.82, 0.784])
    utilities16.append([ 0.252, 0.214, 0.219, 0.253, 0.243])
    utilities32.append([ 0.162, 0.122, 0.12, 0.131, 0.137])
    if not use_lambdas:
        infos.append([-0.193, -0.237, -0.219, -0.234, -0.21])
    else:
        infos.append([2.0 for _ in range(5)])  # Hacking in lambdas
    sizes.append(small_size)

    labels.append('$\lambda_I = 3.0 Onehot')
    utilities2.append([0.74, 0.803, 0.725, 0.862, 0.777])
    utilities16.append([0.213, 0.271, 0.237, 0.327, 0.276])
    utilities32.append([0.121, 0.156, 0.157, 0.198, 0.174])
    if not use_lambdas:
        infos.append([ -0.236, -0.243, -0.191, -0.2, -0.197])
    else:
        infos.append([3.0 for _ in range(5)])  # Hacking in lambdas
    sizes.append(small_size)

    labels.append('$\lambda_I = 10$ Onehot')
    utilities2.append([0.78, 0.818, 0.821, 0.77, 0.753])
    utilities16.append([0.262, 0.246, 0.299, 0.266, 0.208])
    utilities32.append([ 0.164, 0.152, 0.184, 0.163, 0.134])
    if not use_lambdas:
        infos.append([-0.177, -0.215, -0.191, -0.199, -0.216])
    else:
        infos.append([10 for _ in range(5)])  # Hacking in lambdas
    sizes.append(small_size)

    labels.append('$\lambda_I = 100$ Onehot')
    utilities2.append([0.706, 0.847, 0.693, 0.831, 0.752])
    utilities16.append([0.186, 0.311, 0.217, 0.259, 0.223])
    utilities32.append([0.13, 0.182, 0.141, 0.176, 0.141])
    if not use_lambdas:
        infos.append([-0.208, -0.197, -0.205, -0.21, -0.211])
    else:
        infos.append([100 for _ in range(5)])  # Hacking in lambdas
    sizes.append(small_size)

    onehot_util2 = utilities2
    onehot_util16 = utilities16
    onehot_util32 = utilities32
    onehot_comps = infos
    utilities2 = []
    utilities16 = []
    utilities32 = []
    infos = []
    ############################################################################
    #####           N = 1
    ############################################################################
    n = 1

    labels.append('$\lambda_I = 0.1; n = 1$')
    utilities2.append([0.833, 0.757, 0.71, 0.58, 0.695])
    utilities16.append([0.265, 0.22, 0.202, 0.112, 0.197])
    utilities32.append([0.146, 0.123, 0.131, 0.056, 0.12])
    if not use_lambdas:
        infos.append([ -0.251, -0.247, -0.234, -0.274, -0.239])
    else:
        infos.append([0.1 for _ in range(5)])  # Hacking in lambdas
    sizes.append(small_size)

    labels.append('$\lambda_I = 0.5; n = 1$')
    utilities2.append([0.873, 0.756, 0.775, 0.806, 0.853])
    utilities16.append([0.329, 0.266, 0.267, 0.278, 0.339])
    utilities32.append([0.207, 0.143, 0.176, 0.182, 0.225])
    if not use_lambdas:
        infos.append([ -0.189, -0.202, -0.197, -0.199, -0.182])
    else:
        infos.append([0.5 for _ in range(5)])  # Hacking in lambdas
    sizes.append(small_size)

    labels.append('$\lambda_I = 1; n = 1$')
    utilities2.append([ 0.892, 0.856, 0.822, 0.846, 0.872])
    utilities16.append([0.431, 0.343, 0.323, 0.338, 0.386])
    utilities32.append([0.244, 0.218, 0.208, 0.186, 0.262])
    if not use_lambdas:
        infos.append([ -0.168, -0.164, -0.167, -0.174, -0.167])
    else:
        infos.append([1 for _ in range(5)])  # Hacking in lambdas
    sizes.append(small_size)

    labels.append('$\lambda_I = 1.5; n = 1$')
    utilities2.append([0.892, 0.87, 0.821, 0.84, 0.871])
    utilities16.append([0.437, 0.389, 0.32, 0.377, 0.411])
    utilities32.append([0.288, 0.252, 0.206, 0.232, 0.266])
    if not use_lambdas:
        infos.append([ -0.158, -0.149, -0.155, -0.161, -0.156 ])
    else:
        infos.append([1.5 for _ in range(5)])  # Hacking in lambdas
    sizes.append(small_size)

    labels.append('$\lambda_I = 2.0; n = 1$')
    utilities2.append([0.888, 0.876, 0.8, 0.868, 0.89])
    utilities16.append([0.429, 0.383, 0.317, 0.373, 0.432])
    utilities32.append([0.281, 0.261, 0.218, 0.241, 0.305])
    if not use_lambdas:
        infos.append([-0.158, -0.15, -0.149, -0.149, -0.151])
    else:
        infos.append([2 for _ in range(5)])  # Hacking in lambdas
    sizes.append(small_size)

    labels.append('$\lambda_I = 3.0; n = 1$')
    utilities2.append([0.896, 0.881, 0.784, 0.86, 0.896])
    utilities16.append([0.44, 0.407, 0.321, 0.376, 0.478])
    utilities32.append([ 0.289, 0.256, 0.207, 0.244, 0.332])
    if not use_lambdas:
        infos.append([ -0.149, -0.141, -0.147, -0.142, -0.141])
    else:
        infos.append([3 for _ in range(5)])  # Hacking in lambdas
    sizes.append(small_size)

    labels.append('$\lambda_I = 10; n = 1$')
    utilities2.append([0.897, 0.848, 0.814, 0.882, 0.912])
    utilities16.append([0.473, 0.354, 0.344, 0.417, 0.5])
    utilities32.append([0.328, 0.246, 0.232, 0.292, 0.365])
    if not use_lambdas:
        infos.append([ -0.128, -0.125, -0.127, -0.127, -0.124])
    else:
        infos.append([10 for _ in range(5)])  # Hacking in lambdas
    sizes.append(small_size)

    labels.append('$\lambda_I = 100; n = 1$')
    utilities2.append([ 0.905, 0.871, 0.793, 0.863, 0.912])
    utilities16.append([  0.417, 0.357, 0.279, 0.364, 0.453])
    utilities32.append([ 0.276, 0.225, 0.19, 0.251, 0.316])
    if not use_lambdas:
        infos.append([ -0.154, -0.168, -0.142, -0.132])
    else:
        infos.append([100 for _ in range(5)])  # Hacking in lambdas
    sizes.append(small_size)

    vq_util2 = utilities2
    vq_util16 = utilities16
    vq_util32 = utilities32
    vq_comps = infos
    utilities2 = []
    utilities16 = []
    utilities32 = []
    infos = []

    ############################################################################
    #####         VQ-VIB2  N = 1
    ############################################################################
    n = 1
    labels.append('$\lambda_I = 0.1; n = 1$')
    utilities2.append([ 0.503, 0.489, 0.482, 0.494, 0.503])
    utilities16.append([ 0.064, 0.07, 0.061, 0.052, 0.052])
    utilities32.append([ 0.028, 0.032, 0.03, 0.031, 0.032])
    if not use_lambdas:
        infos.append([  -0.373, -0.362, -0.349, -0.304])
    else:
        infos.append([0.1 for _ in range(5)])  # Hacking in lambdas
    sizes.append(small_size)

    labels.append('$\lambda_I = 0.5; n = 1$')
    utilities2.append([ 0.514, 0.516, 0.491, 0.496, 0.503])
    utilities16.append([ 0.058, 0.062, 0.075, 0.053, 0.059])
    utilities32.append([ 0.029, 0.032, 0.032, 0.033, 0.029])
    if not use_lambdas:
        infos.append([ -0.374, -0.365, -0.345, -0.307, -0.318])
    else:
        infos.append([0.5 for _ in range(5)])  # Hacking in lambdas
    sizes.append(small_size)

    labels.append('$\lambda_I = 1; n = 1$')
    utilities2.append([ 0.548, 0.52, 0.642, 0.532, 0.613])
    utilities16.append([ 0.106, 0.078, 0.123, 0.107, 0.137])
    utilities32.append([ 0.057, 0.041, 0.056, 0.045, 0.082])
    if not use_lambdas:
        infos.append([ -0.274, -0.297, -0.278, -0.277, -0.266])  # Done
    else:
        infos.append([1.0 for _ in range(5)])  # Hacking in lambdas
    sizes.append(small_size)

    labels.append('$\lambda_I = 1.5; n = 1$')
    utilities2.append([ 0.77, 0.723, 0.616, 0.73, 0.516])
    utilities16.append([ 0.212, 0.169, 0.134, 0.149, 0.09])
    utilities32.append([ 0.112, 0.077, 0.077, 0.076, 0.048])
    if not use_lambdas:
        infos.append([ -0.269, -0.287, -0.227, -0.264, -0.268])
    else:
        infos.append([1.5 for _ in range(5)])  # Hacking in lambdas
    sizes.append(small_size)

    labels.append('$\lambda_I = 2.0; n = 1$')
    utilities2.append([ 0.8, 0.515, 0.633, 0.745, 0.502])
    utilities16.append([ 0.198, 0.061, 0.13, 0.167, 0.06])
    utilities32.append([ 0.111, 0.033, 0.065, 0.085, 0.029])
    if not use_lambdas:
        infos.append([ -0.237, -0.365, -0.287, -0.263, -0.318])
    else:
        infos.append([2.0 for _ in range(5)])  # Hacking in lambdas
    sizes.append(small_size)

    labels.append('$\lambda_I = 3.0; n = 1$')
    utilities2.append([ 0.808, 0.627, 0.505, 0.692, 0.576])
    utilities16.append([ 0.238, 0.117, 0.087, 0.165, 0.131])
    utilities32.append([ 0.139, 0.062, 0.055, 0.093, 0.083])
    if not use_lambdas:
        infos.append([ -0.228, -0.285, -0.285, -0.237, -0.213])
    else:
        infos.append([3.0 for _ in range(5)])  # Hacking in lambdas
    sizes.append(small_size)

    labels.append('$\lambda_I = 10; n = 1$')
    utilities2.append([ 0.829, 0.838, 0.758, 0.827, 0.791])
    utilities16.append([ 0.286, 0.327, 0.216, 0.3, 0.266])
    utilities32.append([ 0.158, 0.207, 0.139, 0.183, 0.163])
    if not use_lambdas:
        infos.append([ -0.183, -0.183, -0.188, -0.179, -0.173])
    else:
        infos.append([10 for _ in range(5)])  # Hacking in lambdas
    sizes.append(small_size)

    labels.append('$\lambda_I = 100; n = 1$')
    utilities2.append([ 0.895, 0.873, 0.812, 0.881, 0.909])
    utilities16.append([ 0.433, 0.362, 0.324, 0.405, 0.437])
    utilities32.append([ 0.27, 0.237, 0.219, 0.261, 0.318])
    if not use_lambdas:
        infos.append([ -0.147, -0.151, -0.142, -0.147, -0.138])
    else:
        infos.append([100 for _ in range(5)])  # Hacking in lambdas
    sizes.append(small_size)

    vqvib2_util2 = utilities2
    vqvib2_util16 = utilities16
    vqvib2_util32 = utilities32
    vqvib2_comps = infos
    utilities2 = []
    utilities16 = []
    utilities32 = []
    infos = []


    for data, suffix in zip([utilities2, utilities16, utilities32], ['2', '16', '32']):
        plot_multi_trials([infos, data], labels, sizes, ylabel='Utility', colors=None, filename='ood_comp' + suffix + '.png')
    labels = ['$C=2$'] + ['' for _ in utilities2[:-1]] + ['$C=16$'] + ['' for _ in utilities16[:-1]] + ['$C=32$'] + ['' for _ in utilities32[:-1]]
    sizes = [2 * small_size for _ in utilities2] + [2 * small_size for _ in utilities16] + [2 * small_size for _ in utilities32]
    plot_multi_trials([infos + infos + infos, utilities2 + utilities16 + utilities32], labels, sizes, ylabel='OOD Utility', colors=None, filename='ood_info_all_n' + str(n) + '.png')

    labels = ['Onehot'] + ['' for _ in onehot_comps[:-1]] + \
             ['Proto.'] + ['' for _ in proto_comps[:-1]] + \
             ['VQ-VIB$_{\mathregular{norm.}}$'] + ['' for _ in vq_comps[:-1]] +\
            ['VQ-VIB$_{\mathregular{cat.}}$'] + ['' for _ in vqvib2_comps[:-1]]
    sizes = [small_size for _ in onehot_comps] + [small_size for _ in proto_comps] + [small_size for _ in vq_comps] + [small_size for _ in vqvib2_comps]
    num_candidates = 16
    filename = 'ood_info_all_multimodel_' + str(num_candidates)
    filename = filename + '_lambda' if use_lambdas else filename
    xlabel = 'Distortion (MSE)' if not use_lambdas else '$\lambda_I$'
    if num_candidates == 2:
        proto_util = proto_util2
        onehot_util = onehot_util2
        vq_util = vq_util2
        vqvib2_util = vqvib2_util2
    elif num_candidates == 16:
        proto_util = proto_util16
        onehot_util = onehot_util16
        vq_util = vq_util16
        vqvib2_util = vqvib2_util16
    elif num_candidates == 32:
        proto_util = proto_util32
        onehot_util = onehot_util32
        vq_util = vq_util32
        vqvib2_util = vqvib2_util32

    plot_multi_trials([onehot_comps + proto_comps + vq_comps + vqvib2_comps, onehot_util + proto_util + vq_util + vqvib2_util], labels, sizes,
                      ylabel='OOD Utility', xlabel=xlabel, colors=None, filename=('%s.png' % filename))

    # labels = ['Proto.'] + ['' for _ in proto_comps[:-1]] +\
    #          ['VQ-VIB'] + ['' for _ in vq_comps[:-1]]
    # sizes = [2 * small_size for _ in proto_comps] + [2 * small_size for _ in vq_comps]
    # plot_multi_trials([proto_comps + vq_comps, proto_util16 + vq_util16], labels, sizes,
    #                   ylabel='OOD Utility', colors=None, filename='ood_comp_all_multimodel_16.png')

    return infos, utilities2, utilities16, utilities32


if __name__ == '__main__':
    run()
