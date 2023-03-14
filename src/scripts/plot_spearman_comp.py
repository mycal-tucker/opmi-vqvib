from src.utils.plotting import plot_multi_trials


def run():
    labels = []
    alignments = []
    utilities16 = []
    utilities32 = []
    infos = []
    sizes = []
    colors = []
    small_size = 100
    ############################################################################
    #####           Proto
    ############################################################################
    n = 'proto'
    labels.append('$\lambda_I = 0 Proto')
    alignments.append([0.096, 0.044, 0.037, 0.035, 0.024])
    infos.append([-0.362, -0.358, -0.356, -0.367, -0.364])
    sizes.append(small_size)

    labels.append('$\lambda_I = 0.1 Proto')
    alignments.append([0.097, 0.045, 0.037, 0.026, 0.023])
    infos.append([ -0.322, -0.373, -0.347, -0.32, -0.323])
    sizes.append(small_size)

    labels.append('$\lambda_I = 0.5 Proto')
    alignments.append([0.095, 0.024, 0.038, 0.031, 0.023])
    infos.append([-0.294, -0.333, -0.294, -0.317, -0.271])
    sizes.append(small_size)

    labels.append('$\lambda_I = 1.0 Proto')
    alignments.append([0.302, -0.008, -0.149, 0.077, 0.126])
    infos.append([ -0.281, -0.331, -0.256, -0.253, -0.265])
    sizes.append(small_size)

    labels.append('$\lambda_I = 1.5 Proto')
    alignments.append([0.329, 0.021, -0.056, 0.373, 0.12])
    infos.append([-0.281, -0.331, -0.256, -0.253, -0.265])
    sizes.append(small_size)

    labels.append('$\lambda_I = 2.0 Proto')
    alignments.append([0.346, 0.034, -0.022, 0.352, 0.091])
    infos.append([ -0.253, -0.326, -0.249, -0.248, -0.246])
    sizes.append(small_size)

    labels.append('$\lambda_I = 3.0 Proto')
    alignments.append([0.344, 0.14, -0.002, 0.282, 0.068])
    infos.append([-0.252, -0.296, -0.244, -0.233, -0.24])
    sizes.append(small_size)

    labels.append('$\lambda_I = 10.0 Proto')
    alignments.append([0.361, 0.19, -0.016, 0.266, 0.05])
    infos.append([-0.225, -0.253, -0.226, -0.208, -0.218])
    sizes.append(small_size)


    ############################################################################
    #####           Onehot
    ############################################################################
    n = 'onehot'
    labels.append('$\lambda_I = 0 Onehot')
    alignments.append([0.093, 0.056, 0.091, 0.091, 0.096])
    infos.append([-0.35787592064386037, -0.35398460166198664, -0.35506191779680624, -0.3578795228725507, -0.3601705696813769])
    sizes.append(small_size)

    labels.append('$\lambda_I = 0.1 Onehot')
    alignments.append([0.116, 0.153, -0.041, 0.093, 0.091])
    infos.append([-0.317, -0.286, -0.266, -0.347, -0.311])
    sizes.append(small_size)

    labels.append('$\lambda_I = 0.5 Onehot')
    alignments.append([0.197, 0.277, 0.196, 0.107, 0.128])
    infos.append([-0.214, -0.245, -0.204, -0.266, -0.222])
    sizes.append(small_size)

    labels.append('$\lambda_I = 1.0 Onehot')
    alignments.append([0.207, 0.156, 0.05, 0.251, 0.006])
    infos.append([-0.194, -0.274, -0.242, -0.229, -0.254])
    sizes.append(small_size)

    labels.append('$\lambda_I = 1.5 Onehot')
    alignments.append([0.16, -0.005, 0.195, 0.14, -0.086])
    infos.append([-0.203, -0.248, -0.212, -0.221, -0.262])
    sizes.append(small_size)

    labels.append('$\lambda_I = 2.0 Onehot')
    alignments.append([0.245, 0.168, 0.099, 0.118, 0.151])
    infos.append([-0.193, -0.237, -0.219, -0.234, -0.21])
    sizes.append(small_size)

    labels.append('$\lambda_I = 3.0 Onehot')
    alignments.append([0.157, 0.149, 0.12, 0.141, 0.157])
    infos.append([-0.236, -0.243, -0.191, -0.2, -0.197])
    sizes.append(small_size)

    labels.append('$\lambda_I = 10.0 Onehot')
    alignments.append([0.209, 0.174, 0.111, 0.112, 0.096])
    infos.append([-0.177, -0.215, -0.191, -0.199, -0.216])
    sizes.append(small_size)
    ############################################################################
    #####         VQ-VIB  N = 1
    ############################################################################
    n = 1
    labels.append('$\lambda_I = 0; n = 1$')
    alignments.append([0.181, 0.009, 0.033, 0.033, 0.025])
    infos.append([-0.39741656019806076, -0.353978157233614, -0.35842789156572397, -0.3610487512607027, -0.3579979256632083])
    sizes.append(small_size)
    colors.append('xkcd:blue')

    labels.append('$\lambda_I = 0.1; n = 1$')
    alignments.append([0.378, 0.341, 0.395, 0.123, 0.242])
    infos.append([ -0.251, -0.247, -0.234, -0.274, -0.239])
    sizes.append(small_size)
    colors.append('xkcd:blue')

    labels.append('$\lambda_I = 0.5; n = 1$')
    alignments.append([0.379, 0.406, 0.347, 0.245, 0.288])
    infos.append([ -0.189, -0.202, -0.197, -0.199, -0.182])
    sizes.append(small_size)
    colors.append('xkcd:blue')

    labels.append('$\lambda_I = 1; n = 1$')
    alignments.append([0.373, 0.368, 0.332, 0.285, 0.321])
    infos.append([ -0.168, -0.164, -0.167, -0.174, -0.167])
    sizes.append(small_size)
    colors.append('xkcd:blue')

    labels.append('$\lambda_I = 1.5; n = 1$')
    alignments.append([0.366, 0.249, 0.295, 0.205, 0.2])
    infos.append([ -0.158, -0.149, -0.155, -0.161, -0.156 ])
    sizes.append(small_size)
    colors.append('xkcd:blue')

    labels.append('$\lambda_I = 2.0; n = 1$')
    alignments.append([0.277, 0.245, 0.281, 0.213, 0.243])
    infos.append([-0.158, -0.15, -0.149, -0.149, -0.151])
    sizes.append(small_size)
    colors.append('xkcd:blue')

    labels.append('$\lambda_I = 3.0; n = 1$')
    alignments.append([0.184, 0.252, 0.239, 0.17, 0.202])
    infos.append([ -0.149, -0.141, -0.147, -0.142, -0.141])
    sizes.append(small_size)
    colors.append('xkcd:blue')

    labels.append('$\lambda_I = 10; n = 1$')
    alignments.append([0.129, 0.224, 0.24, 0.206, 0.123])
    infos.append([ -0.128, -0.125, -0.127, -0.127, -0.124])
    sizes.append(small_size)
    colors.append('xkcd:blue')

    ############################################################################
    #####         VQ-VIB  N = 2
    ############################################################################
    # n = 2
    # labels.append('$\lambda_I = 0; n = 2$')
    # alignments.append([0.031, 0.012, 0.043, 0.03, 0.017])
    # infos.append(
    #     [-0.36113061682465586, -0.35627751491915355, -0.3623669310914216, -0.3583042439979138, -0.3607637684645006])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 0.1; n = 2$')
    # alignments.append([0.357, 0.252, 0.354, 0.217, 0.194])
    # infos.append([-0.212, -0.242, -0.218, -0.245, -0.224])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 0.5; n = 2$')
    # alignments.append([0.339, 0.324, 0.315, 0.232, 0.338])
    # infos.append([ -0.159, -0.167, -0.163, -0.171, -0.164])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 1; n = 2$')
    # alignments.append([0.33, 0.285, 0.26, 0.268, 0.302])
    # infos.append([ -0.135, -0.132, -0.135, -0.138, -0.138])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 1.5; n = 2$')
    # alignments.append([0.219, 0.295, 0.312, 0.232, 0.226])
    # infos.append([ -0.119, -0.121, -0.12, -0.122, -0.122])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 2.0; n = 2$')
    # alignments.append([0.249, 0.253, 0.219, 0.246, 0.157])
    # infos.append([ -0.109, -0.109, -0.112, -0.108, -0.114])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 3.0; n = 2$')
    # alignments.append([0.188, 0.25, 0.201, 0.175, 0.167])
    # infos.append([ -0.1, -0.095, -0.101, -0.098, -0.1])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 10; n = 2$')
    # alignments.append([0.117, 0.194, 0.16, 0.137, 0.115])
    # infos.append([ -0.081, -0.081, -0.083, -0.081, -0.082])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')

    ############################################################################
    #####         VQ-VIB  N = 4
    ############################################################################
    # n = 4
    # labels.append('$\lambda_I = 0; n = 4$')
    # alignments.append([0.315, 0.014, 0.173, 0.028, 0.028])
    # infos.append(
    #     [-0.33291103337716754, -0.35574541812638083, -0.31681010496121126, -0.3570997573599156, -0.3498205273278654])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 0.1; n = 4$')
    # alignments.append([0.334, 0.261, 0.331, 0.196, 0.282])
    # infos.append([ -0.196, -0.212, -0.202, -0.248, -0.199])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 0.5; n = 4$')
    # alignments.append([0.269, 0.308, 0.26, 0.255, 0.222])
    # infos.append([ -0.137, -0.144, -0.138, -0.143, -0.141])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 1; n = 4$')
    # alignments.append([0.187, 0.274, 0.316, 0.244, 0.186])
    # infos.append([ -0.107, -0.1, -0.113, -0.109, -0.107])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 1.5; n = 4$')
    # alignments.append([0.202, 0.269, 0.238, 0.193, 0.141])
    # infos.append([ -0.092, -0.091, -0.083, -0.083, -0.088])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 2.0; n = 4$')
    # alignments.append([0.212, 0.253, 0.248, 0.25, 0.165])
    # infos.append([ -0.083, -0.076, -0.07, -0.082, -0.075])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 3.0; n = 4$')
    # alignments.append([0.188, 0.222, 0.225, 0.236, 0.2])
    # infos.append([-0.062, -0.061, -0.063, -0.061, -0.063])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 10; n = 4$')
    # alignments.append([0.162, 0.206, 0.223, 0.187, 0.266])
    # infos.append([ -0.042, -0.041, -0.044, -0.041, -0.045])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')

    labels = ['$n=1$'] + ['' for _ in range(7)] + ['$n=2$'] + ['' for _ in range(7)] + ['$n=4$'] + ['' for _ in range(7)]
    plot_multi_trials([infos, alignments], labels, sizes, ylabel='Alignment ($\\rho$)', xlabel='Distortion (MSE)', colors=None, filename='spearman_comp.png')

    # labels = ['Onehot'] + ['' for _ in onehot_comps[:-1]] +\
    #          ['Proto.'] + ['' for _ in proto_comps[:-1]] +\
    #          ['VQ-VIB'] + ['' for _ in vq_comps[:-1]]
    # sizes = [2 * small_size for _ in onehot_comps] + [2 * small_size for _ in proto_comps] + [2 * small_size for _ in vq_comps]
    # plot_multi_trials([onehot_comps + proto_comps + vq_comps, onehot_util2 + proto_util2 + vq_util2], labels, sizes,
    #                   ylabel='OOD Utility', colors=None, filename='ood_lambda_all_multimodel_2.png')



if __name__ == '__main__':
    run()
