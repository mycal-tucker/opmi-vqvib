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
    utilities2.append([])
    utilities16.append([])
    utilities32.append([])
    if not use_lambdas:
        infos.append([ -0.322, -0.373, -0.347, -0.32, -0.323])
    else:
        infos.append([0.1 for _ in range(5)])  # Hacking in lambdas
    sizes.append(small_size)

    labels.append('$\lambda_I = 0.5$ proto')
    utilities2.append([])
    utilities16.append([])
    utilities32.append([])
    if not use_lambdas:
        infos.append([ -0.322, -0.373, -0.347, -0.32, -0.323])
    else:
        infos.append([0.5 for _ in range(5)])  # Hacking in lambdas
    sizes.append(small_size)

    labels.append('$\lambda_I = 1.0$ proto')
    utilities2.append([])
    utilities16.append([])
    utilities32.append([])
    if not use_lambdas:
        infos.append([-0.294, -0.333, -0.294, -0.317, -0.271])
    else:
        infos.append([1.0 for _ in range(5)])  # Hacking in lambdas
    sizes.append(small_size)

    labels.append('$\lambda_I = 1.5$ proto')
    utilities2.append([])
    utilities16.append([])
    utilities32.append([])
    if not use_lambdas:
        infos.append([ -0.281, -0.331, -0.256, -0.253, -0.265])
    else:
        infos.append([1.5 for _ in range(5)])  # Hacking in lambdas
    sizes.append(small_size)

    labels.append('$\lambda_I = 2$ proto')
    utilities2.append([])
    utilities16.append([])
    utilities32.append([])
    if not use_lambdas:
        infos.append([ -0.253, -0.326, -0.249, -0.248, -0.246])
    else:
        infos.append([2 for _ in range(5)])  # Hacking in lambdas
    sizes.append(small_size)

    labels.append('$\lambda_I = 3$ proto')
    utilities2.append([])
    utilities16.append([])
    utilities32.append([])
    if not use_lambdas:
        infos.append([ -0.252, -0.296, -0.244, -0.233, -0.24])
    else:
        infos.append([3 for _ in range(5)])  # Hacking in lambdas
    sizes.append(small_size)

    labels.append('$\lambda_I = 10$ proto')
    utilities2.append([])
    utilities16.append([])
    utilities32.append([])
    if not use_lambdas:
        infos.append([ -0.225, -0.253, -0.226, -0.208, -0.218])
    else:
        infos.append([10 for _ in range(5)])  # Hacking in lambdas
    sizes.append(small_size)

    labels.append('$\lambda_I = 100$ proto')
    utilities2.append([])
    utilities16.append([])
    utilities32.append([])
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
    utilities2.append([])
    utilities16.append([])
    utilities32.append([])
    if not use_lambdas:
        infos.append([ -0.317, -0.286, -0.266, -0.347, -0.311])
    else:
        infos.append([0.1 for _ in range(5)])  # Hacking in lambdas
    sizes.append(small_size)

    labels.append('$\lambda_I = 0.5 Onehot')
    utilities2.append([])
    utilities16.append([])
    utilities32.append([])
    if not use_lambdas:
        infos.append([ -0.214, -0.245, -0.204, -0.266, -0.222])
    else:
        infos.append([0.5 for _ in range(5)])  # Hacking in lambdas
    sizes.append(small_size)

    labels.append('$\lambda_I = 1.0$ Onehot')
    utilities2.append([])
    utilities16.append([])
    utilities32.append([])
    if not use_lambdas:
        infos.append([-0.194, -0.274, -0.242, -0.229, -0.254])
    else:
        infos.append([1.0 for _ in range(5)])  # Hacking in lambdas
    sizes.append(small_size)

    labels.append('$\lambda_I = 1.5$ Onehot')
    utilities2.append([])
    utilities16.append([])
    utilities32.append([])
    if not use_lambdas:
        infos.append([ -0.203, -0.248, -0.212, -0.221, -0.262])
    else:
        infos.append([1.5 for _ in range(5)])  # Hacking in lambdas
    sizes.append(small_size)

    labels.append('$\lambda_I = 2.0 Onehot')
    utilities2.append([])
    utilities16.append([])
    utilities32.append([])
    if not use_lambdas:
        infos.append([-0.193, -0.237, -0.219, -0.234, -0.21])
    else:
        infos.append([2.0 for _ in range(5)])  # Hacking in lambdas
    sizes.append(small_size)

    labels.append('$\lambda_I = 3.0 Onehot')
    utilities2.append([])
    utilities16.append([])
    utilities32.append([])
    if not use_lambdas:
        infos.append([ -0.236, -0.243, -0.191, -0.2, -0.197])
    else:
        infos.append([3.0 for _ in range(5)])  # Hacking in lambdas
    sizes.append(small_size)

    labels.append('$\lambda_I = 10$ Onehot')
    utilities2.append([])
    utilities16.append([])
    utilities32.append([])
    if not use_lambdas:
        infos.append([-0.177, -0.215, -0.191, -0.199, -0.216])
    else:
        infos.append([10 for _ in range(5)])  # Hacking in lambdas
    sizes.append(small_size)

    labels.append('$\lambda_I = 100$ Onehot')
    utilities2.append([])
    utilities16.append([])
    utilities32.append([])
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
    utilities2.append([.71, 0.77, 0.6, 0.82, 0.68])
    utilities16.append([0.1, 0.06, 0.15, 0.15, 0.12])
    utilities32.append([])
    if not use_lambdas:
        infos.append([ -0.251, -0.247, -0.234, -0.274, -0.239])
    else:
        infos.append([0.1 for _ in range(5)])  # Hacking in lambdas
    sizes.append(small_size)

    labels.append('$\lambda_I = 0.5; n = 1$')
    utilities2.append([0.68, 0.8, 0.54, 0.82, 0.67])
    utilities16.append([0.12, 0.09, 0.12, 0.11, 0.13])
    utilities32.append([])
    if not use_lambdas:
        infos.append([ -0.189, -0.202, -0.197, -0.199, -0.182])
    else:
        infos.append([0.5 for _ in range(5)])  # Hacking in lambdas
    sizes.append(small_size)

    labels.append('$\lambda_I = 1; n = 1$')
    utilities2.append([0.7, 0.81, 0.5, 0.82, 0.65])
    utilities16.append([0.09, 0.06, 0.11, 0.15, 0.13])
    utilities32.append([])
    if not use_lambdas:
        infos.append([ -0.168, -0.164, -0.167, -0.174, -0.167])
    else:
        infos.append([1 for _ in range(5)])  # Hacking in lambdas
    sizes.append(small_size)

    labels.append('$\lambda_I = 1.5; n = 1$')
    utilities2.append([0.71, 0.79, 0.53, 0.85, 0.68])
    utilities16.append([0.11, 0.08, 0.18, 0.13, 0.16])
    utilities32.append([])
    if not use_lambdas:
        infos.append([ -0.158, -0.149, -0.155, -0.161, -0.156 ])
    else:
        infos.append([1.5 for _ in range(5)])  # Hacking in lambdas
    sizes.append(small_size)

    labels.append('$\lambda_I = 2.0; n = 1$')
    utilities2.append([0.62, 0.63, 0.58, 0.7, 0.63])
    utilities16.append([0.15, 0.09, 0.12, 0.11, 0.14])
    utilities32.append([])
    if not use_lambdas:
        infos.append([-0.158, -0.15, -0.149, -0.149, -0.151])
    else:
        infos.append([2 for _ in range(5)])  # Hacking in lambdas
    sizes.append(small_size)

    labels.append('$\lambda_I = 3.0; n = 1$')
    utilities2.append([0.6, 0.62, 0.56, 0.74, 0.65])
    utilities16.append([0.15, 0.11, 0.12, 0.14, 0.16])
    utilities32.append([])
    if not use_lambdas:
        infos.append([ -0.149, -0.141, -0.147, -0.142, -0.141])
    else:
        infos.append([3 for _ in range(5)])  # Hacking in lambdas
    sizes.append(small_size)

    labels.append('$\lambda_I = 10; n = 1$')
    utilities2.append([0.67, 0.59, 0.56, 0.7, 0.7])
    utilities16.append([0.15, 0.08, 0.15, 0.17, 0.17])
    utilities32.append([])
    if not use_lambdas:
        infos.append([ -0.128, -0.125, -0.127, -0.127, -0.124])
    else:
        infos.append([10 for _ in range(5)])  # Hacking in lambdas
    sizes.append(small_size)

    labels.append('$\lambda_I = 100; n = 1$')
    utilities2.append([])
    utilities16.append([])
    utilities32.append([])
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
    utilities2.append([])
    utilities16.append([])
    utilities32.append([])
    if not use_lambdas:
        infos.append([  -0.373, -0.362, -0.349, -0.304])
    else:
        infos.append([0.1 for _ in range(5)])  # Hacking in lambdas
    sizes.append(small_size)

    labels.append('$\lambda_I = 0.5; n = 1$')
    utilities2.append([])
    utilities16.append([])
    utilities32.append([])
    if not use_lambdas:
        infos.append([ -0.374, -0.365, -0.345, -0.307, -0.318])
    else:
        infos.append([0.5 for _ in range(5)])  # Hacking in lambdas
    sizes.append(small_size)

    labels.append('$\lambda_I = 1; n = 1$')
    utilities2.append([])
    utilities16.append([])
    utilities32.append([])
    if not use_lambdas:
        infos.append([ -0.274, -0.297, -0.278, -0.277, -0.266])  # Done
    else:
        infos.append([1.0 for _ in range(5)])  # Hacking in lambdas
    sizes.append(small_size)

    labels.append('$\lambda_I = 1.5; n = 1$')
    utilities2.append([])
    utilities16.append([])
    utilities32.append([])
    if not use_lambdas:
        infos.append([ -0.269, -0.287, -0.227, -0.264, -0.268])
    else:
        infos.append([1.5 for _ in range(5)])  # Hacking in lambdas
    sizes.append(small_size)

    labels.append('$\lambda_I = 2.0; n = 1$')
    utilities2.append([])
    utilities16.append([])
    utilities32.append([])
    if not use_lambdas:
        infos.append([ -0.237, -0.365, -0.287, -0.263, -0.318])
    else:
        infos.append([2.0 for _ in range(5)])  # Hacking in lambdas
    sizes.append(small_size)

    labels.append('$\lambda_I = 3.0; n = 1$')
    utilities2.append([])
    utilities16.append([])
    utilities32.append([])
    if not use_lambdas:
        infos.append([ -0.228, -0.285, -0.285, -0.237, -0.213])
    else:
        infos.append([3.0 for _ in range(5)])  # Hacking in lambdas
    sizes.append(small_size)

    labels.append('$\lambda_I = 10; n = 1$')
    utilities2.append([])
    utilities16.append([])
    utilities32.append([])
    if not use_lambdas:
        infos.append([ -0.183, -0.183, -0.188, -0.179, -0.173])
    else:
        infos.append([10 for _ in range(5)])  # Hacking in lambdas
    sizes.append(small_size)

    labels.append('$\lambda_I = 100; n = 1$')
    utilities2.append([])
    utilities16.append([])
    utilities32.append([])
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



    labels = ['VQ-VIB$_{\mathregular{norm.}}$'] + ['' for _ in vq_comps[:-1]]
    sizes = [small_size for _ in vq_comps]
    num_candidates = 2
    filename = 'xlation_ec_to_eng' + str(num_candidates)
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

    plot_multi_trials([vq_comps, vq_util], labels, sizes,
                      ylabel='Translation Utility', xlabel=xlabel, colors=None, filename=('%s.png' % filename))


    return infos, utilities2, utilities16, utilities32


if __name__ == '__main__':
    run()
