from src.utils.plotting import plot_multi_trials


def run():
    labels = []
    alignments = []
    utilities16 = []
    utilities32 = []
    comps = []
    sizes = []
    colors = []
    small_size = 50
    ############################################################################
    #####           Proto
    ############################################################################
    # n = 'proto'
    # labels.append('$\lambda_I = 0.1 Proto')
    # alignments.append([0.461, 0.292, 0.43, 0.372, 0.444])
    # comps.append([1.076, 1.311, 1.254, 1.203, 1.128])
    # sizes.append(small_size)
    #
    # labels.append('$\lambda_I = 1.0$ proto')
    # alignments.append([0.397, 0.362, 0.422, 0.356, 0.438])
    # comps.append([1.3, 1.684, 1.455, 1.599, 1.635])
    # sizes.append(small_size)
    #
    # labels.append('$\lambda_I = 10$ proto')
    # alignments.append([0.336, 0.368, 0.387, 0.347, 0.415])
    # comps.append([1.86, 1.962, 1.89, 1.983, 1.874])
    # sizes.append(small_size)
    #
    # labels.append('$\lambda_I = 100$ proto')
    # alignments.append([0.331, 0.378, 0.393, 0.389, 0.431])
    # comps.append([2.448, 2.474, 2.426, 2.525, 2.605])
    # sizes.append(small_size)

    ############################################################################
    #####           Onehot
    ############################################################################
    # n = 'onehot'
    # labels.append('$\lambda_I = 0.1 Onehot')
    # alignments.append([0.499, 0.661, 0.52, 0.502, 0.486])
    # utilities16.append([0.058, 0.116, 0.066, 0.057, 0.061])
    # utilities32.append([ 0.026, 0.054, 0.03, 0.033, 0.029])
    # # comps.append([0.001, 0.637, 0.298, 0.003, 0.0])
    # comps.append([0.1 for _ in range(5)])  # Hacking in lambdas
    # sizes.append(small_size)
    #
    # labels.append('$\lambda_I = 1.0$ Onehot')
    # alignments.append([0.797, 0.617, 0.583, 0.822, 0.716])
    # utilities16.append([0.253, 0.109, 0.095, 0.274, 0.134])
    # utilities32.append([0.142, 0.056, 0.054, 0.151, 0.08])
    # # comps.append([1.393, 0.489, 0.782, 1.268, 0.651])
    # comps.append([1.0 for _ in range(5)])  # Hacking in lambdas
    # sizes.append(small_size)
    #
    # labels.append('$\lambda_I = 10$ Onehot')
    # alignments.append([0.793, 0.809, 0.774, 0.77, 0.753])
    # utilities16.append([0.25, 0.281, 0.256, 0.266, 0.208])
    # utilities32.append([0.149, 0.17, 0.162, 0.163, 0.134])
    # # comps.append([1.22, 1.432, 1.346, 1.38, 1.105])
    # comps.append([10 for _ in range(5)])  # Hacking in lambdas
    # sizes.append(small_size)
    #
    # labels.append('$\lambda_I = 100$ Onehot')
    # alignments.append([0.706, 0.847, 0.693, 0.831, 0.752])
    # utilities16.append([0.186, 0.311, 0.217, 0.259, 0.223])
    # utilities32.append([0.13, 0.182, 0.141, 0.176, 0.141])
    # # comps.append([1.201, 1.492, 1.216, 1.353, 1.047])
    # comps.append([100 for _ in range(5)])  # Hacking in lambdas
    # sizes.append(small_size)
    #
    # onehot_util2 = alignments
    # onehot_util16 = utilities16
    # onehot_util32 = utilities32
    # onehot_comps = comps
    # alignments = []
    # utilities16 = []
    # utilities32 = []
    # comps = []
    ############################################################################
    #####           N = 1
    ############################################################################
    n = 1
    labels.append('$\lambda_I = 0; n = 1$')
    alignments.append([0.377, 0.378, 0.4, 0.409, 0.408])
    comps.append([1.084, 0.0, -0.0, -0.0, 0.0])
    sizes.append(small_size)
    colors.append('xkcd:blue')

    labels.append('$\lambda_I = 0.1; n = 1$')
    alignments.append([0.316, 0.29, 0.314, 0.34, 0.302])
    comps.append([1.284, 1.141, 1.347, 0.995, 1.142])
    sizes.append(small_size)
    colors.append('xkcd:blue')

    labels.append('$\lambda_I = 0.5; n = 1$')
    alignments.append([0.326, 0.275, 0.325, 0.33, 0.317])
    comps.append([1.823, 1.586, 1.659, 1.662, 1.816])
    sizes.append(small_size)
    colors.append('xkcd:blue')

    labels.append('$\lambda_I = 1; n = 1$')
    alignments.append([0.31, 0.276, 0.288, 0.336, 0.302])
    comps.append([2.13, 1.899, 2.113, 1.94, 2.031])
    sizes.append(small_size)
    colors.append('xkcd:blue')

    labels.append('$\lambda_I = 1.5; n = 1$')
    alignments.append([0.29, 0.284, 0.315, 0.338, 0.323])
    comps.append([2.113, 2.218, 2.153, 2.234, 2.136])
    sizes.append(small_size)
    colors.append('xkcd:blue')

    labels.append('$\lambda_I = 2.0; n = 1$')
    alignments.append([0.332, 0.294, 0.313, 0.335, 0.323])
    comps.append([2.196, 2.206, 2.232, 2.203, 2.277])
    sizes.append(small_size)
    colors.append('xkcd:blue')

    labels.append('$\lambda_I = 3.0; n = 1$')
    alignments.append([0.362, 0.295, 0.33, 0.344, 0.348])
    comps.append([2.307, 2.257, 2.198, 2.247, 2.312])
    sizes.append(small_size)
    colors.append('xkcd:blue')

    labels.append('$\lambda_I = 10; n = 1$')
    alignments.append([0.387, 0.332, 0.331, 0.353, 0.357])
    comps.append([2.598, 2.487, 2.324, 2.482, 2.665])
    sizes.append(small_size)
    colors.append('xkcd:blue')

    ############################################################################
    #####           N = 2
    ############################################################################
    n = 2
    labels.append('$\lambda_I = 0; n = 2$')
    alignments.append([0.414, 0.398, 0.415, 0.425, 0.408])
    comps.append([ -0.0, -0.0, 0.0, 0.0, -0.0])
    sizes.append(small_size)
    colors.append('xkcd:blue')

    labels.append('$\lambda_I = 0.1; n = 2$')
    alignments.append([0.317, 0.318, 0.318, 0.348, 0.325])
    comps.append([1.681, 1.2, 1.549, 1.316, 1.493])
    sizes.append(small_size)
    colors.append('xkcd:blue')

    labels.append('$\lambda_I = 0.5; n = 2$')
    alignments.append([0.336, 0.298, 0.325, 0.337, 0.303])
    comps.append([2.043, 1.908, 2.073, 1.881, 1.993])
    sizes.append(small_size)
    colors.append('xkcd:blue')

    labels.append('$\lambda_I = 1; n = 2$')
    alignments.append([0.326, 0.296, 0.313, 0.328, 0.307])
    comps.append([2.403, 2.3, 2.391, 2.254, 2.298])
    sizes.append(small_size)
    colors.append('xkcd:blue')

    labels.append('$\lambda_I = 1.5; n = 2$')
    alignments.append([0.35, 0.298, 0.314, 0.337, 0.337])
    comps.append([2.425, 2.458, 2.722, 2.454, 2.48])
    sizes.append(small_size)
    colors.append('xkcd:blue')

    labels.append('$\lambda_I = 2.0; n = 2$')
    alignments.append([0.336, 0.314, 0.34, 0.342, 0.333])
    comps.append([2.587, 2.758, 2.467, 2.566, 2.66])
    sizes.append(small_size)
    colors.append('xkcd:blue')

    labels.append('$\lambda_I = 3.0; n = 2$')
    alignments.append([0.351, 0.33, 0.363, 0.351, 0.38])
    comps.append([2.867, 2.847, 2.612, 3.257, 3.258])
    sizes.append(small_size)
    colors.append('xkcd:blue')

    labels.append('$\lambda_I = 10; n = 2$')
    alignments.append([0.385, 0.361, 0.353, 0.361, 0.364])
    comps.append([3.491, 3.085, 3.435, 3.279, 3.665])
    sizes.append(small_size)
    colors.append('xkcd:blue')

    ############################################################################
    #####           N = 4
    ############################################################################
    n = 4
    labels.append('$\lambda_I = 0; n = 4$')
    alignments.append([0.33, 0.392, 0.351, 0.395, 0.397])
    comps.append([1.507, -0.0, 1.517, -0.0, 0.0])
    sizes.append(small_size)
    colors.append('xkcd:blue')

    labels.append('$\lambda_I = 0.1; n = 4$')
    alignments.append([0.332, 0.287, 0.312, 0.375, 0.322])
    comps.append([1.748, 1.476, 1.751, 1.149, 1.65])
    sizes.append(small_size)
    colors.append('xkcd:blue')

    labels.append('$\lambda_I = 0.5; n = 4$')
    alignments.append([0.347, 0.3, 0.335, 0.312, 0.331])
    comps.append([2.382, 2.199, 2.216, 2.225, 2.125])
    sizes.append(small_size)
    colors.append('xkcd:blue')

    labels.append('$\lambda_I = 1; n = 4$')
    alignments.append([0.358, 0.321, 0.32, 0.337, 0.346])
    comps.append([2.734, 2.652, 2.683, 2.763, 2.809])
    sizes.append(small_size)
    colors.append('xkcd:blue')

    labels.append('$\lambda_I = 1.5; n = 4$')
    alignments.append([0.354, 0.292, 0.338, 0.349, 0.349])
    comps.append([2.962, 2.748, 3.395, 2.98, 2.942])
    sizes.append(small_size)
    colors.append('xkcd:blue')

    labels.append('$\lambda_I = 2.0; n = 4$')
    alignments.append([0.356, 0.31, 0.336, 0.327, 0.371])
    comps.append([3.596, 3.426, 3.096, 3.679, 2.883])
    sizes.append(small_size)
    colors.append('xkcd:blue')

    labels.append('$\lambda_I = 3.0; n = 4$')
    alignments.append([0.364, 0.319, 0.33, 0.346, 0.358])
    comps.append([3.123, 3.455, 3.419, 3.86, 3.31])
    sizes.append(small_size)
    colors.append('xkcd:blue')

    labels.append('$\lambda_I = 10; n = 4$')
    alignments.append([0.36, 0.343, 0.335, 0.368, 0.34])
    comps.append([ 4.634, 4.924, 4.915, 4.245, 3.565])
    sizes.append(small_size)
    colors.append('xkcd:blue')

    ############################################################################
    #####           N = 8
    ############################################################################
    # n = 8
    # labels.append('$\lambda_I = 0; n = 8$')
    # alignments.append([0.907, 0.493, 0.485, 0.499, 0.866])
    # utilities16.append([0.333, 0.071, 0.072, 0.063, 0.352])
    # utilities32.append([0.218, 0.038, 0.03, 0.031, 0.222])
    # comps.append([1.553, -0.0, -0.0, -0.0, 1.622])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 0.1; n = 8$')
    # alignments.append([ 0.895, 0.779, 0.834, 0.824, 0.852])
    # utilities16.append([0.363, 0.281, 0.3, 0.293, 0.318])
    # utilities32.append([0.227, 0.168, 0.189, 0.191, 0.195])
    # comps.append([1.813, 1.53, 1.728, 1.776, 1.589])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 0.5; n = 8$')
    # alignments.append([ 0.922, 0.912, 0.85, 0.906, 0.94])
    # utilities16.append([ 0.482, 0.463, 0.378, 0.505, 0.587])
    # utilities32.append([  0.316, 0.328, 0.291, 0.332, 0.438])
    # comps.append([ 2.126, 2.28, 2.201, 2.502, 2.827])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 1; n = 8$')
    # alignments.append([0.948, 0.936, 0.878, 0.944, 0.945])
    # utilities16.append([0.586, 0.545, 0.48, 0.656, 0.602])
    # utilities32.append([0.448, 0.412, 0.38, 0.487, 0.475])
    # comps.append([ 2.884, 3.16, 3.221, 3.308, 3.062])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 1.5; n = 8$')
    # alignments.append([0.964, 0.939, 0.929, 0.966, 0.983])
    # utilities16.append([ 0.672, 0.586, 0.602, 0.716, 0.834])
    # utilities32.append([ 0.536, 0.455, 0.511, 0.596, 0.759])
    # comps.append([3.124, 3.448, 4.164, 3.883, 3.793])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 2.0; n = 8$')
    # alignments.append([ 0.96, 0.942, 0.93, 0.973, 0.984])
    # utilities16.append([  0.664, 0.605, 0.611, 0.786, 0.813])
    # utilities32.append([0.53, 0.477, 0.532, 0.678, 0.727])
    # comps.append([ 3.612, 3.618, 3.957, 4.084, 3.863])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 3.0; n = 8$')
    # alignments.append([ 0.976, 0.95, 0.933, 0.974, 0.992])
    # utilities16.append([ 0.777, 0.694, 0.632, 0.748, 0.912])
    # utilities32.append([ 0.665, 0.592, 0.54, 0.637, 0.845])
    # comps.append([ 4.456, 3.311, 4.16, 4.543, 4.493])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 10; n = 8$')
    # alignments.append([0.995, 0.981, 0.986, 0.992, 0.996])
    # utilities16.append([0.914, 0.828, 0.857, 0.924, 0.958])
    # utilities32.append([0.854, 0.736, 0.793, 0.874, 0.912])
    # comps.append([4.703, 4.502, 5.082, 5.502, 4.655])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')

    ############################################################################
    #####           N = 16
    ############################################################################
    # n = 16
    # labels.append('$\lambda_I = 0; n = 16$')
    # alignments.append([0.914, 0.848, 0.486, 0.502, 0.907])
    # utilities16.append([0.391, 0.29, 0.071, 0.063, 0.516])
    # utilities32.append([0.239, 0.173, 0.029, 0.031, 0.368])
    # comps.append([1.928, 1.227, 0.0, -0.0, 1.932])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 0.1; n = 16$')
    # alignments.append([0.91, 0.889, 0.867, 0.498, 0.907])
    # utilities16.append([ 0.419, 0.428, 0.384, 0.063, 0.425])
    # utilities32.append([0.273, 0.284, 0.271, 0.03, 0.301])
    # comps.append([1.862, 1.579, 1.955, -0.0, 1.679])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 0.5; n = 16$')
    # alignments.append([ 0.947, 0.922, 0.864, 0.924, 0.954])
    # utilities16.append([ 0.547, 0.492, 0.429, 0.529, 0.638])
    # utilities32.append([ 0.401, 0.347, 0.329, 0.37, 0.497])
    # comps.append([2.394, 2.143, 2.707, 2.454, 2.685])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 1; n = 16$')
    # alignments.append([0.958, 0.934, 0.902, 0.941, 0.973])
    # utilities16.append([0.634, 0.548, 0.536, 0.621, 0.75])
    # utilities32.append([ 0.478, 0.405, 0.427, 0.505, 0.63])
    # comps.append([2.915, 2.485, 3.422, 2.821, 3.037])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 1.5; n = 16$')
    # alignments.append([0.97, 0.932, 0.968, 0.927, 0.993])
    # utilities16.append([ 0.702, 0.578, 0.783, 0.586, 0.92])
    # utilities32.append([ 0.547, 0.45, 0.715, 0.466, 0.868])
    # comps.append([3.31, 3.021, 4.041, 3.5, 3.319])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 2.0; n = 16$')
    # alignments.append([ 0.977, 0.931, 0.987, 0.959, 0.992])
    # utilities16.append([ 0.73, 0.568, 0.861, 0.724, 0.909])
    # utilities32.append([0.63, 0.436, 0.797, 0.606, 0.846])
    # comps.append([ 3.847, 3.184, 4.175, 3.823, 3.878])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 3.0; n = 16$')
    # alignments.append([0.982, 0.952, 0.986, 0.973, 0.992])
    # utilities16.append([ 0.808, 0.707, 0.824, 0.773, 0.92])
    # utilities32.append([0.723, 0.587, 0.768, 0.647, 0.866])
    # comps.append([4.35, 4.017, 4.986, 3.709, 2.619])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 10; n = 16$')
    # alignments.append([ 0.992, 0.992, 0.986, 0.997, 0.998])
    # utilities16.append([ 0.913, 0.906, 0.85, 0.952, 0.976])
    # utilities32.append([0.863, 0.869, 0.805, 0.916, 0.951])
    # comps.append([4.31, 4.316, 4.773, 3.781, 4.276])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')

    labels = ['$n=1$'] + ['' for _ in range(7)] + ['$n=2$'] + ['' for _ in range(7)] + ['$n=4$'] + ['' for _ in range(7)]
    plot_multi_trials([comps, alignments], labels, sizes, ylabel='Alignment', colors=None, filename='alignment_comp.png')

    # labels = ['Onehot'] + ['' for _ in onehot_comps[:-1]] +\
    #          ['Proto.'] + ['' for _ in proto_comps[:-1]] +\
    #          ['VQ-VIB'] + ['' for _ in vq_comps[:-1]]
    # sizes = [2 * small_size for _ in onehot_comps] + [2 * small_size for _ in proto_comps] + [2 * small_size for _ in vq_comps]
    # plot_multi_trials([onehot_comps + proto_comps + vq_comps, onehot_util2 + proto_util2 + vq_util2], labels, sizes,
    #                   ylabel='OOD Utility', colors=None, filename='ood_lambda_all_multimodel_2.png')

    return comps, alignments, utilities16, utilities32


if __name__ == '__main__':
    run()
