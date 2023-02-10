from src.utils.plotting import plot_multi_trials


def run():
    labels = []
    utilities2 = []
    utilities16 = []
    utilities32 = []
    comps = []
    sizes = []
    colors = []
    small_size = 50

    ############################################################################
    #####           Proto
    ############################################################################
    n='proto'
    labels.append('$\lambda_I = 0.1 proto')
    utilities2.append([0.918, 0.931, 0.95, 0.943, 0.929])
    utilities16.append([0.436, 0.498, 0.518, 0.534, 0.491])
    utilities32.append([0.277, 0.325, 0.364, 0.359, 0.299])
    comps.append([1.076, 1.311, 1.254, 1.203, 1.128])
    sizes.append(small_size)

    labels.append('$\lambda_I = 1.0$ proto')
    utilities2.append([0.96, 0.959, 0.963, 0.96, 0.957])
    utilities16.append([0.693, 0.651, 0.657, 0.616, 0.639])
    utilities32.append([0.503, 0.466, 0.51, 0.435, 0.479])
    comps.append([1.3, 1.684, 1.455, 1.599, 1.635])
    sizes.append(small_size)

    labels.append('$\lambda_I = 10$ proto')
    utilities2.append([0.971, 0.974, 0.985, 0.972, 0.975])
    utilities16.append([0.761, 0.819, 0.817, 0.751, 0.743])
    utilities32.append([0.624, 0.692, 0.707, 0.632, 0.611])
    comps.append([1.86, 1.962, 1.89, 1.983, 1.874])
    sizes.append(small_size)

    labels.append('$\lambda_I = 100$ proto')
    utilities2.append([ 0.972, 0.984, 0.985, 0.972, 0.989])
    utilities16.append([ 0.743, 0.848, 0.826, 0.771, 0.898])
    utilities32.append([0.612, 0.736, 0.715, 0.642, 0.82])
    comps.append([2.448, 2.474, 2.426, 2.525, 2.605])
    sizes.append(small_size)

    ############################################################################
    #####           Onehot
    ############################################################################
    # n='onehot'
    # labels.append('$\lambda_I = 0.1 Onehot')
    # utilities2.append([0.501, 0.718, 0.606, 0.517, 0.513])
    # utilities16.append([0.066, 0.162, 0.125, 0.063, 0.063])
    # utilities32.append([0.031, 0.103, 0.066, 0.044, 0.034])
    # comps.append([0.001, 0.637, 0.298, 0.003, 0.0])
    # sizes.append(small_size)
    #
    # labels.append('$\lambda_I = 1.0$ Onehot')
    # utilities2.append([0.898, 0.665, 0.747, 0.87, 0.772])
    # utilities16.append([0.355, 0.159, 0.197, 0.334, 0.158])
    # utilities32.append([0.217, 0.099, 0.107, 0.219, 0.088])
    # comps.append([1.393, 0.489, 0.782, 1.268, 0.651])
    # sizes.append(small_size)
    #
    # labels.append('$\lambda_I = 10$ Onehot')
    # utilities2.append([0.869, 0.907, 0.885, 0.897, 0.884])
    # utilities16.append([0.384, 0.462, 0.4, 0.377, 0.352])
    # utilities32.append([0.234, 0.255, 0.295, 0.263, 0.207])
    # comps.append([1.22, 1.432, 1.346, 1.38, 1.105])
    # sizes.append(small_size)
    #
    # labels.append('$\lambda_I = 100$ Onehot')
    # utilities2.append([0.876, 0.893, 0.927, 0.885, 0.871])
    # utilities16.append([0.383, 0.414, 0.461, 0.419, 0.354])
    # utilities32.append([0.243, 0.288, 0.242, 0.246, 0.213])
    # comps.append([1.201, 1.492, 1.216, 1.353, 1.047])
    # sizes.append(small_size)

    ############################################################################
    #####           N = 1
    ############################################################################
    # n = 1
    # labels.append('$\lambda_I = 0; n = 1$')
    # utilities2.append([0.837, 0.502, 0.503, 0.493, 0.499])
    # utilities16.append([0.23, 0.07, 0.062, 0.068, 0.058])
    # utilities32.append([0.12, 0.028, 0.039, 0.03, 0.031])
    # comps.append([1.084, 0.0, -0.0, -0.0, 0.0])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 0.1; n = 1$')
    # utilities2.append([0.879, 0.85, 0.878, 0.812, 0.865])
    # utilities16.append([0.308, 0.252, 0.351, 0.211, 0.312])
    # utilities32.append([0.173, 0.134, 0.195, 0.106, 0.177])
    # comps.append([1.284, 1.141, 1.347, 0.995, 1.142])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 0.5; n = 1$')
    # utilities2.append([0.935, 0.903, 0.909, 0.922, 0.929])
    # utilities16.append([ 0.521, 0.384, 0.468, 0.47, 0.473])
    # utilities32.append([0.347, 0.251, 0.323, 0.298, 0.323])
    # comps.append([1.823, 1.586, 1.659, 1.662, 1.816])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 1; n = 1$')
    # utilities2.append([0.947, 0.941, 0.936, 0.923, 0.933])
    # utilities16.append([ 0.593, 0.55, 0.559, 0.541, 0.539])
    # utilities32.append([ 0.432, 0.388, 0.418, 0.384, 0.377])
    # comps.append([2.13, 1.899, 2.113, 1.94, 2.031])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 1.5; n = 1$')
    # utilities2.append([0.947, 0.939, 0.943, 0.935, 0.945])
    # utilities16.append([0.621, 0.605, 0.638, 0.605, 0.59])
    # utilities32.append([0.486, 0.437, 0.479, 0.438, 0.45])
    # comps.append([2.113, 2.218, 2.153, 2.234, 2.136])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 2.0; n = 1$')
    # utilities2.append([0.946, 0.937, 0.95, 0.948, 0.942])
    # utilities16.append([ 0.655, 0.591, 0.648, 0.66, 0.626])
    # utilities32.append([ 0.488, 0.434, 0.505, 0.513, 0.477])
    # comps.append([2.196, 2.206, 2.232, 2.203, 2.277])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 3.0; n = 1$')
    # utilities2.append([0.951, 0.94, 0.946, 0.946, 0.948])
    # utilities16.append([0.668, 0.611, 0.643, 0.655, 0.652])
    # utilities32.append([0.527, 0.468, 0.506, 0.529, 0.523])
    # comps.append([2.307, 2.257, 2.198, 2.247, 2.312])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 10; n = 1$')
    # utilities2.append([0.963, 0.944, 0.96, 0.96, 0.961])
    # utilities16.append([0.721, 0.636, 0.689, 0.704, 0.687])
    # utilities32.append([0.58, 0.496, 0.565, 0.573, 0.557])
    # comps.append([2.598, 2.487, 2.324, 2.482, 2.665])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')

    ############################################################################
    #####           N = 2
    ############################################################################
    # n = 2
    # labels.append('$\lambda_I = 0; n = 2$')
    # utilities2.append([0.508, 0.504, 0.507, 0.495, 0.499])
    # utilities16.append([0.066, 0.071, 0.059, 0.067, 0.059])
    # utilities32.append([0.033, 0.027, 0.037, 0.031, 0.031])
    # comps.append([-0.0, -0.0, 0.0, 0.0, -0.0])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 0.1; n = 2$')
    # utilities2.append([0.917, 0.849, 0.901, 0.866, 0.875])
    # utilities16.append([ 0.404, 0.248, 0.389, 0.291, 0.331])
    # utilities32.append([0.237, 0.121, 0.231, 0.165, 0.194])
    # comps.append([1.681, 1.2, 1.549, 1.316, 1.493])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 0.5; n = 2$')
    # utilities2.append([  0.961, 0.93, 0.949, 0.934, 0.946])
    # utilities16.append([0.588, 0.5, 0.589, 0.514, 0.551])
    # utilities32.append([0.432, 0.323, 0.43, 0.334, 0.386])
    # comps.append([2.043, 1.908, 2.073, 1.881, 1.993])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 1; n = 2$')
    # utilities2.append([ 0.966, 0.948, 0.958, 0.954, 0.955])
    # utilities16.append([ 0.696, 0.627, 0.664, 0.622, 0.62])
    # utilities32.append([0.516, 0.473, 0.525, 0.461, 0.463])
    # comps.append([2.403, 2.3, 2.391, 2.254, 2.298])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 1.5; n = 2$')
    # utilities2.append([ 0.97, 0.958, 0.965, 0.971, 0.961])
    # utilities16.append([0.704, 0.664, 0.696, 0.712, 0.684])
    # utilities32.append([0.565, 0.511, 0.573, 0.588, 0.556])
    # comps.append([2.425, 2.458, 2.722, 2.454, 2.48])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 2.0; n = 2$')
    # utilities2.append([ 0.975, 0.963, 0.962, 0.971, 0.964])
    # utilities16.append([ 0.754, 0.685, 0.703, 0.765, 0.695])
    # utilities32.append([0.601, 0.558, 0.575, 0.618, 0.567])
    # comps.append([2.587, 2.758, 2.467, 2.566, 2.66])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 3.0; n = 2$')
    # utilities2.append([ 0.979, 0.969, 0.977, 0.978, 0.972])
    # utilities16.append([0.789, 0.727, 0.776, 0.798, 0.757])
    # utilities32.append([0.641, 0.572, 0.645, 0.693, 0.641])
    # comps.append([2.867, 2.847, 2.612, 3.257, 3.258])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 10; n = 2$')
    # utilities2.append([ 0.987, 0.975, 0.984, 0.986, 0.979])
    # utilities16.append([ 0.824, 0.773, 0.847, 0.868, 0.82])
    # utilities32.append([ 0.729, 0.661, 0.764, 0.789, 0.722])
    # comps.append([3.491, 3.085, 3.435, 3.279, 3.665])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')


    ############################################################################
    #####           N = 4
    ############################################################################
    # n = 4
    # labels.append('$\lambda_I = 0; n = 4$')
    # utilities2.append([0.909, 0.504, 0.912, 0.494, 0.498])
    # utilities16.append([0.361, 0.07, 0.39, 0.066, 0.059])
    # utilities32.append([0.2, 0.028, 0.248, 0.031, 0.031])
    # comps.append([1.507, -0.0, 1.517, -0.0, 0.0])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 0.1; n = 4$')
    # utilities2.append([0.94, 0.899, 0.923, 0.865, 0.907])
    # utilities16.append([0.459, 0.364, 0.452, 0.29, 0.415])
    # utilities32.append([0.288, 0.218, 0.289, 0.166, 0.261])
    # comps.append([1.748, 1.476, 1.751, 1.149, 1.65])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 0.5; n = 4$')
    # utilities2.append([0.968, 0.946, 0.961, 0.954, 0.949])
    # utilities16.append([0.678, 0.586, 0.629, 0.622, 0.605])
    # utilities32.append([0.5, 0.425, 0.476, 0.462, 0.451])
    # comps.append([2.382, 2.199, 2.216, 2.225, 2.125])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 1; n = 4$')
    # utilities2.append([0.98, 0.961, 0.972, 0.977, 0.974])
    # utilities16.append([ 0.763, 0.675, 0.724, 0.779, 0.722])
    # utilities32.append([0.63, 0.535, 0.592, 0.634, 0.598])
    # comps.append([2.734, 2.652, 2.683, 2.763, 2.809])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 1.5; n = 4$')
    # utilities2.append([ 0.983, 0.971, 0.985, 0.983, 0.977])
    # utilities16.append([0.795, 0.735, 0.828, 0.854, 0.768])
    # utilities32.append([0.658, 0.606, 0.732, 0.741, 0.655])
    # comps.append([2.962, 2.748, 3.395, 2.98, 2.942])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 2.0; n = 4$')
    # utilities2.append([ 0.983, 0.981, 0.989, 0.987, 0.98])
    # utilities16.append([0.819, 0.789, 0.864, 0.862, 0.828])
    # utilities32.append([0.71, 0.668, 0.796, 0.785, 0.725])
    # comps.append([3.596, 3.426, 3.096, 3.679, 2.883])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 3.0; n = 4$')
    # utilities2.append([ 0.991, 0.983, 0.993, 0.993, 0.989])
    # utilities16.append([0.887, 0.843, 0.886, 0.927, 0.874])
    # utilities32.append([0.804, 0.748, 0.811, 0.876, 0.812])
    # comps.append([3.123, 3.455, 3.419, 3.86, 3.31])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # # Just for 100k
    # labels.append('$\lambda_I = 10; n = 4$')
    # utilities2.append([ 0.996, 0.99, 0.996, 0.998, 0.994])
    # utilities16.append([0.944, 0.929, 0.958, 0.975, 0.935])
    # utilities32.append([0.915, 0.865, 0.932, 0.954, 0.895])
    # comps.append([4.634, 4.924, 4.915, 4.245, 3.565])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 100; n = 4$')
    # utilities2.append([0.996, 0.995, 0.996, 0.999, 0.995])
    # utilities16.append([ 0.938, 0.927, 0.963, 0.979, 0.95])
    # utilities32.append([0.9, 0.877, 0.941, 0.958, 0.916])
    # comps.append([ 4.812, 4.593, 3.557, 4.942, 4.373])
    # sizes.append(small_size)
    # colors.append('xkcd:orange')


    # labels.append('$\lambda_I = 1; n = 8$')
    # utilities2.append([0.971, 0.985, 0.984, 0.976])
    # utilities16.append([0.727, 0.838, 0.84, 0.811])
    # utilities32.append([0.611, 0.738, 0.721, 0.704])
    # comps.append([3.092, 3.625, 3.259, 3.166])
    # sizes.append(small_size)
    # colors.append('xkcd:green')
    #
    # labels.append('$\lambda_I = 10; n = 8$')
    # utilities2.append([ 0.997, 0.996, 0.999, 0.998, 0.996])
    # utilities16.append([0.976, 0.967, 0.985, 0.988, 0.973])
    # utilities32.append([0.958, 0.947, 0.977, 0.978, 0.955])
    # comps.append([4.681, 4.35, 4.025, 4.363, 4.747])
    # sizes.append(small_size)
    # colors.append('xkcd:green')
    #
    #
    # labels.append('$\lambda_I = 100; n = 8$')
    # utilities2.append([0.998, 0.998, 0.991, 0.999, 0.997])
    # utilities16.append([0.985, 0.985, 0.891, 0.993, 0.979])
    # utilities32.append([ 0.973, 0.97, 0.83, 0.984, 0.962])
    # comps.append([4.038, 5.248, 3.155, 5.119, 3.319])
    # sizes.append(small_size)
    # colors.append('xkcd:black')

    for data, suffix in zip([utilities2, utilities16, utilities32], ['2', '16', '32']):
        plot_multi_trials([comps, data], labels, sizes, ylabel='Utility', colors=None, filename='nondist_comp' + suffix + '.png')

    labels = ['$C=2$'] + ['' for _ in utilities2[:-1]] + ['$C=16$'] + ['' for _ in utilities16[:-1]] + ['$C=32$'] + ['' for _ in utilities32[:-1]]
    sizes = [4 * small_size for _ in utilities2] + [2 * small_size for _ in utilities16] + [small_size for _ in utilities32]
    plot_multi_trials([comps + comps + comps, utilities2 + utilities16 + utilities32], labels, sizes, ylabel='In-Distribution Utility', colors=None, filename='nondist_comp_all_n' + str(n) + '.png')
    return comps, utilities2, utilities16, utilities32

if __name__ == '__main__':
    run()
