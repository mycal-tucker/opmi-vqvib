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
    #####           Proto Old
    ############################################################################
    # n = 'proto'
    # labels.append('$\lambda_I = 0.1 Proto')
    # utilities2.append([ 0.794, 0.877, 0.86, 0.91, 0.837])
    # utilities16.append([0.319, 0.44, 0.352, 0.408, 0.294])
    # utilities32.append([0.209, 0.295, 0.236, 0.277, 0.168])
    # # comps.append([1.076, 1.311, 1.254, 1.203, 1.128])
    # comps.append([0.1 for _ in range(5)])  # Hacking in lambdas
    # sizes.append(small_size)
    #
    # labels.append('$\lambda_I = 1.0$ proto')
    # utilities2.append([ 0.867, 0.895, 0.864, 0.921, 0.902])
    # utilities16.append([0.407, 0.484, 0.396, 0.47, 0.424])
    # utilities32.append([0.3, 0.3, 0.28, 0.332, 0.256])
    # # comps.append([1.3, 1.684, 1.455, 1.599, 1.635])
    # comps.append([1.0 for _ in range(5)])  # Hacking in lambdas
    # sizes.append(small_size)
    #
    # labels.append('$\lambda_I = 10$ proto')
    # utilities2.append([0.907, 0.941, 0.91, 0.95, 0.908])
    # utilities16.append([0.488, 0.661, 0.505, 0.59, 0.431])
    # utilities32.append([0.363, 0.487, 0.392, 0.471, 0.311])
    # # comps.append([1.86, 1.962, 1.89, 1.983, 1.874])
    # comps.append([10 for _ in range(5)])  # Hacking in lambdas
    # sizes.append(small_size)
    #
    # labels.append('$\lambda_I = 100$ proto')
    # utilities2.append([0.866, 0.941, 0.922, 0.948, 0.958])
    # utilities16.append([0.423, 0.659, 0.53, 0.618, 0.664])
    # utilities32.append([0.298, 0.497, 0.386, 0.471, 0.515])
    # # comps.append([2.448, 2.474, 2.426, 2.525, 2.605])
    # comps.append([100 for _ in range(5)])  # Hacking in lambdas
    # sizes.append(small_size)
    #
    # proto_util2 = utilities2
    # proto_util16 = utilities16
    # proto_util32 = utilities32
    # proto_comps = comps
    # utilities2 = []
    # utilities16 = []
    # utilities32 = []
    # comps = []
    ############################################################################
    #####           Proto Redone March 2023
    ############################################################################
    n = 'proto'
    labels.append('$\lambda_I = 0.1 Proto')
    utilities2.append([ 0.888, 0.887, 0.805, 0.873, 0.902])
    utilities16.append([0.39, 0.421, 0.294, 0.404, 0.452])
    utilities32.append([0.261, 0.304, 0.184, 0.268, 0.301])
    # comps.append([2.17, 3.06, 2.46, 2.71, 2.62])
    comps.append([0.1 for _ in range(5)])  # Hacking in lambdas
    sizes.append(small_size)

    labels.append('$\lambda_I = 0.5$ proto')
    utilities2.append([0.942, 0.881, 0.845, 0.895, 0.927])
    utilities16.append([0.574, 0.367, 0.338, 0.458, 0.536])
    utilities32.append([0.417, 0.228, 0.242, 0.314, 0.375])
    # comps.append([2.92, 2.01, 2.70, 3.12, 3.00])
    comps.append([0.5 for _ in range(5)])  # Hacking in lambdas
    sizes.append(small_size)

    labels.append('$\lambda_I = 1.0$ proto')
    utilities2.append([0.927, 0.871, 0.81, 0.887, 0.886])
    utilities16.append([ 0.533, 0.379, 0.396, 0.47, 0.424])
    utilities32.append([ 0.396, 0.275, 0.28, 0.332, 0.256])
    # comps.append([2.74, 2.62, 2.91, 3.01, 2.35])
    comps.append([1.0 for _ in range(5)])  # Hacking in lambdas
    sizes.append(small_size)

    labels.append('$\lambda_I = 1.5$ proto')
    utilities2.append([0.927, 0.895, 0.857, 0.889, 0.907])
    utilities16.append([0.514, 0.392, 0.382, 0.439, 0.447])
    utilities32.append([0.374, 0.273, 0.292, 0.318, 0.307])
    # comps.append([2.96, 2.25, 3.07, 3.23, 2.51])
    comps.append([1.5 for _ in range(5)])  # Hacking in lambdas
    sizes.append(small_size)

    labels.append('$\lambda_I = 2$ proto')
    utilities2.append([ 0.929, 0.9, 0.882, 0.92, 0.928])
    utilities16.append([0.574, 0.384, 0.405, 0.512, 0.548])
    utilities32.append([0.444, 0.262, 0.293, 0.362, 0.395])
    # comps.append([3.03, 2.12, 2.94, 3.50, 2.87])
    comps.append([2 for _ in range(5)])  # Hacking in lambdas
    sizes.append(small_size)

    labels.append('$\lambda_I = 3$ proto')
    utilities2.append([0.93, 0.886, 0.857, 0.916, 0.92])
    utilities16.append([0.545, 0.374, 0.377, 0.519, 0.519])
    utilities32.append([ 0.407, 0.239, 0.302, 0.388, 0.368])
    # comps.append([2.64, 1.87, 3.04, 2.98, 2.68])
    comps.append([3 for _ in range(5)])  # Hacking in lambdas
    sizes.append(small_size)

    labels.append('$\lambda_I = 10$ proto')
    utilities2.append([0.941, 0.893, 0.891, 0.889, 0.92])
    utilities16.append([ 0.609, 0.418, 0.453, 0.442, 0.512])
    utilities32.append([ 0.493, 0.289, 0.342, 0.328, 0.384])
    # comps.append([3.32, 2.70, 3.45, 3.15, 2.77])
    comps.append([10 for _ in range(5)])  # Hacking in lambdas
    sizes.append(small_size)

    labels.append('$\lambda_I = 100$ proto')
    utilities2.append([ 0.843, 0.821, 0.787, 0.857, 0.873])
    utilities16.append([ 0.634, 0.484, 0.379, 0.539, 0.626])
    utilities32.append([  0.312, 0.262, 0.167, 0.266, 0.315])
    # comps.append([3.10, 3.20, 3.13, 3.39, 3.49])
    comps.append([100 for _ in range(5)])  # Hacking in lambdas
    sizes.append(small_size)

    proto_util2 = utilities2
    proto_util16 = utilities16
    proto_util32 = utilities32
    proto_comps = comps
    utilities2 = []
    utilities16 = []
    utilities32 = []
    comps = []
    ############################################################################
    #####           Onehot
    ############################################################################
    n = 'onehot'
    labels.append('$\lambda_I = 0.1 Onehot')
    utilities2.append([0.499, 0.661, 0.52, 0.502, 0.486])
    utilities16.append([0.058, 0.116, 0.066, 0.057, 0.061])
    utilities32.append([ 0.026, 0.054, 0.03, 0.033, 0.029])
    # comps.append([0.001, 0.637, 0.298, 0.003, 0.0])  # Done.
    comps.append([0.1 for _ in range(5)])  # Hacking in lambdas
    sizes.append(small_size)

    labels.append('$\lambda_I = 0.5 Onehot')
    utilities2.append([ 0.784, 0.793, 0.755, 0.82, 0.723])
    utilities16.append([ 0.249, 0.234, 0.217, 0.213, 0.185])
    utilities32.append([ 0.129, 0.136, 0.13, 0.114, 0.112])
    # comps.append([1.34, 1.79, 1.88, 1.26, 1.44])  # Done
    comps.append([0.5 for _ in range(5)])  # Hacking in lambdas
    sizes.append(small_size)

    labels.append('$\lambda_I = 1.0$ Onehot')
    utilities2.append([0.797, 0.617, 0.583, 0.822, 0.716])
    utilities16.append([0.253, 0.109, 0.095, 0.274, 0.134])
    utilities32.append([0.142, 0.056, 0.054, 0.151, 0.08])
    # comps.append([1.93, 0.48, 0.68, 1.36, 0.75])  # Done
    comps.append([1.0 for _ in range(5)])  # Hacking in lambdas
    sizes.append(small_size)

    labels.append('$\lambda_I = 1.5$ Onehot')
    utilities2.append([0.801, 0.601, 0.799, 0.844, 0.638])
    utilities16.append([0.259, 0.14, 0.241, 0.289, 0.148])
    utilities32.append([0.142, 0.09, 0.154, 0.164, 0.097])
    # comps.append([1.58, 0.93, 1.75, 1.86, 0.89])  # Done
    comps.append([1.5 for _ in range(5)])  # Hacking in lambdas
    sizes.append(small_size)

    labels.append('$\lambda_I = 2.0 Onehot')
    utilities2.append([ 0.777, 0.785, 0.763, 0.82, 0.784])
    utilities16.append([ 0.252, 0.214, 0.219, 0.253, 0.243])
    utilities32.append([ 0.162, 0.122, 0.12, 0.131, 0.137])
    # comps.append([1.97, 1.66, 1.76, 1.61, 1.51]) # Done
    comps.append([2.0 for _ in range(5)])  # Hacking in lambdas
    sizes.append(small_size)

    labels.append('$\lambda_I = 3.0 Onehot')
    utilities2.append([0.74, 0.803, 0.725, 0.862, 0.777])
    utilities16.append([0.213, 0.271, 0.237, 0.327, 0.276])
    utilities32.append([0.121, 0.156, 0.157, 0.198, 0.174])
    # comps.append([1.65, 1.59, 1.86, 1.93, 2.05])  # Done
    comps.append([3.0 for _ in range(5)])  # Hacking in lambdas
    sizes.append(small_size)

    labels.append('$\lambda_I = 10$ Onehot')
    utilities2.append([0.78, 0.818, 0.821, 0.77, 0.753])
    utilities16.append([0.262, 0.246, 0.299, 0.266, 0.208])
    utilities32.append([ 0.164, 0.152, 0.184, 0.163, 0.134])
    # comps.append([2.06, 1.90, 1.97, 1.72, 1.49])  # Done
    comps.append([10 for _ in range(5)])  # Hacking in lambdas
    sizes.append(small_size)

    labels.append('$\lambda_I = 100$ Onehot')
    utilities2.append([0.706, 0.847, 0.693, 0.831, 0.752])
    utilities16.append([0.186, 0.311, 0.217, 0.259, 0.223])
    utilities32.append([0.13, 0.182, 0.141, 0.176, 0.141])
    # comps.append([1.62, 2.07, 1.76, 1.73, 1.52])  # Now
    comps.append([100 for _ in range(5)])  # Hacking in lambdas
    sizes.append(small_size)

    onehot_util2 = utilities2
    onehot_util16 = utilities16
    onehot_util32 = utilities32
    onehot_comps = comps
    utilities2 = []
    utilities16 = []
    utilities32 = []
    comps = []
    ############################################################################
    #####           N = 1
    ############################################################################
    n = 1
    # labels.append('$\lambda_I = 0; n = 1$')
    # utilities2.append([0.817, 0.498, 0.504, 0.509, 0.492])
    # utilities16.append([0.207, 0.066, 0.068, 0.07, 0.063])
    # utilities32.append([0.106, 0.034, 0.029, 0.028, 0.03])
    # comps.append([1.084, 0.0, -0.0, -0.0, 0.0])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')

    labels.append('$\lambda_I = 0.1; n = 1$')
    utilities2.append([0.833, 0.757, 0.71, 0.58, 0.695])
    utilities16.append([0.265, 0.22, 0.202, 0.112, 0.197])
    utilities32.append([0.146, 0.123, 0.131, 0.056, 0.12])
    # comps.append([1.284, 1.141, 1.347, 0.995, 1.142])
    comps.append([0.1 for _ in range(5)])  # Hacking in lambdas
    sizes.append(small_size)
    colors.append('xkcd:blue')

    labels.append('$\lambda_I = 0.5; n = 1$')
    utilities2.append([0.873, 0.756, 0.775, 0.806, 0.853])
    utilities16.append([0.329, 0.266, 0.267, 0.278, 0.339])
    utilities32.append([0.207, 0.143, 0.176, 0.182, 0.225])
    # comps.append([1.823, 1.586, 1.659, 1.662, 1.816])
    comps.append([0.5 for _ in range(5)])  # Hacking in lambdas
    sizes.append(small_size)
    colors.append('xkcd:blue')

    labels.append('$\lambda_I = 1; n = 1$')
    utilities2.append([ 0.892, 0.856, 0.822, 0.846, 0.872])
    utilities16.append([0.431, 0.343, 0.323, 0.338, 0.386])
    utilities32.append([0.244, 0.218, 0.208, 0.186, 0.262])
    # comps.append([2.13, 1.899, 2.113, 1.94, 2.031])
    comps.append([1 for _ in range(5)])  # Hacking in lambdas
    sizes.append(small_size)
    colors.append('xkcd:blue')

    labels.append('$\lambda_I = 1.5; n = 1$')
    utilities2.append([0.892, 0.87, 0.821, 0.84, 0.871])
    utilities16.append([0.437, 0.389, 0.32, 0.377, 0.411])
    utilities32.append([0.288, 0.252, 0.206, 0.232, 0.266])
    # comps.append([2.113, 2.218, 2.153, 2.234, 2.136])
    comps.append([1.5 for _ in range(5)])  # Hacking in lambdas
    sizes.append(small_size)
    colors.append('xkcd:blue')

    labels.append('$\lambda_I = 2.0; n = 1$')
    utilities2.append([0.888, 0.876, 0.8, 0.868, 0.89])
    utilities16.append([0.429, 0.383, 0.317, 0.373, 0.432])
    utilities32.append([0.281, 0.261, 0.218, 0.241, 0.305])
    # comps.append([2.196, 2.206, 2.232, 2.203, 2.277])
    comps.append([2 for _ in range(5)])  # Hacking in lambdas
    sizes.append(small_size)
    colors.append('xkcd:blue')

    labels.append('$\lambda_I = 3.0; n = 1$')
    utilities2.append([0.896, 0.881, 0.784, 0.86, 0.896])
    utilities16.append([0.44, 0.407, 0.321, 0.376, 0.478])
    utilities32.append([ 0.289, 0.256, 0.207, 0.244, 0.332])
    # comps.append([2.307, 2.257, 2.198, 2.247, 2.312])
    comps.append([3 for _ in range(5)])  # Hacking in lambdas
    sizes.append(small_size)
    colors.append('xkcd:blue')

    labels.append('$\lambda_I = 10; n = 1$')
    utilities2.append([0.897, 0.848, 0.814, 0.882, 0.912])
    utilities16.append([0.473, 0.354, 0.344, 0.417, 0.5])
    utilities32.append([0.328, 0.246, 0.232, 0.292, 0.365])
    # comps.append([2.598, 2.487, 2.324, 2.482, 2.665])
    comps.append([10 for _ in range(5)])  # Hacking in lambdas
    sizes.append(small_size)
    colors.append('xkcd:blue')

    labels.append('$\lambda_I = 100; n = 1$')
    utilities2.append([ 0.879, 0.88, 0.811, 0.791, 0.879])
    utilities16.append([ 0.245, 0.214, 0.187, 0.189, 0.254])
    utilities32.append([ 0.245, 0.214, 0.187, 0.189, 0.254])
    # comps.append([2.598, 2.487, 2.324, 2.482, 2.665])
    comps.append([100 for _ in range(5)])  # Hacking in lambdas
    sizes.append(small_size)
    colors.append('xkcd:blue')

    ############################################################################
    #####           N = 2
    ############################################################################
    # n = 2
    # labels.append('$\lambda_I = 0; n = 2$')
    # utilities2.append([0.51, 0.493, 0.506, 0.509, 0.493])
    # utilities16.append([0.067, 0.066, 0.071, 0.07, 0.063])
    # utilities32.append([0.038, 0.034, 0.03, 0.027, 0.031])
    # comps.append([ -0.0, -0.0, 0.0, 0.0, -0.0])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 0.1; n = 2$')
    # utilities2.append([ 0.866, 0.727, 0.8, 0.705, 0.722])
    # utilities16.append([ 0.296, 0.214, 0.253, 0.185, 0.194])
    # utilities32.append([ 0.157, 0.11, 0.15, 0.091, 0.127])
    # comps.append([1.681, 1.2, 1.549, 1.316, 1.493])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 0.5; n = 2$')
    # utilities2.append([ 0.907, 0.873, 0.824, 0.823, 0.879])
    # utilities16.append([0.441, 0.375, 0.342, 0.318, 0.382])
    # utilities32.append([0.28, 0.229, 0.219, 0.189, 0.25])
    # comps.append([2.043, 1.908, 2.073, 1.881, 1.993])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 1; n = 2$')
    # utilities2.append([ 0.922, 0.888, 0.838, 0.862, 0.892])
    # utilities16.append([0.497, 0.422, 0.353, 0.392, 0.443])
    # utilities32.append([0.321, 0.293, 0.248, 0.264, 0.314])
    # comps.append([2.403, 2.3, 2.391, 2.254, 2.298])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 1.5; n = 2$')
    # utilities2.append([ 0.931, 0.902, 0.851, 0.884, 0.92])
    # utilities16.append([0.499, 0.452, 0.402, 0.426, 0.522])
    # utilities32.append([ 0.345, 0.282, 0.284, 0.296, 0.405])
    # comps.append([2.425, 2.458, 2.722, 2.454, 2.48])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 2.0; n = 2$')
    # utilities2.append([ 0.932, 0.909, 0.852, 0.891, 0.932])
    # utilities16.append([ 0.53, 0.477, 0.417, 0.49, 0.534])
    # utilities32.append([0.367, 0.334, 0.274, 0.311, 0.404])
    # comps.append([2.587, 2.758, 2.467, 2.566, 2.66])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 3.0; n = 2$')
    # utilities2.append([ 0.947, 0.902, 0.869, 0.904, 0.945])
    # utilities16.append([0.581, 0.475, 0.404, 0.529, 0.589])
    # utilities32.append([0.405, 0.343, 0.302, 0.369, 0.437])
    # comps.append([2.867, 2.847, 2.612, 3.257, 3.258])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 10; n = 2$')
    # utilities2.append([ 0.951, 0.918, 0.9, 0.917, 0.959])
    # utilities16.append([ 0.622, 0.481, 0.485, 0.543, 0.642])
    # utilities32.append([ 0.46, 0.368, 0.375, 0.4, 0.516])
    # comps.append([3.491, 3.085, 3.435, 3.279, 3.665])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')

    ############################################################################
    #####           N = 4
    ############################################################################
    # n = 4
    # labels.append('$\lambda_I = 0; n = 4$')
    # utilities2.append([0.867, 0.498, 0.823, 0.51, 0.491])
    # utilities16.append([0.306, 0.067, 0.304, 0.071, 0.062])
    # utilities32.append([0.178, 0.034, 0.167, 0.027, 0.031])
    # comps.append([1.507, -0.0, 1.517, -0.0, 0.0])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 0.1; n = 4$')
    # utilities2.append([0.884, 0.782, 0.82, 0.766, 0.83])
    # utilities16.append([0.346, 0.283, 0.274, 0.234, 0.293])
    # utilities32.append([0.188, 0.156, 0.171, 0.124, 0.184])
    # comps.append([1.99, 1.78, 2.04, 1.149, 1.88])  # Done
    # # comps.append([0.1 for _ in range(5)])  # Hacking in lambdas
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 0.5; n = 4$')
    # utilities2.append([0.918, 0.898, 0.845, 0.877, 0.902])
    # utilities16.append([0.505, 0.435, 0.372, 0.398, 0.441])
    # utilities32.append([0.325, 0.293, 0.257, 0.263, 0.318])
    # comps.append([2.182, 2.499, 2.316, 2.325, 2.125])  # Done
    # # comps.append([0.5 for _ in range(5)])  # Hacking in lambdas
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 1; n = 4$')
    # utilities2.append([ 0.933, 0.918, 0.859, 0.902, 0.941])
    # utilities16.append([0.547, 0.509, 0.404, 0.5, 0.588])
    # utilities32.append([0.393, 0.357, 0.288, 0.346, 0.458])
    # comps.append([2.834, 2.652, 2.783, 2.963, 2.859])  # Done
    # # comps.append([1.0 for _ in range(5)])  # Hacking in lambdas
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 1.5; n = 4$')
    # utilities2.append([ 0.946, 0.92, 0.909, 0.927, 0.959])
    # utilities16.append([0.604, 0.525, 0.508, 0.565, 0.66])
    # utilities32.append([0.427, 0.361, 0.379, 0.432, 0.525])
    # comps.append([2.962, 2.648, 3.195, 3.28, 3.142])  # Done
    # # comps.append([1.5 for _ in range(5)])  # Hacking in lambdas
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 2.0; n = 4$')
    # utilities2.append([ 0.958, 0.921, 0.916, 0.938, 0.97])
    # utilities16.append([0.652, 0.524, 0.559, 0.628, 0.724])
    # utilities32.append([0.481, 0.381, 0.437, 0.475, 0.605])
    # comps.append([3.696, 3.526, 3.096, 3.699, 3.283])  # Done
    # # comps.append([2.0 for _ in range(5)])  # Hacking in lambdas
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 3.0; n = 4$')
    # utilities2.append([0.969, 0.948, 0.923, 0.964, 0.981])
    # utilities16.append([0.733, 0.619, 0.582, 0.728, 0.805])
    # utilities32.append([0.582, 0.495, 0.456, 0.592, 0.689])
    # comps.append([3.223, 3.555, 3.439, 3.96, 3.41])  # Done
    # # comps.append([3.0 for _ in range(5)])  # Hacking in lambdas
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 10; n = 4$')
    # utilities2.append([0.982, 0.965, 0.95, 0.985, 0.99])
    # utilities16.append([0.833, 0.731, 0.698, 0.863, 0.89])
    # utilities32.append([0.704, 0.628, 0.605, 0.78, 0.818])
    # comps.append([ 4.634, 5.124, 4.955, 4.345, 3.665])  # Done
    # # comps.append([10 for _ in range(5)])  # Hacking in lambdas
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 100; n = 4$')
    # utilities2.append([0.982, 0.966, 0.952, 0.984, 0.993])
    # utilities16.append([ 0.805, 0.73, 0.733, 0.851, 0.903])
    # utilities32.append([0.716, 0.623, 0.644, 0.763, 0.851])
    # comps.append([  4.912, 4.693, 3.577, 4.992, 4.423])  # Done
    # # comps.append([100 for _ in range(5)])  # Hacking in lambdas
    # sizes.append(small_size)
    # colors.append('xkcd:blue')

    vq_util2 = utilities2
    vq_util16 = utilities16
    vq_util32 = utilities32
    vq_comps = comps
    utilities2 = []
    utilities16 = []
    utilities32 = []
    comps = []
    # labels.append('$\lambda_I = 1; n = 8$')
    # utilities2.append([0.924, 0.882, 0.933, 0.962])
    # utilities16.append([0.524, 0.503, 0.591, 0.679])
    # utilities32.append([0.389, 0.379, 0.424, 0.574])
    # comps.append([3.092, 3.625, 3.259, 3.166])
    # sizes.append(small_size)
    # colors.append('xkcd:green')
    #
    # labels.append('$\lambda_I = 10; n = 8$')
    # utilities2.append([0.992, 0.981, 0.984, 0.986, 0.996])
    # utilities16.append([ 0.924, 0.863, 0.868, 0.881, 0.957])
    # utilities32.append([0.869, 0.796, 0.797, 0.827, 0.93])
    # comps.append([4.681, 4.35, 4.025, 4.363, 4.747])
    # sizes.append(small_size)
    # colors.append('xkcd:green')
    #
    # labels.append('$\lambda_I = 100; n = 8$')
    # utilities2.append([0.997, 0.991, 0.895, 0.994, 0.998])
    # utilities16.append([0.948, 0.905, 0.507, 0.954, 0.969])
    # utilities32.append([0.911, 0.858, 0.436, 0.916, 0.939])
    # comps.append([4.038, 5.248, 3.155, 5.119, 3.319])
    # sizes.append(small_size)
    # colors.append('xkcd:black')

    ############################################################################
    #####           N = 8
    ############################################################################
    # n = 8
    # labels.append('$\lambda_I = 0; n = 8$')
    # utilities2.append([0.907, 0.493, 0.485, 0.499, 0.866])
    # utilities16.append([0.333, 0.071, 0.072, 0.063, 0.352])
    # utilities32.append([0.218, 0.038, 0.03, 0.031, 0.222])
    # comps.append([1.553, -0.0, -0.0, -0.0, 1.622])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 0.1; n = 8$')
    # utilities2.append([ 0.895, 0.779, 0.834, 0.824, 0.852])
    # utilities16.append([0.363, 0.281, 0.3, 0.293, 0.318])
    # utilities32.append([0.227, 0.168, 0.189, 0.191, 0.195])
    # comps.append([1.813, 1.53, 1.728, 1.776, 1.589])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 0.5; n = 8$')
    # utilities2.append([ 0.922, 0.912, 0.85, 0.906, 0.94])
    # utilities16.append([ 0.482, 0.463, 0.378, 0.505, 0.587])
    # utilities32.append([  0.316, 0.328, 0.291, 0.332, 0.438])
    # comps.append([ 2.126, 2.28, 2.201, 2.502, 2.827])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 1; n = 8$')
    # utilities2.append([0.948, 0.936, 0.878, 0.944, 0.945])
    # utilities16.append([0.586, 0.545, 0.48, 0.656, 0.602])
    # utilities32.append([0.448, 0.412, 0.38, 0.487, 0.475])
    # comps.append([ 2.884, 3.16, 3.221, 3.308, 3.062])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 1.5; n = 8$')
    # utilities2.append([0.964, 0.939, 0.929, 0.966, 0.983])
    # utilities16.append([ 0.672, 0.586, 0.602, 0.716, 0.834])
    # utilities32.append([ 0.536, 0.455, 0.511, 0.596, 0.759])
    # comps.append([3.124, 3.448, 4.164, 3.883, 3.793])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 2.0; n = 8$')
    # utilities2.append([ 0.96, 0.942, 0.93, 0.973, 0.984])
    # utilities16.append([  0.664, 0.605, 0.611, 0.786, 0.813])
    # utilities32.append([0.53, 0.477, 0.532, 0.678, 0.727])
    # comps.append([ 3.612, 3.618, 3.957, 4.084, 3.863])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 3.0; n = 8$')
    # utilities2.append([ 0.976, 0.95, 0.933, 0.974, 0.992])
    # utilities16.append([ 0.777, 0.694, 0.632, 0.748, 0.912])
    # utilities32.append([ 0.665, 0.592, 0.54, 0.637, 0.845])
    # comps.append([ 4.456, 3.311, 4.16, 4.543, 4.493])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 10; n = 8$')
    # utilities2.append([0.995, 0.981, 0.986, 0.992, 0.996])
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
    # utilities2.append([0.914, 0.848, 0.486, 0.502, 0.907])
    # utilities16.append([0.391, 0.29, 0.071, 0.063, 0.516])
    # utilities32.append([0.239, 0.173, 0.029, 0.031, 0.368])
    # comps.append([1.928, 1.227, 0.0, -0.0, 1.932])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 0.1; n = 16$')
    # utilities2.append([0.91, 0.889, 0.867, 0.498, 0.907])
    # utilities16.append([ 0.419, 0.428, 0.384, 0.063, 0.425])
    # utilities32.append([0.273, 0.284, 0.271, 0.03, 0.301])
    # comps.append([1.862, 1.579, 1.955, -0.0, 1.679])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 0.5; n = 16$')
    # utilities2.append([ 0.947, 0.922, 0.864, 0.924, 0.954])
    # utilities16.append([ 0.547, 0.492, 0.429, 0.529, 0.638])
    # utilities32.append([ 0.401, 0.347, 0.329, 0.37, 0.497])
    # comps.append([2.394, 2.143, 2.707, 2.454, 2.685])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 1; n = 16$')
    # utilities2.append([0.958, 0.934, 0.902, 0.941, 0.973])
    # utilities16.append([0.634, 0.548, 0.536, 0.621, 0.75])
    # utilities32.append([ 0.478, 0.405, 0.427, 0.505, 0.63])
    # comps.append([2.915, 2.485, 3.422, 2.821, 3.037])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 1.5; n = 16$')
    # utilities2.append([0.97, 0.932, 0.968, 0.927, 0.993])
    # utilities16.append([ 0.702, 0.578, 0.783, 0.586, 0.92])
    # utilities32.append([ 0.547, 0.45, 0.715, 0.466, 0.868])
    # comps.append([3.31, 3.021, 4.041, 3.5, 3.319])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 2.0; n = 16$')
    # utilities2.append([ 0.977, 0.931, 0.987, 0.959, 0.992])
    # utilities16.append([ 0.73, 0.568, 0.861, 0.724, 0.909])
    # utilities32.append([0.63, 0.436, 0.797, 0.606, 0.846])
    # comps.append([ 3.847, 3.184, 4.175, 3.823, 3.878])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 3.0; n = 16$')
    # utilities2.append([0.982, 0.952, 0.986, 0.973, 0.992])
    # utilities16.append([ 0.808, 0.707, 0.824, 0.773, 0.92])
    # utilities32.append([0.723, 0.587, 0.768, 0.647, 0.866])
    # comps.append([4.35, 4.017, 4.986, 3.709, 2.619])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 10; n = 16$')
    # utilities2.append([ 0.992, 0.992, 0.986, 0.997, 0.998])
    # utilities16.append([ 0.913, 0.906, 0.85, 0.952, 0.976])
    # utilities32.append([0.863, 0.869, 0.805, 0.916, 0.951])
    # comps.append([4.31, 4.316, 4.773, 3.781, 4.276])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')

    ############################################################################
    #####         VQ-VIB2  N = 1
    ############################################################################
    n = 1
    labels.append('$\lambda_I = 0.1; n = 1$')
    utilities2.append([ 0.503, 0.489, 0.482, 0.494])
    utilities16.append([ 0.064, 0.07, 0.061, 0.052])
    utilities32.append([ 0.028, 0.032, 0.03, 0.031])
    # comps.append([])  # Done
    comps.append([0.1 for _ in range(5)])  # Hacking in lambdas
    sizes.append(small_size)
    colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 0.5; n = 1$')
    # utilities2.append([])
    # utilities16.append([])
    # utilities32.append([])
    # comps.append([])  # Done
    # # comps.append([0.5 for _ in range(5)])  # Hacking in lambdas
    # sizes.append(small_size)
    # colors.append('xkcd:blue')

    labels.append('$\lambda_I = 1; n = 1$')
    utilities2.append([ 0.548, 0.52, 0.642, 0.532, 0.613])
    utilities16.append([ 0.106, 0.078, 0.123, 0.107, 0.137])
    utilities32.append([ 0.057, 0.041, 0.056, 0.045, 0.082])
    # comps.append([])  # Done
    comps.append([1.0 for _ in range(5)])  # Hacking in lambdas
    sizes.append(small_size)
    colors.append('xkcd:blue')

    # labels.append('$\lambda_I = 1.5; n = 1$')
    # utilities2.append([])
    # utilities16.append([])
    # utilities32.append([])
    # comps.append([])  # Done
    # # comps.append([1.5 for _ in range(5)])  # Hacking in lambdas
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 2.0; n = 1$')
    # utilities2.append([])
    # utilities16.append([])
    # utilities32.append([])
    # comps.append([])  # Done
    # # comps.append([2.0 for _ in range(5)])  # Hacking in lambdas
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 3.0; n = 1$')
    # utilities2.append([])
    # utilities16.append([])
    # utilities32.append([])
    # comps.append([])  # Done
    # # comps.append([3.0 for _ in range(5)])  # Hacking in lambdas
    # sizes.append(small_size)
    # colors.append('xkcd:blue')

    labels.append('$\lambda_I = 10; n = 1$')
    utilities2.append([ 0.829, 0.838, 0.758, 0.827, 0.791])
    utilities16.append([ 0.286, 0.327, 0.216, 0.3, 0.266])
    utilities32.append([ 0.158, 0.207, 0.139, 0.183, 0.163])
    # comps.append([])  # Done
    comps.append([10 for _ in range(5)])  # Hacking in lambdas
    sizes.append(small_size)
    colors.append('xkcd:blue')

    labels.append('$\lambda_I = 100; n = 1$')
    utilities2.append([ 0.895, 0.873, 0.812, 0.881, 0.909])
    utilities16.append([ 0.433, 0.362, 0.324, 0.405, 0.437])
    utilities32.append([ 0.27, 0.237, 0.219, 0.261, 0.318])
    # comps.append([])
    comps.append([100 for _ in range(5)])  # Hacking in lambdas
    sizes.append(small_size)
    colors.append('xkcd:blue')

    vqvib2_util2 = utilities2
    vqvib2_util16 = utilities16
    vqvib2_util32 = utilities32
    vqvib2_comps = comps
    utilities2 = []
    utilities16 = []
    utilities32 = []
    comps = []

    ############################################################################
    #####         VQ-VIB2  N = 2
    ############################################################################
    # n = 2
    # labels.append('$\lambda_I = 0.1; n = 2$')
    # utilities2.append([ 0.513, 0.515, 0.495, 0.499, 0.505])
    # utilities16.append([ 0.06, 0.061, 0.074, 0.052, 0.061])
    # utilities32.append([ 0.029, 0.032, 0.031, 0.034, 0.028])
    # comps.append([])  # Done
    # # comps.append([0.1 for _ in range(5)])  # Hacking in lambdas
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    # #
    # # labels.append('$\lambda_I = 0.5; n = 2$')
    # # utilities2.append([])
    # # utilities16.append([])
    # # utilities32.append([])
    # # comps.append([])  # Done
    # # # comps.append([0.5 for _ in range(5)])  # Hacking in lambdas
    # # sizes.append(small_size)
    # # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 1; n = 2$')
    # utilities2.append([ 0.824, 0.83, 0.696, 0.707, 0.836])
    # utilities16.append([ 0.25, 0.311, 0.215, 0.193, 0.287])
    # utilities32.append([ 0.143, 0.2, 0.141, 0.118, 0.196])
    # comps.append([])  # Done
    # # comps.append([1.0 for _ in range(5)])  # Hacking in lambdas
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # # labels.append('$\lambda_I = 1.5; n = 2$')
    # # utilities2.append([])
    # # utilities16.append([])
    # # utilities32.append([])
    # # comps.append([])  # Done
    # # # comps.append([1.5 for _ in range(5)])  # Hacking in lambdas
    # # sizes.append(small_size)
    # # colors.append('xkcd:blue')
    # #
    # # labels.append('$\lambda_I = 2.0; n = 2$')
    # # utilities2.append([])
    # # utilities16.append([])
    # # utilities32.append([])
    # # comps.append([])  # Done
    # # # comps.append([2.0 for _ in range(5)])  # Hacking in lambdas
    # # sizes.append(small_size)
    # # colors.append('xkcd:blue')
    # #
    # # labels.append('$\lambda_I = 3.0; n = 2$')
    # # utilities2.append([])
    # # utilities16.append([])
    # # utilities32.append([])
    # # comps.append([])  # Done
    # # # comps.append([3.0 for _ in range(5)])  # Hacking in lambdas
    # # sizes.append(small_size)
    # # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 10; n = 2$')
    # utilities2.append([ 0.912, 0.917, 0.833, 0.885, 0.921])
    # utilities16.append([  0.494, 0.452, 0.347, 0.431, 0.496])
    # utilities32.append([  0.341, 0.309, 0.248, 0.285, 0.369])
    # comps.append([])  # Done
    # # comps.append([10 for _ in range(5)])  # Hacking in lambdas
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 100; n = 2$')
    # utilities2.append([  0.923, 0.925, 0.855, 0.931, 0.949])
    # utilities16.append([  0.559, 0.504, 0.382, 0.548, 0.633])
    # utilities32.append([  0.403, 0.367, 0.271, 0.393, 0.517])
    # comps.append([])
    # # comps.append([100 for _ in range(5)])  # Hacking in lambdas
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # vqvib2_util2 = utilities2
    # vqvib2_util16 = utilities16
    # vqvib2_util32 = utilities32
    # vqvib2_comps = comps
    # utilities2 = []
    # utilities16 = []
    # utilities32 = []
    # comps = []

    ############################################################################
    #####         VQ-VIB2  N = 4
    ############################################################################
    # n = 4
    # labels.append('$\lambda_I = 0.1; n = 4$')
    # utilities2.append([ 0.512, 0.515, 0.493, 0.493, 0.5])
    # utilities16.append([ 0.059, 0.061, 0.07, 0.052, 0.059])
    # utilities32.append([ 0.03, 0.032, 0.032, 0.033, 0.029])
    # comps.append([])  # Done
    # # comps.append([0.1 for _ in range(5)])  # Hacking in lambdas
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    # #
    # # labels.append('$\lambda_I = 0.5; n = 4$')
    # # utilities2.append([])
    # # utilities16.append([])
    # # utilities32.append([])
    # # comps.append([])  # Done
    # # # comps.append([0.5 for _ in range(5)])  # Hacking in lambdas
    # # sizes.append(small_size)
    # # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 1; n = 4$')
    # utilities2.append([ 0.912, 0.884, 0.829, 0.832, 0.873])
    # utilities16.append([ 0.426, 0.397, 0.331, 0.339, 0.373])
    # utilities32.append([ 0.296, 0.263, 0.222, 0.207, 0.235])
    # comps.append([])  # Done
    # # comps.append([1.0 for _ in range(5)])  # Hacking in lambdas
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # # labels.append('$\lambda_I = 1.5; n = 4$')
    # # utilities2.append([])
    # # utilities16.append([])
    # # utilities32.append([])
    # # comps.append([])  # Done
    # # # comps.append([1.5 for _ in range(5)])  # Hacking in lambdas
    # # sizes.append(small_size)
    # # colors.append('xkcd:blue')
    # #
    # # labels.append('$\lambda_I = 2.0; n = 4$')
    # # utilities2.append([])
    # # utilities16.append([])
    # # utilities32.append([])
    # # comps.append([])  # Done
    # # # comps.append([2.0 for _ in range(5)])  # Hacking in lambdas
    # # sizes.append(small_size)
    # # colors.append('xkcd:blue')
    # #
    # # labels.append('$\lambda_I = 3.0; n = 4$')
    # # utilities2.append([])
    # # utilities16.append([])
    # # utilities32.append([])
    # # comps.append([])  # Done
    # # # comps.append([3.0 for _ in range(5)])  # Hacking in lambdas
    # # sizes.append(small_size)
    # # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 10; n = 4$')
    # utilities2.append([ 0.934, 0.921, 0.875, 0.92, 0.949])
    # utilities16.append([ 0.6, 0.562, 0.433, 0.556, 0.62])
    # utilities32.append([ 0.431, 0.399, 0.334, 0.412, 0.486])
    # comps.append([])  # Done
    # # comps.append([10 for _ in range(5)])  # Hacking in lambdas
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 100; n = 4$')
    # utilities2.append([ 0.946, 0.95, 0.922, 0.95, 0.975])
    # utilities16.append([ 0.622, 0.632, 0.543, 0.686, 0.77])
    # utilities32.append([ 0.462, 0.496, 0.436, 0.553, 0.677])
    # comps.append([])
    # # comps.append([100 for _ in range(5)])  # Hacking in lambdas
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # vqvib2_util2 = utilities2
    # vqvib2_util16 = utilities16
    # vqvib2_util32 = utilities32
    # vqvib2_comps = comps
    # utilities2 = []
    # utilities16 = []
    # utilities32 = []
    # comps = []

    for data, suffix in zip([utilities2, utilities16, utilities32], ['2', '16', '32']):
        plot_multi_trials([comps, data], labels, sizes, ylabel='Utility', colors=None, filename='ood_comp' + suffix + '.png')
    labels = ['$C=2$'] + ['' for _ in utilities2[:-1]] + ['$C=16$'] + ['' for _ in utilities16[:-1]] + ['$C=32$'] + ['' for _ in utilities32[:-1]]
    sizes = [2 * small_size for _ in utilities2] + [2 * small_size for _ in utilities16] + [2 * small_size for _ in utilities32]
    plot_multi_trials([comps + comps + comps, utilities2 + utilities16 + utilities32], labels, sizes, ylabel='OOD Utility', colors=None, filename='ood_comp_all_n' + str(n) + '.png')

    labels = ['Proto.'] + ['' for _ in proto_comps[:-1]] +\
             ['Onehot'] + ['' for _ in onehot_comps[:-1]] +\
             ['VQ-VIB'] + ['' for _ in vq_comps[:-1]] +\
            ['VQ-VIB$^*$'] + ['' for _ in vqvib2_comps[:-1]]
    sizes = [2 * small_size for _ in proto_comps] + [2 * small_size for _ in onehot_comps] + [2 * small_size for _ in vq_comps] + [2 * small_size for _ in vqvib2_comps]
    plot_multi_trials([proto_comps + onehot_comps + vq_comps + vqvib2_comps, proto_util32 + onehot_util32 + vq_util32 + vqvib2_util32], labels, sizes,
                      ylabel='OOD Utility', colors=None, filename='ood_comp_all_multimodel_32.png')

    # labels = ['Proto.'] + ['' for _ in proto_comps[:-1]] +\
    #          ['VQ-VIB'] + ['' for _ in vq_comps[:-1]]
    # sizes = [2 * small_size for _ in proto_comps] + [2 * small_size for _ in vq_comps]
    # plot_multi_trials([proto_comps + vq_comps, proto_util16 + vq_util16], labels, sizes,
    #                   ylabel='OOD Utility', colors=None, filename='ood_comp_all_multimodel_16.png')

    return comps, utilities2, utilities16, utilities32


if __name__ == '__main__':
    run()
