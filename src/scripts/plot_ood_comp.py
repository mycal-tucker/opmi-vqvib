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

    # labels.append('$\lambda_I = 0.1; n = 1$')
    # utilities2.append([0.833, 0.757, 0.71, 0.58, 0.695])
    # utilities16.append([0.265, 0.22, 0.202, 0.112, 0.197])
    # utilities32.append([0.146, 0.123, 0.131, 0.056, 0.12])
    # comps.append([1.284, 1.141, 1.347, 0.995, 1.142])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')
    #
    # labels.append('$\lambda_I = 1; n = 1$')
    # utilities2.append([0.892, 0.856, 0.822, 0.846, 0.872])
    # utilities16.append([0.431, 0.343, 0.323, 0.338, 0.386])
    # utilities32.append([0.244, 0.218, 0.208, 0.186, 0.262])
    # comps.append([2.13, 1.899, 2.113, 1.94, 2.031])
    # sizes.append(small_size)
    # colors.append('xkcd:blue')

    labels.append('$\lambda_I = 0; n = 4$')
    utilities2.append([0.867, 0.498, 0.823, 0.51, 0.491])
    utilities16.append([0.306, 0.067, 0.304, 0.071, 0.062])
    utilities32.append([0.178, 0.034, 0.167, 0.027, 0.031])
    comps.append([1.507, -0.0, 1.517, -0.0, 0.0])
    sizes.append(small_size)
    colors.append('xkcd:blue')


    labels.append('$\lambda_I = 0.1; n = 4$')
    utilities2.append([0.884, 0.782, 0.82, 0.766, 0.83])
    utilities16.append([0.346, 0.283, 0.274, 0.234, 0.293])
    utilities32.append([0.188, 0.156, 0.171, 0.124, 0.184])
    comps.append([1.748, 1.476, 1.751, 1.149, 1.65])
    sizes.append(small_size)
    colors.append('xkcd:blue')

    labels.append('$\lambda_I = 1; n = 4$')
    utilities2.append([ 0.933, 0.918, 0.859, 0.902, 0.941])
    utilities16.append([0.547, 0.509, 0.404, 0.5, 0.588])
    utilities32.append([0.393, 0.357, 0.288, 0.346, 0.458])
    comps.append([2.734, 2.652, 2.683, 2.763, 2.809])
    sizes.append(small_size)
    colors.append('xkcd:blue')


    labels.append('$\lambda_I = 10; n = 4$')
    utilities2.append([0.982, 0.965, 0.95, 0.985, 0.99])
    utilities16.append([0.833, 0.731, 0.698, 0.863, 0.89])
    utilities32.append([0.704, 0.628, 0.605, 0.78, 0.818])
    comps.append([ 4.634, 4.924, 4.915, 4.245, 3.565])
    sizes.append(small_size)
    colors.append('xkcd:blue')

    labels.append('$\lambda_I = 100; n = 4$')
    utilities2.append([ 0.982, 0.966, 0.952, 0.984, 0.993])
    utilities16.append([0.805, 0.73, 0.733, 0.851, 0.903])
    utilities32.append([0.716, 0.623, 0.644, 0.763, 0.851])
    comps.append([ 4.812, 4.593, 3.557, 4.942, 4.373])
    sizes.append(small_size)
    colors.append('xkcd:orange')

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

    for data, suffix in zip([utilities2, utilities16, utilities32], ['2', '16', '32']):
        plot_multi_trials([comps, data], labels, sizes, ylabel='Utility', colors=None, filename='ood_comp' + suffix + '.png')


if __name__ == '__main__':
    run()
