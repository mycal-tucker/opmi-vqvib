from src.utils.plotting import plot_multi_trials


def run():
    labels = []
    infos = []
    comps = []
    sizes = []
    colors = []

    big_size = 200
    small_size = 100

    # English topname
    labels.append('English (Topname)')
    infos.append([-0.20])
    comps.append([1.9])
    sizes.append(big_size)
    colors.append('xkcd:black')

    # English VG Domain.
    labels.append('English (VG)')
    infos.append([-0.28])
    comps.append([1.3])
    sizes.append(big_size)
    colors.append('xkcd:gray')

    labels.append('$\lambda_I=0.1; n=1$')
    infos.append([-0.313, -0.291, -0.297, -0.303, -0.301])
    comps.append([1.154, 1.097, 1.166, 1.064, 1.046])
    sizes.append(small_size)
    colors.append('xkcd:green')

    labels.append('$\lambda_I=1.0; n=1$')
    infos.append([-0.213, -0.212, -0.209, -0.21, -0.211])
    comps.append([1.872, 1.853, 1.883, 1.957, 1.882])
    sizes.append(small_size)
    colors.append('xkcd:teal')

    labels.append('$\lambda_I=0.1; n=4$')
    infos.append([-0.311, -0.278, -0.298, -0.296, -0.29])
    comps.append([1.389, 1.407, 1.332, 1.314, 1.322])
    sizes.append(small_size)
    colors.append('xkcd:orange')

    labels.append('$\lambda_I=1.0; n=4$')
    infos.append([-0.149, -0.149, -0.143, -0.153, -0.155])
    comps.append([2.642, 2.551, 2.651, 2.694, 2.458])
    sizes.append(small_size)
    colors.append('xkcd:red')

    plot_multi_trials([comps, infos], labels, sizes, ylabel='Negative MSE', colors=colors, filename='dummyinfoplane.png')


if __name__ == '__main__':
    run()
