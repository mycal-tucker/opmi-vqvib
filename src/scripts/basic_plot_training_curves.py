import matplotlib.pyplot as plt
import math


def run():
    labels = []
    epochs = []
    utilities = []
    stds = []
    colors = []
    markers = []

    basic_epoch = [9999, 19999, 29999, 39999, 49999]

    # VG Domain VQ alpha0 1tok
    labels.append('VQ $\lambda_I=0$ 1tok')
    epochs.append(basic_epoch)
    utilities.append([0.875, 0.898, 0.897, 0.897, 0.894])
    stds.append([0.068, 0.072, 0.069, 0.069, 0.069])
    colors.append('xkcd:green')
    markers.append('s')

    # VG Domain VQ alpha10 1tok
    labels.append('VQ $\lambda_I=10$ 1tok')
    epochs.append(basic_epoch)
    utilities.append([0.879, 0.932, 0.937, 0.946, 0.946])
    stds.append([0.004, 0.002, 0.003, 0.002, 0.001])
    colors.append('xkcd:teal')
    markers.append('o')

    # VG Domain VQ alpha0 8tok
    labels.append('VQ $\lambda_I=0$ 8tok')
    epochs.append(basic_epoch)
    utilities.append([0.924, 0.953, 0.964, 0.961, 0.967])
    stds.append([0.006, 0.007, 0.003, 0.003, 0.002])
    colors.append('xkcd:salmon')
    markers.append('s')

    # VG Domain VQ alpha10 8tok
    labels.append('VQ $\lambda_I=10$ 8tok')
    epochs.append(basic_epoch)
    utilities.append([0.948, 0.985, 0.992, 0.996, 0.997])
    stds.append([0.004, 0.001, 0.001, 0.000, 0.000])
    colors.append('xkcd:magenta')
    markers.append('o')

    font = {'family': 'normal',
            'size': 20}

    plt.rc('font', **font)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xticks([e + 1 for e in basic_epoch])
    for l, e, u, s, c, m in zip(labels, epochs, utilities, stds, colors, markers):
        plt.errorbar(e, u, yerr=s, c=c, label=l, marker=m)
    plt.xlabel('Epoch')
    plt.ylabel('Percent Accuracy')
    plt.ylim(0.80, 1.0)
    # plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('/home/mycal/trainingcurves.png')
    plt.show()


if __name__ == '__main__':
    run()
