import matplotlib.pyplot as plt
import math


def run():
    labels = []
    epochs = []
    utilities = []
    stds = []
    colors = []
    markers = []

    basic_epoch = [4999, 9999, 14999, 19999]

    labels.append('$\lambda_I=0.1; n=1$')
    epochs.append(basic_epoch)
    utilities.append([0.503, 0.779, 0.854, 0.861])
    stds.append([0.003, 0.005, 0.003, 0.002])
    colors.append('xkcd:green')
    markers.append('s')

    labels.append('$\lambda_I=1.0; n=1$')
    epochs.append(basic_epoch)
    utilities.append([0.861, 0.894, 0.916, 0.923])
    stds.append([0.004, 0.003, 0.004, 0.002])
    colors.append('xkcd:teal')
    markers.append('o')

    labels.append('$\lambda_I=0.1; n=4$')
    epochs.append(basic_epoch)
    utilities.append([0.810, 0.884, 0.878, 0.878])
    stds.append([0.020, 0.002, 0.005, 0.005])
    colors.append('xkcd:orange')
    markers.append('s')

    labels.append('$\lambda_I=1.0; n=4$')
    epochs.append(basic_epoch)
    utilities.append([0.929, 0.945, 0.943, 0.954])
    stds.append([0.004, 0.002, 0.001, 0.003])
    colors.append('xkcd:red')
    markers.append('o')


    font = {'family': 'normal',
            'size': 20}

    plt.rc('font', **font)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xticks([e + 1 for e in basic_epoch])
    for l, e, u, s, c, m in zip(labels, epochs, utilities, stds, colors, markers):
        plt.errorbar(e, u, yerr=s, c=c, label=l, marker=m)
    plt.xlabel('Epoch')
    plt.ylabel('Utility')
    plt.ylim(0.65, 1.0)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('trainingcurves.png')
    plt.show()


if __name__ == '__main__':
    run()
