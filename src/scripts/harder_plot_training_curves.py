import matplotlib.pyplot as plt
import math


# Non-distinct

def run():
    labels = []
    epochs = []
    utilities = []
    stds = []
    colors = []
    markers = []

    basic_epoch = [9999, 19999, 29999, 39999, 49999]

    # VQ
    # VG Domain, alpha 10 8tok 2 candidates
    labels.append('VQ $C=2$')
    epochs.append(basic_epoch)
    utilities.append([0.930, 0.978, 0.988, 0.994, 0.994])
    stds.append([0.006, 0.002, 0.001, 0.000, 0.000])
    colors.append('xkcd:blue')
    markers.append('o')

    # VG Domain, alpha 10 8tok 16 candidates
    labels.append('VQ $C=16$')
    epochs.append(basic_epoch)
    utilities.append([0.532, 0.755, 0.85, 0.908, 0.930])
    stds.append([0.023, 0.022, 0.014, 0.004, 0.004])
    colors.append('xkcd:magenta')
    markers.append('o')

    # VG Domain, alpha 10 8tok 32 candidates
    labels.append('VQ $C=32$')
    epochs.append(basic_epoch)
    utilities.append([0.359, 0.623, 0.756, 0.850, 0.869])
    stds.append([0.026, 0.030, 0.024, 0.008, 0.005])
    colors.append('xkcd:red')
    markers.append('o')

    # Onehot
    # VG Domain, alpha 10 8tok 2 candidates
    labels.append('Onehot $C=2$')
    epochs.append(basic_epoch)
    utilities.append([0.963, 0.981, 0.985, 0.986, 0.986])
    stds.append([0.002, 0.001, 0.001, 0.000, 0.000])
    colors.append('xkcd:blue')
    markers.append('s')

    # VG Domain, alpha 10 8tok 16 candidates
    labels.append('Onehot $C=16$')
    epochs.append(basic_epoch)
    utilities.append([0.685, 0.800, 0.822, 0.877, 0.871])
    stds.append([0.012, 0.012, 0.010, 0.009, 0.003])
    colors.append('xkcd:magenta')
    markers.append('s')

    # VG Domain, alpha 10 8tok 32 candidates
    labels.append('Onehot $C=32$')
    epochs.append(basic_epoch)
    utilities.append([0.508, 0.672, 0.707, 0.788, 0.784])
    stds.append([0.007, 0.014, 0.016, 0.010, 0.006])
    colors.append('xkcd:red')
    markers.append('s')

    font = {'family': 'normal',
            'size': 20}

    plt.rc('font', **font)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xticks([e + 1 for e in basic_epoch])
    for l, e, u, s, c, m in zip(labels, epochs, utilities, stds, colors, markers):
        plt.errorbar(e, u, yerr=s, c=c, label=l, marker=m)
    plt.xlabel('Epoch')
    plt.ylabel('Percent Accuracy')
    plt.ylim(0.0, 1.0)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('/home/mycal/hardertrainingcurves.png')
    plt.show()


if __name__ == '__main__':
    run()
