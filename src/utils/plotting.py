import matplotlib.pyplot as plt


def plot_metrics(metrics, labels):
    for i, label in enumerate(labels):
        metric_data = [metric[i] for metric in metrics]
        plt.plot(metric_data, label=label)
    plt.legend()
    plt.savefig('metrics.png')
    plt.close()
