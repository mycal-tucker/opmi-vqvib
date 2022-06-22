import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import euclidean_distances
from sklearn.manifold import MDS, TSNE


def plot_metrics(metrics, labels):
    for i, label in enumerate(labels):
        metric_data = [metric[i] for metric in metrics]
        plt.plot(metric_data, label=label)
    plt.legend()
    plt.savefig('metrics.png')
    plt.close()


def plot_mds(all_data, labels=None):
    mds_embedder = MDS(n_components=2)
    catted = np.vstack(all_data)
    if catted.shape[0] > 500:
        print("Warning, data very long. Truncating")
        catted = catted[:500]
    # Hacky, but to visualize all the points around a cluster, we need to add noise to spread them out a tiny bit.
    # catted = np.random.normal(catted, 0.1)
    similarities = euclidean_distances(catted.astype(np.float64))
    transformed = mds_embedder.fit_transform(similarities)
    x = transformed[:, 0]
    y = transformed[:, 1]
    fig, ax = plt.subplots()
    last_idx = 0
    for data_idx, data in enumerate(all_data):
        label = None if labels is None else labels[data_idx]
        sub_x = x[last_idx: last_idx + len(data)]
        sub_y = y[last_idx: last_idx + len(data)]
        pcm = ax.scatter(sub_x, sub_y, s=20, marker='o', label=label)
        last_idx += len(data)
    plt.legend()
    plt.show()


def plot_tsne(all_data, labels=None):
    tsne = TSNE(n_components=2, learning_rate='auto', init='random')
    # Concat all data
    catted = np.vstack(all_data)
    transformed = tsne.fit_transform(catted)
    x = transformed[:, 0]
    y = transformed[:, 1]
    fig, ax = plt.subplots()
    last_idx = 0
    for data_idx, data in enumerate(all_data):
        label = None if labels is None else labels[data_idx]
        sub_x = x[last_idx: last_idx + len(data)]
        sub_y = y[last_idx: last_idx + len(data)]
        pcm = ax.scatter(sub_x, sub_y, s=20, marker='o', label=label)
        last_idx += len(data)
    plt.legend()
    plt.show()