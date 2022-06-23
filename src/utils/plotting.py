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


def invert_permutation(p):
    p = np.asanyarray(p)
    s = np.empty_like(p)
    s[p] = np.arange(p.size)
    return s


def plot_mds(all_data, labels=None, savepath=None):
    mds_embedder = MDS(n_components=2, random_state=0)
    catted = np.vstack(all_data)
    max_entries = 1000
    if catted.shape[0] > max_entries:
        print("Warning, data very long. Truncating")
        catted = catted[:max_entries]
    # Sort the data for reproducibility.
    sort_permutation = catted[:, 0].argsort()
    undo_permutation = invert_permutation(sort_permutation)
    catted = catted[sort_permutation]
    similarities = euclidean_distances(catted.astype(np.float64))
    transformed = mds_embedder.fit_transform(similarities)
    transformed = transformed[undo_permutation]  # Undo the permutation for plotting, so it lines up with labels.
    x = transformed[:, 0]
    y = transformed[:, 1]
    fig, ax = plt.subplots()
    last_idx = 0
    for data_idx, data in enumerate(all_data):
        label = None if labels is None else labels[data_idx]
        sub_x = x[last_idx: last_idx + len(data)]
        sub_y = y[last_idx: last_idx + len(data)]
        if label == -1 or label == '-1':
            pcm = ax.scatter(sub_x, sub_y, s=20, marker='o', facecolors='none', edgecolors='black', label=label)
        else:
            pcm = ax.scatter(sub_x, sub_y, s=20, marker='o', label=label)
        last_idx += len(data)
    plt.legend()
    if savepath is not None:
        plt.savefig(savepath)
    else:
        plt.show()
    plt.close()


def plot_tsne(all_data, labels=None, savepath=None):
    tsne = TSNE(n_components=2, learning_rate='auto', random_state=0)
    # Concat all data
    catted = np.vstack(all_data)
    # Sort the data for reproducibility.
    sort_permutation = catted[:, 0].argsort()
    undo_permutation = invert_permutation(sort_permutation)
    catted = catted[sort_permutation]
    transformed = tsne.fit_transform(catted)
    transformed = transformed[undo_permutation]  # Undo the permutation for plotting, so it lines up with labels.
    x = transformed[:, 0]
    y = transformed[:, 1]
    fig, ax = plt.subplots()
    last_idx = 0
    for data_idx, data in enumerate(all_data):
        label = None if labels is None else labels[data_idx]
        sub_x = x[last_idx: last_idx + len(data)]
        sub_y = y[last_idx: last_idx + len(data)]
        if label == -1 or label == '-1':
            pcm = ax.scatter(sub_x, sub_y, s=20, marker='o', facecolors='none', edgecolors='black', label=label)
        else:
            pcm = ax.scatter(sub_x, sub_y, s=20, marker='o', label=label)
        last_idx += len(data)
    plt.legend()
    if savepath is not None:
        plt.savefig(savepath)
    else:
        plt.show()
    plt.close()
