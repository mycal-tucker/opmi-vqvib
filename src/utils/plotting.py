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


# Helper function from stackoverflow to adjust a color's lightness.
def adjust_lightness(color, amount=0.5):
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])


def plot_naming(all_data, viz_method, labels=None, savepath=None, plot_all_colors=False):
    # The only difference between different plotting methods is the embedding version. Coloring, labeling, etc.
    # are all the same.
    assert viz_method in ['mds', 'tsne'], "Only support mds or tsne visualization"
    is_mds = viz_method == 'mds'
    embedder = MDS(n_components=2, random_state=0) if is_mds else TSNE(n_components=2, learning_rate='auto', random_state=0)
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
    transformed = embedder.fit_transform(similarities)
    transformed = transformed[undo_permutation]  # Undo the permutation for plotting, so it lines up with labels.
    x = transformed[:, 0]
    y = transformed[:, 1]
    # Rescale to be within a smaller range
    x = x / (max(x) - min(x))
    x = x - min(x)
    y = y / (max(y) - min(y))
    y = y - min(y)
    cmap = plt.get_cmap('hsv')
    colors = cmap(x)
    # Transform color by the y coordinate as well to make lower values darker
    darkness = y / 2 + 0.5
    for i, dark in enumerate(darkness):
        colors[i, :3] = adjust_lightness(colors[i, :3], dark)
    if plot_all_colors:
        # No labels, just color things.
        fig, ax = plt.subplots()
        pcm = ax.scatter(x, y, s=20, color=colors, edgecolors='black')
        plt.savefig('all_colors_' + viz_method + '.png')
        plt.close()
    fig, ax = plt.subplots()
    last_idx = 0
    for data_idx, data in enumerate(all_data):
        label = None if labels is None else labels[data_idx]
        sub_x = x[last_idx: last_idx + len(data)]
        sub_y = y[last_idx: last_idx + len(data)]
        sub_colors = colors[last_idx: last_idx + len(data)]
        mean_color = np.mean(sub_colors, axis=0)
        if label == -1 or label == '-1':
            pcm = ax.scatter(sub_x, sub_y, s=20, marker='o', facecolors='none', edgecolors='black')
        else:
            pcm = ax.scatter(sub_x, sub_y, s=20, marker='o', color=mean_color, label=label)
        last_idx += len(data)
    plt.legend()
    if savepath is not None:
        plt.savefig(savepath)
    else:
        plt.show()
    plt.close()
