
from src.data_utils.read_data import get_feature_data
from src.utils.plotting import plot_naming
import numpy as np


def run(data, cutoff):
    features = data['features']
    topnames = data['topname']
    responses = data['responses']
    # Rewrite the topnames entry to be -1 if no single name achieved greater than some threshold.
    for i, response in enumerate(responses):
        total_count = sum(response.values())
        max_count = max(response.values())
        if max_count / total_count < cutoff:
            topnames[i] = '-1'
    regrouped_data = []
    labels = []
    for g in np.unique(topnames):
        ix = np.where(topnames == g)[0]
        matching_features = np.vstack(features[ix].values)
        averaged = np.mean(matching_features, axis=0, keepdims=True)
        plot_features = averaged if plot_mean else matching_features
        regrouped_data.append(plot_features)
        labels.append(g)
    plot_naming(regrouped_data, viz_method='mds', labels=labels, savepath='english_mds_' + str(cutoff) + '.png', plot_all_colors=True)
    plot_naming(regrouped_data, viz_method='tsne', labels=labels, savepath='english_tsne_' + str(cutoff) + '.png', plot_all_colors=True)


if __name__ == '__main__':
    # Specify which classes you want to use.
    # If you don't want to specify, you'll end up with tons of classes.
    # But you can do so by removing the 'desired_names' field in the call to get_feature_data()
    viz_names = ['airplane', 'plane',
                 'animal', 'cow', 'dog', 'cat']
    features_filename = 'data/features_nobox.csv'
    plot_mean = False  # Do you want to plot all the individual points or the average for each class?
    full_data = get_feature_data(features_filename, desired_names=viz_names, max_per_class=40)
    for cutoff_likelihood in [0.4, 0.5, 0.6, 0.7, 0.8]:
        run(full_data, cutoff_likelihood)
