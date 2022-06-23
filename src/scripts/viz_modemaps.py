
from src.data_utils.read_data import get_feature_data
from src.utils.plotting import plot_mds, plot_tsne
import numpy as np


def run():
    data = get_feature_data(features_filename, desired_names=viz_names, max_per_class=40)
    features = data['features']
    names = data['topname']
    regrouped_data = []
    labels = []
    for g in np.unique(names):
        ix = np.where(names == g)[0]
        matching_features = np.vstack(features[ix].values)
        averaged = np.mean(matching_features, axis=0, keepdims=True)
        plot_features = averaged if plot_mean else matching_features
        regrouped_data.append(plot_features)
        labels.append(g)
    plot_mds(regrouped_data, labels=labels, savepath='english_mds')
    plot_tsne(regrouped_data, labels=labels, savepath='english_tsne')


if __name__ == '__main__':
    # Specify which classes you want to use.
    # If you don't want to specify, you'll end up with tons of classes.
    # But you can do so by removing the 'desired_names' field in the call to get_feature_data()
    viz_names = ['airplane', 'plane',
                 'animal', 'cow', 'dog', 'cat',
                 'chair', 'counter', 'table']
    features_filename = 'data/features_nobox.csv'
    plot_mean = False  # Do you want to plot all the individual points or the average for each class?
    run()
