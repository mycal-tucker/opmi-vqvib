# conll_vqvib

All scripts should be run from the root directory, i.e. ``conll_vqvib``

## Downloading the data

This code is built to run on the ManyNames dataset.
The repo includes a tsv file that describes the dataset, but the raw images need to be downloaded.
Download the images via the src/data_utils/read_data.py script.
That script will iterate over the tsv file, downloading images from the url, and then extract features by passing the images through a resnet.
Get the tsv file from the manynames repo (https://raw.githubusercontent.com/amore-upf/manynames/master/manynames.tsv) and put it under ``conll_vqvib/data``

The key variable to toggle inside the script is with_bbox, which specifies whether images should be downloaded with or without their bounding boxes.

You also need to download glove embeddings from here: https://www.kaggle.com/danielwillgeorge/glove6b100dtxt
Just put the file under the ``data`` folder.

## Training agents

Once the data are downloaded, almost trian agents.

First, you need to train a VAE to autoencode features, so that we can regenerate features to be fed to the speaker.
Train the VAE using the src/scripts/train_vae.py script.
Note that within the VAE class we take a linear combination of reconstruction and KL loss, as in a 
beta VAE framework.
The script ends by saving the trained model.

Now we can train agents to play a reference game via the src/scripts/main.py script.
That script creates some neural agents (speaker, listener, and decoder) and assembles them into a team.
Agents are trained by sampling random target and distractor images.

## Metrics
We have several metrics to consider, only some of which are implemented.

1. Complexity/Informativeness are calculated using Mutual Information Neural Estimation (MINE).
2. In-distribution utility. How often does the listener pick the right image, given the types of inputs it's trained on?
3. Out-of-distribution utility. If we hold out a specific set of images (say, with the most common label ''car'') from training, how well does the team do on those types of images?

Other metrics that have yet to be implemented are things like gNID, optimality, etc.

## Visualization
We plot training metrics in an image named ``metrics.png``

We visualize communication naming schemes in figures named ``training_mds.png`` and ``training_tsne.png``.
These color 2D versions of resnet features of images according to their most commonly associated tokens.
To compare to English naming, run ``viz_modemaps.py``, which generates similar plots using English naming schemes.

## Current experiment results

I can train agents to high task success for continuous, onehot, or VQ-VIB communication.
