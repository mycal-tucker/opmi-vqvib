# conll_vqvib

All scripts should be run from the root directory, i.e. ``conll_vqvib``

## Downloading the data

This code is built to run on the ManyNames dataset.
The repo includes a tsv file that describes the dataset, but the raw images need to be downloaded.
Download the images via the src/data_utils/read_data.py script.
That script will iterate over the tsv file, downloading images from the url, and then extract features by passing the images through a resnet.
Get the tsv file from the manynames repo (https://raw.githubusercontent.com/amore-upf/manynames/master/manynames.tsv) and put it under ``conll_vqvib/data``

The key variable to toggle inside the script is with_bbox, which specifies whether images should be downloaded with or without their bounding boxes.

## Training agents

Once the data are downloaded, you can train agents to play a reference game by running ``src/scripts/main.py``

That script creates some neural agents (speaker, listener, and decoder) and assembles them into a team.
Agents are trained by sampling random target and distractor images.

## Current experiment results

So far, I've gotten the continuous communication working to about 95% accuracy.
I'm just starting to test VQ-VIB agents but haven't spent any time tweaking hyperparameters or anything, but it's not learning immediately.
I have ideas for a different way of sampling that seems cleaner and I think will converge faster.