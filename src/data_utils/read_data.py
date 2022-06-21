# %% ---- DEPENDENCIES
import os
import shutil
import sys
import time
from multiprocessing.pool import ThreadPool

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image


# %% ---- FUNCTION TO LOAD MANYNAMES.TSV
def load_cleaned_results(filename="../manynames.tsv", sep="\t",
                         index_col=None):
    # read tsv
    resdf = pd.read_csv(filename, sep=sep, index_col=index_col)

    # remove any old index columns
    columns = [col for col in resdf.columns if not col.startswith("Unnamed")]
    resdf = resdf[columns]

    # run eval on nested lists/dictionaries
    evcols = ['vg_same_object', 'vg_inadequacy_type',
              'bbox_xywh', 'clusters', 'responses', 'singletons',
              'same_object', 'adequacy_mean', 'inadequacy_type']
    for icol in evcols:
        if icol in resdf:
            resdf[icol] = resdf[icol].apply(lambda x: eval(x))
    return resdf


def download_url(args):
    t0 = time.time()
    url, filename = args[0], args[1]
    try:
        print("Downloading url", url, "to", filename)
        res = requests.get(url, stream=True)
        if res.status_code == 200:
            with open(filename, 'wb') as f:
                shutil.copyfileobj(res.raw, f)
        return(url, time.time() - t0)
    except Exception as e:
        print('Exception in download_url():', e)


def download_parallel(args):
    cpus = 64
    results = ThreadPool(cpus - 1).imap_unordered(download_url, args)
    for result in results:
        print('url:', result[0], 'time (s):', result[1])


def download_img():
    existing_ids = []
    for existing_img in os.listdir(image_directory):
        existing_ids.append(existing_img.split('.')[0])
    inputs = []
    id_to_url = {}
    for url, img_id in zip(manynames['link_mn'], manynames['vg_image_id']):
        id_to_url[img_id] = url
        if str(img_id) in existing_ids:
            # print("Skipping", img_id)
            continue
        inputs.append((url, image_directory + str(img_id) + '.png'))
    print("Total number to do", len(inputs))
    download_parallel(inputs)
    return id_to_url


def img_features(id_to_url):
    # Get a pretrained model
    resnet18 = models.resnet18(pretrained=True)
    modules = list(resnet18.children())[:-1]
    resnet18 = nn.Sequential(*modules)
    for p in resnet18.parameters():
        p.requires_grad = False
    count = 0
    for img in os.listdir(image_directory):
        print("reading", img, "number", count, "of", len(manynames))
        count += 1
        pil_image = Image.open(image_directory + img)
        try:
            pil_image = pil_image.resize((224, 224))
        except OSError:
            print("Downloading replacement for truncated image")
            img_id = int(img.split('.')[0])
            download_url((id_to_url.get(img_id), image_directory + str(img_id) + '.png'))
            pil_image = Image.open(image_directory + img)
            pil_image = pil_image.resize((224, 224))
        image = np.asarray(pil_image)[:, :, :-1]
        img_tensor = np.moveaxis(image, -1, 0)
        img_tensor = torch.Tensor(img_tensor)
        img_tensor = torch.unsqueeze(img_tensor, 0)  # Batch size 1
        features = resnet18(img_tensor)[0, :, 0, 0]
        with open('data/features.csv', 'a') as f:
            f.write(img.split('.')[0] + ', ')
            f.write(', '.join([str(e) for e in features.cpu().detach().numpy()]))
            f.write('\n')


# %% ---- DIRECTLY RUN
if __name__ == "__main__":
    image_directory = 'data/images/'
    if len(sys.argv) > 1:
        fn = sys.argv[1]
    else:
        fn = "data/manynames.tsv"
    print("Loading data from", fn)
    manynames = load_cleaned_results(filename=fn)
    print(manynames.head())
    print(len(manynames))
    url_map = download_img()
    img_features(url_map)
