import numpy as np
import src.settings as settings
import torch


def gen_batch(all_data, batch_size, vae=None, glove_data=None, see_distractors=False, num_dist=None, preset_targ_idx=None):
    # Given the dataset, creates a batch of inputs.
    # That's:
    # 1) The speaker's observation
    # 2) The listener's observation
    # 3) The label (which is the index of the speaker's observation).
    # 4) Word embeddings for the target, if glove_data is not None
    speaker_obs = []
    listener_obs = []
    labels = []
    embeddings = []

    all_features = all_data['features']
    all_words = all_data['topname']
    if num_dist is None:
        num_dist = settings.num_distractors
    for _ in range(batch_size):
        targ_idx = int(np.random.random() * len(all_features)) if preset_targ_idx is None else preset_targ_idx
        # Get the word embedding
        if glove_data is not None:
            word = all_words[targ_idx]
            emb = get_glove_embedding(glove_data, word)
            if emb is not None:
                emb = emb.to_numpy()
            embeddings.append(emb)
        targ_features = all_features[targ_idx]
        distractor_features = [all_features[int(np.random.random() * len(all_features))] for _ in range(num_dist)]
        obs_targ_idx = int(np.random.random() * (num_dist + 1))  # Pick where to slide the target observation into.
        l_obs = np.expand_dims(np.vstack(distractor_features[:obs_targ_idx] + [targ_features] + distractor_features[obs_targ_idx:]), axis=0)
        listener_obs.append(l_obs)
        labels.append(obs_targ_idx)
        s_obs = targ_features if not see_distractors else np.expand_dims(np.vstack([targ_features] + distractor_features), axis=0)
        speaker_obs.append(s_obs)
    speaker_tensor = torch.Tensor(np.vstack(speaker_obs)).to(settings.device)
    listener_tensor = torch.Tensor(np.vstack(listener_obs)).to(settings.device)
    if vae is not None:
        with torch.no_grad():
            speaker_tensor, _ = vae(speaker_tensor)
            listener_tensor, _ = vae(listener_tensor)
    label_tensor = torch.Tensor(labels).long().to(settings.device)
    return speaker_tensor, listener_tensor, label_tensor, embeddings


def get_unique_labels(dataset):
    unique_topnames = set()
    for topname in dataset['topname']:
        unique_topnames.add(topname)
    unique_responses = set()
    for responses in dataset['responses']:
        for k in responses.keys():
            unique_responses.add(k)
    return unique_topnames, unique_responses


def get_embedding_batch(all_data, embed_data, batch_size, vae=None):
    all_features = all_data['features']
    features = []
    embeddings = []
    while len(features) < batch_size:
        targ_idx = int(np.random.random() * len(all_features))
        # Get the features
        features.append(all_features[targ_idx])
        # Get the embedding for the word
        responses = all_data['responses'][targ_idx]
        words = []
        probs = []
        for k, v in responses.items():
            parsed_word = k.split(' ')
            if len(parsed_word) > 1:
                # Skip "words" like "tennis player" etc. because
                continue
            words.append(k)
            probs.append(v)
        if len(words) == 0:
            # Failed to find any legal words (e.g., all like "tennis player")
            continue
        total = np.sum(probs)
        probs = [p / total for p in probs]
        sampled_word = np.random.choice(words, p=probs)
        # sampled_word = words[np.argmax(probs)]
        embedding = get_glove_embedding(embed_data, sampled_word)
        embeddings.append(embedding)
    feature_tensor = torch.Tensor(np.vstack(features)).to(settings.device)
    if vae is not None:
        with torch.no_grad():
            feature_tensor, _ = vae(feature_tensor)
    emb_tensor = torch.Tensor(np.vstack(embeddings)).to(settings.device)
    return feature_tensor, emb_tensor


def get_glove_embedding(dataset, word):
    try:
        cached_embed = settings.embedding_cache.get(word)
        if cached_embed is not None:
            return cached_embed
        embed = dataset.loc[word]
        settings.embedding_cache[word] = embed
        return embed
    except KeyError:
        # print("Couldn't find word", word)
        return None


def get_all_embeddings(glove_dataset, words):
    all_embeddings = []
    for word in words:
        emb = get_glove_embedding(glove_dataset, word)
        if emb is None:
            continue
        all_embeddings.append(emb.to_numpy())
    stacked_embeddings = np.vstack(all_embeddings)
    return stacked_embeddings
