import numpy as np
import src.settings as settings
import torch


def gen_batch(all_features, batch_size, num_distractors, vae=None):
    # Given the dataset of all features, creates a batch of inputs.
    # That's:
    # 1) The speaker's observation
    # 2) The listener's observation
    # 3) The label (which is the index of the speaker's observation).
    speaker_obs = []
    listener_obs = []
    labels = []
    for _ in range(batch_size):
        targ_idx = int(np.random.random() * len(all_features))
        targ_features = all_features[targ_idx]
        distractor_features = [all_features[int(np.random.random() * len(all_features))] for _ in range(num_distractors)]
        obs_targ_idx = int(np.random.random() * (num_distractors + 1))  # Pick where to slide the target observation into.
        speaker_obs.append(targ_features)
        l_obs = np.expand_dims(np.vstack(distractor_features[:obs_targ_idx] + [targ_features] + distractor_features[obs_targ_idx:]), axis=0)
        listener_obs.append(l_obs)
        labels.append(obs_targ_idx)
    speaker_tensor = torch.Tensor(np.vstack(speaker_obs)).to(settings.device)
    listener_tensor = torch.Tensor(np.vstack(listener_obs)).to(settings.device)
    if vae is not None:
        with torch.no_grad():
            speaker_tensor, _ = vae(speaker_tensor)
            listener_tensor, _ = vae(listener_tensor)
    label_tensor = torch.Tensor(labels).long().to(settings.device)
    return speaker_tensor, listener_tensor, label_tensor


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
        print("Couldn't find word", word)
        return np.zeros(100)
