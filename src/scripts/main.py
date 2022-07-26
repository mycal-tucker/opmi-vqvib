import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LinearRegression
import random

import src.settings as settings
from src.data_utils.helper_fns import gen_batch, get_glove_embedding, get_unique_labels
from src.data_utils.read_data import get_feature_data, get_glove_vectors
from src.models.decoder import Decoder
from src.models.listener import Listener
from src.models.team import Team
from src.models.vae import VAE
from src.models.vq import VQ
from src.models.mlp import MLP
from src.utils.mine import get_info
from src.utils.plotting import plot_metrics, plot_naming, plot_scatter
from src.utils.performance_metrics import PerformanceMetrics


def evaluate(model, dataset, batch_size, vae, num_dist=None):
    model.eval()
    num_test_batches = 10
    num_correct = 0
    total_recons_loss = 0
    num_total = 0
    for _ in range(num_test_batches):
        with torch.no_grad():
            speaker_obs, listener_obs, labels = gen_batch(dataset, batch_size, vae=vae, see_distractors=settings.see_distractor, num_dist=num_dist)
            outputs, _, _, recons = model(speaker_obs, listener_obs)
        pred_labels = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        num_correct += np.sum(pred_labels == labels.cpu().numpy())
        num_total += pred_labels.size
        total_recons_loss += torch.mean(((speaker_obs - recons) ** 2)).item()
    acc = num_correct / num_total
    total_recons_loss = total_recons_loss / num_total
    print("Evaluation accuracy", acc)
    print("Evaluation recons loss", total_recons_loss)
    return acc, total_recons_loss


def plot_comms(model, dataset, basepath):
    num_tests = 1000  # Generate lots of samples for the same input because it's not deterministic.
    labels = []
    for f in dataset:
        speaker_obs = torch.Tensor(np.array(f)).to(settings.device)
        speaker_obs = torch.unsqueeze(speaker_obs, 0)
        speaker_obs = speaker_obs.repeat(num_tests, 1)
        likelihoods = model.speaker.get_token_dist(speaker_obs)
        top_comm_idx = np.argmax(likelihoods)
        top_likelihood = likelihoods[top_comm_idx]
        # print("Max likelihood", top_likelihood)
        label = top_comm_idx if top_likelihood > 0.4 else -1
        labels.append(label)
    features = np.vstack(dataset)
    label_np = np.reshape(np.array(labels), (-1, 1))
    all_np = np.hstack([label_np, features])
    # all_np = all_np[all_np[:, 0].argsort()]
    regrouped_data = []
    plot_labels = []
    plot_mean = False
    for c in np.unique(labels):
        ix = np.where(all_np[:, 0] == c)
        matching_features = np.vstack(all_np[ix, 1:])
        averaged = np.mean(matching_features, axis=0, keepdims=True)
        plot_features = averaged if plot_mean else matching_features
        regrouped_data.append(plot_features)
        plot_labels.append(c)
    plot_naming(regrouped_data, viz_method='mds', labels=plot_labels, savepath=basepath + 'training_mds')
    plot_naming(regrouped_data, viz_method='tsne', labels=plot_labels, savepath=basepath + 'training_tsne')


def get_embedding_alignment(model, dataset):
    num_tests = 1
    comms = []
    embeddings = []
    features = []
    for f, word in zip(dataset['features'], dataset['topname']):
        speaker_obs = torch.Tensor(np.array(f)).to(settings.device)
        speaker_obs = torch.unsqueeze(speaker_obs, 0)
        with torch.no_grad():
            speaker_obs = speaker_obs.repeat(num_tests, 1)
        comm, _, _ = model.speaker(speaker_obs)
        comms.append(np.mean(comm.detach().cpu().numpy(), axis=0))
        embeddings.append(get_glove_embedding(glove_data, word).to_numpy())
        features.append(np.array(f))
    comms = np.vstack(comms)
    embeddings = np.vstack(embeddings)
    # Now just find a linear mapping from comms to embeddings.
    reg = LinearRegression()
    reg.fit(comms, embeddings)
    comm_to_embed_r2 = reg.score(comms, embeddings)
    print("Comm to word embedding regression score\t\t", comm_to_embed_r2)
    # And compute regression between pairwise distances in feature space and comm space.
    comm_dists = np.zeros((len(comms), len(comms)))
    feature_dists = np.zeros((len(comms), len(comms)))
    for i, c1 in enumerate(comms):
        for j, c2 in enumerate(comms):
            comm_dists[i, j] = np.linalg.norm(c1 - c2)
            feature_dists[i, j] = np.linalg.norm(features[i] - features[j])
    reg = LinearRegression()
    comm_dists = np.reshape(comm_dists, (-1, 1))
    feature_dists = np.reshape(feature_dists, (-1, 1))
    reg.fit(comm_dists, feature_dists)
    pairwise_r2 = reg.score(comm_dists, feature_dists)
    print("Pairwise dist regression score", pairwise_r2)
    return comm_to_embed_r2, pairwise_r2


# Manually calculate the complexity of communication by sampling some inputs and comparing the conditional communication
# to the marginal communication.
def get_probs(speaker, dataset):
    num_samples = 1000
    w_m = np.zeros((num_samples, speaker.num_tokens))
    for i in range(num_samples):
        speaker_obs, _, _ = gen_batch(dataset, batch_size=1)
        with torch.no_grad():
            likelihoods = speaker.get_token_dist(speaker_obs)
        w_m[i] = likelihoods
    # Calculate the divergence from the marginal over tokens. Here we just calculate the marginal
    # from observations.
    marginal = np.average(w_m, axis=0)
    complexities = []
    for likelihood in w_m:
        summed = 0
        for l, p in zip(likelihood, marginal):
            if l == 0:
                continue
            summed += l * (np.log(l) - np.log(p))
        complexities.append(summed)
    comp = np.average(complexities)  # Note that this is in nats (not bits)
    print("Sampled communication complexity", comp)


def train(model, train_data, val_data, viz_data, vae, savepath, comm_dim, num_epochs=3000, batch_size=1024, burnin_epochs=500,
          val_period=200, plot_comms_flag=False, calculate_complexity=False):
    train_features = train_data['features']
    val_features = val_data['features']
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    running_acc = 0
    running_mse = 0
    num_cand_to_metrics = {2: [], 4: [], 8: []}
    for empty_list in num_cand_to_metrics.values():
        empty_list.extend([PerformanceMetrics(), PerformanceMetrics()])  # Train metrics, validation metrics
    for epoch in range(num_epochs):
        if epoch > burnin_epochs:
            settings.kl_weight += settings.kl_incr
        print('epoch', epoch, 'of', num_epochs)
        print("Kl weight", settings.kl_weight)
        speaker_obs, listener_obs, labels = gen_batch(train_features, batch_size, vae=vae, see_distractors=settings.see_distractor)
        optimizer.zero_grad()
        outputs, speaker_loss, info, recons = model(speaker_obs, listener_obs)
        loss = criterion(outputs, labels)
        if len(speaker_obs.shape) == 2:
            speaker_obs = torch.unsqueeze(speaker_obs, 1)
        recons_loss = torch.mean(((speaker_obs - recons) ** 2))
        loss += settings.alpha * recons_loss
        loss += speaker_loss
        print("Speaker loss fraction:\t", speaker_loss.item() / loss.item())
        print("Recons loss fraction:\t", settings.alpha * recons_loss.item() / loss.item())
        loss.backward()
        optimizer.step()

        # Metrics
        pred_labels = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        num_correct = np.sum(pred_labels == labels.cpu().numpy())
        num_total = pred_labels.size
        running_acc = running_acc * 0.95 + 0.05 * num_correct / num_total
        running_mse = running_mse * 0.95 + 0.05 * recons_loss.item() / num_total
        print("Running acc", running_acc)
        print("Running mse", running_mse)

        if epoch % val_period == val_period - 1:
            # Create a directory to save information, models, etc.
            basepath = savepath + str(epoch) + '/'
            if not os.path.exists(basepath):
                os.makedirs(basepath)
            # Calculate efficiency values like complexity and informativeness.
            # Can estimate complexity by sampling inputs and measuring communication probabilities.
            # get_probs(model.speaker, train_data)
            # Or we can use MINE to estimate complexity and informativeness.
            if calculate_complexity:
                train_complexity = get_info(model, train_features, targ_dim=comm_dim, comm_targ=True)
                # val_complexity = get_info(model, val_features, targ_dim=comm_dim, comm_targ=True)
                val_complexity = None
                print("Train complexity", train_complexity)
                print("Val complexity", val_complexity)
            else:
                train_complexity = None
                val_complexity = None
            complexities = [train_complexity, val_complexity]
            eval_batch_size = 256
            for feature_idx, features in enumerate([train_features, val_features]):
                for num_candidates in num_cand_to_metrics.keys():
                    acc, recons = evaluate(model, features, eval_batch_size, vae, num_dist=num_candidates - 1)
                    relevant_metrics = num_cand_to_metrics.get(num_candidates)[feature_idx]
                    relevant_metrics.add_data(epoch, complexities[feature_idx], -1 * recons, acc, settings.kl_weight)
            # Plot some of the metrics for online visualization
            comm_accs = []
            labels = []
            epoch_idxs = None
            for feature_idx, label in enumerate(['train', 'val']):
                for num_candidates in sorted(num_cand_to_metrics.keys()):
                    comm_accs.append(num_cand_to_metrics.get(num_candidates)[feature_idx].comm_accs)
                    labels.append(" ".join([label, str(num_candidates), "utility"]))
                    if epoch_idxs is None:
                        epoch_idxs = num_cand_to_metrics.get(num_candidates)[feature_idx].epoch_idxs
            plot_metrics(comm_accs, labels, epoch_idxs, basepath=basepath)
            # Visualize some of the communication
            try:
                if plot_comms_flag:
                    plot_comms(model, viz_data['features'], basepath)
            except AssertionError:
                print("Can't plot comms for whatever reason (e.g., continuous communication makes categorizing hard)")
            # And compare to english word embeddings
            # word_embed_r2, pairwise_r2 = get_embedding_alignment(model, viz_data)
            # Save the model and metrics to files.
            torch.save(model.state_dict(), basepath + 'model.pt')
            for feature_idx, label in enumerate(['train', 'val']):
                for num_candidates in sorted(num_cand_to_metrics.keys()):
                    metric = num_cand_to_metrics.get(num_candidates)[feature_idx]
                    metric.to_file(basepath + "_".join([label, str(num_candidates), "metrics"]))

def run():
    num_imgs = 1 if not settings.see_distractor else (num_distractors + 1)
    if speaker_type == 'cont':
        speaker = MLP(feature_len, c_dim, num_layers=3, onehot=False, variational=variational, num_imgs=num_imgs)
    elif speaker_type == 'onehot':
        speaker = MLP(feature_len, c_dim, num_layers=3, onehot=True, variational=variational, num_imgs=num_imgs)
    elif speaker_type == 'vq':
        # speaker = VQ(feature_len, c_dim, num_layers=3, num_protos=1763, num_simultaneous_tokens=1, variational=variational, num_imgs=num_imgs)
        speaker = VQ(feature_len, c_dim, num_layers=3, num_protos=32, num_simultaneous_tokens=8, variational=variational, num_imgs=num_imgs)
    listener = Listener(feature_len)
    decoder = Decoder(c_dim, feature_len, num_layers=3, num_imgs=num_imgs)
    model = Team(speaker, listener, decoder)
    model.to(settings.device)

    train_data = get_feature_data(features_filename, selected_fraction=train_fraction)
    train_topnames, train_responses = get_unique_labels(train_data)
    val_data = get_feature_data(features_filename, excluded_names=train_responses)
    # val_data = train_data  # For debugging, it's faster to just reuse datasets
    # test_data = get_feature_data(features_filename, desired_names=test_classes)
    # viz_data = get_feature_data(features_filename, desired_names=viz_names, max_per_class=40) if do_plot_comms else train_data
    viz_data = train_data  # For debugging, it's faster to just reuse datasets
    train(model, train_data, val_data, viz_data, vae=vae_model, savepath=save_loc, comm_dim=c_dim, num_epochs=n_epochs,
          batch_size=b_size, burnin_epochs=num_burnin, val_period=v_period, plot_comms_flag=do_plot_comms,
          calculate_complexity=do_calc_complexity)


if __name__ == '__main__':
    feature_len = 512
    settings.see_distractor = False
    num_distractors = 1
    settings.num_distractors = num_distractors
    n_epochs = 3000
    v_period = 200  # How often to test on the validation set and calculate various info metrics.
    num_burnin = 500
    b_size = 1024
    c_dim = 128
    variational = True
    # Measuring complexity takes a lot of time. For debugging other features, set to false.
    do_calc_complexity = False
    do_plot_comms = False
    settings.alpha = 0
    # settings.kl_weight = 0.00001  # For cont
    settings.kl_weight = 0.0  # For cont
    settings.kl_incr = 0.0
    settings.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    settings.learned_marginal = False
    with_bbox = False
    train_fraction = 0.5
    test_classes = ['couch', 'counter', 'bowl']
    viz_names = ['airplane', 'plane',
                 'animal', 'cow', 'dog', 'cat']
    features_filename = 'data/features.csv' if with_bbox else 'data/features_nobox.csv'
    np.random.seed(0)
    glove_data = get_glove_vectors(c_dim)
    vae_model = VAE(512, 32)
    vae_model.load_state_dict(torch.load('saved_models/vae0.001.pt'))
    vae_model.to(settings.device)
    settings.embedding_cache = {}
    settings.sample_first = True
    speaker_type = 'vq'  # Options are 'vq', 'cont', or 'onehot'
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    save_loc = 'saved_models/' + speaker_type + '/seed' + str(seed) + '/'
    run()
