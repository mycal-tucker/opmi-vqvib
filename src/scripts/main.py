from src.models.vq import VQ
from src.models.team import Team
from src.models.mlp import MLP
from src.models.listener import Listener
from src.models.decoder import Decoder
from src.models.vae import VAE
import src.settings as settings
from src.data_utils.read_data import get_feature_data, get_glove_vectors
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from src.data_utils.helper_fns import gen_batch, get_glove_embedding
from src.utils.mine import get_info
from src.utils.plotting import plot_metrics, plot_naming, plot_scatter
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression


def evaluate(model, dataset):
    model.eval()
    num_test_batches = 20
    num_correct = 0
    total_recons_loss = 0
    num_total = 0
    for _ in range(num_test_batches):
        speaker_obs, listener_obs, labels = gen_batch(dataset, batch_size, num_distractors, vae=vae)
        with torch.no_grad():
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


def plot_comms(model, dataset):
    num_tests = 100
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
    plot_naming(regrouped_data, viz_method='mds', labels=plot_labels, savepath='training_mds')
    plot_naming(regrouped_data, viz_method='tsne', labels=plot_labels, savepath='training_tsne')


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
        speaker_obs, _, _ = gen_batch(dataset, batch_size=1, num_distractors=num_distractors)
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


def train(model, train_data, val_data, viz_data):
    train_features = train_data['features']
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    running_acc = 0
    metric_labels = ['train_acc', 'val_acc', 'complexity (nats)', 'informativeness (Recons loss)']
    metrics = []
    for epoch in range(num_epochs):
        if epoch > burnin_epochs:
            settings.kl_weight += kl_incr
        print('epoch', epoch, 'of', num_epochs)
        print("Kl weight", settings.kl_weight)
        speaker_obs, listener_obs, labels = gen_batch(train_features, batch_size, num_distractors, vae=vae)
        optimizer.zero_grad()
        outputs, speaker_loss, info, recons = model(speaker_obs, listener_obs)
        loss = criterion(outputs, labels)
        recons_loss = torch.mean(((speaker_obs - recons) ** 2))
        loss += settings.alpha * recons_loss
        loss += speaker_loss
        loss.backward()
        optimizer.step()

        # Metrics
        pred_labels = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        num_correct = np.sum(pred_labels == labels.cpu().numpy())
        num_total = pred_labels.size
        running_acc = running_acc * 0.95 + 0.05 * num_correct / num_total
        print("Running acc", running_acc)

        if epoch % val_period == val_period - 1:
            # Evaluate on the validation set
            val_acc, val_recons = evaluate(model, val_data['features'])
            # Visualize some of the communication
            try:
                plot_comms(model, viz_data['features'])
            except AssertionError:
                print("Can't plot comms for whatever reason (e.g., continuous communication makes categorizing hard)")
            # And compare to english word embeddings
            word_embed_r2, pairwise_r2 = get_embedding_alignment(model, viz_data)
            # And calculate efficiency values like complexity and informativeness.
            # Can estimate complexity by sampling inputs and measuring communication probabilities.
            # get_probs(model.speaker, train_data)
            # Or we can use MINE to estimate complexity and informativeness.
            complexity = get_info(model, train_features, num_distractors, targ_dim=comm_dim, comm_targ=True)
            # informativeness = get_info(model, train_features, num_distractors, targ_dim=feature_len, comm_targ=False)
            informativeness = -1 * val_recons
            print("Complexity", complexity)
            print("Informativeness", informativeness)
            metrics.append([running_acc, val_acc, complexity, informativeness])
            plot_metrics(metrics, metric_labels)
            plot_scatter([[metric[2] for metric in metrics], [metric[3] for metric in metrics]],
                         [metric_labels[2], metric_labels[3]], savepath='info_plane.png')


def run():
    speaker_inp_dim = feature_len if not see_distractor else (num_distractors + 1) * feature_len
    # speaker = MLP(speaker_inp_dim, comm_dim, num_layers=3, onehot=False, variational=variational)
    speaker = VQ(speaker_inp_dim, comm_dim, num_layers=3, num_protos=330, variational=variational)
    listener = Listener(feature_len, num_distractors + 1, num_layers=2)
    decoder = Decoder(comm_dim, speaker_inp_dim, num_layers=3)
    model = Team(speaker, listener, decoder)
    model.to(settings.device)

    train_data = get_feature_data(features_filename, excluded_names=val_classes + test_classes)
    val_data = get_feature_data(features_filename, desired_names=val_classes)
    # test_data = get_feature_data(features_filename, desired_names=test_classes)
    viz_data = get_feature_data(features_filename, desired_names=viz_names, max_per_class=40)
    train(model, train_data, val_data, viz_data)


if __name__ == '__main__':
    feature_len = 512
    see_distractor = False
    num_distractors = 1
    num_epochs = 5000
    val_period = 100  # How often to test on the validation set and calculate various info metrics.
    batch_size = 1024
    comm_dim = 32
    # kl_incr = 0.0001  # For continuous communication
    kl_incr = 0.0000001  # For VQ-VIB
    # kl_incr = 0.0
    burnin_epochs = 500
    variational = True
    settings.alpha = 1
    settings.sample_first = True
    settings.kl_weight = 0.00001
    settings.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    settings.learned_marginal = False
    with_bbox = False
    val_classes = ['car', 'carpet']
    test_classes = ['couch', 'counter', 'bowl']
    viz_names = ['airplane', 'plane',
                 'animal', 'cow', 'dog', 'cat']
    features_filename = 'data/features.csv' if with_bbox else 'data/features_nobox.csv'
    np.random.seed(0)
    glove_data = get_glove_vectors()
    vae = VAE(512, 32)
    vae.load_state_dict(torch.load('saved_models/vae.pt'))
    vae.to(settings.device)
    run()
