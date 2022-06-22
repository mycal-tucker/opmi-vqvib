from src.models.vq import VQ
from src.models.team import Team
from src.models.mlp import MLP
from src.models.listener import Listener
from src.models.decoder import Decoder
import src.settings as settings
from src.data_utils.read_data import get_feature_data
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from src.data_utils.helper_fns import gen_batch
from src.utils.mine import get_info
from src.utils.plotting import plot_metrics


def evaluate(model, dataset):
    model.eval()
    num_test_batches = 20
    num_correct = 0
    num_total = 0
    for _ in range(num_test_batches):
        speaker_obs, listener_obs, labels = gen_batch(dataset, batch_size, num_distractors)
        with torch.no_grad():
            outputs, _, _, _ = model(speaker_obs, listener_obs)
        pred_labels = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        num_correct += np.sum(pred_labels == labels.cpu().numpy())
        num_total += pred_labels.size
    acc = num_correct / num_total
    print("Evaluation accuracy", acc)
    return acc


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


def train(model, train_data, val_data):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    running_acc = 0
    metric_labels = ['train_acc', 'val_acc', 'complexity (nats)', 'informativeness (nats)']
    metrics = []
    for epoch in range(num_epochs):
        settings.kl_weight += kl_incr
        print('epoch', epoch, 'of', num_epochs)
        print("Kl weight", settings.kl_weight)
        speaker_obs, listener_obs, labels = gen_batch(train_data, batch_size, num_distractors)
        optimizer.zero_grad()
        outputs, speaker_loss, info, recons = model(speaker_obs, listener_obs)
        loss = criterion(outputs, labels)
        print("Classification loss", loss)
        recons_loss = torch.mean(((speaker_obs - recons) ** 2))
        print("Recons loss", recons_loss)
        loss += settings.alpha * recons_loss
        loss += speaker_loss
        loss.backward()
        optimizer.step()
        print("Loss", loss)

        # Metrics
        pred_labels = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        num_correct = np.sum(pred_labels == labels.cpu().numpy())
        num_total = pred_labels.size
        running_acc = running_acc * 0.95 + 0.05 * num_correct / num_total
        print("Running acc", running_acc)

        # Evaluate on the validation set
        if epoch % val_period == val_period - 1:
            val_acc = evaluate(model, val_data)
            # And calculate efficiency values like complexity and informativeness.
            # Can estimate complexity by sampling inputs and measuring communication probabilities.
            # get_probs(model.speaker, train_data)
            # Or we can use MINE to estimate complexity and informativeness.
            complexity = get_info(model, train_data, num_distractors, targ_dim=comm_dim, comm_targ=True)
            informativeness = get_info(model, train_data, num_distractors, targ_dim=feature_len, comm_targ=False)
            print("Complexity", complexity)
            print("Informativeness", informativeness)
            metrics.append([running_acc, val_acc, complexity, informativeness])
            plot_metrics(metrics, metric_labels)


def run():
    speaker_inp_dim = feature_len if not see_distractor else (num_distractors + 1) * feature_len
    # speaker = MLP(speaker_inp_dim, comm_dim, num_layers=3, onehot=True, variational=variational)
    speaker = VQ(speaker_inp_dim, comm_dim, num_layers=3, num_protos=330, variational=variational)
    listener = Listener(comm_dim, feature_len, num_distractors + 1, num_layers=2)
    decoder = Decoder(comm_dim, speaker_inp_dim, num_layers=3)
    model = Team(speaker, listener, decoder)
    model.to(settings.device)

    train_data = get_feature_data(features_filename, excluded_names=val_classes + test_classes)
    val_data = get_feature_data(features_filename, desired_names=val_classes)
    test_data = get_feature_data(features_filename, desired_names=test_classes)
    train(model, train_data['features'], val_data['features'])


if __name__ == '__main__':
    feature_len = 512
    see_distractor = False
    num_distractors = 1
    num_epochs = 5000
    val_period = 100  # How often to test on the validation set and calculate various info metrics.
    batch_size = 1024
    comm_dim = 32
    # kl_incr = 0.00001
    kl_incr = 0.0
    variational = True
    settings.alpha = 1
    settings.sample_first = True
    settings.kl_weight = 0.00001
    settings.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    settings.learned_marginal = False
    with_bbox = False
    val_classes = ['car', 'carpet']
    test_classes = ['couch', 'counter', 'bowl']
    features_filename = 'data/features.csv' if with_bbox else 'data/features_nobox.csv'
    run()
