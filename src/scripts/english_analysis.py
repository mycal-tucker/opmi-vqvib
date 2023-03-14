
import random
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
import numpy as np
import src.settings as settings
from src.data_utils.helper_fns import get_embedding_batch, gen_batch
from src.data_utils.read_data import get_feature_data, get_glove_vectors
from src.models.decoder import Decoder
from src.models.vae import VAE
from src.models.listener import Listener
from src.models.team import Team
from src.utils.mine import Net
import matplotlib.pyplot as plt
import torch.nn as nn


def train(data, model):
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    running_mse = 0
    log_data = []
    for epoch in range(num_epochs):
        print("Epoch", epoch, "of", num_epochs)
        features, embeddings = get_embedding_batch(data, glove_data, batch_size, fieldname=fieldname, vae=vae)
        optimizer.zero_grad()
        recons = model(embeddings)
        recons = torch.squeeze(recons, dim=1)
        loss = torch.mean(((features - recons) ** 2))
        loss.backward()
        optimizer.step()
        running_mse = 0.95 * running_mse + 0.05 * loss.item()
        print("Loss", running_mse)
        log_data.append(loss.item())
    plt.plot(log_data)
    plt.show()
    # Now evaluate
    total_loss = 0
    num_eval_epochs = 20
    for epoch in range(num_eval_epochs):
        print("Eval epoch", epoch, "of", num_eval_epochs)
        features, embeddings = get_embedding_batch(data, glove_data, batch_size, fieldname=fieldname, vae=vae)
        with torch.no_grad():
            recons = model(embeddings)
            recons = torch.squeeze(recons, dim=1)
            loss = torch.mean(((features - recons) ** 2))
            total_loss += loss.item()
    print(total_loss / num_eval_epochs)


def get_complexity(data):
    mine_net = Net(512, comm_dim)
    mine_net.to(settings.device)
    optimizer = optim.Adam(mine_net.parameters(), lr=0.001)
    scheduler = ExponentialLR(optimizer, gamma=0.9)
    comps = []
    for epoch in range(5000):
        features, embeddings = get_embedding_batch(data, glove_data, batch_size, fieldname=fieldname, vae=vae)
        emb_shuffled = torch.Tensor(np.random.permutation(embeddings.cpu().numpy())).to(settings.device)
        optimizer.zero_grad()
        pred_xy = mine_net(features, embeddings)
        pred_x_y = mine_net(features, emb_shuffled)
        ret = torch.mean(pred_xy) - torch.log(torch.mean(torch.exp(pred_x_y)))
        loss = -ret  # maximize
        loss.backward()
        optimizer.step()
        comps.append(ret.item())
        if epoch % 100 == 0:
            print(ret.item())
        if epoch % 500 == 0:
            print("Stepping scheduler")
            scheduler.step()
    # Now evaluate
    summed_loss = 0
    num_eval_epochs = 20
    for epoch in range(num_eval_epochs):
        features, embeddings = get_embedding_batch(data, glove_data, batch_size, fieldname=fieldname, vae=vae)
        emb_shuffled = torch.Tensor(np.random.permutation(embeddings.cpu().numpy())).to(settings.device)
        with torch.no_grad():
            pred_xy = mine_net(features, embeddings)
            pred_x_y = mine_net(features, emb_shuffled)
            ret = torch.mean(pred_xy) - torch.log(torch.mean(torch.exp(pred_x_y)))
            summed_loss += ret.item()
    mutual_info = summed_loss / num_eval_epochs
    return mutual_info, comps


def train_listener(data):
    settings.see_distractor = False
    settings.num_distractors = 1

    dec = Decoder(comm_dim, 512, num_layers=3)
    listener = Listener(512)
    team = Team(None, listener, dec)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(team.parameters())
    running_acc = 0
    running_mse = 0
    for epoch in range(10000):
        if epoch % 10 == 0:
            print("Epoch", epoch)
            print("Running acc", running_acc)
            print("Running mse", running_mse)
        speaker_obs, listener_obs, labels, embeddings = gen_batch(data, batch_size, fieldname='topname', vae=vae, glove_data=glove_data,
                                                         see_distractors=settings.see_distractor)
        optimizer.zero_grad()
        embs = torch.Tensor(np.vstack(embeddings))
        # Add noise?
        # noise = torch.tensor(np.random.normal(0, 0.05, (embs.shape[0], 64)))
        # embs = embs + noise
        recons = dec(embs)
        pred = listener(recons, listener_obs)
        recons_loss = torch.mean(((speaker_obs - recons) ** 2))
        pred_loss = criterion(pred, labels)

        loss = pred_loss + 0.0 * recons_loss
        loss.backward()
        optimizer.step()
        pred_labels = np.argmax(pred.detach().cpu().numpy(), axis=1)
        num_correct = np.sum(pred_labels == labels.cpu().numpy())
        num_total = pred_labels.size
        running_acc = running_acc * 0.95 + 0.05 * num_correct / num_total
        running_mse = running_mse * 0.95 + 0.05 * recons_loss.item()
    if fieldname == 'vg_domain':
        torch.save(dec, 'english_vg_dec64.pt')
        torch.save(listener, 'english_vg_list64.pt')
    else:
        torch.save(dec, 'english_resp2_dec64.pt')
        torch.save(listener, 'english_resp2_list64.pt')



def run():
    s = 4
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    # train_data = get_feature_data(features_filename, selected_fraction=0.2)
    train_data = get_feature_data(features_filename, selected_fraction=1.0)
    # Calculate complexity
    # settings.distinct_words = False
    # complexity, training_log = get_complexity(train_data)
    # print("Complexity", complexity)
    # plt.plot(training_log)
    # plt.xlabel("Training epoch")
    # plt.ylabel("Complexity")
    # plt.savefig("mine_training.png")
    # plt.close()
    # plt.show()
    # Calculate informativeness via an autoencoder
    # model = Decoder(comm_dim, 512, num_layers=3)
    # model.to(settings.device)
    # train(train_data, model)
    # torch.save(model, 'english64.pt')
    # Calculate utility via a decoder and listener
    settings.distinct_words = True
    train_listener(train_data)


if __name__ == '__main__':
    features_filename = 'data/features_nobox.csv'
    comm_dim = 64  # Align with glove embedding size
    num_epochs = 3000
    batch_size = 32
    # batch_size = 512
    # Instead of just topname, it would be great to use a randomly-drawn response.
    fieldname = 'responses'
    # fieldname = 'topname'
    # fieldname = 'vg_domain'
    # settings.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    settings.device = 'cpu'
    glove_data = get_glove_vectors(comm_dim)
    settings.embedding_cache = {}
    vae = VAE(512, 32)
    vae.load_state_dict(torch.load('saved_models/vae0.001.pt'))
    vae.to(settings.device)
    print("Analysis for fieldname:\t", fieldname)
    run()
