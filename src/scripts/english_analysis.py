
import torch
import torch.optim as optim
import numpy as np
import src.settings as settings
from src.data_utils.helper_fns import get_embedding_batch
from src.data_utils.read_data import get_feature_data, get_glove_vectors
from src.models.decoder import Decoder
from src.models.vae import VAE
from src.utils.mine import Net


def train(data, model):
    optimizer = optim.Adam(model.parameters())
    running_mse = 0
    for epoch in range(num_epochs):
        print("Epoch", epoch, "of", num_epochs)
        features, embeddings = get_embedding_batch(data, glove_data, batch_size, vae)
        optimizer.zero_grad()
        recons = model(embeddings)
        loss = torch.mean(((features - recons) ** 2)) / batch_size
        loss.backward()
        optimizer.step()
        running_mse = 0.95 * running_mse + 0.05 * loss.item()
        print("Loss", running_mse)


def get_complexity(data):
    mine_net = Net(512, comm_dim)
    mine_net.to(settings.device)
    optimizer = optim.Adam(mine_net.parameters())
    for epoch in range(200):
        features, embeddings = get_embedding_batch(data, glove_data, batch_size, vae)
        emb_shuffled = torch.Tensor(np.random.permutation(embeddings.cpu().numpy())).to(settings.device)
        optimizer.zero_grad()
        pred_xy = mine_net(features, embeddings)
        pred_x_y = mine_net(features, emb_shuffled)
        ret = torch.mean(pred_xy) - torch.log(torch.mean(torch.exp(pred_x_y)))
        loss = -ret  # maximize
        loss.backward()
        optimizer.step()
    # Now evaluate
    summed_loss = 0
    num_eval_epochs = 20
    for epoch in range(num_eval_epochs):
        features, embeddings = get_embedding_batch(data, glove_data, batch_size, vae)
        emb_shuffled = torch.Tensor(np.random.permutation(embeddings.cpu().numpy())).to(settings.device)
        with torch.no_grad():
            pred_xy = mine_net(features, embeddings)
            pred_x_y = mine_net(features, emb_shuffled)
            ret = torch.mean(pred_xy) - torch.log(torch.mean(torch.exp(pred_x_y)))
            summed_loss += ret.item()
    mutual_info = summed_loss / num_eval_epochs
    return mutual_info


def run():
    train_data = get_feature_data(features_filename)
    complexity = get_complexity(train_data)
    print("Complexity", complexity)
    model = Decoder(comm_dim, 512, num_layers=3)
    model.to(settings.device)
    train(train_data, model)


if __name__ == '__main__':
    features_filename = 'data/features_nobox.csv'
    comm_dim = 100  # Align with glove embedding size
    num_epochs = 500
    batch_size = 1024
    # settings.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    settings.device = 'cpu'
    glove_data = get_glove_vectors(comm_dim)
    settings.embedding_cache = {}
    vae = VAE(512, 32)
    vae.load_state_dict(torch.load('saved_models/vae.pt'))
    vae.to(settings.device)
    run()
