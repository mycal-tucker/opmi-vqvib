import torch
import torch.optim as optim

import src.settings as settings
from src.data_utils.helper_fns import gen_batch
from src.data_utils.read_data import get_feature_data
from src.models.vae import VAE


def run():
    model = VAE(512, 32)
    model.to(settings.device)
    train_data = get_feature_data(features_filename)['features']
    optimizer = optim.Adam(model.parameters())
    for epoch in range(num_epochs):
        features, _, _ = gen_batch(train_data, batch_size, 0)
        optimizer.zero_grad()
        reconstruction, loss = model(features)
        loss.backward()
        optimizer.step()
        print("Loss", loss)
    torch.save(model.state_dict(), savepath)


if __name__ == '__main__':
    features_filename = 'data/features_nobox.csv'
    savepath = 'saved_models/vae.pt'
    num_epochs = 100
    batch_size = 128
    settings.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    run()
