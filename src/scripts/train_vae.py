import torch
import torch.optim as optim

import src.settings as settings
from src.data_utils.helper_fns import gen_batch
from src.data_utils.read_data import get_feature_data
from src.models.vae import VAE


def run():
    model = VAE(512, 32)
    model.to(settings.device)
    train_data = get_feature_data(features_filename)
    optimizer = optim.Adam(model.parameters())
    for epoch in range(num_epochs):
        print("Epoch", epoch, "of", num_epochs)
        features, _, _, _ = gen_batch(train_data, batch_size, fieldname='topname')  # Fieldname doesn't matter
        optimizer.zero_grad()
        reconstruction, loss = model(features)
        loss.backward()
        optimizer.step()
        print("Loss", loss)
    torch.save(model.state_dict(), savepath)


if __name__ == '__main__':
    features_filename = 'data/features_nobox.csv'
    savepath = 'saved_models/vae0.01.pt'
    num_epochs = 10000
    batch_size = 128
    settings.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    run()
