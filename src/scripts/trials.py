import src.settings as settings
from src.data_utils.helper_fns import get_unique_labels, get_all_embeddings, get_unique_by_field
from src.data_utils.read_data import get_feature_data, get_glove_vectors
from src.models.decoder import Decoder
from src.models.listener import Listener
from src.models.mlp import MLP
from src.models.team import Team
from src.models.vq import VQ
from src.models.vqvib2 import VQVIB2
from src.models.proto import ProtoNetwork
from src.models.vq_after import VQAfter
import numpy as np
from src.models.vae import VAE
import torch
import random
from src.scripts.main import train


def run_trial():
    # Load data
    glove_data = get_glove_vectors(comm_dim)
    train_data = get_feature_data(features_filename, selected_fraction=train_fraction)
    vgs = get_unique_by_field(train_data, 'vg_domain')
    print("Vgs", vgs)
    train_topnames, train_responses = get_unique_labels(train_data)
    if train_fraction == 1.0:
        val_data = train_data
    else:
        val_data = get_feature_data(features_filename, excluded_names=train_responses)
    # if len(train_data) < 1000 or len(val_data) < 1000:
    #     return
    # viz_data = get_feature_data(features_filename, desired_names=viz_names, max_per_class=40)
    viz_data = val_data  # Turn off viz data because we don't use it during trials.

    all_embeddings = None if not use_embed_tokens else get_all_embeddings(glove_data, train_topnames)

    # Initialize the agents for training
    num_imgs = 1 if not settings.see_distractor else (num_distractors + 1)
    if speaker_type == 'cont':
        speaker = MLP(feature_len, comm_dim, num_layers=3, onehot=False, variational=variational, num_imgs=num_imgs)
    elif speaker_type == 'onehot':
        speaker = MLP(feature_len, comm_dim, num_layers=3, onehot=True, num_simultaneous_tokens=num_tokens,
                      variational=variational, num_imgs=num_imgs)
    elif speaker_type == 'vq':
        speaker = VQ(feature_len, comm_dim, num_layers=3, num_protos=num_prototypes, specified_tok=all_embeddings,
                     num_simultaneous_tokens=num_tokens, variational=variational, num_imgs=num_imgs)
    elif speaker_type == 'vq2':
        speaker = VQVIB2(feature_len, comm_dim, num_layers=3, num_protos=num_prototypes, num_simultaneous_tokens=num_tokens)
    elif speaker_type == 'vq_after':
        speaker = VQAfter(feature_len, comm_dim, num_layers=3, num_protos=num_prototypes, specified_tok=all_embeddings,
                     num_simultaneous_tokens=num_tokens, variational=variational, num_imgs=num_imgs)
    elif speaker_type == 'proto':
        speaker = ProtoNetwork(feature_len, comm_dim, num_layers=3, num_protos=num_prototypes, variational=True)
    listener = Listener(feature_len)
    decoder = Decoder(comm_dim, feature_len, num_layers=3, num_imgs=num_imgs)
    model = Team(speaker, listener, decoder)
    model.to(settings.device)

    train(model, train_data, val_data, viz_data, glove_data, vae=vae, savepath=savepath, comm_dim=comm_dim, fieldname=fieldname, num_epochs=num_epochs,
          batch_size=batch_size, burnin_epochs=num_burnin, val_period=val_period,
          plot_comms_flag=False, calculate_complexity=False)


if __name__ == '__main__':
    feature_len = 512
    settings.see_distractor = False
    num_distractors = 1
    num_epochs = 100000  # For hold-out
    # num_epochs = 20000  # For hold-out faster version
    # num_epochs = 20000  # 1000 is way too short, but it's quick for debugging.e
    num_burnin = num_epochs
    # val_period = 10000  # For holdout
    val_period = 5000  # How often to test on the validation set and calculate various info metrics.
    batch_size = 32  # TODO: try bigger batch size. I think it'll improve capacity measures.
    comm_dim = 1024  # Normally, 64. But for onehot, make it 1024?
    features_filename = 'data/features_nobox.csv'

    # field_setup = 'vg_domain'
    field_setup = 'topname'
    # field_setup = 'all'
    settings.distinct_words = field_setup != 'all'
    fieldname = field_setup if field_setup != 'all' else 'topname'

    if num_distractors != 1:
        field_setup += str(num_distractors)

    train_fraction = 0.2
    settings.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # settings.kl_weight = 0.001  # For cont
    # settings.kl_weight = 0.01  # For VQ 1 token, 0.001
    # settings.kl_incr = 0.0  # For VQ 1 token 0.00001 works, but is slow.
    # settings.kl_weight = 0.01  # For VQ 8 tokens
    # settings.kl_incr = 0.0003  # For VQ 8 token 0.0001 is good but a little slow, but 0.001 is too fast.

    # VQ-VIB2
    settings.kl_weight = 0.0  # Doesn't matter what you set it to.
    settings.kl_incr = 0.0  # All guessing.

    # Onehot
    # settings.kl_weight = 0.001
    # settings.kl_incr = 0.00001  # For onehot with 1 token
    # settings.kl_weight = 0.0  # Having a non-zero starting weight is very important to encourage exploration
    # settings.kl_incr = 0.0  # For onehot with 8 tokens .001 is too fast. 0.0001 is good but a little slow.
    # num_burnin = 3000

    settings.num_distractors = num_distractors
    settings.learned_marginal = False
    settings.embedding_cache = {}
    settings.sample_first = True
    variational = True
    settings.supervision_weight = 0.0
    settings.hardcoded_vq = False
    settings.entropy_weight = None
    settings.max_num_align_data = 1

    use_embed_tokens = False
    if use_embed_tokens:
        assert comm_dim <= 100, "Can't use 100D glove embeddings in greater than 100 comm dim"

    vae = VAE(512, 32)
    vae_beta = 0.001
    vae.load_state_dict(torch.load('saved_models/vae' + str(vae_beta) +'.pt'))
    vae.to(settings.device)

    num_prototypes = 1024  # Normally, 1024
    # num_prototypes = 100

    seeds = [i for i in range(0, 1)]
    # comm_types = ['vq', 'cont']
    comm_types = ['onehot']
    # if comm_types == ['onehot']:
    #     settings.kl_weight = 0.0
    # elif comm_types == ['vq']:
    #     settings.kl_weight = 0.01

    settings.lr = 0.001  # Default  FIXME
    if comm_types == ['onehot'] or comm_types == ['proto']:
        settings.lr = 0.0001
    elif comm_types == ['vq2']:
        settings.lr = 0.001  # We like 0.001 over 0.0001, it seems.

    # entropy_weight = 0.0
    entropy_weight = 0.0
    starting_weight = settings.kl_weight
    for num_tokens in [1]:
        # for alpha in [0.5, 1.5, 2, 3]:
        for alpha in [0, 0.1, 1, 0.5, 1.5, 2, 3, 10, 100]:
        # for alpha in [0]:
            if alpha == 0:
                variational = True
                # settings.kl_weight = 0.0
                # starting_weight = 0.0
                # if comm_types == ['vq']:
                #     settings.lr = 0.0001
            settings.alpha = alpha
            # for entropy_weight in [0.0, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06]:
            for starting_weight in [0.01]:
                settings.kl_weight = starting_weight
                settings.entropy_weight = entropy_weight
                for seed in seeds:
                    for speaker_type in comm_types:
                        print("Training comm type", speaker_type, "seed", seed, "for", num_tokens, "num tokens and", alpha, "alpha", "klweight", settings.kl_weight)
                        random.seed(seed)
                        np.random.seed(seed)
                        torch.manual_seed(seed)
                        settings.kl_weight = starting_weight
                        # savepath = 'saved_models/' + field_setup + '/trainfrac' + str(train_fraction) + '/' + speaker_type + '/alpha' + str(settings.alpha) + '/' + str(num_tokens) + 'tok/klweight' + str(settings.kl_weight) + '/seed' + str(seed) + '/'
                        savepath = 'saved_models/' + field_setup + '/trainfrac' + str(train_fraction) + '/' + speaker_type + '/alpha' + str(settings.alpha) + '/' + str(num_tokens) + 'tok/klweight' + str(settings.kl_weight) + '/seed' + str(seed) + '/'
                        run_trial()
