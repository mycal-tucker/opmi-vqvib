import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LinearRegression
import random

import src.settings as settings
from src.data_utils.helper_fns import gen_batch, get_glove_embedding, get_unique_labels, get_entry_for_labels
from src.data_utils.read_data import get_feature_data
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
            speaker_obs, listener_obs, labels, _ = gen_batch(dataset, batch_size, vae=vae, see_distractors=settings.see_distractor, num_dist=num_dist)
            outputs, _, _, recons = model(speaker_obs, listener_obs)
            recons = torch.squeeze(recons, dim=1)
        pred_labels = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        num_correct += np.sum(pred_labels == labels.cpu().numpy())
        num_total += pred_labels.size
        total_recons_loss += torch.mean(((speaker_obs - recons) ** 2)).item()
    acc = num_correct / num_total
    total_recons_loss = total_recons_loss / num_test_batches
    print("Evaluation accuracy", acc)
    print("Evaluation recons loss", total_recons_loss)
    return acc, total_recons_loss


def evaluate_with_english(model, dataset, vae, embed_to_tok, glove_data, use_top=True, num_dist=None, eng_dec=None, eng_list=None,
                          tok_to_embed=None, use_comm_idx=False, comm_map=None):
    topwords = dataset['topname']
    responses = dataset['responses']
    model.eval()
    num_nosnap_correct = 0
    num_snap_correct = 0
    num_total = 0
    eng_correct = 0
    # eval_dataset_size = len(dataset)
    eval_dataset_size = 1000
    print("Evaluating English performance on", eval_dataset_size, "examples")
    for targ_idx in range(eval_dataset_size):
        speaker_obs, listener_obs, labels, _ = gen_batch(dataset, 1, vae=vae, see_distractors=settings.see_distractor,
                                            num_dist=num_dist, preset_targ_idx=targ_idx)
        labels = labels.cpu().numpy()
        # Can pick just the topname, or one of the random responses.
        topword = topwords[targ_idx]
        if not use_top:
            all_responses = list(responses[targ_idx])
            if len(all_responses) == 1:
                continue  # Can't use a synonym if we only know one word
            word = topword
            while word == topword:
                word = random.choice(list(responses[targ_idx]))
        else:
            word = topword
        # Create the embedding, and then the token from the embedding.
        try:
            embedding = get_glove_embedding(glove_data, word).to_numpy()
        except AttributeError:  # If we don't have an embedding for the word, we get None back, so let's just move on.
            continue
        token = embed_to_tok.predict(np.expand_dims(embedding, 0))
        with torch.no_grad():
            tensor_token = torch.Tensor(token).to(settings.device)
            nosnap_prediction = model.pred_from_comms(tensor_token, listener_obs)
            # Snap the token to the nearest acceptable communication token
            tensor_token = model.speaker.snap_comms(tensor_token)
            snap_prediction = model.pred_from_comms(tensor_token, listener_obs)
        nosnap_pred_labels = np.argmax(nosnap_prediction.detach().cpu().numpy(), axis=1)
        num_nosnap_correct += np.sum(nosnap_pred_labels == labels)
        snap_pred_labels = np.argmax(snap_prediction.detach().cpu().numpy(), axis=1)
        num_snap_correct += np.sum(snap_pred_labels == labels)
        num_total += num_nosnap_correct.size

        # If parameters are provided, also evaluate EC -> English
        if eng_dec is None:
            continue
        with torch.no_grad():
            ec_comm, _, _ = model.speaker(speaker_obs)
            # comm_id = model.speaker.get_comm_id(ec_comm).detach().cpu().numpy()
            # relevant_comm = ec_comm if not use_comm_idx else comm_id

            ec_key = tuple(np.round(ec_comm.detach().cpu().squeeze(dim=0).numpy(), 3))

            if use_comm_idx and comm_map.get(ec_key) is None:
                # print("Couldn't find this comm! Marking as random chance")
                print("Using comm idx")
                eng_correct += 0.5
                continue
            # Convert comm id to onehot
            ec_comm = ec_comm.detach().cpu().numpy()
            relevant_comm = ec_comm
            if use_comm_idx:
                print("Using comm idx")
                dummy = np.zeros((1, len(comm_map.keys())))
                dummy[0, comm_map.get(ec_key)] = 1
                relevant_comm = dummy
            eng_comm = tok_to_embed.predict(relevant_comm)
            eng_comm = torch.Tensor(eng_comm).to(settings.device)
            # TODO: snap to best english embedding?
            recons = eng_dec(eng_comm)
            # Log the MSE just for fun
            # mse = torch.mean((speaker_obs - recons) ** 2)
            # print("Recons error", mse)
            # prediction = eng_list(recons, listener_obs)
            prediction = model.listener(recons, listener_obs)
            pred_label = np.argmax(prediction.detach().cpu().numpy(), axis=1)
            eng_correct += np.sum(pred_label == labels)
    return num_nosnap_correct / num_total, num_snap_correct / num_total, eng_correct / num_total


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
        label = top_comm_idx if top_likelihood > 0.4 else -1
        labels.append(label)
    features = np.vstack(dataset)
    label_np = np.reshape(np.array(labels), (-1, 1))
    all_np = np.hstack([label_np, features])
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


def get_embedding_alignment(model, dataset, glove_data, use_comm_idx=False):
    num_tests = 1
    comms = []
    embeddings = []
    features = []
    comm_ids = []
    comm_to_id = {}
    max_num_align_data = 1000   # FIXME
    for f, word in zip(dataset['features'], dataset['topname']):
        speaker_obs = torch.Tensor(np.array(f)).to(settings.device)
        speaker_obs = torch.unsqueeze(speaker_obs, 0)
        with torch.no_grad():
            speaker_obs = speaker_obs.repeat(num_tests, 1)
            comm, _, _ = model.speaker(speaker_obs)
        try:
            embedding = get_glove_embedding(glove_data, word).to_numpy()
        except AttributeError:
            continue
        # comm_id = model.speaker.get_comm_id(comm).detach().cpu().numpy()
        # comm_ids.append(comm_id)
        np_comm = np.mean(comm.detach().cpu().numpy(), axis=0)
        comms.append(np_comm)

        dict_comm = tuple(np.round(np_comm, 3))
        matching_id = comm_to_id.get(dict_comm)
        if matching_id is None:
            matching_id = len(comm_to_id.keys())
            comm_to_id[dict_comm] = matching_id
        comm_ids.append(matching_id)

        embeddings.append(embedding)
        features.append(np.array(f))
        if len(embeddings) > max_num_align_data:
            print("Breaking after", len(embeddings), "examples for word alignment")
            break

    comms = np.vstack(comms)
    # comm_ids = np.vstack(comm_ids)
    # relevant_comms = comm_ids if use_comm_idx else comms

    relevant_comms = comms
    if use_comm_idx:
        # Convert to onehot
        relevant_comms = np.zeros((len(comm_ids), len(comm_to_id.keys())))
        for i, c_id in enumerate(comm_ids):
            relevant_comms[i, c_id] = 1
    embeddings = np.vstack(embeddings)
    # Now learn two linear regressions to map to/from tokens and word embeddings.
    reg1 = LinearRegression()
    reg1.fit(relevant_comms, embeddings)
    tok_to_embed_r2 = reg1.score(relevant_comms, embeddings)
    print("Tok to word embedding regression score\t\t", tok_to_embed_r2)
    # reg1.fit(comms, comms)  # Actually, if we just want to hardwire it in, "refit" the regression to be 1-1
    reg2 = LinearRegression()
    reg2.fit(embeddings, comms)
    embed_to_tok_r2 = reg2.score(embeddings, comms)
    print("Word embedding to token regression score\t\t", embed_to_tok_r2)
    # reg2.fit(embeddings, embeddings)  # Actually, if we just want to hardwire it in, "refit" the regression to be 1-1
    print("Found", len(comm_to_id), "unique comm vectors!")
    # Gross. Instead of regression score, pass back the number of unique vectors
    return reg1, reg2, tok_to_embed_r2, len(comm_to_id), comm_to_id


# Manually calculate the complexity of communication by sampling some inputs and comparing the conditional communication
# to the marginal communication.
def get_probs(speaker, dataset):
    num_samples = 1000
    w_m = np.zeros((num_samples, speaker.num_tokens))
    for i in range(num_samples):
        speaker_obs, _, _, _ = gen_batch(dataset, batch_size=1)
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


def eval_model(model, vae, comm_dim, train_data, val_data, viz_data, glove_data, num_cand_to_metrics, savepath,
               epoch, calculate_complexity=False, plot_comms_flag=False, alignment_dataset=None, save_model=True):
    # Create a directory to save information, models, etc.
    basepath = savepath + str(epoch) + '/'
    if not os.path.exists(basepath):
        os.makedirs(basepath)
    # Calculate efficiency values like complexity and informativeness.
    # Can estimate complexity by sampling inputs and measuring communication probabilities.
    # get_probs(model.speaker, train_data)
    # Or we can use MINE to estimate complexity and informativeness.
    if calculate_complexity:
        train_complexity = get_info(model, train_data, targ_dim=comm_dim, num_epochs=200, comm_targ=True)
        # val_complexity = get_info(model, val_features, targ_dim=comm_dim, comm_targ=True)
        val_complexity = None
        print("Train complexity", train_complexity)
        print("Val complexity", val_complexity)
    else:
        train_complexity = None
        val_complexity = None
    # And compare to english word embeddings (doesn't depend on number of distractors)
    align_data = train_data if alignment_dataset is None else alignment_dataset
    tok_to_embed, embed_to_tok, tokr2, embr2, _ = get_embedding_alignment(model, align_data, glove_data)
    eval_batch_size = 256
    val_is_train = len(train_data) == len(val_data)  # Not actually true, but close enough
    if val_is_train:
        print("WARNING: ASSUMING VALIDATION AND TRAIN ARE SAME")

    complexities = [train_complexity, val_complexity]
    for feature_idx, data in enumerate([train_data, val_data]):
        if feature_idx == 1 and val_is_train:
            pass
        for num_candidates in num_cand_to_metrics.keys():
            if feature_idx == 1 and val_is_train:
                pass  # Just save the values from last time.
            else:
                acc, recons = evaluate(model, data, eval_batch_size, vae, num_dist=num_candidates - 1)
                # Using the embed_to_tok, map English words to tokens to see if the listener can do well.
                # During training, just set to none. It's so noisy that we just run eval_trials.py for this.
                eng_train_top_score = None
                eng_train_syn_score = None
                eng_val_top_score = None
                eng_val_syn_score = None
                # eng_train_top_score = evaluate_with_english(model, train_data, vae, embed_to_tok, glove_data,
                #                                             use_top=True,
                #                                             num_dist=num_candidates - 1)
                # print("English topname train accuracy", eng_train_top_score)
                # eng_train_syn_score = evaluate_with_english(model, train_data, vae, embed_to_tok, glove_data,
                #                                             use_top=False,
                #                                             num_dist=num_candidates - 1)
                # print("English synonym train accuracy", eng_train_syn_score)
                # eng_val_top_score = eng_train_top_score if val_is_train else evaluate_with_english(model, val_data, vae,
                #                                                                                    embed_to_tok,
                #                                                                                    glove_data,
                #                                                                                    use_top=True,
                #                                                                                    num_dist=num_candidates - 1)
                # print("English topname val accuracy", eng_val_top_score)
                # eng_val_syn_score = eng_train_syn_score if val_is_train else evaluate_with_english(model, val_data, vae,
                #                                                                                    embed_to_tok,
                #                                                                                    glove_data,
                #                                                                                    use_top=False,
                #                                                                                    num_dist=num_candidates - 1)
                # print("English synonym val accuracy", eng_val_syn_score)
            relevant_metrics = num_cand_to_metrics.get(num_candidates)[feature_idx]
            relevant_metrics.add_data(epoch, complexities[feature_idx], -1 * recons, acc, settings.kl_weight,
                                      tokr2, embr2, eng_train_top_score, eng_train_syn_score, eng_val_top_score, eng_val_syn_score)
    # Plot some of the metrics for online visualization
    comm_accs = []
    top_eng_scores = []
    syn_eng_scores = []
    top_val_eng_scores = []
    syn_val_eng_scores = []
    regressions = []
    labels = []
    epoch_idxs = None
    for feature_idx, label in enumerate(['train', 'val']):
        for num_candidates in sorted(num_cand_to_metrics.keys()):
            comm_accs.append(num_cand_to_metrics.get(num_candidates)[feature_idx].comm_accs)
            top_eng_scores.append(num_cand_to_metrics.get(num_candidates)[feature_idx].top_eng_acc)
            syn_eng_scores.append(num_cand_to_metrics.get(num_candidates)[feature_idx].syn_eng_acc)
            top_val_eng_scores.append(num_cand_to_metrics.get(num_candidates)[feature_idx].top_val_eng_acc)
            syn_val_eng_scores.append(num_cand_to_metrics.get(num_candidates)[feature_idx].syn_val_eng_acc)
            regressions.append(num_cand_to_metrics.get(num_candidates)[feature_idx].embed_r2)
            labels.append(" ".join([label, str(num_candidates), "utility"]))
            if epoch_idxs is None:
                epoch_idxs = num_cand_to_metrics.get(num_candidates)[feature_idx].epoch_idxs
    plot_metrics(comm_accs, labels, epoch_idxs, basepath=basepath)
    plot_metrics(regressions, ['r2 score'], epoch_idxs, basepath=basepath + 'regression_')
    plot_metrics(top_eng_scores, ['top eng score'], epoch_idxs, basepath=basepath + 'top_eng_')
    plot_metrics(syn_eng_scores, ['syn eng score'], epoch_idxs, basepath=basepath + 'syn_eng_')
    plot_metrics(top_val_eng_scores, ['top val eng score'], epoch_idxs, basepath=basepath + 'top_val_eng_')
    plot_metrics(syn_val_eng_scores, ['syn val eng score'], epoch_idxs, basepath=basepath + 'syn_val_eng_')
    # Visualize some of the communication
    try:
        if plot_comms_flag:
            plot_comms(model, viz_data['features'], basepath)
    except AssertionError:
        print("Can't plot comms for whatever reason (e.g., continuous communication makes categorizing hard)")
    # Save the model and metrics to files.
    for feature_idx, label in enumerate(['train', 'val']):
        for num_candidates in sorted(num_cand_to_metrics.keys()):
            metric = num_cand_to_metrics.get(num_candidates)[feature_idx]
            metric.to_file(basepath + "_".join([label, str(num_candidates), "metrics"]))
    if not save_model:
        return
    torch.save(model.state_dict(), basepath + 'model.pt')
    torch.save(model, basepath + 'model_obj.pt')


def get_supervised_data(train_data, num_examples, glove_data, vae):
    speaker_obs = []
    embs = []
    for i in range(len(train_data)):
        if len(embs) == num_examples:
            break
        obs, _, _, emb = gen_batch(train_data, 1, glove_data=glove_data, vae=vae, preset_targ_idx=i)
        if emb[0] is not None:  # It's a list
            embs.append(emb)
            speaker_obs.append(obs.cpu())  # Put on the cpu to get into numpy for moving. Awkward, I know, but whatever.
    speaker_obs = torch.Tensor(np.vstack(speaker_obs)).to(settings.device)
    embs = torch.Tensor(np.vstack(embs)).to(settings.device)
    return [speaker_obs, embs]


def get_super_loss(supervised_data, speaker):
    supervision_crit = nn.MSELoss()
    speaker.eval_mode = True
    comms, _, _ = speaker(supervised_data[0])
    speaker.eval_mode = False
    supervised_loss = supervision_crit(comms, supervised_data[1])
    # print("Supervised loss", supervised_loss)
    return supervised_loss


def train(model, train_data, val_data, viz_data, glove_data, vae, savepath, comm_dim, num_epochs=3000, batch_size=1024,
          burnin_epochs=500, val_period=200, plot_comms_flag=False, calculate_complexity=False):
    unique_topnames, _ = get_unique_labels(train_data)
    sup_dataset = pd.concat([get_entry_for_labels(train_data, unique_topnames) for _ in range(3)])
    sup_dataset = sup_dataset.sample(frac=1).reset_index(drop=True)  # Shuffle the data.
    supervised_data = get_supervised_data(sup_dataset, 1000, glove_data, vae)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    running_acc = 0
    running_mse = 0
    num_cand_to_metrics = {2: [], 8: [], 16: []}
    for empty_list in num_cand_to_metrics.values():
        empty_list.extend([PerformanceMetrics(), PerformanceMetrics()])  # Train metrics, validation metrics
    settings.epoch = 0
    for epoch in range(num_epochs):
        settings.epoch += 1
        if epoch > burnin_epochs:
            settings.kl_weight += settings.kl_incr
        speaker_obs, listener_obs, labels, _ = gen_batch(train_data, batch_size, vae=vae, see_distractors=settings.see_distractor)
        optimizer.zero_grad()
        outputs, speaker_loss, info, recons = model(speaker_obs, listener_obs)

        loss = criterion(outputs, labels)
        supervised_loss = 0
        if settings.supervision_weight != 0:  # Don't even bother computing the loss if it's not used.
            supervised_loss = get_super_loss(supervised_data, model.speaker)
            loss = loss + settings.supervision_weight * supervised_loss
        if len(speaker_obs.shape) == 2:
            speaker_obs = torch.unsqueeze(speaker_obs, 1)
        recons_loss = torch.mean(((speaker_obs - recons) ** 2))
        loss += settings.alpha * recons_loss
        loss += speaker_loss
        # print("Speaker loss fraction:\t", speaker_loss.item() / loss.item())
        # print("Recons loss fraction:\t", settings.alpha * recons_loss.item() / loss.item())
        loss.backward()
        optimizer.step()

        # Metrics
        pred_labels = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        num_correct = np.sum(pred_labels == labels.cpu().numpy())
        num_total = pred_labels.size
        running_acc = running_acc * 0.95 + 0.05 * num_correct / num_total
        running_mse = running_mse * 0.95 + 0.05 * recons_loss.item()
        if epoch % 20 == 0:
            print('epoch', epoch, 'of', num_epochs)
            print("Overall loss", loss.item())
            print("Kl weight", settings.kl_weight)
            print("Running acc", running_acc)
            print("Running mse", running_mse)
            print("Supervised loss", supervised_loss)

        if epoch % val_period == val_period - 1:
        # if epoch > burnin_epochs and 0.17 < running_mse < 0.22 and np.random.random() < 0.05:
            eval_model(model, vae, comm_dim, train_data, val_data, viz_data, glove_data, num_cand_to_metrics,
                       savepath, epoch, calculate_complexity=calculate_complexity, plot_comms_flag=plot_comms_flag)


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
