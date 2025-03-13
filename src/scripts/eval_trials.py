import os
import random

import numpy as np
import torch
import csv
import src.settings as settings
from src.data_utils.helper_fns import get_unique_labels, get_entry_for_labels, get_unique_by_field, get_rand_entries
from src.data_utils.read_data import get_feature_data, get_glove_vectors
from src.models.decoder import Decoder
from src.models.proto import ProtoNetwork
from src.models.listener import Listener
from src.models.mlp import MLP
from src.models.team import Team
from src.models.vae import VAE
from src.models.vq import VQ
from src.models.vq2 import VQ2
from src.models.vq_after import VQAfter
from src.scripts.main import eval_model, get_embedding_alignment, evaluate_with_english, get_relative_embedding
from src.utils.performance_metrics import PerformanceMetrics
from src.utils.plotting import plot_scatter, plot_multi_trials


# For a given particular setup, load and evaluate all the checkpoints
def eval_run(basepath, num_tok, speaker_type, train_data, alignment_datasets):
    list_of_files = os.listdir(basepath)
    checkpoints = sorted([int(elt) for elt in list_of_files])

    complexities = []
    mses = []
    # eng_decoder = torch.load('english_vg_dec64.pt').to(settings.device)
    # eng_listener = torch.load('english_vg_list64.pt').to(settings.device)
    # eng_decoder = torch.load('english_resp_dec64.pt').to(settings.device)
    # eng_listener = torch.load('english_resp_list64.pt').to(settings.device)
    eng_decoder = torch.load('english_resp2_dec64.pt').to(settings.device)
    eng_listener = torch.load('english_resp2_list64.pt').to(settings.device)
    checkpoint = checkpoints[-1]
    # checkpoint = 19999
    # checkpoint = 99999
    print("Evaluating checkpoint", checkpoint)
    # if checkpoint != 19999:
    #     print("Skipping checkpoint", checkpoint)
    #     return
    # print("\n\n\nCheckpoint", checkpoint)
    # Load the model
    try:
        team = torch.load(basepath + str(checkpoint) + '/model_obj.pt')
    except FileNotFoundError:
        print("Failed to load the preferred full model; falling back to statedict.")
        feature_len = 512
        if speaker_type == 'vq':
            speaker = VQ(feature_len, comm_dim, num_layers=3, num_protos=1024, specified_tok=None,
                         num_simultaneous_tokens=num_tok,
                         variational=True, num_imgs=1)
        elif speaker_type == 'onehot':
            speaker = MLP(feature_len, comm_dim, num_layers=3, onehot=True, num_simultaneous_tokens=num_tok,
                          variational=True, num_imgs=1)
        elif speaker_type == 'proto':
            speaker = ProtoNetwork(feature_len, comm_dim, num_layers=3, num_protos=1024, variational=True)
        listener = Listener(feature_len)
        decoder = Decoder(comm_dim, feature_len, num_layers=3, num_imgs=1)
        team = Team(speaker, listener, decoder)
        team.load_state_dict(torch.load(basepath + str(checkpoint) + '/model.pt'))
        team.to(settings.device)
    # And evaluate it
    metric = PerformanceMetrics.from_file(basepath + str(checkpoint) + '/train_True_2_metrics')
    comps = metric.complexities
    print("Comps", comps)
    if comps[-1] is not None:
        complexities.append(comps[-1])
        mses.append(-1 * metric.recons[-1])
    else:  # If we didn't calculate complexity during the training run
        print("Running full eval to get complexity")
        mses.append(None)
        complexities.append(None)
        # eval_model(team, vae, comm_dim, train_data, train_data, None, glove_data,
        #            num_cand_to_metrics=num_cand_to_metrics, savepath=basepath, epoch=checkpoint,
        #            calculate_complexity=True, alignment_dataset=alignment_dataset, save_model=False)
        # metric = PerformanceMetrics.from_file(basepath + str(checkpoint) + '/train_2_metrics')
        # complexities.append(metric.complexities[-1])
    top_eng_to_ec_snap_accs = []
    top_eng_to_ec_nosnap_accs = []
    top_ec_to_eng_comm_id_accs = []
    top_ec_to_eng_accs = []
    tok_to_embed_r2s = []
    embed_to_tok_r2s = []
    for j, align_data in enumerate(alignment_datasets):
        for use_comm_idx in [False]:
            use_top = True
            dummy_eng = english_fieldname
            if english_fieldname == 'responses':
                use_top = False
                dummy_eng = 'topname'
            # consistency_score = get_relative_embedding(team, align_data, glove_data, train_data, fieldname='responses')
            # print("consistency percent", consistency_score)
            print("Not calculating consistency.")
            tok_to_embed, embed_to_tok, tok_to_embed_r2, embed_to_tok_r2, comm_map = get_embedding_alignment(team, align_data, glove_data,
                                                                                 fieldname=alignment_fieldname)
            nosnap, snap, ec_to_eng = evaluate_with_english(team, train_data, vae,
                                                            embed_to_tok,
                                                            # embed_to_tok_fn,
                                                            glove_data,
                                                            fieldname=experiment_fieldname,
                                                            eng_fieldname=dummy_eng,
                                                            use_top=use_top,
                                                            # num_dist=settings.num_distractors,
                                                            eng_dec=eng_decoder,
                                                            # eng_dec=team.decoder,
                                                            # eng_dec=None,  # Set to None because we don't want to bother
                                                            eng_list=eng_listener,
                                                            tok_to_embed=tok_to_embed,
                                                            use_comm_idx=use_comm_idx, comm_map=comm_map)
            if use_comm_idx:
                top_ec_to_eng_comm_id_accs.append(ec_to_eng)
            else:
                top_ec_to_eng_accs.append(ec_to_eng)
                # Just to avoid duplicates, pick either use comm idx or not for this.
                top_eng_to_ec_nosnap_accs.append(nosnap)
                top_eng_to_ec_snap_accs.append(snap)
                tok_to_embed_r2s.append(tok_to_embed_r2)
                embed_to_tok_r2s.append(embed_to_tok_r2)
            # If calculating consistency instead of translation stuff.
            # top_eng_to_ec_nosnap_accs.append(consistency_score)
    num_runs = len(alignment_datasets)
    return complexities[-1],\
           (np.median(top_ec_to_eng_comm_id_accs), np.std(top_ec_to_eng_comm_id_accs) / np.sqrt(num_runs)),\
           (np.median(top_ec_to_eng_accs), np.std(top_ec_to_eng_accs) / np.sqrt(num_runs)),\
           (np.median(top_eng_to_ec_nosnap_accs), np.std(top_eng_to_ec_nosnap_accs) / np.sqrt(num_runs)),\
           (np.median(top_eng_to_ec_snap_accs), np.std(top_eng_to_ec_snap_accs) / np.sqrt(num_runs)),\
           (np.median(tok_to_embed_r2s), np.std(tok_to_embed_r2s) / np.sqrt(num_runs)),\
           (np.median(embed_to_tok_r2s), np.std(embed_to_tok_r2s) / np.sqrt(num_runs))


# Iterate over combinations of hyperparameters and seeds.
def run():
    base = 'saved_models/topname/trainfrac0.2/'
    # base = 'saved_models/all/trainfrac1.0/'
    # base = 'saved_models/vg_domain/trainfrac1.0/'
    for speaker_type in model_types:
        for alpha in alphas:
            for num_tok in num_tokens:
                all_comps = []
                all_top_ec_to_eng_comm_ids = [[], []]  # Mean, std.
                all_top_ec_to_eng = [[], []]
                all_top_eng_to_ec_nosnap = [[], []]
                all_top_eng_to_ec_snap = [[], []]
                all_tok_to_embed_r2 = [[], []]
                all_embed_to_tok_r2 = [[], []]
                labels = []
                filename = '_'.join([str(num_examples), str(num_tok), alignment_fieldname, experiment_fieldname, english_fieldname])
                with open(filename + '.csv', 'w', newline='') as f:
                    writer = csv.writer(f, delimiter=',')
                    writer.writerow(['lambda C', 'Seed', 'Complexity',
                                     'EC to Eng Comm Id Mean', 'EC to Eng Comm Id Std',
                                     'EC to Eng mean', 'EC to Eng Std',
                                     'Eng to EC nosnap mean', 'Eng to EC nosnap Std',
                                     'Eng to EC snap mean', 'Eng to EC snap Std',
                                     'Tok to Embed r2 mean', 'Tok to Embed r2 Std',
                                     'Embed to Tok r2 mean', 'Embed to Tok Std'])
                for kl_weight in kl_weights:
                    comps = []
                    top_ec_to_eng_comm_ids = []
                    top_ec_to_engs = []
                    top_eng_to_ecs_nosnaps = []
                    top_eng_to_ecs_snaps = []
                    tok_to_embed_r2s = []
                    embed_to_tok_r2s = []
                    basepath = '/'.join([base, speaker_type, 'alpha' + str(alpha), str(num_tok) + 'tok', 'klweight' + str(kl_weight)]) + '/'
                    for s in seeds:
                        print("Seed", s)
                        random.seed(s)
                        np.random.seed(s)
                        torch.manual_seed(s)
                        if s in seed_to_state.keys():
                            print("Loading a saved state!")
                            np.random.set_state(seed_to_state.get(s))
                        else:
                            glove_data = get_glove_vectors(embed_dim)  # Need to do this for random seed stuff
                        train_data = get_feature_data(features_filename,
                                                      selected_fraction=0.2)
                        unique_topnames = get_unique_by_field(train_data, alignment_fieldname)
                        print(len(unique_topnames), "unique classes for fieldname", alignment_fieldname)
                        alignment_datasets = [get_rand_entries(train_data, num_examples) for _ in range(num_rand_trial)]

                        comp, top_ec_to_eng_comm_id, top_ec_to_eng, top_eng_to_ec_nosnap, top_eng_to_ec_snap, tok_to_embed_r2, embed_to_tok_r2 = eval_run(basepath + '/seed' + str(s) + '/', num_tok, speaker_type, train_data, alignment_datasets)
                        comps.append(comp)
                        top_ec_to_eng_comm_ids.append(top_ec_to_eng_comm_id)
                        top_ec_to_engs.append(top_ec_to_eng)
                        top_eng_to_ecs_nosnaps.append(top_eng_to_ec_nosnap)
                        top_eng_to_ecs_snaps.append(top_eng_to_ec_snap)
                        tok_to_embed_r2s.append(tok_to_embed_r2)
                        embed_to_tok_r2s.append(embed_to_tok_r2)
                    all_comps.append(comps)
                    for i in range(2):
                        all_top_ec_to_eng_comm_ids[i].append([elt[i] for elt in top_ec_to_eng_comm_ids])
                        all_top_ec_to_eng[i].append([elt[i] for elt in top_ec_to_engs])
                        all_top_eng_to_ec_nosnap[i].append([elt[i] for elt in top_eng_to_ecs_nosnaps])
                        all_top_eng_to_ec_snap[i].append([elt[i] for elt in top_eng_to_ecs_snaps])
                        all_tok_to_embed_r2[i].append([elt[i] for elt in tok_to_embed_r2s])
                        all_embed_to_tok_r2[i].append([elt[i] for elt in embed_to_tok_r2s])
                    labels.append('$\lambda_C =$ ' + str(kl_weight))
                    # plot_multi_trials([all_comps, all_top_ec_to_eng_comm_ids[0], all_top_ec_to_eng_comm_ids[1]],
                    #                   labels,
                    #                   [size for _ in labels],
                    #                   ylabel='Utility EC to Eng Top Comm ID',
                    #                   filename='/'.join([basepath + '../', '_'.join([rand_string, str(num_examples), str(num_tok), alignment_fieldname, experiment_fieldname, english_fieldname, 'EC_to_Eng_Top_Comm_ID'])]))
                    # plot_multi_trials([all_comps, all_top_ec_to_eng[0], all_top_ec_to_eng[1]],
                    #                   labels,
                    #                   [size for _ in labels],
                    #                   ylabel='Utility EC to Eng Top',
                    #                   filename='/'.join([basepath + '../', '_'.join([rand_string, str(num_examples), str(num_tok), alignment_fieldname, experiment_fieldname, english_fieldname, 'EC_to_Eng_Top'])]))
                    # plot_multi_trials([all_comps, all_top_eng_to_ec_nosnap[0], all_top_eng_to_ec_nosnap[1]],
                    #                   labels,
                    #                   [size for _ in labels],
                    #                   ylabel='Utility Eng to EC Top No Snap',
                    #                   filename='/'.join([basepath + '../', '_'.join([rand_string, str(num_examples), str(num_tok), alignment_fieldname, experiment_fieldname, english_fieldname, 'Eng_to_EC_Top_Nosnap'])]))
                    # plot_multi_trials([all_comps, all_top_eng_to_ec_snap[0], all_top_eng_to_ec_snap[1]],
                    #                   labels,
                    #                   [size for _ in labels],
                    #                   ylabel='Utility Eng to EC TopSnap',
                    #                   filename='/'.join([basepath + '../', '_'.join([rand_string, str(num_examples), str(num_tok), alignment_fieldname, experiment_fieldname, english_fieldname, 'Eng_to_EC_Top_Snap'])]))
                    # plot_multi_trials([all_comps, all_tok_to_embed_r2[0], all_tok_to_embed_r2[1]],
                    #                   labels,
                    #                   [size for _ in labels],
                    #                   ylabel='Tok to Embed r2',
                    #                   filename='/'.join([basepath + '../', '_'.join([rand_string, str(num_examples), str(num_tok), alignment_fieldname, experiment_fieldname, english_fieldname, 'tok_to_embed_r2'])]))
                    # plot_multi_trials([all_comps, all_embed_to_tok_r2[0], all_embed_to_tok_r2[1]],
                    #                   labels,
                    #                   [size for _ in labels],
                    #                   ylabel='Embed to Tok r2',
                    #                   filename='/'.join([basepath + '../', '_'.join([rand_string, str(num_examples), str(num_tok), alignment_fieldname, experiment_fieldname, english_fieldname, 'embed_to_tok_r2'])]))
                    # with open(filename + '.csv', 'a', newline='') as f:
                    #     writer = csv.writer(f, delimiter=',')
                    #     for i in range(len(seeds)):
                    #         writer.writerow([kl_weight, seeds[i], np.round(comps[i], 3),
                    #                          np.round(top_ec_to_eng_comm_ids[i][0], 3),
                    #                          np.round(top_ec_to_eng_comm_ids[i][1], 3),
                    #                          np.round(top_ec_to_engs[i][0], 3),
                    #                          np.round(top_ec_to_engs[i][1], 3),
                    #                          np.round(top_eng_to_ecs_nosnaps[i][0], 3),
                    #                          np.round(top_eng_to_ecs_nosnaps[i][1], 3),
                    #                          np.round(top_eng_to_ecs_snaps[i][0], 3),
                    #                          np.round(top_eng_to_ecs_snaps[i][1], 3),
                    #                          np.round(tok_to_embed_r2s[i][0], 3),
                    #                          np.round(tok_to_embed_r2s[i][1], 3),
                    #                          np.round(embed_to_tok_r2s[i][0], 3),
                    #                          np.round(embed_to_tok_r2s[i][1], 3)])
                print("Num align data:", num_examples)
                print("Alpha:", alpha)
                print("Alignment fieldname:", alignment_fieldname)
                print("Expt fieldname:", experiment_fieldname)
                print("English fieldname:", english_fieldname)
                print("Num distractors:", settings.num_distractors)
                print("KL Weights:\n", kl_weights)
                # print("Complexities\n", [np.round(e, 3).tolist() for e in all_comps])
                # print("EC to Eng Comm id mean\n", np.round(all_top_ec_to_eng_comm_ids[0], 3))
                # print("EC to Eng Comm id std\n", np.round(all_top_ec_to_eng_comm_ids[1], 3))
                print("EC to Eng mean\n", np.round(all_top_ec_to_eng[0], 3).tolist())
                print("EC to Eng std\n", np.round(all_top_ec_to_eng[1], 3))
                print("Eng to EC nosnap mean\n", np.round(all_top_eng_to_ec_nosnap[0], 3).tolist())
                print("Eng to EC nosnap std\n", np.round(all_top_eng_to_ec_nosnap[1], 3))
                # print("Eng to EC snap mean\n", np.round(all_top_eng_to_ec_snap[0], 3))
                # print("Eng to EC snap std\n", np.round(all_top_eng_to_ec_snap[1], 3))
                print("\n\n\n")


if __name__ == '__main__':
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    num_rand_trial = 5  # FIXME: 10

    settings.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    settings.see_distractor = False
    settings.learned_marginal = False
    settings.embedding_cache = {}
    settings.sample_first = True
    settings.hardcoded_vq = False
    settings.kl_weight = 0.0
    settings.epoch = 0
    settings.alpha = 0
    settings.distinct_words = False  # FIXME
    settings.entropy_weight = 0.0
    settings.max_num_align_data = 10000

    # comm_dim = 64
    comm_dim = 64
    features_filename = 'data/features_nobox.csv'

    # Load the dataset
    embed_dim = 64

    glove_data = get_glove_vectors(embed_dim)

    seed_to_state = {0: ('MT19937', np.array([4211200251,    1495080, 1333446588, 2357801020, 1130639478,
        487638447, 3725283550,   39146648, 1065937415, 4255066115,
        397770284, 3826074799, 4254744370,  794858159, 3056368334,
       2235351281, 3910665812, 2597282274, 3589355900, 3782463336,
       1653104972, 3719242961, 4217544166, 2333173009,  537997638,
       3178400034, 4150235944, 1433053781,  644355714, 2062458388,
       2537280040, 3631506064, 2606224160,  363396502, 3421024527,
       2016730423, 1152831051, 3635708581, 2291951660, 4085251013,
       2533279432, 3121070769,   72228808, 3922069077, 2978706684,
       1921789229, 4255584465, 1403289122, 1058102511, 2714553678,
        937828538, 2251299646, 3120280598, 3190328546, 3813602289,
       2093991894, 1388523655, 1028418503, 1906264818, 1755267690,
       1573280321, 2306329935, 2840465768, 1859258182,  812674069,
        552369520, 1370456121,  238393402, 2149283928, 3953932322,
       2187842840, 1323333621, 4229078222, 3038091324, 2526601601,
       1874120584, 3294698320, 2128476949, 2697447323, 3501276835,
       1742707385,  872354648, 2230513495,  224635975, 2147931798,
       1029634825,  839684931, 1339666365, 2253848533, 1691934552,
       1824818488, 1076022551,  511996302, 1451644266,  946795651,
       3597799702, 2253607502, 1335956678, 2276519545, 2969199915,
       3387407988, 1277661772, 1913898469,  461696811, 1956867707,
        925691488, 4105861711, 3463158482,  213356189, 1539338586,
       3212791222, 4145533075,  543054484, 2076231118, 1770888336,
       3478435577, 2359333104, 4181022464, 3109430959, 2149737074,
       2874436700, 3896314371, 2903224029,  659291336, 3860663918,
       3398287562, 3226494562, 4062163016, 1276693831, 3898916998,
       4148420520, 3588619132,  341835051, 4214484692, 1635591462,
       1721842533,  807007664, 3142156751, 4088337250, 3075410797,
       4272432751, 2101738597,  204715327,  294462827,  743357291,
        247168789, 2451339365, 2477567948, 3024583456, 2082779688,
       1163860555, 3424567364,  723450015,  412340537, 1884848179,
        102487990, 1291967873, 3180119436,    5186252, 2280744690,
       3074188014, 1484928632, 2837255609, 1721151752, 2422344038,
       1698466205, 1901919879,   84660249, 1079114812,  912540815,
       2672346314, 3042437725,  527615275, 2996966015, 3626376046,
       1814392805, 2732704640, 2620974434,   84012383, 1796526074,
       2950783893, 3674639056, 3319669400,  690808062, 1877127597,
       1846757389, 1415793595, 1970848219, 3379368234, 2477337183,
       3023030609, 3910509373, 3898663551, 2282724366, 2320098614,
       2081479078,  120635847, 1354909038, 2819657434,  323396770,
        355779272, 4082591619, 3945807002, 2558699366, 4271890792,
        846320721, 2236769186,   28229477, 2661593013, 2792667759,
       2797719449, 3588826542,  592448801,   28966370, 3795275998,
       3987480774,  292209029, 2262321793,  802232958, 1526245319,
         44195887, 1089302178, 3185344277, 4199715533, 2412991257,
       3786799721, 3237967105, 1549621572, 1122215046, 4205892045,
       2016277121, 3236819088, 1723170200,  247827647, 3137943946,
       1064854731, 3728707232, 2984969440, 2742569332, 3517279500,
       3052961260, 1977806382, 2429897199, 3665322211, 4265559941,
       3628745968, 2244611483,  375370236, 2015354676, 4202030277,
       3747148599, 2855992575,  595096068, 1702533169,  291357877,
       3866641341, 1610543355,  759739812, 2356765625, 2965217823,
       2773354973,  681622462, 3879519186, 2096173455,  403537335,
        423957961, 3041477466, 3281651616, 3530603932,  150480227,
         92466439, 3433679109, 2798323843,  270819191, 3959586160,
       1773956305, 3952293479, 2008278822, 4084378699, 2572056019,
       3031540939, 2864618691, 3754074973,  584219803,  864252806,
       1004655632,  806666342, 4219899466, 1769283357, 3482985097,
       1047226465, 1801211502, 2717391222, 2097488407, 3977246827,
       2637866754, 4162802863, 1603643471,  110924356,  645089689,
        906233461,  112605356, 4196061250, 1504967203, 1832935494,
       1507722609, 3568233512, 4237041047, 2666064688, 4195897391,
       3118827938, 2278647928,  388551184, 1624120039, 2769724149,
       4157865705,  198036532, 3789790400, 3430979309,  278076178,
       2399401463, 2118313086,  857216817, 1029776709, 3464691794,
        206998500,  185897143, 1694662842, 2883285732,  686631845,
       4260902402, 4236102630, 2436903333, 4124429344, 4038695460,
       3495408420, 3130891935, 4151028845, 1721057633, 3324054820,
        736687483, 1285887570, 4018312534, 2397750872, 4026748645,
       3631361993, 3484698875, 3959623116, 1125849120, 1378246025,
       2718336623,  810063843, 1671608834,  220123835,  363625101,
        213541444, 2936958849, 3143381517,  452385728, 4229183829,
       2545254774, 1828785016, 3693810118, 2083551911, 3777819688,
       3265939838, 2233017808, 1232324917, 2567484748, 3247212685,
       1575876414, 4026033131, 1898565173, 3368071762, 1926024167,
       3582900724, 3032093552, 4248693753, 3668903982, 1389808526,
       3150509040,   13909188, 2627752476, 4202076285, 1068261721,
       1195295214, 2591269463, 2099000010, 3717063286, 3223439373,
        604641984, 4144480048, 1838068745, 3433044286, 1652631689,
       1957595461, 2350159484, 1427680634,  254899780,   56116156,
       2160061178, 3471716615, 2130156709, 3731861441,  167081596,
       1429445759,  833064799, 2460593016, 1491259581, 1827415565,
       3949761947, 1219801561, 2243545811, 1792496248,  565881243,
       1419896964, 3718141468, 2767926969, 1295793937, 3882165622,
       3521394574, 2595491201, 2105453773, 1196792528, 1526227818,
       1272351584, 2794339883, 1041577388, 4076308959, 2624647147,
       3200991274, 2452800036, 3307395380, 2452455428, 1163218816,
       1213879487, 4065523647, 1399355409, 3545891102,  306184607,
       3006821523, 3809057068,  508744494,  750779595, 3622171951,
       2209591577, 1376783698, 1677653138,  799469241, 3545166056,
       1326402105, 3239152674, 3293812057, 2669430849, 2583360532,
       2251439614,  771250877, 3154492835, 2430659770,  265247511,
       3014140786,   70069363,  153975957, 1433305065, 1170694360,
       3667663489, 2441901480, 1509850994, 2491695994,  204329350,
       1194421031,  729297093, 1589640920, 4056824299, 1186393325,
       1446811366, 3492158269,  662721861, 2299335249, 4184404916,
       3017489729, 4260827752, 1295658713,  284451116, 1707203165,
       2731173642, 1371755134, 3132235084,   77442265, 2803046647,
       2362629637, 1132328676,   53502997, 3715458918, 2921852443,
       3507209073,  265319612, 2651726586, 2209892581,  624286762,
       1727843392,   14874806, 4052000078, 1823608555, 3954634882,
       2881954529, 1966540553, 1610070556, 3441531658, 1124855100,
        704818576, 2373930141, 2218118841,  892136021,  206352861,
       1403805803, 3410336448,  278956508, 4005559451,  895579135,
       1861916247,  534917982, 1117797790, 1384867540,  309721277,
       2543558140, 1622251994,  418512021, 2913307829, 3027576246,
       2920720503, 1497654794, 2022410510, 3632311723, 4156365430,
       2508188024, 2130815625, 2353238325, 1816457622, 2292736412,
       3785043472, 2728110479,  864754622, 1709978412,  833128467,
       1150076119, 1239586924, 4206981551, 1799316644, 2376337942,
       2120708859, 4066353235, 1602412123, 2317379298, 3815664964,
       1518037249,  133529045, 3450400932, 1299535805,  416854737,
        474238824, 1916414351, 3237409116,  857646267, 3610445188,
        752957829, 3327630100, 3027008281, 3428690206,  253627260,
       2167241332,  382607666, 2153482203, 3129862697, 3219968626,
       2121254630, 2951608643, 1949170423, 3786458636,  340596248,
       1026105152, 2675158762,  526038136, 3522297153, 1463534362,
        695340585,  954210010, 1583170521, 2368800757,  603708052,
       1330465769, 1949360046,  852025325,  275041373, 1784362136,
        886070820, 2448295470, 1716433547, 3104308286, 3232067626,
        147426956, 1135603570, 3583626519, 3464959294, 4129645210,
       2801940226, 2601922310, 3459726859, 1502563060,  685243120,
         15751571, 3962257555, 3205424312, 1160530268, 4102723039,
        370627734, 1135151232, 1460977026, 1684760120, 1687231092,
        204570933,  536540689, 4036693697, 1755107038]), 136, 0, 0.0),
                     1: ('MT19937', np.array([1655280075,  620203506, 2039850331,  199525209, 4191149037,
        107501193,  248933574, 3530956325, 3024222202, 1565670825,
       3541361751,  927439732, 1655898207,  714849686, 4059898989,
       2012649327, 1458023531, 1931338976, 3540332280, 4037889240,
       1678358134, 3772001662, 1150687099, 3248522305, 3270051110,
       2166970700,  398401403,  767949497, 2353616967, 3774480609,
       1254334821, 1308818412, 2145518974,  532222385,   96212495,
       1060609765, 4039815067, 3952590618, 1453650669, 3856054597,
       3290938864,  483815720,  763686221, 3281387703, 2717380631,
        703577647, 3241359213,   79311109, 4025166240, 2191907798,
       2707486281, 3340104497, 4180538216, 3762730760,  523741225,
       3419377184,  520313247, 2376752586, 1189704031, 3969270725,
        905641601, 1937577740,  567285773,  783861393, 1024386678,
       4062536639, 4044677439, 2741189521, 1083562270, 2032437999,
        525099128, 2362273294,  995273489, 3983237590,  661954631,
        853110096,  734049919, 2580675383, 1846208313,  154485268,
       2369497968, 1365649588, 4264413908, 2167484896,  344622418,
        523901983,  357033634,  790627002, 2410745456, 3679544068,
        638531138, 3752048546, 4091968601, 3820734268, 3526790842,
       3175012623, 4119898361, 4267595936, 1471108770,   93241425,
       1561924721,  787259913, 3204818180, 1798603164,  702704856,
       2657892201,  656310048, 2572204183, 3405743186, 3679944528,
       3048545806, 2364545007,  531387887, 1647722020, 3800947236,
       1391070738, 2779057503, 1204006800, 1120350246,   73069965,
       3981662264, 3246497219,  276915992,  987317986, 2521169690,
         12494005,   96300634, 2524365188, 2989400297, 3460030652,
       1573230142, 1756057937, 2930556197,  672797695, 4021937464,
       3006083417, 3555475989, 3519404033, 2325022743, 3438286824,
       1792318857, 3206052243, 2348927000, 3691675408,  569350294,
        842618130, 2404160922, 1289437611, 3806537519, 3597724168,
        285560764,  918007668, 2619423986,  894217799,  272055691,
       2575340478, 1091426641, 3290909239,  294865858,  730515774,
        764662150, 3290540916,   79763804, 2811079851,  411206173,
       1445869544, 1105011545, 3418655490,  116766801,   32841463,
        253429892,   46759110, 1942630556, 2272360702, 3447768932,
       1735200451, 3672400737, 1534149734, 2410698669, 1828922716,
       1641392055, 2492613095, 4011625038, 2348906109, 1335176015,
        518243380, 1159757363, 1954350245, 1470693168,  649635570,
        390930488, 1510145106, 2545514469, 3971224393, 1596368503,
       2310233823,  352568498, 1662246027, 1098927551, 3145471531,
       4135119112, 2423568024, 3073602608, 2878345505, 1130699580,
       4127427085,  665410745, 1312310457, 2641041691,  382263493,
        254685938,   16229627, 1029128148, 1116762332,  800477924,
       2699360253, 3362563424, 3561855555, 3695600285, 3268579624,
        465556057, 2876341822, 2284800423, 3599360147, 4266993153,
       2753215400, 4246479968, 1731287779, 4103105296, 2879632060,
       1579356622, 2473835098, 1730658375, 2570187074, 2356321361,
       1562181719, 2530686317, 1493481997, 3446674885, 2622772983,
       3799890638, 4217749196, 1479018980, 1429461018, 3249419152,
       3200015845,  894556977,  978016206, 4219714567, 1629703937,
        571348533, 1854190256, 2922639469, 2618912848, 2745952312,
       2008030000, 3060032294, 3101699173,  904076583, 1176923392,
       1716509267,  954675056, 2504220660,  915491313, 3918384553,
       1973662818, 1145605730,  190934254, 3617550429,  676532613,
       1911238043, 3141177709, 1530606454, 1919836589, 2776298898,
       2883844253, 1068053880,  840260666, 1870958426, 2176142211,
       3718962194, 1880452132, 2744839629, 3430764342, 1359252377,
       2190858977, 3530933234, 1586753814,  895618809, 2373288797,
       1362615108, 1513797099, 3771635451,  530613581,  516080470,
       2964736127, 3573189081, 3601352435, 4059255183,  237758754,
       3949584915, 3467290277, 3152538652, 2766286630, 3049115654,
       3878130689, 2985360181, 3598707911, 1994400132,  738388837,
       2557099463,  360360745, 1554189615,  162738249, 1901928589,
         26245973, 3278374735,  897954563, 1584783881,  379909198,
        706784972, 1427878384, 2008703752, 2550783264, 3851902090,
       4081729146, 2468099885, 3734803660, 3367527701, 2124908225,
       1062654835, 1551970947, 1524307287, 2174958396, 1404165662,
       3046160961, 3001370609, 2697001144,   67005173,  589062286,
        297554564, 3712760530, 1176699341, 2001015307, 1068919658,
        144280688, 3628258785, 1075741142, 1275484846,  583609669,
       2748166134, 1875939181, 1847921067,  666932547, 1418792857,
       1084256401,  485364222,  936413836, 2318559327,  945909872,
       3591884570, 1342134356,   30759196, 3505310249, 3486385395,
          7999006, 1734671186, 2698441227, 2065238285,  717325881,
       3486169393, 3732888944,  279281836, 3817253195, 4112837894,
       4096029566, 1233999878, 3721665777, 3528745853, 1071158804,
       1282271533, 2080824413,  821960812, 1321991407,  880388764,
       2077080808, 2069371158, 4241082428,  106755896, 1123605023,
       1462695753, 1088847780,  628876901, 3562406999, 1851129861,
       3909891018, 3726922261, 3261017297, 2095360054,  565959500,
       3626919747, 2095044136, 1732004003, 1279821677, 2058803915,
       3421770002, 1874157037, 4203859076, 4221791682, 4256409706,
       1592266976,    4711506, 3445879774, 2092375678,  575147433,
       1125083976,  104094574, 4241310361, 2224817435, 2992297025,
       3225059663, 1349760255, 1583487826, 2475008903, 3508790779,
        364547709, 2148861910, 4255984837, 1028355932, 1510571863,
       3150777315, 3052382602,  644320543,  429746815, 1544285209,
       3736019482, 4236850033, 3869291040, 2934522559, 4170144477,
       4262293494, 3276142707, 3924455232, 4202844918, 2974351027,
       2221791662, 2471651095,  994848575, 1508780312, 3544063636,
       1181353757, 2369413585,  576696736,  281512698, 3679769162,
       4106687234, 2450417135, 2533240268, 3060018101, 1334549538,
       2214371189, 1267442074, 1684990636,  516065705, 3599778220,
       3364397736, 3988507115, 2792865886, 2244160579, 3296795981,
       3884742118,  919716854, 4005508290, 3224235339, 3342261643,
       3925733675, 4191662037, 3599370399, 3027193310, 1635576397,
       3000373543,  921786382, 4249865146, 2728977815, 2082029426,
       1907737353, 3285939113, 3137592264, 1513561525,  310764861,
       2423676172, 2294547935, 1677681694, 2976671459, 2978226941,
        374196716, 2196333139, 2551260554, 2950661327, 2129263716,
        433182671,  988499788, 1496246722, 1455039792, 3668433562,
       1925973014, 2713183347,  966330087, 1014749422, 1560419995,
       1884129873, 1448271109, 2502771542, 3173869312,  670679789,
       1139979377,  262816214, 1077691876, 1450696586, 3348150859,
         19293693, 2561303724,  297279983,   17960688, 2266594652,
       1717834091, 1397547411, 1380237985, 3328435042, 3741379245,
       2884223488,  584972210, 3381736449, 2373912268, 3928105305,
        712131674, 2722951568, 2457374775,  367321925, 2223022777,
       4091320709, 3821596818, 1354755725,  390814787,  204794878,
        214637341, 2845540451,  577061865,  628426549, 2143182949,
       2898053094,   64445580, 4011735268, 2509183499, 4047140725,
       3003964605, 1615139055, 4148100276,  551671879, 1075519102,
       1133560206, 1403090666, 2452151194, 3514026291, 2754772108,
        642307722, 4282385421, 2169609992,  340347919, 3356142506,
        769109099, 3300932660, 3632143850, 3366619809, 3122063828,
       2384334847, 2819552319, 1267340945, 4005104469, 1790792912,
        246232304, 3986597303, 2100210965, 1211600388, 2658645759,
       2701200454, 2633699445, 1155456289, 3778853246, 4259747143,
        647630003, 2307452773, 1370805690,  689396246, 2693341129,
       2696836232, 2086833200, 2835906866, 2381880743, 2339853208,
       1232576318,   84758145, 1335306253, 3978475245, 3806351883,
       3932837824, 2021500345, 3413559859, 1394223231, 3510385892,
       2850872678, 3936099951,  108435989,  831013205, 1452317104,
       2889454330, 1240150768, 2630310815, 3986132940,  685251799,
       3067891058, 2499667518, 2832383112, 1987586607]), 52, 0, 0.0),
                     2: ('MT19937', np.array([ 485961815, 3249695217, 3776728881, 3489714065, 4012733282,
       4205869014, 1677068306, 1906036593, 4140797375, 2326131532,
       1614035608,  122504310, 2613982108, 3269245306,   77074267,
       2103093680, 2123647636,  247356557, 1870611324, 1595616109,
       2094866047,  434850869, 1955942181, 2079483742,  838614822,
        586225502, 2669599935, 2656079861, 3434578178, 2654542364,
        253202553,   65671499, 3668268169, 1268953422, 1004637775,
       3432717283, 3777317937, 2404077733, 1169402687, 2876528891,
        844631014, 1863098439, 3963314654, 1647150320,  157084454,
       3952871610, 1026692148,  756663145, 4294642045,  564487814,
       2126825540,  631953837, 1206952091, 2592213731,  735500497,
       1378636518, 3097683154, 2764939149, 2783061528, 2772194408,
       3045845510, 1820011412, 2737742656, 1419898499, 3992576849,
       2713377068, 3744819550, 1170546409,  249427485, 1303505841,
        791671377, 3787564434,  426989960, 2102327715, 4193494864,
       2397611400, 2819481553, 3018955402, 2507946628, 1919088350,
       4124298931, 4294381425, 1877621214,  144947695, 3852780793,
        658309423, 1213830925, 2655425305,  383611064,  270081297,
       4021700555, 1512455185,  739350175,  443949016, 3261062642,
       2451384559,  644772637, 1051354409, 2892474230, 3004858873,
       3283752279,  403534841, 1363227842, 1120861108,  627613367,
       1725322608, 3983998372, 1985677503, 1760945262,  664593950,
       2294471808, 3540956911,  602370937, 1881858687, 3818966271,
       2158523327, 3540940401, 3874098224,  875247428, 1732029276,
       4169240072, 2297606311, 2891161056,  146203186, 1390525052,
       2320460814, 3941939722, 4280946821, 2331822920, 2233193399,
       3943946519,  558619284, 2320263008, 4023115296, 4080746052,
       3818970015, 2116560953, 1323411414, 4218101396, 4184645086,
       4078144670, 1956029854, 2716931552,  876814009, 3869576199,
       1441269362,  438678929,  546144338, 1230966909, 3780403212,
        978168534, 1017864529, 3731710971,   36163235,  468420420,
       3845180583,  501077711, 2023482520,  944100625, 2024527660,
       3698099618,  640818533, 1298219187,  100244546, 2716873148,
       2306046296, 3154182688, 2711858530, 2181811175,  454228840,
       2037818071,  359064090, 3098874189,  624882625,  751943906,
       2914179874, 1318134830, 3867697382, 3911568366, 3602635803,
       1633335902, 1391376060,  505147767, 3012276792, 3256364035,
       2443472369, 2460588564, 3680918080,  363568062, 3627337373,
       4096539139, 2456734577, 2395663206, 2215159003, 3648669244,
       2537676114,  930863320, 1131020189, 2463481718,  854935848,
       3425067267, 2681310626,  373035655, 3498240935,  429699998,
        705757514, 2604089503,  695550577, 4243296944, 3210714586,
       1333138246, 1406373331, 1720641474, 1558426538, 1476882999,
       4148683311, 2632106505,  992569082, 4224794732, 3081893852,
       2737722324,  889193296, 1812009894, 1529854558, 2146504629,
       2159011444,  599515942, 2660471834, 1751372147, 3568965094,
       2109206498, 3176439151, 2623729446, 4037261706,  480099633,
        475002933,  640026783,  607482413,  888909364, 2987377359,
       2592461050,  291263745,  551851597, 2193431778, 2204775343,
        922228506, 3917627485, 1294550855,  654704516, 3027915453,
       3078281379, 3802302265, 1636635851,  781446067, 3691127665,
       1762931927,   66795712, 3572458446, 1820811738, 4112629953,
        242290396, 1473551370,  808823672, 1192852627, 3013125411,
        654845106, 3545205926, 3398360433, 1095181622, 4173517664,
       1547409827, 3093307658, 1867812661, 2812091866, 2223194420,
        558943238, 1040095968, 2519061036, 2026331215, 2596089102,
       1135237905, 1891878761, 4027865958,  979325719,  490393293,
        790201889, 1552229410, 3559612569,  399054512, 3113170731,
       2369002351, 2623803029,  571458204, 1580045493, 2009042865,
       2605885731,  764087843, 2157493715, 1956508100, 1166181756,
       4240058114, 2794865466, 1026319061,  304896827,  482854900,
       1995678255, 3032905319, 1420657214, 3895064640, 3453694841,
       4292898183, 2288873974, 1469208176, 4245560567, 1963017659,
        314781941,   77644182,  191767356,  627166761, 3611049259,
        961545804, 4206737409,  253912773, 4269241773,  498844059,
       4281186339,  112457022, 1129280303, 1832405584, 1224882209,
       1726471341, 4118434849, 2953771976, 1130657665,  606186532,
       3018236949, 4291056759, 3108975381,  520007341, 1213006412,
       2089608013, 1576295288, 1774127150, 3774869338,  631338715,
       3370326359,    5943232,  515545689,  530726225, 3401067750,
       3194935098,  416148782, 1450817429, 3570786350, 3590396715,
       2242923888, 2996108445, 3519437253, 2490714109,  647732919,
       4058348357, 3487572605, 1134022848, 2295927380, 4196959813,
       1777302092,  743812270, 1402570035, 3127784742,  519907772,
       4173363051, 2181711543,  220161706,  202151487, 2320466389,
       2617736072, 1135546532, 2527450747, 2000023175, 3450830030,
        685747584, 2157719847,  763775545, 3208036586,  596303748,
       2317613101, 3659383871, 3315803713,  685837596,  763323786,
       2805226222, 1091555258, 2292140453,  476089289, 2195638969,
        209881133, 3752180975, 1918195531, 1600274656,  841312033,
       4251929692, 3317548919, 1461198613, 2976940027, 1184659646,
       1160867795, 1173877033, 3447347220, 1021915536, 3201200092,
       2937387674, 2061365942, 4182046111, 2188335411,  858000552,
       1926029310, 3149526905, 3790171950, 1116982294,  573902434,
       3856990753, 4191904418, 2001481104, 2473810867, 1474204524,
       1470764001,  212549375, 1189872861, 4173838353, 1525531738,
       2428661654,  457751652, 3869920626, 1640425394, 2596574941,
       3155264645, 2539113878, 2059426383, 1338912318, 3512142963,
       3469616104, 1759221244,  763343506, 2515605164, 3011899855,
       3388336634, 3875305278, 3214287423, 2695166771, 2736407594,
       1820111311,  636141722, 1436678685, 1858305105, 3688386286,
        708025846,  717338489,  564789170,  906952158, 3538424550,
       4120393140, 2985924291, 2774497647, 1508578201, 2159405619,
       3355279101, 2708828460, 1379030956, 1258350751, 1928423796,
       1228942719, 2086418294, 1040298067, 4047899485, 1445800241,
       2585924202, 3903675563, 1618912862, 3807263069, 3603618300,
       1230062994,  823689427, 1291234635, 2210032866, 3962720002,
       1733169055, 2865594271, 1826605812, 3697995878, 2217397909,
        779464727, 3747729982, 1569644938, 3855673293, 2930721953,
       2795059648, 2702225277, 4140605224,   32950174, 2453480039,
       1226420533, 1600170378, 2103827233,  648383296,  334236169,
       4135815492, 2132461632, 2356753337, 2094302250, 1518022524,
       3722320739, 2627187287, 2019349514, 2700964784, 2957066808,
       1837586972, 2022469790, 2152498880, 3788600254,  610483476,
       2163966739, 2220202574,   54614739,  797833857,  941674346,
       2015363097,  959622845, 3189483279, 3124920638,  333939421,
       1329148079, 3306123520, 1163782211, 3398529304, 1143321511,
        454044755, 2265989396, 4229049366, 4191413836, 2449479459,
       1000566559,  879390115,  455211476,  699693482, 3970760220,
       3616089831, 1589918061, 3309266857, 2581724829, 2724944170,
       3983234462, 3841464082, 3008278201, 1235006030,  320164871,
       3562995326,   53990353, 3917512321, 1026876298, 1456064726,
       3784037618, 3727877488, 3505195461,  691722673,  271785107,
       1072832226,  318306351, 1228729649, 3542227795, 2413626555,
       1225296978, 2233937157, 1859965757, 1799438749,  237176334,
        567533740, 1010506667, 3108474242, 1657730398,  823946179,
       2845006613,  902913590, 3106631492, 4247433090,  726825704,
       1550097301, 2780698262, 2045614494, 2722875959, 2513114612,
       2051325219, 3672837848, 1038260722, 2633129741, 1016732125,
       2763176614,  933255705, 4215171254,  885439966, 3477381522,
        339207876,  724632653, 3493245823, 2515328135,  553748956,
       3510126751,  866337352,  515030454, 3257586034, 2303578241,
       4025079017, 1076746352, 3743645538, 1413660673, 2527705782,
        832934151, 3038818916, 1728609416, 1563374119, 2517861667,
       2292290314, 2804147876, 2073909175,  148280347]), 276, 0, 0.0),
                     3: ('MT19937', np.array([1405275500, 1696215139, 2160817242, 3931188011, 4056358294,
       2154463164, 1704962202, 2920510341, 1384292332, 3448258199,
        639823590, 3994991644,  398045696, 4206020700, 3447591246,
       2528599551, 1844517584, 2228530772, 2370703270, 3442254591,
       1608986533, 2888981885,  603744061, 2690942405, 4076030251,
       2973193131, 3833210346, 3066356518, 2520794136, 1079839415,
       3503675938,  187076358, 1592579522, 3553353746, 1629998734,
       1585386693, 1286646484, 2608642009,  335350048, 2967244657,
        836782456,  676848607, 2443328045,  470634664,  458398716,
       2751581763,  424891212, 1959791869, 2413639257,  593495111,
       2883332229, 1194404606, 2226335233, 2278018068, 3953951808,
       3573124044, 3924649012, 1024618945,  978610399, 1618151405,
       3756154916, 3100951575, 1439209837, 3979597311,  343852910,
       3676853027, 1578246512, 1546811539, 1570091220, 1806455470,
       3579664690, 1536241562, 2993743082,  271591238, 2768231300,
       4049631445,  179525029, 2737436227, 3265153792,  828358747,
       1407909687,  747469438, 1826040874, 1570696407, 3158196330,
       3951383115, 1457931416, 3231061880, 2354245569, 1525011066,
        377879309, 2898111190, 3253222318, 2956821554, 3905177072,
       2926453431, 1347955207, 1244942550, 2234801278, 2015389592,
       4123931782, 2027183218,  318703944, 4202805222, 4058954077,
       1158558288, 1657244003, 3439201183, 2188048348, 3305582907,
       1848150638, 2697258771,   31969270, 1138064838, 1653262482,
       2601285786, 2488947944,  264810926, 3631453923,  379422757,
       4143318870,  884486404, 2905493222, 3515000421,   65648864,
       3801131449,  117372933,  832079936, 3416060699,  956584922,
        629514587, 2626081849, 2288036136, 3664793600, 3183788162,
       1680447558, 4010233394, 2493404581, 1636944073, 2260964241,
       2652886017, 3335464712, 1491928913, 4106529278, 1816304310,
       1868118275, 4102919772, 1360628843, 2414068537,  401858887,
       1207047664, 1497980571, 2681945876, 1889545927,  491943322,
        999232808, 3893268011, 2753814852,   28224808, 3506073111,
       2008573622,  883290449,  291974169, 2109943721, 1774384393,
       1695090535,  906069626, 1691092430, 3038758011, 4175478766,
       4024399533, 3917887724, 1083118207, 1417341294, 3808824101,
       2797034603, 2822818630, 3319363875, 1144642233,  258991793,
       2912092175, 1131539817, 2103814732,  553682696,  687966952,
       4269013729, 3583129125, 3615651912, 2323521041, 1954471135,
       3840242827, 3320005111, 1179097164,   28314630, 1595047913,
       1889040987, 4141917099,  401791545, 2823687697, 2784748122,
       1225301717, 3737829169, 1775618891, 4035038028,  899843498,
       2931692269, 1732014397, 2615020616, 1765353511, 3761859996,
         46264367, 1437294977, 3543366572, 2489805314, 2504843536,
       1195882169, 1902501300,  208116325, 3971744451, 2670896210,
        471752846, 4038196880, 3314055084, 2424104271,   22789236,
       1048561886,  214874165, 3656999594, 2301454542, 3424082704,
        540161904, 2518815483,  750139221, 1777142800, 3990079182,
       3674959571,  473892635, 3791509505, 3358454824, 3694018218,
       1913992350, 2402751203, 3886186085, 2858830247,  686683113,
        332989550, 3517494818, 3265949261,  459265221, 3864324263,
       2515716658,  770739910, 2897329748, 4070862855, 2728288547,
       3831378621,  956418793,  469397582, 2843866284,  465636328,
       1355396548, 2392906156, 3006772293, 2285474992, 3925292391,
       2097584523, 2264431192, 2796151371, 1492602402, 4194174728,
       1915882539, 1966514469, 2526767877, 1129586537, 3776606137,
       3570293206, 2235938264, 1813896975, 1328115119, 3151771899,
       1896207454,  360980983, 3526457923,  569258738, 4061598431,
       3445355745,  257765464, 3180467070, 4225623568,  131924462,
       2002991901, 4202741012, 3019317012, 3331816990,  824743422,
       2631094431, 2805943100, 1654082981, 1210906076, 3644152115,
       3027200140, 4107469241, 4199330140,  408953967,  558239809,
       2499650225, 1088086261, 3770187834,   63320712, 3753037257,
       3618621358, 3849947800, 4291762346, 2568479101, 2480550892,
       3839190448, 2485104424, 1091247183, 3965389591, 1762420664,
       3916609532, 4040186377, 4127168356, 1890392205, 2903249646,
       3222243941,   56579831, 1060802266, 3865614984, 2374351569,
       1326933397, 2898342516, 2273218239, 1240464979, 4026780824,
       3996841936, 1987175547, 1594515251, 2973769009, 1556285421,
       1708189155,  897208918, 4134903928, 3034440855,  684094642,
       2104239516, 2432414695,  785836262, 4238042319, 2071128681,
        230619180, 3873203562, 3103479654, 2536249355,  429924768,
       3540304030,  849159392,  308341060, 4003395031,   76698677,
       1917223797, 2870917986, 3867319840, 2172187733, 1790816446,
       2974839020, 2578017134, 2389319242, 1054773306, 1455665361,
       3761243343,  883784521,  779848822, 1914067126, 2774936164,
       1781882579, 2367724973, 3692842630, 3983926445, 3051852624,
        791356224,  801000616, 2980177952, 4258683471, 4006156190,
       4204580152, 2725317184, 2961271160, 1713395881, 1916760343,
       3783330776, 1713951543, 2536856631,  121200770, 3753444319,
       3131145764, 1053724926, 2201656385, 1745355787, 3358922923,
       1488909128,  492970501, 1773627238, 1130840926, 2145312744,
       3168008764, 2131834080,  180160956,   67613084, 3143333104,
       2387443492, 1021230141, 1787938219, 1051790849, 1838625903,
       3160203182, 3657286757, 2655174659, 3267094678, 2794151946,
       3160636401, 1872540955,  277934574, 3975908222, 1726155097,
       1431041654, 4058659826, 2936354833,  970535512, 1490069927,
         94091863,  391266291, 2925720771, 2528602916,  198784149,
       2391870070, 2374775846,  430170835, 2901660434, 2236790882,
       1915060037, 3972674732, 2367674946, 3094267081, 2669652677,
       2432659279, 4104441308, 1040180541, 3702407529, 1268912346,
       3466423762,  259381688, 2385982258,  958340277, 3653400734,
       2647351169, 3273793272,  474932291, 2182816531,  332555804,
       2824551148, 3523028845, 3181517754,  403818768, 2811287853,
        930975823, 3155217572, 2312885095, 1972919551,  968382798,
        448352901,  972546494, 1338968155, 1064209269, 3744774549,
        915067009, 3976372290, 1268494607, 1986058869, 2935313321,
       2472234994, 3397161671,  137122402, 4221130176, 1170603654,
       3798461573, 2643634207, 4291185112, 1048282288, 2057910020,
       1934094269, 2754311304, 2319619929, 2507886180, 1404936788,
       1936219684,  874121733, 2801029016, 1633870960, 1643745497,
       1046232547, 3372386734,  886867554, 3876929772, 2989602237,
        843491477, 1296016064, 2420172972, 2172043131,  635141864,
       1442037996,  973482219, 3533540972, 3438157194,  758278149,
       3282014711,  417504798,  505747267, 3464677002,  336867233,
       1196198093, 1615305891,  793893612, 2462261339, 2153014021,
       1826521206, 3608933959, 3394077059,   98867685, 1820448502,
        662189147, 2149361247, 3037660979,  819881403,  895946838,
       2183204445,  984744542, 1636165182, 1199558978, 3699071367,
       2296575039,  222648173, 3460139312, 2198783647,  931098540,
       1721627936,  808392768,  127440504, 3920952406, 1933765200,
       2152579814, 2602669165,   68777350,  763072727, 1848429340,
       1736134839, 1700627809, 3043715502, 3402144552, 1878213128,
       4110465522,  329287990, 2604718256, 3524359148, 1173503748,
       3148685727, 2538402470,  596360870, 4146865102,  244258847,
       2109917551, 3516002263, 3850788395, 2750777297, 1417941237,
       1685113637, 2393879339, 4042034416, 1252961885,  736173014,
        799712935, 4090849601,    8645166, 1050669642, 1849797638,
       3559746583,  407763832, 2503952024,  797865133, 2148741774,
       2466540045, 2400036346, 3558457000, 1829623535, 2196442905,
         99479307, 1251070300, 1298277042, 2108000499,  218925633,
       3593614788, 1771673764, 3839006774,  556814804, 1636798903,
        987116518, 1762062935, 2057102898, 2242580358, 3026834666,
       1989127871, 1057569233, 2415266571, 2672785529, 2277045182,
       2729237769,  120320869, 2382746480,  379886484, 1017578359,
       1906548039, 3122785827, 2884293480,  389019208]), 24, 0, 0.0),
                     4: ('MT19937', np.array([3372491929, 2635326110, 3241059280, 4086508811, 1819829828,
       3610557117, 2685300798, 2861275482, 2631756241, 1314887231,
       4198292195, 1948547621, 2302882495, 2738794337,   84301716,
        285602507, 1134964742, 1388907180, 4062256299, 2572654688,
       3475973257, 3249629209,  535181219, 4047479148, 1006662873,
       2066569477, 2704042584, 3641911403,   93028343, 2332960834,
       2476209620, 2552243400, 2412444064, 2976897471, 2515290347,
       4192406733, 3451217581,  531034865, 1671049365,  712620944,
        560866050, 1578524610,  708774733,  459238334, 3541778848,
       4271476424, 1470502357,   34556253, 3670340946, 3303487872,
       3302839859,  405624739, 2673589783,  223789336,   37571180,
         66761965, 3422297741,   88008902, 3955721260, 4004944498,
       2136045657, 1427296003,  725326374,  823995427, 2660927919,
       1269581553, 1719986019, 2867590478, 1149267859, 3558027008,
       2249304019, 2776111649, 3128957087, 1260805424,  468287073,
       3040161749,   21796691, 2431234981,  565089147, 1246692539,
       2285639153, 4200447371, 2827196906,  633801096, 3005934656,
       1046227322, 2914348557, 1706515361, 3303482036,  796980526,
        803737785,  136335961,  489147316, 1263792578, 1196459347,
       2479534889, 3851257194, 1063975605, 3295177869, 4108168385,
       3974853161, 1267147228, 1585451072,  374564743, 2628188855,
       1908516503, 2017673243, 1040950108, 3799353146, 2206203521,
       1419525587,  574379262,  691468314, 2967407046,  555472180,
       2854646216, 2154317147,   49563524, 3196245684,   24672573,
       2204507339, 3462747900, 3010864656,  284479532, 2598179291,
       3407225657, 2480521430,  272179879, 1098882729, 2772520176,
       3615523680, 1779795580, 2723221934, 3139891890, 3015083200,
       2294156925, 2179914004,  491762224,  130649565, 4103368084,
        933767293, 3540149819, 1675288518, 4011985304, 1307597889,
       3940381253, 2536595105, 4111431892, 1970757974, 2588718501,
       1746396620, 2816968597, 4248384136, 1090017269, 2310444910,
       2762563300, 3790390578, 4063977651, 3810051537, 2943818376,
       3168648504, 3696172492, 1238883902, 3792088153, 3152499333,
       1520040376, 4143807177, 2456744149, 3610435961, 1571918702,
        153168324,  871085659,  579917856, 1095404084, 4080746464,
       1225229692,  676616316, 1143163124, 1465984036, 1495189522,
       1831158702, 2816949323, 2569531615, 1396074919,  531060564,
       3282111473,  300955354,  833877995, 3599068924, 2493214118,
       4267229074, 3409434773, 3289742126, 1728157999, 1496872711,
        384761656, 3555631842,    3697274,  537894422,  793732916,
       2723921689, 3785850028, 2943940749, 3603232002, 2944473667,
       3197539699, 3267320569, 1448449327, 1124799182, 2712689472,
       4072620464, 2913013547, 3052690723,  974289377,  296693176,
       1884461186, 2036919082,  228593054, 3481979518, 2071437662,
        409966262, 3505321500, 3207869318,  615189209,  266077035,
       2135452355,  682521987, 1642461215,  813953585, 2205779768,
        645893432,   23594660, 3151746337, 1709921343, 4136094459,
       2660969247, 3524369555, 2259167279, 1750814773, 3747998099,
       2949531844, 1176632085, 3320434555, 1459224217, 3409287503,
       3603318191, 1638089948, 3381083941, 4114124002, 2404281655,
       2928570117, 3241367585, 4159966156, 2842785445, 2064731097,
       4157853196, 2763797947,   50316847,  571092764, 1026297254,
       1600708964, 3596338628,  415881692, 2373163136,  757442163,
        424223858, 1959236504, 1641432716, 1461451286, 3585506304,
       2805061198, 3182615586, 3232133380, 2269426296, 2461974271,
       1173886628,  956039639, 3298506808, 2851942630, 2742868738,
       4196550725,  153459482, 2731942280, 3462181865,  797082690,
       1100138582, 3132248011,  123521097,  567136936, 2835738776,
       4220872706,  535394549, 1164161052,  631792353, 4284667307,
       4153485727, 1483489960, 2788534942, 3753796365, 2308860846,
       2473601033, 1088607217, 1131824956, 3022912689,  275955723,
       2144970467, 2693195506, 3067055864,  777075801, 1151765460,
       2270387551, 1211727778,  393966790,  452556990,  517028485,
       3576742739,  311116056, 4245746695, 4235780714, 3733985972,
        718880549, 1590151375, 1870475116,  554829441, 2058060427,
       3291071793, 2574662130,  647898856, 4064938949, 2332119249,
       1179089330, 1044120114, 1867230127, 1665972760, 3768626627,
        877897373,  474642029, 1992453403,  726287373, 1854938180,
        132345224,  137934643, 3412967297, 3876749571,  862297060,
       2748062023, 1769689218,  134644001, 2345856360,  841776864,
       4045422581, 1383056293, 4161977752,  265481798, 3430791214,
        524423509, 3673459066, 1505921772, 1246407031, 3703793548,
       2370381671, 4200943144, 2329600543, 1895584110, 1805854497,
       1220739901,  243588791, 2477735474, 4080315996, 3424140335,
       1662020430, 1974920293,  757828312,   63806316,  625092996,
       3667219500, 2744585943, 2858288981, 2215333462, 3070923658,
        311132325, 3440582594,  436607432, 4125494159, 2558605191,
        667468766, 3877684897, 1287437133, 1192487171, 1650455400,
       1780410569, 3823223478,  345051163,  424233944,  213021493,
       1410171257,  189230667, 3055575982, 3901166272,  374725295,
       3498896434, 2357453848, 4187770423, 2983197787, 4167243802,
        746867268, 3466097600, 2592236682, 1133577849, 2659332228,
       3600155972,  704706930, 1780687954,  787629964, 4161439304,
        183930929,  571765914, 1872329330, 4131034747, 3401270524,
       3716047284,  747428716, 2543171544,  474059682, 1036691621,
       3321608404,  577040185, 3882605400,  889333297, 2422890266,
       1823021193, 1382668028, 3302273945, 1241453635,  282337762,
        279900601,    5358270, 3656226546, 3872382981, 1541049533,
        812071798, 2165230179, 1047179422, 1126285997, 3845249578,
       1427615518, 2324528545, 1452783410,  661793108,  533813160,
       2239716911, 3204787793, 1861593300, 3087277719,  916077214,
       1994774424, 2988912904, 1003710078, 1732018837,  183596096,
       4012385049, 4071256964, 2299424809, 2350417567, 1880854988,
       3288943482, 1475824269, 2057833940, 1420124843, 4027528536,
       3780258361,  587621272,  970318990, 1961673359,  577682675,
       2249428827, 1823695237, 3188354811, 1556549916,  202837096,
       2379292019, 1734426466, 1275962131, 1670389225,  190931322,
       1290804223, 3615729664, 3481995602, 4167607862, 2084709925,
         11032778,  316083995, 3512214465, 2750976281, 2537954695,
       3533667649, 3309878519, 4100536786, 1433192339, 2220725697,
       2397019041, 2442071072, 3652378263,   14849502, 2965222858,
        954076850,  609864489, 3649160986, 2588385458, 3834674992,
        985048717,  361668609, 2423049906, 1972013208, 2566333969,
       2979234372, 3009269094,  605069574, 3100210520, 3711439857,
        112203401, 2522001925, 1155808571,  144262847, 2777516193,
       3130654853,  336231178, 2649270758, 2332106843, 3302025144,
       3315333894, 1431223796, 1192624544, 3861740749, 2697616165,
       1864027067, 3907059773, 3124103931, 1832262376, 3499025855,
       4020315529, 1091098986, 1432412381, 3760777908, 1203276904,
       1823508630, 1169826350, 3475519550,   58991685, 4268768504,
        347165483, 4278393592, 3792214895, 3620815093, 2175392370,
        526407083,  617166278, 3226329125, 3072759232, 4101328006,
       1643227696, 2980598315, 3385602129,  860544726, 2403956649,
       1405915512, 3102325500, 3187807754, 2903532463,  251928707,
        352511864,   11996284, 4168298533, 2322522229, 1311738609,
       3732256477, 1114943092,  443119705, 3098967499, 1077161733,
       1801263496, 2872220036, 3309374193, 3121807477,  399908544,
       1919901324, 3788331598,  586628294, 1619920784,  426476775,
        102753475, 4072822555, 1945760257, 1161170606,  253371824,
       2805238320, 2315592774, 2129566589, 3976092041, 1675706451,
       2989119877,  683593480,  728274236, 1972695164, 3886908747,
       1900248455,  874053278, 2606369852, 1322582396, 1086493237,
       3543088974, 2003317899,  726770915, 1043826895, 4041203015,
       2507354097, 2606355560,  540677909,  439643892, 1405407088,
       3586957810, 2887759163, 3028309908, 4140682456]), 360, 0, 0.0)
}

    # Use hardcoded subsets based on index
    # num_align_data = 32
    # alignment_dataset = train_data[:num_align_data]
    # alignment_datasets = [train_data[i * num_align_data: (i + 1) * num_align_data] for i in range(3)]
    # Use a dataset generated as one label for each English topname or vg_domain label
    for num_dist in [31]:
        settings.num_distractors = num_dist
        for num_examples in [10, 50, 1000]:
            for alignment_fieldname in ['topname']:  # What data we use to train the translator
                for experiment_fieldname in ['topname']:
                    for english_fieldname in ['responses']:  # What the english speaker outputs
                    # for english_fieldname in ['topname']:  # What the english speaker outputs
                        vae = VAE(512, 32)
                        vae_beta = 0.001
                        vae.load_state_dict(torch.load('saved_models/vae' + str(vae_beta) + '.pt'))
                        vae.to(settings.device)

                        # model_types = ['vq']
                        model_types = ['onehot']
                        # seeds = [i for i in range(10)]
                        seeds = [i for i in range(0, 5)]
                        # seeds = [i for i in range(5, 10)]  # For proto.
                        # kl_weights = [0.01]  # For VQ-VIB
                        kl_weights = [0.00]
                        alphas = [0.1, 0.5, 1, 1.5, 2, 3, 10, 100]  # For VQ-VIB
                        # alphas = [0.1, 1, 10, 100]
                        # alphas = [100]
                        # alphas = [2]
                        # alphas = [0.5, 1.5, 2, 3]
                        num_tokens = [1]
                        run()
