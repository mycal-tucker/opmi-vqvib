import csv

from src.utils.plotting import plot_multi_trials


def run():
    filename = '_'.join([str(num_align_data), str(num_tok), alignment_fieldname, experiment_fieldname, english_fieldname])
    results_dict = {}
    with open(filename + '.csv', 'r', newline='') as f:
        print("Opening", filename)
        reader = csv.reader(f, delimiter=',')
        row_idx = 0
        for row in reader:
            row_idx += 1
            if row_idx == 1:
                continue
            lambda_c, seed, comp = [float(e) for e in row[:3]]
            ec_to_eng_comm_id_mu, ec_to_eng_comm_id_std = [float(e) for e in row[3: 5]]
            ec_to_eng_mu, ec_to_eng_std = [float(e) for e in row[5: 7]]
            eng_to_ec_nosnap_mu, eng_to_ec_nosnap_std = [float(e) for e in row[7: 9]]
            eng_to_ec_snap_mu, eng_to_ec_snap_std = [float(e) for e in row[9:11]]
            tok_to_embed_r2_mu, tok_to_embed_r2_std = [float(e) for e in row[11:13]]
            embed_to_tok_r2_mu, embed_to_tok_r2_std = [float(e) for e in row[13:]]
            if lambda_c not in results_dict.keys():
                results_dict[lambda_c] = [[comp], [(ec_to_eng_comm_id_mu, ec_to_eng_comm_id_std)],
                                   [(ec_to_eng_mu, ec_to_eng_std)],
                                   [(eng_to_ec_nosnap_mu, eng_to_ec_nosnap_std)],
                                   [(eng_to_ec_snap_mu, eng_to_ec_snap_std)],
                                   [(tok_to_embed_r2_mu, tok_to_embed_r2_std)],
                                   [(embed_to_tok_r2_mu, embed_to_tok_r2_std)]]
                continue
            results_dict[lambda_c][0].append(comp)
            results_dict[lambda_c][1].append((ec_to_eng_comm_id_mu, ec_to_eng_comm_id_std))
            results_dict[lambda_c][2].append((ec_to_eng_mu, ec_to_eng_std))
            results_dict[lambda_c][3].append((eng_to_ec_nosnap_mu, eng_to_ec_nosnap_std))
            results_dict[lambda_c][4].append((eng_to_ec_snap_mu, eng_to_ec_snap_std))
            results_dict[lambda_c][5].append((tok_to_embed_r2_mu, tok_to_embed_r2_std))
            results_dict[lambda_c][6].append((embed_to_tok_r2_mu, embed_to_tok_r2_std))
    size = 20
    all_comps = []
    all_top_ec_to_eng_mu = []
    all_top_ec_to_eng_std = []
    all_top_eng_to_ec_nosnap_mu = []
    all_top_eng_to_ec_nosnap_std = []
    all_top_eng_to_ec_snap_mu = []
    all_top_eng_to_ec_snap_std = []
    all_tok_to_embed_r2_mu = []
    all_tok_to_embed_r2_std = []
    all_embed_to_tok_r2_mu = []
    all_embed_to_tok_r2_std = []
    labels = []
    for key, res in sorted(results_dict.items()):
        labels.append("$\lambda_C=$" + str(key))
        all_comps.append(res[0])
        # all_top_ec_to_eng_comm_ids.append(res[1])
        all_top_ec_to_eng_mu.append([e[0] for e in res[2]])
        all_top_ec_to_eng_std.append([e[1] for e in res[2]])
        all_top_eng_to_ec_nosnap_mu.append([e[0] for e in res[3]])
        all_top_eng_to_ec_nosnap_std.append([e[1] for e in res[3]])
        all_top_eng_to_ec_snap_mu.append([e[0] for e in res[4]])
        all_top_eng_to_ec_snap_std.append([e[1] for e in res[4]])
        all_tok_to_embed_r2_mu.append([e[0] for e in res[5]])
        all_tok_to_embed_r2_std.append([e[1] for e in res[5]])
        all_embed_to_tok_r2_mu.append([e[0] for e in res[6]])
        all_embed_to_tok_r2_std.append([e[1] for e in res[6]])
    plot_multi_trials([all_comps, all_top_ec_to_eng_mu, all_top_ec_to_eng_std],
                      labels,
                      [size for _ in labels],
                      ylabel='Utility EC to Eng Top',
                      filename='/'.join(['_'.join(
                          [str(num_align_data), str(num_tok), alignment_fieldname, experiment_fieldname,
                           english_fieldname, 'EC_to_Eng_Top'])]))
    plot_multi_trials([all_comps, all_top_eng_to_ec_nosnap_mu, all_top_eng_to_ec_nosnap_std],
                      labels,
                      [size for _ in labels],
                      ylabel='Utility Eng to EC Top No Snap',
                      filename='/'.join(['_'.join(
                          [str(num_align_data), str(num_tok), alignment_fieldname, experiment_fieldname,
                           english_fieldname, 'Eng_to_EC_Top_Nosnap'])]))
    plot_multi_trials([all_comps, all_top_eng_to_ec_snap_mu, all_top_eng_to_ec_snap_std],
                      labels,
                      [size for _ in labels],
                      ylabel='Utility Eng to EC TopSnap',
                      filename='/'.join(['_'.join(
                          [str(num_align_data), str(num_tok), alignment_fieldname, experiment_fieldname,
                           english_fieldname, 'Eng_to_EC_Top_Snap'])]))
    plot_multi_trials([all_comps, all_tok_to_embed_r2_mu, all_tok_to_embed_r2_std],
                      labels,
                      [size for _ in labels],
                      ylabel='Tok to Embed r2',
                      filename='/'.join(['_'.join(
                          [str(num_align_data), str(num_tok), alignment_fieldname, experiment_fieldname,
                           english_fieldname, 'tok_to_embed_r2'])]))
    plot_multi_trials([all_comps, all_embed_to_tok_r2_mu, all_embed_to_tok_r2_std],
                      labels,
                      [size for _ in labels],
                      ylabel='Embed to tok r2',
                      filename='/'.join(['_'.join(
                          [str(num_align_data), str(num_tok), alignment_fieldname, experiment_fieldname,
                           english_fieldname, 'embed_to_tok_r2'])]))


if __name__ == '__main__':
    num_align_data = 100
    num_tok = 8
    alignment_fieldname = 'vg_domain'
    experiment_fieldname = 'vg_domain'
    english_fieldname = 'vg_domain'
    run()