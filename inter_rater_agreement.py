import csv
import numpy as np
import os

# from statsmodels.stats.inter_rater import fleiss_kappa


def fleiss_kappa(M):
    """
    See `Fleiss' Kappa <https://en.wikipedia.org/wiki/Fleiss%27_kappa>`_.
    :param M: a matrix of shape (:attr:`N`, :attr:`k`) where `N` is the number of subjects and `k` is the number of
    categories into which assignments are made. `M[i, j]` represent the number of raters who assigned the `i`th subject
    to the `j`th category.
    :type M: numpy matrix
    """
    N, k = M.shape  # N is # of items, k is # of categories
    n_annotators = float(np.sum(M[0, :]))  # # of annotators

    p = np.sum(M, axis=0) / (N * n_annotators)
    P = (np.sum(M * M, axis=1) - n_annotators) / (n_annotators * (n_annotators - 1))
    Pbar = np.sum(P) / N
    PbarE = np.sum(p * p)

    chance_agreement = PbarE
    actual_above_chance = Pbar - PbarE
    max_above_chance = 1 - PbarE
    kappa = actual_above_chance / max_above_chance

    # return chance_agreement, actual_above_chance, max_above_chance, kappa
    return kappa if kappa < 1. else 1.


base_path = os.getcwd()
data_path = os.path.join(base_path, '../data/UserStudyLabeling')
datasets = ['BML', 'CMU', 'Human3.6M', 'ICT', 'RGB', 'SIG', 'UNC_RGB']
filter_step = 0
raw_file = 'Responses/outputStep{:d}.csv'.format(filter_step)
num_emotions = 4
num_choices = 1
happy_idx = np.arange(0, 1)
angry_idx = np.arange(1, 2)
sad_idx = np.arange(2, 3)
neutral_idx = np.arange(3, 4)
# happy_idx = np.arange(0, 5)
# angry_idx = np.arange(5, 10)
# sad_idx = np.arange(10, 15)
# neutral_idx = np.arange(15, 20)
fk_scores_all = np.zeros((len(datasets), num_emotions * num_emotions - 1))

def compute_fk_scores(responses_across_users):
    fk_scores = np.zeros(num_emotions * num_emotions - 1)
    fk_scores[0] = fleiss_kappa(responses_across_users[:, happy_idx])
    fk_scores[1] = fleiss_kappa(responses_across_users[:, angry_idx])
    fk_scores[2] = fleiss_kappa(responses_across_users[:, sad_idx])
    fk_scores[3] = fleiss_kappa(responses_across_users[:, neutral_idx])
    fk_scores[4] = fleiss_kappa(responses_across_users[:, np.concatenate((happy_idx, angry_idx))])
    fk_scores[5] = fleiss_kappa(responses_across_users[:, np.concatenate((happy_idx, sad_idx))])
    fk_scores[6] = fleiss_kappa(responses_across_users[:, np.concatenate((happy_idx, neutral_idx))])
    fk_scores[7] = fleiss_kappa(responses_across_users[:, np.concatenate((angry_idx, sad_idx))])
    fk_scores[8] = fleiss_kappa(responses_across_users[:, np.concatenate((angry_idx, neutral_idx))])
    fk_scores[9] = fleiss_kappa(responses_across_users[:, np.concatenate((sad_idx, neutral_idx))])
    fk_scores[10] = fleiss_kappa(responses_across_users[:, np.concatenate((happy_idx, angry_idx, sad_idx))])
    fk_scores[11] = fleiss_kappa(responses_across_users[:, np.concatenate((happy_idx, angry_idx, neutral_idx))])
    fk_scores[12] = fleiss_kappa(responses_across_users[:, np.concatenate((happy_idx, sad_idx, neutral_idx))])
    fk_scores[13] = fleiss_kappa(responses_across_users[:, np.concatenate((angry_idx, sad_idx, neutral_idx))])
    fk_scores[14] = fleiss_kappa(responses_across_users)
    return fk_scores

if filter_step == 0:
    responses_across_datasets = []
    responses_across_users_across_datasets = []
    for d_idx, dataset in enumerate(datasets):
        emotions = []
        responses_curr = []
        user_names = []
        with open(os.path.join(data_path, dataset, raw_file)) as df:
            data = csv.reader(df)
            for r_idx, row in enumerate(data):
                if r_idx == 0:
                    continue
                elif r_idx == 1:
                    for entry in row[1:-1] if len(row) % 4 == 2 else row[1:]:
                        emotions.append(entry.split(' ')[-1])
                else:
                    num_qs = int(len(emotions) / num_emotions)
                    responses_per_user = np.zeros((num_qs, num_emotions, num_choices))
                    user_names.append(row[0])
                    for e_idx, entry in enumerate(row[1:-1] if len(row) % 4 == 2 else row[1:]):
                        response_idx = int(np.floor(e_idx / 4))
                        # if len(entry) > 0:
                        #     if emotions[e_idx] == 'Angry':
                        #         responses_per_user[response_idx, 0, int(entry) - 1] += 1
                        #     elif emotions[e_idx] == 'Happy':
                        #         responses_per_user[response_idx, 1, int(entry) - 1] += 1
                        #     if emotions[e_idx] == 'Neutral':
                        #         responses_per_user[response_idx, 2, int(entry) - 1] += 1
                        #     if emotions[e_idx] == 'Sad':
                        #         responses_per_user[response_idx, 3, int(entry) - 1] += 1
                        # if len(entry) > 0:
                        #     if emotions[e_idx] == 'Angry':
                        #         responses_per_user[response_idx, 0] += 1
                        #     elif emotions[e_idx] == 'Happy':
                        #         responses_per_user[response_idx, 1] += 1
                        #     if emotions[e_idx] == 'Neutral':
                        #         responses_per_user[response_idx, 2] += 1
                        #     if emotions[e_idx] == 'Sad':
                        #         responses_per_user[response_idx, 3] += 1
                        if len(entry) > 0:
                            if emotions[e_idx] == 'Happy':
                                responses_per_user[response_idx, 0] = float(entry)
                            elif emotions[e_idx] == 'Angry':
                                responses_per_user[response_idx, 1] = float(entry)
                            if emotions[e_idx] == 'Sad':
                                responses_per_user[response_idx, 2] = float(entry)
                            if emotions[e_idx] == 'Neutral':
                                responses_per_user[response_idx, 3] = float(entry)
                        if e_idx % 4 == 3 and np.max(responses_per_user[response_idx]) > 0:
                            if responses_per_user[response_idx].max() - responses_per_user[response_idx].min() <= 1:
                                responses_per_user[response_idx] = 0.
                            # # max_idx = np.argwhere(responses_per_user[response_idx] == responses_per_user[response_idx].max())
                            # # responses_per_user[response_idx] = 0.
                            # # responses_per_user[response_idx, max_idx] = 1.
                            elif np.sum(responses_per_user[response_idx] == responses_per_user[response_idx].max()) > 1:
                                responses_per_user[response_idx] = 0.
                                responses_per_user[response_idx, 2] = 1.
                            else:
                                max_idx = np.argmax(responses_per_user[response_idx])
                                responses_per_user[response_idx] = 0.
                                responses_per_user[response_idx, max_idx] = 1.

                    responses_curr.append(responses_per_user)
        responses_curr = np.reshape(np.stack(responses_curr), (len(responses_curr), len(responses_curr[0]), -1))
        responses_across_users = np.sum(responses_curr, axis=0)
        fk_scores_all[d_idx] = compute_fk_scores(responses_across_users)
        responses_across_datasets.append(responses_curr)
        if len(responses_across_users_across_datasets) == 0:
            responses_across_users_across_datasets = np.copy(responses_across_users)
        else:
            responses_across_users_across_datasets = np.concatenate((responses_across_users_across_datasets, responses_across_users))
        print('{}:\t\t{}'.format(datasets[d_idx], fk_scores_all[d_idx, 14]))
elif filter_step == 1:
    responses_across_users_across_datasets = []
    for d_idx, dataset in enumerate(datasets):
        emotions = []
        qs_list = []
        responses_across_users = []
        num_responses = []
        with open(os.path.join(data_path, dataset, raw_file)) as df:
            data = csv.reader(df)
            for r_idx, row in enumerate(data):
                if r_idx == 0:
                    for entry in row[2:]:
                        emotions.append(entry.split(' ')[-1])
                else:
                    if row[1] not in qs_list:
                        qs_list.append(row[1])
                    q_idx = qs_list.index(row[1])
                    # responses_curr = [1 if float(r) > 3 else 0 for r in row[2:-1]]
                    responses_curr = [float(r) for r in row[2:-1]]
                    # responses_diff = []
                    # for i in range(len(responses_curr)):
                    #     for j in range(i + 1, len(responses_curr)):
                    #         responses_diff.append(np.abs(responses_curr[i] - responses_curr[j]))
                    # if min(responses_diff) < 1:
                    #     responses_curr = [0] * len(responses_curr)
                    if 0 <= q_idx < len(responses_across_users):
                        responses_across_users[q_idx] = [responses_across_users[q_idx][i] + responses_curr[i] for i in range(len(responses_curr))]
                        num_responses[q_idx] += 1
                    elif q_idx == len(responses_across_users):
                        responses_across_users.append(responses_curr)
                        num_responses.append(0)
        responses_across_users = np.stack(responses_across_users)
        fk_scores_all[d_idx] = compute_fk_scores(responses_across_users)
        if len(responses_across_users_across_datasets) == 0:
            responses_across_users_across_datasets = np.copy(responses_across_users)
        else:
            responses_across_users_across_datasets = np.concatenate((responses_across_users_across_datasets, responses_across_users))
        print('{}:\t\t{}'.format(datasets[d_idx], fk_scores_all[d_idx, 14]))
    stop = 1
