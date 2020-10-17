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
raw_file = 'Responses/outputStep0.csv'
num_emotions = 4
responses_across_datasets = []
responses_across_users_across_datasets = []
fk_scores = np.zeros((len(datasets), num_emotions * num_emotions - 1))

for d_idx, dataset in enumerate(datasets):
    emotions = []
    responses = []
    with open(os.path.join(os.path.join(data_path, dataset), raw_file)) as df:
        data = csv.reader(df)
        for r_idx, row in enumerate(data):
            if r_idx == 0:
                continue
            elif r_idx == 1:
                for entry in row[1:-1] if len(row) % 4 == 2 else row[1:]:
                    emotions.append(entry.split(' ')[-1])
            else:
                num_qs = int(len(emotions) / num_emotions)
                responses_per_user = np.zeros((num_qs, num_emotions))
                for e_idx, entry in enumerate(row[1:-1] if len(row) % 4 == 2 else row[1:]):
                    response_idx = int(np.floor(e_idx / 4))
                    if len(entry) > 0:
                        if emotions[e_idx] == 'Angry':
                            responses_per_user[response_idx, 0] = float(entry)
                        elif emotions[e_idx] == 'Happy':
                            responses_per_user[response_idx, 1] = float(entry)
                        if emotions[e_idx] == 'Neutral':
                            responses_per_user[response_idx, 2] = float(entry)
                        if emotions[e_idx] == 'Sad':
                            responses_per_user[response_idx, 3] = float(entry)
                    if e_idx % 4 == 3 and np.max(responses_per_user[response_idx]) > 0:
                        if np.sum(responses_per_user[response_idx] == responses_per_user[response_idx].max()) > 1:
                            responses_per_user[response_idx] = 0.
                            responses_per_user[response_idx, 2] = 1.
                        else:
                            max_idx = np.argmax(responses_per_user[response_idx])
                            responses_per_user[response_idx] = 0.
                            responses_per_user[response_idx, max_idx] = 1.

                responses.append(responses_per_user)
    responses = np.stack(responses)
    responses_across_users = np.sum(responses, axis=0)
    fk_scores[d_idx, 0] = fleiss_kappa(responses_across_users[:, 0:1])
    fk_scores[d_idx, 1] = fleiss_kappa(responses_across_users[:, 1:2])
    fk_scores[d_idx, 2] = fleiss_kappa(responses_across_users[:, 2:3])
    fk_scores[d_idx, 3] = fleiss_kappa(responses_across_users[:, 3:])
    fk_scores[d_idx, 4] = fleiss_kappa(responses_across_users[:, [0, 1]])
    fk_scores[d_idx, 5] = fleiss_kappa(responses_across_users[:, [0, 2]])
    fk_scores[d_idx, 6] = fleiss_kappa(responses_across_users[:, [0, 3]])
    fk_scores[d_idx, 7] = fleiss_kappa(responses_across_users[:, [1, 2]])
    fk_scores[d_idx, 8] = fleiss_kappa(responses_across_users[:, [1, 3]])
    fk_scores[d_idx, 9] = fleiss_kappa(responses_across_users[:, [2, 3]])
    fk_scores[d_idx, 10] = fleiss_kappa(responses_across_users[:, [0, 1, 2]])
    fk_scores[d_idx, 11] = fleiss_kappa(responses_across_users[:, [0, 1, 3]])
    fk_scores[d_idx, 12] = fleiss_kappa(responses_across_users[:, [0, 2, 3]])
    fk_scores[d_idx, 13] = fleiss_kappa(responses_across_users[:, [1, 2, 3]])
    fk_scores[d_idx, 14] = fleiss_kappa(responses_across_users)
    responses_across_datasets.append(responses)
    responses_across_users_across_datasets.append(responses_across_users)
temp = 1
