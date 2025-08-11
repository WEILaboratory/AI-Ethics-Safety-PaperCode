import numpy as np
from Adult import generate_Adult_sample
from COMPAS import generate_Compas_sample
from Bank import generate_bank_sample
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC


def random_sampling(D, sampling_size):
    selected_rows = np.random.choice(D.shape[0], size=int(sampling_size), replace=True)
    subset = D[selected_rows]
    D = np.delete(D, selected_rows, axis=0)
    return subset, D


def ON_RDS(providers_num, budget, price, D_providers, D_init, clf, z_index, gamma):
    action_space = range(providers_num)
    alfa = np.ones(providers_num)
    beta = np.ones(providers_num)

    S = np.zeros(providers_num)
    F = np.zeros(providers_num)

    S_last = np.zeros(providers_num)
    F_last = np.zeros(providers_num)

    theta = np.zeros(providers_num)

    db = 0.02 * budget

    subsets = [[] for i in range(providers_num)]

    D_all = D_init

    dp = 0

    reward_record = [[] for i in range(providers_num)]
    while budget > 0:
        # print("budget remaining:", budget)
        clf.fit(D_all[:, :-1], D_all[:, -1])

        for a in action_space:
            theta[a] = np.random.beta(alfa[a], beta[a])
            reward_record[a].append(theta[a])
            if D_providers[a].shape[0] < (db / price[a]):
                theta[a] = 0
        best_action = np.argmax(theta)

        new_set, D_providers[best_action] = random_sampling(
            D_providers[best_action], db / price[best_action]
        )
        subsets[best_action].append(new_set)

        nr = 0
        for ii in range(new_set.shape[0]):

            y_pred = clf.predict(new_set[ii : ii + 1, :-1])
            if y_pred[0] != new_set[ii : ii + 1, -1][0]:
                if dp < 0:
                    if new_set[ii, z_index] > 0 and y_pred[0] > 0:
                        nr += 1
                    if new_set[ii, z_index] <= 0 and y_pred[0] <= 0:
                        nr += 1
                if dp > 0:
                    if new_set[ii, z_index] > 0 and y_pred[0] <= 0:
                        nr += 1
                    if new_set[ii, z_index] <= 0 and y_pred[0] > 0:
                        nr += 1
                if dp == 0:
                    nr += 1

        for ii in range(providers_num):
            if ii == best_action:
                S[ii] = gamma * S_last[ii] + nr
                F[ii] = gamma * F_last[ii] + new_set.shape[0] - nr
            else:
                S[ii] = gamma * S_last[ii]
                F[ii] = gamma * F_last[ii]

            alfa[ii] += S[ii]
            beta[ii] += F[ii]

        S_last = S
        F_last = F

        y_pred = clf.predict(new_set[:, :-1])

        y_pred_z1 = y_pred[np.where(new_set[:, z_index] > 0)[0]]
        y_pred_z0 = y_pred[np.where(new_set[:, z_index] <= 0)[0]]
        dp = np.abs(
            np.count_nonzero(y_pred_z1 == 1)
            / (np.count_nonzero(new_set[:, z_index] > 0) + 0.001)
            - np.count_nonzero(y_pred_z0 == 1)
            / (np.count_nonzero(new_set[:, z_index] <= 0) + 0.001)
        )

        acquired_set = [
            np.concatenate(subsets[i], axis=0)
            for i in range(providers_num)
            if len(subsets[i]) != 0
        ]

        D_obtained = np.concatenate(acquired_set, axis=0)
        D_all = np.concatenate((D_init, D_obtained), axis=0)
        budget -= db

    return acquired_set, D_all, reward_record


if __name__ == "__main__":
    z_index = [-6, 4, 2]
    func = [generate_Adult_sample, generate_Compas_sample, generate_bank_sample]
    clf = [
        LogisticRegression(random_state=0),
        LogisticRegression(random_state=0),
        MLPClassifier(random_state=0),
    ]
    gamma_list = [0.1, 0.6, 0.9]

    for kk in range(0, 3):
        test_accuracy_before = []
        test_dp_before = []

        test_accuracy_after = []
        test_dp_after = []

        (D_init_x, D_init_y, D_provider_x, D_provider_y, D_test_x, D_test_y,) = func[
            kk
        ](D_init_num=1, D_providers_num=5, add_noise_bool=True)

        D_init = np.concatenate((D_init_x[0], D_init_y[0]), axis=1)
        D_provider = []
        providers_num = 5
        for ii in range(providers_num):
            D_provider.append(
                np.concatenate((D_provider_x[ii], D_provider_y[ii]), axis=1)
            )
        D_test = np.concatenate((D_test_x[0], D_test_y[0]), axis=1)

        budgets = [0.1 * D_provider_x[0].shape[0], D_provider_x[0].shape[0]]
        price = [1, 1, 1, 1, 1]

        clf[kk].fit(D_init[:, :-1], D_init[:, -1])

        # test before data acquisition
        y_pred = clf[kk].predict(D_test[:, :-1])
        accuracy_test = accuracy_score(D_test[:, -1], y_pred)

        y_pred_z1 = y_pred[np.where(D_test[:, z_index[kk]] > 0)[0]]
        y_pred_z0 = y_pred[np.where(D_test[:, z_index[kk]] <= 0)[0]]
        dp = np.abs(
            np.count_nonzero(y_pred_z1 == 1)
            / np.count_nonzero(D_test[:, z_index[kk]] > 0)
            - np.count_nonzero(y_pred_z0 == 1)
            / np.count_nonzero(D_test[:, z_index[kk]] <= 0)
        )
        test_accuracy_before.append(accuracy_test)
        test_dp_before.append(dp)
        print(
            "test before data acquisiton:  test_accuracy:",
            accuracy_test,
            "   test_dp:",
            dp,
        )
        reward_all = []
        for budget in budgets:
            accuracy_test = 0
            dp = 0
            repeat_times = 10
            for rr in range(repeat_times):
                acquired_set, D_all, reward_record = ON_RDS(
                    providers_num,
                    budget,
                    price,
                    D_provider.copy(),
                    D_init,
                    clf[kk],
                    z_index[kk],
                    0.9,
                )
                reward_all.append(reward_record)
                clf[kk].fit(D_all[:, :-1], D_all[:, -1])

                y_pred = clf[kk].predict(D_test[:, :-1])
                accuracy_test += accuracy_score(D_test[:, -1], y_pred)

                y_pred_z1 = y_pred[np.where(D_test[:, z_index[kk]] > 0)[0]]
                y_pred_z0 = y_pred[np.where(D_test[:, z_index[kk]] <= 0)[0]]
                dp += np.abs(
                    np.count_nonzero(y_pred_z1 == 1)
                    / np.count_nonzero(D_test[:, z_index[kk]] > 0)
                    - np.count_nonzero(y_pred_z0 == 1)
                    / np.count_nonzero(D_test[:, z_index[kk]] <= 0)
                )
            test_accuracy_after.append(accuracy_test / repeat_times)
            test_dp_after.append(dp / repeat_times)
            print(
                "test after data acquisiton:  test_accuracy:",
                accuracy_test / repeat_times,
                "   test_dp:",
                dp / repeat_times,
            )
        np.save(
            "record/budget_unbias_test_accuracy_before_on" + str(kk) + ".npy",
            np.array(test_accuracy_before),
        )
        np.save(
            "record/budget_unbias_test_dp_before_on" + str(kk) + ".npy",
            np.array(test_dp_before),
        )
        np.save(
            "record/budget_unbias_test_accuracy_after_on" + str(kk) + ".npy",
            np.array(test_accuracy_after),
        )
        np.save(
            "record/budget_unbias_test_dp_after_on" + str(kk) + ".npy",
            np.array(test_dp_after),
        )

        print("end")
