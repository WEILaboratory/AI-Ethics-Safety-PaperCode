from gekko import GEKKO
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# user-defined
from generateSynthet import generate_dataset_1, generate_dataset_2
from samplingRatCal import IterativeCal
from utils import Ua, pz_0, pz_1, Uf
from Adult import generate_Adult_sample
from COMPAS import generate_Compas_sample
from Bank import generate_bank_sample
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from scipy.optimize import minimize, differential_evolution
from scipy.optimize import NonlinearConstraint, Bounds
from on_rds import ON_RDS
import time
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier


def scipy_solve(
    D_size: None,
    u_estimate: None,
    p_z1: None,
    D_size_z1: None,
    p_z0: None,
    D_size_z0: None,
    mu_min: None,
    price: None,
    budget: None,
    providers_num: 5,
    epsilon: 0.3,
):
    # try:
    # constant
    a_0 = D_size * u_estimate
    c_0 = np.sum(mu_min * D_size * u_estimate)
    a_1 = p_z1 * D_size_z1
    c_1 = np.sum(p_z1 * mu_min * D_size_z1)
    a_2 = D_size_z1
    c_2 = np.sum(mu_min * D_size_z1)
    a_3 = p_z0 * D_size_z0
    c_3 = np.sum(p_z0 * mu_min * D_size_z0)
    a_4 = D_size_z0
    c_4 = np.sum(mu_min * D_size_z0)
    a_5 = D_size * price
    c_5 = budget - np.sum(mu_min * D_size * price)

    if c_5 <= 0:
        return np.zeros(providers_num)
    # Define the objective function
    def objective(mu_further):
        return -sum([a_0[i] * mu_further[i] for i in range(len(mu_further))]) - c_0

    # Define the first constraint
    constraint1 = lambda mu_further: abs(
        (sum([a_1[i] * mu_further[i] for i in range(len(mu_further))]) + c_1)
        / (sum([a_2[i] * mu_further[i] for i in range(len(mu_further))]) + c_2)
        - (sum([a_3[i] * mu_further[i] for i in range(len(mu_further))]) + c_3)
        / (sum([a_4[i] * mu_further[i] for i in range(len(mu_further))]) + c_4)
    )
    # Define the second constraint
    def constraint2(mu_further):
        return sum([a_5[i] * mu_further[i] for i in range(len(mu_further))])

    nlc = [
        NonlinearConstraint(constraint1, -np.inf, epsilon),
        NonlinearConstraint(constraint2, -np.inf, c_5),
    ]

    # Define the bounds for the decision variables
    bounds = Bounds(
        [0.0 for i in range(len(a_0))], [1.0 - mu_min[i] for i in range(len(a_0))]
    )

    # Define the constraints

    # Use the SLSQP optimization algorithm to solve the problem
    # res = minimize(objective, mu_further0, method='SLSQP', bounds=bounds, constraints=cons)
    res = differential_evolution(
        objective, bounds=bounds, constraints=nlc, tol=1e-7, disp=False
    )
    # Print the result
    return res.x
    # except:
    #     print("None")
    #     return None


def gekko_solve(
    D_size: None,
    u_estimate: None,
    p_z1: None,
    D_size_z1: None,
    p_z0: None,
    D_size_z0: None,
    mu_min: None,
    price: None,
    budget: None,
    providers_num: 5,
    epsilon: 0.3,
):
    try:
        # constant
        a_0 = D_size * u_estimate
        c_0 = np.sum(mu_min * D_size * u_estimate)
        a_1 = p_z1 * D_size_z1
        c_1 = np.sum(p_z1 * mu_min * D_size_z1)
        a_2 = D_size_z1
        c_2 = np.sum(mu_min * D_size_z1)
        a_3 = p_z0 * D_size_z0
        c_3 = np.sum(p_z0 * mu_min * D_size_z0)
        a_4 = D_size_z0
        c_4 = np.sum(mu_min * D_size_z0)
        a_5 = D_size * price
        c_5 = budget - np.sum(mu_min * D_size * price)

        m = GEKKO()
        m.options.MAX_ITER = 10000
        m.options.SOLVER = 1
        # m.options.OTOL=1e-10
        x = m.Array(m.Var, providers_num)

        for ii in range(providers_num):
            x[ii].lower = 0.0
            x[ii].upper = 1.0 - mu_min[ii]

        # change initial values
        # x2.value = 5; x3.value = 5
        expr1 = (np.sum(a_1 * x) + c_1) / (np.sum(a_2 * x) + c_2)
        expr2 = (np.sum(a_3 * x) + c_3) / (np.sum(a_4 * x) + c_4)

        m.Equation(abs(expr1 - expr2) <= epsilon)
        m.Equation(np.sum(a_5 * x) - c_5 <= 0)
        m.Obj(-(np.sum(a_0 * x)))
        # m.options.IMODE = 3 #steady state optimization
        m.solve(disp=False)
        # Print results
        print("Optimal solution:")

        x_result = np.zeros((1, providers_num))
        for i in range(providers_num):
            x_result[0][i] = float(x[i].value[0])
            print("x[{}]: {}".format(i, x[i].value))
        print("Objective function value: {}".format(m.options.objfcnval))

        print("NLKP solved")
        return x
    except:
        print("None")
        return None


if __name__ == "__main__":
    z_index = [-6, 4, 2]
    func = [generate_Adult_sample, generate_Compas_sample, generate_bank_sample]
    clf = [
        LogisticRegression(random_state=0),
        LogisticRegression(random_state=0),
        LogisticRegression(random_state=0),
    ]
    for kk in range(0, 1):
        print("the " + str(kk) + "-th case:")

        # D_test = generate_dataset_2(1000,  np.pi / 4) #biased
        # price=[0.5,0.5,0.5,0.5,0.5]
        price = [1, 1, 1, 1, 1]
        # budgets = [1 * D_provider_x[0].shape[0]]
        # epsilons=[1,0.5,0.1,0.01,0.001]

        ######################## OFF-KDS #######################################################
        epsilon = 1
        test_accuracy_before = []
        test_dp_before = []

        test_accuracy_after = []
        test_dp_after = []

        # coverges = [0.1, 0.3, 0.5]
        coverges = [0.1]
        for coverage in coverges:
            # dataset initialize
            (
                D_init_x,
                D_init_y,
                D_provider_x,
                D_provider_y,
                D_test_x,
                D_test_y,
            ) = func[kk](
                D_init_num=1, D_providers_num=5, add_noise_bool=True, coverage=coverage
            )

            D_init = np.concatenate((D_init_x[0], D_init_y[0]), axis=1)
            D_provider = []
            providers_num = 5
            for ii in range(providers_num):
                D_provider.append(
                    np.concatenate((D_provider_x[ii], D_provider_y[ii]), axis=1)
                )
            D_test = np.concatenate((D_test_x[0], D_test_y[0]), axis=1)  # unbiased
            budget = 1 * D_provider_x[0].shape[0]
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
            start = time.time()
            # calculate miu_min
            D_size = []
            D_size_z1 = []
            D_size_z0 = []
            miu_min = []
            D_mined = []
            Ua_list = []
            pz1_list = []
            pz0_list = []
            remained_dataset_list = D_provider

            for ii in range(providers_num):
                D_size.append(D_provider[ii].shape[0])
                (
                    min_samplingratio,
                    subset,
                    remained_dataset,
                    real_miu,
                    min_miu_record,
                    miu_a_record,
                    miu_f_record,
                    miu_pz1_record,
                    miu_pz0_record,
                ) = IterativeCal(
                    D_provider[ii], 0.1, 0.1, 1e-3, 1e-3, clf[kk].predict, z_index[kk]
                )
                miu_min.append(min_samplingratio)

                # continous sampling until miu=miu_min
                if subset.shape[0] < int(min_samplingratio * D_provider[ii].shape[0]):
                    selected_rows = np.random.choice(
                        remained_dataset.shape[0],
                        size=int(min_samplingratio * D_provider[ii].shape[0])
                        - subset.shape[0],
                        replace=True,
                    )
                    c_subset = remained_dataset[selected_rows]
                    remained_dataset = np.delete(
                        remained_dataset, selected_rows, axis=0
                    )

                    D_mined.append(np.concatenate((subset, c_subset), axis=0))

                else:
                    D_mined.append(subset)

                remained_dataset_list[ii] = remained_dataset
                Ua_estimate = Ua(D_mined[ii], clf[kk].predict, z_index[kk])
                pz1_estimate = pz_1(D_mined[ii], clf[kk].predict, z_index[kk])
                pz0_estimate = pz_0(D_mined[ii], clf[kk].predict, z_index[kk])
                Ua_list.append(Ua_estimate)
                pz1_list.append(pz1_estimate)
                pz0_list.append(pz0_estimate)
                D_size_z1.append(
                    D_provider[ii].shape[0]
                    * (np.count_nonzero(D_mined[ii][:, z_index[kk]] > 0))
                    / (
                        np.count_nonzero(D_mined[ii][:, z_index[kk]] > 0)
                        + np.count_nonzero(D_mined[ii][:, z_index[kk]] <= 0)
                    )
                )

                D_size_z0.append(
                    D_provider[ii].shape[0]
                    * (np.count_nonzero(D_mined[ii][:, z_index[kk]] <= 0))
                    / (
                        np.count_nonzero(D_mined[ii][:, z_index[kk]] > 0)
                        + np.count_nonzero(D_mined[ii][:, z_index[kk]] <= 0)
                    )
                )
            print("parameter installed")
            D_mined = np.array(D_mined)
            # change type
            D_size = np.array(D_size)
            D_size_z1 = np.array(D_size_z1)
            D_size_z0 = np.array(D_size_z0)
            miu_min = np.array(miu_min)

            Ua_list = np.array(Ua_list)
            pz1_list = np.array(pz1_list)
            pz0_list = np.array(pz0_list)
            # solve NLKP
            # mu_continue=None
            # while mu_continue==None:
            mu_continue = scipy_solve(
                D_size,
                Ua_list,
                pz1_list,
                D_size_z1,
                pz0_list,
                D_size_z0,
                miu_min,
                price,
                budget,
                providers_num,
                epsilon,
            )
            end = time.time()
            print("time:", end - start)
            # continue sampling
            repeat_times = 1
            accuracy_test = 0
            dp = 0
            for rr in range(repeat_times):
                for ii in range(providers_num):
                    selected_rows = np.random.choice(
                        remained_dataset_list[ii].shape[0],
                        size=int(mu_continue[ii] * D_provider[ii].shape[0]),
                        replace=True,
                    )
                    c_subset = remained_dataset_list[ii][selected_rows]
                    if ii < D_mined.shape[0]:
                        D_mined[ii] = np.concatenate((D_mined[ii], c_subset), axis=0)

                D_all = np.concatenate(D_mined, axis=0)
                D_all = np.concatenate((D_init, D_all), axis=0)
                # testing
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
            "record/coverage_unbias_test_accuracy_before_off" + str(kk) + ".npy",
            np.array(test_accuracy_before),
        )
        np.save(
            "record/coverage_unbias_test_dp_before_off" + str(kk) + ".npy",
            np.array(test_dp_before),
        )
        np.save(
            "record/coverage_unbias_test_accuracy_after_off" + str(kk) + ".npy",
            np.array(test_accuracy_after),
        )
        np.save(
            "record/coverage_unbias_test_dp_after_off" + str(kk) + ".npy",
            np.array(test_dp_after),
        )
        print("end")
        ########################    ON-RDS #######################################################

        test_accuracy_after = []
        test_dp_after = []

        clf[kk].fit(D_init[:, :-1], D_init[:, -1])

        for coverage in coverges:
            # dataset initialize
            (
                D_init_x,
                D_init_y,
                D_provider_x,
                D_provider_y,
                D_test_x,
                D_test_y,
            ) = func[kk](
                D_init_num=1, D_providers_num=5, add_noise_bool=True, coverage=coverage
            )

            D_init = np.concatenate((D_init_x[0], D_init_y[0]), axis=1)
            D_provider = []
            providers_num = 5
            for ii in range(providers_num):
                D_provider.append(
                    np.concatenate((D_provider_x[ii], D_provider_y[ii]), axis=1)
                )
            D_test = np.concatenate((D_test_x[0], D_test_y[0]), axis=1)  # unbiased
            budget = 1 * D_provider_x[0].shape[0]

            accuracy_test = 0
            dp = 0
            repeat_times = 1
            for rr in range(repeat_times):
                start = time.time()
                acquired_set, D_all, reward_record = ON_RDS(
                    providers_num,
                    budget,
                    price,
                    D_provider.copy(),
                    D_init,
                    clf[kk],
                    z_index[kk],
                    0.1,
                )
                end = time.time()
                print("time:", end - start)
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
            "record/coverage_unbias_test_accuracy_after_On" + str(kk) + ".npy",
            np.array(test_accuracy_after),
        )
        np.save(
            "record/coverage_unbias_test_dp_after_On" + str(kk) + ".npy",
            np.array(test_dp_after),
        )
        print("end")
