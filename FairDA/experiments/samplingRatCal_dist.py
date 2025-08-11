import numpy as np
from scipy.stats import t
from utils import Ua, pz_0, pz_1, Uf, p_z0_y0, p_z0_y1, p_z1_y0, p_z1_y1, U_eo
from generateSynthet import generate_dataset_1
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MaxNLocator
from Adult import generate_Adult_sample
from COMPAS import generate_Compas_sample
from Bank import generate_bank_sample

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import matplotlib.ticker as ticker
import random
import matplotlib

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

end_var = 0.002
initial_ratio = 0.002
sample_step = 0.002
end_ratio = 0.0005


def Cal_SamplingRatio(origin_size, dataset, epsilon, delta, func, M, z_index):
    u = func(dataset, M, z_index)
    freedom_degree = dataset.shape[0] - 1
    t_value = t.ppf(delta / 2, freedom_degree)
    A = -t_value
    S = np.sqrt(u * (1 - u))
    miu = (A * S / epsilon) ** 2 / origin_size
    return miu


def IterativeCal(dataset, epsilon_a, epsilon_f, delta_a, delta_f, M, z_index):
    origin_size = dataset.shape[0]
    selected_rows = np.random.choice(
        dataset.shape[0], size=int(initial_ratio * dataset.shape[0]), replace=True
    )
    subset = dataset[selected_rows]
    dataset = np.delete(dataset, selected_rows, axis=0)

    miu_a = Cal_SamplingRatio(origin_size, subset, epsilon_a, delta_a, Ua, M, z_index)
    # miu_pz1 = Cal_SamplingRatio(
    #     origin_size, subset, epsilon_f, delta_f / 2, pz_1, M, z_index
    # )
    # miu_pz0 = Cal_SamplingRatio(
    #     origin_size, subset, epsilon_a, delta_f / 2, pz_0, M, z_index
    # )
    # miu_f = max(miu_pz1, miu_pz0)

    miu_pz1_y1 = Cal_SamplingRatio(
        origin_size, subset, epsilon_f, delta_f / 4, p_z1_y1, M, z_index
    )
    miu_pz0_y0 = Cal_SamplingRatio(
        origin_size, subset, epsilon_a, delta_f / 4, p_z0_y0, M, z_index
    )

    miu_pz1_y0 = Cal_SamplingRatio(
        origin_size, subset, epsilon_a, delta_f / 4, p_z1_y0, M, z_index
    )
    miu_pz0_y1 = Cal_SamplingRatio(
        origin_size, subset, epsilon_a, delta_f / 4, p_z0_y1, M, z_index
    )

    miu_f = max(miu_pz1_y1, miu_pz0_y0, miu_pz0_y1, miu_pz1_y0)

    min_miu = max(miu_a, miu_f)

    last_miu_a = 0
    last_miu_f = 0

    # record
    real_miu = [initial_ratio]
    min_miu_record = [min_miu]
    miu_a_record = [miu_a]
    miu_f_record = [miu_f]

    add_miu_a = True
    add_miu_f = True

    while add_miu_a or add_miu_f:
        add_miu_a = False
        add_miu_f = False
        selected_rows = np.random.choice(
            dataset.shape[0], size=int(sample_step * dataset.shape[0]), replace=True
        )
        new_subset = dataset[selected_rows]
        dataset = np.delete(dataset, selected_rows, axis=0)

        subset = np.concatenate((subset, new_subset), axis=0)
        if abs(miu_a - last_miu_a) > end_ratio:
            if abs(miu_f - last_miu_f) > end_ratio or (
                abs(miu_f - last_miu_f) <= end_ratio and miu_a >= miu_f
            ):
                last_miu_a = miu_a
                miu_a = Cal_SamplingRatio(
                    origin_size, subset, epsilon_a, delta_a, Ua, M, z_index
                )
                miu_a_record.append(miu_a)
                add_miu_a = True

        if abs(miu_f - last_miu_f) > end_ratio:
            if abs(miu_a - last_miu_a) > end_ratio or (
                abs(miu_a - last_miu_a) <= end_ratio and miu_f >= miu_a
            ):
                last_miu_f = miu_f
                miu_pz1 = Cal_SamplingRatio(
                    origin_size, subset, epsilon_f, delta_f / 2, pz_1, M, z_index
                )
                miu_pz0 = Cal_SamplingRatio(
                    origin_size, subset, epsilon_a, delta_f / 2, pz_0, M, z_index
                )
                miu_f = max(miu_pz1, miu_pz0)
                miu_f_record.append(miu_f)

                add_miu_f = True
        real_miu.append(real_miu[-1] + sample_step)
        min_miu_record.append(max(miu_f, miu_a))

    min_samplingratio = max(miu_a, miu_f)
    return (
        min_samplingratio,
        subset,
        dataset,
        real_miu,
        min_miu_record,
        miu_a_record,
        miu_f_record,
    )


def IterativeCal_dp(dataset, epsilon_a, epsilon_f, delta_a, delta_f, M, z_index):
    origin_size = dataset.shape[0]
    selected_rows = np.random.choice(
        dataset.shape[0], size=int(initial_ratio * dataset.shape[0]), replace=True
    )
    subset = dataset[selected_rows]
    dataset = np.delete(dataset, selected_rows, axis=0)

    miu_a = Cal_SamplingRatio(origin_size, subset, epsilon_a, delta_a, Ua, M, z_index)
    miu_pz1 = Cal_SamplingRatio(
        origin_size, subset, epsilon_f, delta_f / 2, pz_1, M, z_index
    )
    miu_pz0 = Cal_SamplingRatio(
        origin_size, subset, epsilon_a, delta_f / 2, pz_0, M, z_index
    )
    miu_f = max(miu_pz1, miu_pz0)

    # miu_pz1_y1 = Cal_SamplingRatio(
    #     origin_size, subset, epsilon_f, delta_f / 4, p_z1_y1, M, z_index
    # )
    # miu_pz0_y0 = Cal_SamplingRatio(
    #     origin_size, subset, epsilon_a, delta_f / 4, p_z0_y0, M, z_index
    # )

    # miu_pz1_y0 = Cal_SamplingRatio(
    #     origin_size, subset, epsilon_a, delta_f / 4, p_z1_y0, M, z_index
    # )
    # miu_pz0_y1 = Cal_SamplingRatio(
    #     origin_size, subset, epsilon_a, delta_f / 4, p_z0_y1, M, z_index
    # )

    # miu_f = max(miu_pz1_y1, miu_pz0_y0, miu_pz0_y1, miu_pz1_y0)

    min_miu = max(miu_a, miu_f)

    last_miu_a = 0
    last_miu_f = 0

    # record
    real_miu = [initial_ratio]
    min_miu_record = [min_miu]
    miu_a_record = [miu_a]
    miu_f_record = [miu_f]

    add_miu_a = True
    add_miu_f = True

    while add_miu_a or add_miu_f:
        add_miu_a = False
        add_miu_f = False
        selected_rows = np.random.choice(
            dataset.shape[0], size=int(sample_step * dataset.shape[0]), replace=True
        )
        new_subset = dataset[selected_rows]
        dataset = np.delete(dataset, selected_rows, axis=0)

        subset = np.concatenate((subset, new_subset), axis=0)
        if abs(miu_a - last_miu_a) > end_ratio:
            if abs(miu_f - last_miu_f) > end_ratio or (
                abs(miu_f - last_miu_f) <= end_ratio and miu_a >= miu_f
            ):
                last_miu_a = miu_a
                miu_a = Cal_SamplingRatio(
                    origin_size, subset, epsilon_a, delta_a, Ua, M, z_index
                )
                miu_a_record.append(miu_a)
                add_miu_a = True

        if abs(miu_f - last_miu_f) > end_ratio:
            if abs(miu_a - last_miu_a) > end_ratio or (
                abs(miu_a - last_miu_a) <= end_ratio and miu_f >= miu_a
            ):
                last_miu_f = miu_f
                miu_pz1 = Cal_SamplingRatio(
                    origin_size, subset, epsilon_f, delta_f / 2, pz_1, M, z_index
                )
                miu_pz0 = Cal_SamplingRatio(
                    origin_size, subset, epsilon_a, delta_f / 2, pz_0, M, z_index
                )
                miu_f = max(miu_pz1, miu_pz0)
                miu_f_record.append(miu_f)

                add_miu_f = True
        real_miu.append(real_miu[-1] + sample_step)
        min_miu_record.append(max(miu_f, miu_a))

    min_samplingratio = max(miu_a, miu_f)
    return (
        min_samplingratio,
        subset,
        dataset,
        real_miu,
        min_miu_record,
        miu_a_record,
        miu_f_record,
    )


if __name__ == "__main__":
    # axs = plt.figure(figsize=(9, 6)).subplot_mosaic(
    #     [
    #         # ["adult", "compas", "bank"]
    #         ["adult_bin1", "compas_bin1", "bank_bin1"],
    #         ["adult_bin2", "compas_bin2", "bank_bin2"],
    #         ["adult_bin3", "compas_bin3", "bank_bin3"],
    #     ]
    # )
    axs = plt.figure(figsize=(10, 2)).subplot_mosaic(
        [
            # ["adult", "compas", "bank"]
            ["adult_bin1", "adult_bin2", "adult_bin3"],
        ]
    )
    # x_axis = axs["adult"].xaxis
    # x_axis.set_major_locator(plt.MaxNLocator(integer=True))
    # x_axis = axs["compas"].xaxis
    # x_axis.set_major_locator(plt.MaxNLocator(integer=True))
    # x_axis = axs["bank"].xaxis
    # x_axis.set_major_locator(plt.MaxNLocator(integer=True))

    axs["adult_bin1"].set_xlabel("$U_{Acc}$ Estimation error", fontsize=18)
    axs["adult_bin1"].set_ylabel("Counts", fontsize=18)
    axs["adult_bin1"].tick_params(axis="x", labelsize=16)
    axs["adult_bin1"].tick_params(axis="y", labelsize=16)
    # axs["compas_bin1"].set_xlabel("$U_{Acc}$ Estimation error")
    # axs["bank_bin1"].set_xlabel("$U_{Acc}$ Estimation error")

    # axs["adult_bin1"].set_title("AdultCensus", loc="center")
    # axs["compas_bin1"].set_title("COMPAS", loc="center")
    # axs["bank_bin1"].set_title("Bank", loc="center")

    axs["adult_bin2"].set_xlabel("$U_{EO}$ Estimation error", fontsize=18)
    # axs["adult_bin2"].set_ylabel("Counts",fontsize=18)
    axs["adult_bin2"].tick_params(axis="x", labelsize=16)
    axs["adult_bin2"].tick_params(axis="y", labelsize=16)
    # axs["compas_bin2"].set_xlabel("$U_{EO}$ Estimation error")
    # axs["bank_bin2"].set_xlabel("$U_{EO}$ Estimation error")

    # axs["adult_bin2"].set_title("AdultCensus", loc="center")
    # axs["compas_bin2"].set_title("COMPAS", loc="center")
    # axs["bank_bin2"].set_title("Bank", loc="center")

    axs["adult_bin3"].set_xlabel("$U_{DP}$ Estimation error", fontsize=18)
    # axs["adult_bin3"].set_ylabel("Counts",fontsize=18)
    axs["adult_bin3"].tick_params(axis="x", labelsize=16)
    axs["adult_bin3"].tick_params(axis="y", labelsize=16)
    # axs["compas_bin3"].set_xlabel("$U_{DP}$ Estimation error")
    # axs["bank_bin3"].set_xlabel("$U_{DP}$ Estimation error")

    # axs["adult_bin3"].set_title("AdultCensus", loc="center")

    # axs["compas_bin3"].set_title("COMPAS", loc="center")
    # axs["bank_bin3"].set_title("Bank", loc="center")

    # create and train SVM
    clf = LogisticRegression(random_state=0)

    # Adult
    (
        D_init_x,
        D_init_y,
        D_provider_x,
        D_provider_y,
        D_test_x,
        D_test_y,
    ) = generate_Adult_sample(D_init_num=1, D_providers_num=1, add_noise_bool=True)

    # pre-test
    clf.fit(D_init_x[0], D_init_y[0])

    new_dataset = np.concatenate((D_provider_x[0], D_provider_y[0]), axis=1)
    z_index = -6

    # real_miu_list = []
    # min_miu_record_list = []
    # for ii in range(100):
    (
        min_samplingratio,
        _,
        _,
        _,
        _,
        _,
        _,
    ) = IterativeCal(new_dataset, 0.1, 0.03, 1e-3, 1e-4, clf.predict, z_index=z_index)

    (
        min_samplingratio2,
        _,
        _,
        _,
        _,
        _,
        _,
    ) = IterativeCal_dp(
        new_dataset, 0.1, 0.03, 1e-3, 1e-4, clf.predict, z_index=z_index
    )

    #     real_miu_list.append(real_miu)
    #     min_miu_record_list.append(min_miu_record)
    # real_miu_list = np.array(real_miu_list)
    # min_miu_record_list = np.array(min_miu_record_list)

    # min_miu_record_min = np.mean(min_miu_record_list, axis=0)
    # min_miu_record_std = np.std(min_miu_record_list, axis=0)

    # axs["adult"].plot(
    #     list(range(1, len(min_miu_record) + 1)),
    #     real_miu,
    #     "-o",
    #     label="Samples that have been acquired",
    # )

    # axs["adult"].plot(
    #     list(range(1, len(min_miu_record) + 1)),
    #     min_miu_record,
    #     "-o",
    #     label="Calculated minimal sample ratio",
    # )

    Ua_success = 0
    U_dp_success = 0
    U_eo_success = 0
    Ua_error_record = []
    Udp_error_record = []
    Ueo_error_record = []
    for ii in range(1000):

        Ua_real = Ua(new_dataset, clf.predict, z_index)
        U_dp_real = Uf(new_dataset, clf.predict, z_index)
        U_eo_real = U_eo(new_dataset, clf.predict, z_index)
        selected_rows = np.random.choice(
            new_dataset.shape[0],
            size=int(min_samplingratio * new_dataset.shape[0]),
            replace=True,
        )
        subset = new_dataset[selected_rows]
        Ua_estimate = Ua(subset, clf.predict, z_index)
        U_eo_estimate = U_eo(subset, clf.predict, z_index)

        selected_rows2 = np.random.choice(
            new_dataset.shape[0],
            size=int(min_samplingratio2 * new_dataset.shape[0]),
            replace=True,
        )
        subset2 = new_dataset[selected_rows2]
        U_dp_estimate = Uf(subset2, clf.predict, z_index)

        Ua_error_record.append(abs(abs(Ua_real - Ua_estimate)))
        Udp_error_record.append(abs(abs(U_dp_real - U_dp_estimate)))
        Ueo_error_record.append(abs(abs(U_eo_real - U_eo_estimate)))
    # Ua_error_record = np.array([Ua_error_record])
    # Uf_error_record = np.array([Uf_error_record])
    offsets1 = Ua_error_record
    offsets2 = Ueo_error_record
    offsets3 = Udp_error_record
    offsets = np.concatenate((offsets1, offsets2, offsets3), axis=0).T
    axs["adult_bin1"].hist(
        offsets1,
        bins=20,
        stacked=True,
        alpha=0.5,
        color="green",
        # label=["Estimation error of $U_a$", "Estimation error of $U_f$"],
    )
    axs["adult_bin2"].hist(
        offsets2,
        bins=20,
        stacked=True,
        alpha=0.5,
        color="blue",
        # label=["Estimation error of $U_a$", "Estimation error of $U_f$"],
    )

    axs["adult_bin3"].hist(
        offsets3,
        bins=20,
        stacked=True,
        alpha=0.5,
        color="red",
        # label=["Estimation error of $U_a$", "Estimation error of $U_f$"],
    )

    # axs["adult_bin2"].hist(offsets1, bins=20, color="blue", alpha=0.5)
    # COMPAS

    (
        D_init_x,
        D_init_y,
        D_provider_x,
        D_provider_y,
        D_test_x,
        D_test_y,
    ) = generate_Compas_sample(D_init_num=1, D_providers_num=1, add_noise_bool=True)

    # pre-test
    clf.fit(D_init_x[0], D_init_y[0])

    new_dataset = np.concatenate((D_provider_x[0], D_provider_y[0]), axis=1)
    z_index = 4

    (
        min_samplingratio,
        _,
        _,
        _,
        _,
        _,
        _,
    ) = IterativeCal(new_dataset, 0.1, 0.02, 1e-3, 1e-4, clf.predict, z_index=z_index)

    (
        min_samplingratio2,
        _,
        _,
        _,
        _,
        _,
        _,
    ) = IterativeCal_dp(
        new_dataset, 0.1, 0.05, 1e-3, 1e-4, clf.predict, z_index=z_index
    )

    Ua_success = 0
    U_dp_success = 0
    U_eo_success = 0
    Ua_error_record = []
    Udp_error_record = []
    Ueo_error_record = []
    for ii in range(1000):

        Ua_real = Ua(new_dataset, clf.predict, z_index)
        U_dp_real = Uf(new_dataset, clf.predict, z_index)
        U_eo_real = U_eo(new_dataset, clf.predict, z_index)
        selected_rows = np.random.choice(
            new_dataset.shape[0],
            size=int(min_samplingratio * new_dataset.shape[0]),
            replace=True,
        )
        subset = new_dataset[selected_rows]
        Ua_estimate = Ua(subset, clf.predict, z_index)
        U_eo_estimate = U_eo(subset, clf.predict, z_index)

        selected_rows2 = np.random.choice(
            new_dataset.shape[0],
            size=int(min_samplingratio2 * new_dataset.shape[0]),
            replace=True,
        )
        subset2 = new_dataset[selected_rows2]
        U_dp_estimate = Uf(subset2, clf.predict, z_index)

        Ua_error_record.append(abs(abs(Ua_real - Ua_estimate)))
        Udp_error_record.append(abs(abs(U_dp_real - U_dp_estimate)))
        Ueo_error_record.append(abs(abs(U_eo_real - U_eo_estimate)))
    # Ua_error_record = np.array([Ua_error_record])
    # Uf_error_record = np.array([Uf_error_record])
    offsets1 = Ua_error_record
    offsets2 = Ueo_error_record
    offsets3 = Udp_error_record
    offsets = np.concatenate((offsets1, offsets2, offsets3), axis=0).T
    # axs["compas_bin1"].hist(
    #     offsets1,
    #     bins=20,
    #     stacked=True,
    #     alpha=0.5,
    #     color="green"
    #     # label=["Estimation error of $U_a$", "Estimation error of $U_f$"],
    # )
    # axs["compas_bin2"].hist(
    #     offsets2,
    #     bins=20,
    #     stacked=True,
    #     alpha=0.5,
    #     color="blue"
    #     # label=["Estimation error of $U_a$", "Estimation error of $U_f$"],
    # )
    # axs["compas_bin3"].hist(
    #     offsets3,
    #     bins=20,
    #     stacked=True,
    #     alpha=0.5,
    #     color="red"
    #     # label=["Estimation error of $U_a$", "Estimation error of $U_f$"],
    # )
    # axs["compas"].plot(
    #     list(range(1, len(min_miu_record) + 1)),
    #     real_miu,
    #     "-o",
    #     label="Samples that have been acquired",
    # )

    # axs["compas"].plot(
    #     list(range(1, len(min_miu_record) + 1)),
    #     min_miu_record,
    #     "-o",
    #     label="Calculated minimal sample ratio",
    # )

    # bank
    # clf = MLPClassifier(random_state=1, hidden_layer_sizes=200)

    (
        D_init_x,
        D_init_y,
        D_provider_x,
        D_provider_y,
        D_test_x,
        D_test_y,
    ) = generate_Compas_sample(D_init_num=1, D_providers_num=1, add_noise_bool=True)

    # pre-test
    clf.fit(D_init_x[0], D_init_y[0])

    new_dataset = np.concatenate((D_provider_x[0], D_provider_y[0]), axis=1)
    z_index = 4

    (
        min_samplingratio,
        _,
        _,
        _,
        _,
        _,
        _,
    ) = IterativeCal(new_dataset, 0.1, 0.02, 1e-3, 1e-4, clf.predict, z_index=z_index)

    (
        min_samplingratio2,
        _,
        _,
        _,
        _,
        _,
        _,
    ) = IterativeCal_dp(
        new_dataset, 0.1, 0.05, 1e-3, 1e-4, clf.predict, z_index=z_index
    )

    Ua_success = 0
    U_dp_success = 0
    U_eo_success = 0
    Ua_error_record = []
    Udp_error_record = []
    Ueo_error_record = []
    for ii in range(1000):

        Ua_real = Ua(new_dataset, clf.predict, z_index)
        U_dp_real = Uf(new_dataset, clf.predict, z_index)
        U_eo_real = U_eo(new_dataset, clf.predict, z_index)
        selected_rows = np.random.choice(
            new_dataset.shape[0],
            size=int(min_samplingratio * new_dataset.shape[0]),
            replace=True,
        )
        subset = new_dataset[selected_rows]
        Ua_estimate = Ua(subset, clf.predict, z_index)
        U_eo_estimate = U_eo(subset, clf.predict, z_index)

        selected_rows2 = np.random.choice(
            new_dataset.shape[0],
            size=int(min_samplingratio2 * new_dataset.shape[0]),
            replace=True,
        )
        subset2 = new_dataset[selected_rows2]
        U_dp_estimate = Uf(subset2, clf.predict, z_index)

        Ua_error_record.append(abs(abs(Ua_real - Ua_estimate)))
        Udp_error_record.append(abs(abs(U_dp_real - U_dp_estimate)))
        Ueo_error_record.append(abs(abs(U_eo_real - U_eo_estimate)))
    # Ua_error_record = np.array([Ua_error_record])
    # Uf_error_record = np.array([Uf_error_record])
    offsets1 = Ua_error_record
    offsets2 = Ueo_error_record
    offsets3 = Udp_error_record
    offsets = np.concatenate((offsets1, offsets2, offsets3), axis=0).T
    # axs["bank_bin1"].hist(
    #     offsets1,
    #     bins=20,
    #     stacked=True,
    #     alpha=0.5,
    #     color="green"
    #     # label=["Estimation error of $U_a$", "Estimation error of $U_f$"],
    # )
    # axs["bank_bin2"].hist(
    #     offsets2,
    #     bins=20,
    #     stacked=True,
    #     alpha=0.5,
    #     color="blue"
    #     # label=["Estimation error of $U_a$", "Estimation error of $U_f$"],
    # )
    # axs["bank_bin3"].hist(
    #     offsets3,
    #     bins=20,
    #     stacked=True,
    #     alpha=0.5,
    #     color="red"
    #     # label=["Estimation error of $U_a$", "Estimation error of $U_f$"],
    # )
    # axs["bank"].plot(
    #     list(range(1, len(min_miu_record) + 1)),
    #     real_miu,
    #     "-o",
    #     label="Samples that have \n  been acquired",
    # )

    # axs["bank"].plot(
    #     list(range(1, len(min_miu_record) + 1)),
    #     min_miu_record,
    #     "-o",
    #     label="Calculated minimal \n sample  ratio",
    # )

    # error_a_a = []
    # error_f_f = []
    # for ii in range(len(miu_a_record)):
    #     error_a = []
    #     for jj in range(20):
    #         Ua_real = Ua(new_dataset, clf.predict, 2)
    #         selected_rows = np.random.choice(
    #             new_dataset.shape[0],
    #             size=int(min_samplingratio * new_dataset.shape[0]),
    #             replace=True,
    #         )
    #         subset = new_dataset[selected_rows]
    #         Ua_estimate = Ua(subset, clf.predict, 2)
    #         error_a.append(abs(Ua_real - Ua_estimate))
    #     error_a_a.append(error_a)
    # for ii in range(len(miu_f_record)):
    #     error_f = []
    #     for jj in range(20):
    #         Uf_real = Uf(new_dataset, clf.predict, 2)
    #         selected_rows = np.random.choice(
    #             new_dataset.shape[0],
    #             size=int(min_samplingratio * new_dataset.shape[0]),
    #             replace=True,
    #         )
    #         subset = new_dataset[selected_rows]
    #         Uf_estimate = Uf(subset, clf.predict, 2)
    #         error_f.append(abs(Uf_real - Uf_estimate))
    #     error_f_f.append(error_f)
    # estimation error

    # plot
    # axs = plt.figure(figsize=(8, 4)).subplot_mosaic(
    #     [
    #         ["zoom1", "zoom2"],
    #         ["main", "main"],
    #     ]
    # )
    # axs["zoom1"].plot(list(range(1, len(miu_a_record) + 1)), miu_a_record, color="pink")
    # axs["zoom1"].set_xlabel("Iterations")
    # axs["zoom1"].set_ylabel("$\mu_{min,a}$")
    # axs["zoom2"].plot(
    #     list(range(1, len(miu_f_record) + 1)), miu_f_record, color="lightblue"
    # )
    # axs["zoom2"].set_xlabel("Iterations")
    # axs["zoom2"].set_ylabel("$\mu_{min,f}$")

    # bplot1 = axs["main"].boxplot(np.array(error_a_a).T, vert=True, patch_artist=True)
    # bplot2 = axs["main"].boxplot(np.array(error_f_f).T, vert=True, patch_artist=True)
    # colors = ["pink", "lightblue", "lightgreen"]
    # for bplot, color in zip((bplot1, bplot2), colors):
    #     for patch in bplot["boxes"]:
    #         patch.set_facecolor(color)
    # axs["main"].set_xlabel("Iterations")
    # axs["main"].set_ylabel("Estimation error")

    # plt.tight_layout()
    # plt.savefig("fig/estimation_error_syn.jpg")
    # print("end")

    # axs["zoom2"].plot(
    #     list(range(1, len(miu_f_record) + 1)), miu_f_record, color="lightblue"
    # )
    # axs["zoom2"].set_xlabel("Iterations")
    # axs["zoom2"].set_ylabel("Calculated minimal sample ratio")

    # bplot1 = axs["main"].boxplot(np.array(error_a_a).T, vert=True, patch_artist=True)
    # bplot2 = axs["main"].boxplot(np.array(error_f_f).T, vert=True, patch_artist=True)
    # colors = ["pink", "lightblue", "lightgreen"]
    # for bplot, color in zip((bplot1, bplot2), colors):
    #     for patch in bplot["boxes"]:
    #         patch.set_facecolor(color)
    # axs["main"].set_xlabel("Iterations")
    # axs["main"].set_ylabel("Estimation error")
    # axs["adult"].legend(
    #     # bbox_to_anchor=(0.2, -5, -1.0, 0.102),
    #     loc="lower left",
    #     ncol=2,
    #     borderaxespad=0.0,
    #     framealpha=1,
    # )
    # axs["adult_bin"].legend(loc="best")
    # axs["bank_bin"].legend(
    #     bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0, prop={"size": 8}
    # )
    plt.tight_layout()
    plt.savefig("fig/estimation_error_iterate2.jpg")
    plt.savefig("fig/estimation_error_iterate2.eps", format="eps")
    plt.savefig("fig/estimation_error_iterate2.svg", format="svg")
    plt.savefig("fig/estimation_error_iterate2.pdf", format="pdf")
    print("end")
