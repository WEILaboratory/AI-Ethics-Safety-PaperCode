import numpy as np
from scipy.stats import t
from utils import Ua, pz_0, pz_1, Uf, p_z0_y0, p_z0_y1, p_z1_y0, p_z1_y1
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


def IterativeCal_EO(dataset, epsilon_a, epsilon_f, delta_a, delta_f, M, z_index):
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


def IterativeCal_DP(dataset, epsilon_a, epsilon_f, delta_a, delta_f, M, z_index):
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
    # axs = plt.figure(figsize=(9, 3)).subplot_mosaic(
    #     [
    #         ["adult", "compas", "bank"]
    #         #  , ["adult_bin", "compas_bin", "bank_bin"]
    #     ]
    # )
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))

    axs[0].set_xlabel("Iterations", fontsize=14)
    axs[0].set_ylabel("Sampling size", fontsize=14)
    axs[0].set_title("$U_{EO}$", loc="center", fontsize=14)

    axs[1].set_xlabel("Iterations", fontsize=14)
    axs[1].set_ylabel("Sampling size", fontsize=14)
    axs[1].set_title("$U_{DP}$", loc="center", fontsize=14)

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

    (
        min_samplingratio,
        subset,
        remained_dataset,
        real_miu,
        min_miu_record,
        miu_a_record,
        miu_f_record,
    ) = IterativeCal_EO(new_dataset, 0.1, 0.1, 1e-3, 1e-3, clf.predict, z_index=z_index)
    axs[0].plot(
        list(range(1, len(min_miu_record) + 1)),
        [_ * 10000 for _ in real_miu],
        "-o",
        label="Samples that have been acquired",
    )

    axs[0].plot(
        list(range(1, len(min_miu_record) + 1)),
        [_ * 10000 for _ in min_miu_record],
        "-o",
        label="Calculated minimal sample size",
    )

    # ##############################
    # create and train SVM
    clf = LogisticRegression(random_state=0)

    # Adult
    # (
    #     D_init_x,
    #     D_init_y,
    #     D_provider_x,
    #     D_provider_y,
    #     D_test_x,
    #     D_test_y,
    # ) = generate_Adult_sample(D_init_num=1, D_providers_num=1, add_noise_bool=True)

    # pre-test
    clf.fit(D_init_x[0], D_init_y[0])

    new_dataset = np.concatenate((D_provider_x[0], D_provider_y[0]), axis=1)
    z_index = -6

    (
        min_samplingratio,
        subset,
        remained_dataset,
        real_miu,
        min_miu_record,
        miu_a_record,
        miu_f_record,
    ) = IterativeCal_DP(new_dataset, 0.1, 0.1, 1e-3, 1e-3, clf.predict, z_index=z_index)
    axs[1].plot(
        list(range(1, len(min_miu_record) + 1)),
        [_ * 10000 for _ in real_miu],
        "-o",
        label="Samples that have been acquired",
    )

    axs[1].plot(
        list(range(1, len(min_miu_record) + 1)),
        [_ * 10000 for _ in min_miu_record],
        "-o",
        label="Calculated minimal sample size",
    )

    axs[0].legend(
        loc="upper center",
        bbox_to_anchor=(1.1, 1.2),
        ncol=2,
        borderaxespad=0.0,
        prop={"size": 12},
    )
    # plt.tight_layout()
    # plt.savefig("fig/miu_min_iterate_2.jpg")
    # plt.savefig("fig/miu_min_iterate_2.eps", format="eps")
    # plt.savefig("fig/miu_min_iterate_2.svg", format="svg")
    plt.savefig("fig/miu_min_iterate.jpg")
    plt.savefig("fig/miu_min_iterate.pdf")
    plt.savefig("fig/miu_min_iterate.svg", format="svg")
    print("end")
